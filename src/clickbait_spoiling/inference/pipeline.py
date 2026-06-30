import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from clickbait_spoiling.inference.predict_classifier import predict_spoiler_types
from clickbait_spoiling.inference.predict_qa import predict_qa_logits
from clickbait_spoiling.models.classification_model import build_classifier
from clickbait_spoiling.models.qa_model import build_qa_model
from clickbait_spoiling.nlp.similarity import SpanSimilarityScorer
from clickbait_spoiling.postprocessing import postprocess
from clickbait_spoiling.schema import ClickbaitPost

logger = logging.getLogger(__name__)


def run_pipeline(
    posts: List[ClickbaitPost],
    classifier_dir: Optional[str],
    qa_model_dirs: Dict[
        str, str
    ],  # {"phrase": path, "passage": path, "multi": path} OR {"all": path}
    output_path: str,
    use_gold_tags: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """End-to-end pipeline: predicting spoiler types, routing to QA models, postprocessing and saving."""
    # 1. Routing classification
    if use_gold_tags:
        logger.info("Using gold tags for Task 1 routing.")
        predicted_types = {
            post.uuid: post.gold_tag() for post in posts if post.gold_tag() is not None
        }
        for post in posts:
            if post.uuid not in predicted_types:
                predicted_types[post.uuid] = "phrase"
    else:
        if not classifier_dir:
            raise ValueError(
                "Classifier directory must be provided if use_gold_tags is False."
            )
        logger.info(f"Loading classifier from {classifier_dir}")
        model, tokenizer = build_classifier(classifier_dir, num_labels=3)
        predicted_types = predict_spoiler_types(model, tokenizer, posts, device=device)

    # Route posts to correct buckets
    posts_by_type: Dict[str, List[ClickbaitPost]] = {
        "phrase": [],
        "passage": [],
        "multi": [],
    }
    for post in posts:
        tag = predicted_types.get(post.uuid, "phrase")
        posts_by_type[tag].append(post)

    qa_results: Dict[str, dict] = {}
    similarity_scorer = SpanSimilarityScorer(method="combined")

    # 2. QA logits prediction
    if "all" in qa_model_dirs:
        universal_dir = qa_model_dirs["all"]
        logger.info(f"Running universal QA model from: {universal_dir}")
        model, tokenizer = build_qa_model(universal_dir)
        qa_results.update(
            predict_qa_logits(
                model, tokenizer, posts, device=device, max_length=384, doc_stride=128
            )
        )
    else:
        for tag, subtype_posts in posts_by_type.items():
            if not subtype_posts:
                continue
            model_dir = qa_model_dirs.get(tag)
            if not model_dir:
                model_dir = list(qa_model_dirs.values())[0]
                logger.warning(
                    f"Specialized model for '{tag}' not found. Falling back to {model_dir}"
                )

            logger.info(
                f"Running specialized QA model for '{tag}' (count={len(subtype_posts)}) from: {model_dir}"
            )
            model, tokenizer = build_qa_model(model_dir)
            qa_results.update(
                predict_qa_logits(
                    model,
                    tokenizer,
                    subtype_posts,
                    device=device,
                    max_length=384,
                    doc_stride=128,
                )
            )

    # 3. Postprocess & Write outputs
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing final prediction records to {output_path}")
    counts = {"phrase": 0, "passage": 0, "multi": 0}
    empty_spoilers_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for post in posts:
            uuid = post.uuid
            tag = predicted_types.get(uuid, "phrase")

            res = qa_results.get(uuid)
            if res is None:
                spoiler_text = ""
            else:
                article_text = "\n".join(post.target_paragraphs)
                spoiler_text = postprocess(
                    spoiler_type=tag,
                    start_logits=res["start_logits"],
                    end_logits=res["end_logits"],
                    offset_mapping=res["offset_mapping"],
                    context=res["context"],
                    post_text=res["post_text"],
                    article_text=article_text,
                    similarity_scorer=similarity_scorer,
                )

            if not spoiler_text.strip():
                empty_spoilers_count += 1

            counts[tag] += 1

            record = {"uuid": uuid, "spoilerType": tag, "spoiler": spoiler_text}
            out_f.write(json.dumps(record) + "\n")

    logger.info(f"Pipeline executed successfully. Outputs distribution: {counts}")
    if empty_spoilers_count > 0:
        logger.warning(
            f"Extracted {empty_spoilers_count} empty spoilers. Check log warnings for details."
        )
