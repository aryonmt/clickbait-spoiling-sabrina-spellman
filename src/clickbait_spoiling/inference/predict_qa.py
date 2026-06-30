import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import datasets
from clickbait_spoiling.schema import ClickbaitPost
from clickbait_spoiling.data.preprocessing import build_qa_example
from clickbait_spoiling.data.tokenization import prepare_validation_features
from clickbait_spoiling.postprocessing.span_selector import get_n_best_spans


def predict_qa_logits(
    model,
    tokenizer,
    posts: List[ClickbaitPost],
    device: str,
    max_length: int,
    doc_stride: int,
    batch_size: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """Batched QA inference returning maximum scoring logits and mappings per post uuid."""
    model.to(device)
    model.eval()

    # Create empty mock annotations where answers are not available (inference)
    qa_examples = []
    for post in posts:
        example = build_qa_example(post)
        if example is None:
            example = {
                "uuid": post.uuid,
                "question": post.joined_post_text(),
                "context": "",
                "answers": {"answer_start": [], "text": []},
            }
        qa_examples.append(example)

    dataset = datasets.Dataset.from_list(qa_examples)

    # Prepare features
    tokenized_dataset = dataset.map(
        lambda ex: prepare_validation_features(ex, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=dataset.column_names,
    )

    results: Dict[str, Dict[str, Any]] = {}
    uuid_to_context = {ex["uuid"]: ex["context"] for ex in qa_examples}
    uuid_to_post_text = {ex["uuid"]: ex["question"] for ex in qa_examples}

    for idx in tqdm(
        range(0, len(tokenized_dataset), batch_size), desc="Predicting QA Spans"
    ):
        batch = tokenized_dataset[idx : idx + batch_size]

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)

        kwargs = {}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        for j in range(len(batch["input_ids"])):
            uuid = batch["example_id"][j]
            offsets = batch["offset_mapping"][j]

            cleaned_offsets = [tuple(o) if o is not None else None for o in offsets]

            s_log = start_logits[j]
            e_log = end_logits[j]

            context = uuid_to_context[uuid]

            # Find candidate scores in this sliding window
            spans = get_n_best_spans(
                s_log,
                e_log,
                cleaned_offsets,
                context,
                n_best_size=20,
                max_answer_length=64,
            )
            best_score = spans[0].score if spans else -9999.0

            # If this sliding window has a higher score than previously evaluated windows of the same UUID
            if uuid not in results or best_score > results[uuid]["best_score"]:
                results[uuid] = {
                    "start_logits": s_log,
                    "end_logits": e_log,
                    "offset_mapping": cleaned_offsets,
                    "context": context,
                    "post_text": uuid_to_post_text[uuid],
                    "best_score": best_score,
                }

    # Ensure all inputs contain some valid structural output arrays
    for post in posts:
        if post.uuid not in results:
            context, _ = post.joined_context()
            results[post.uuid] = {
                "start_logits": np.zeros(max_length),
                "end_logits": np.zeros(max_length),
                "offset_mapping": [None] * max_length,
                "context": context,
                "post_text": post.joined_post_text(),
                "best_score": -9999.0,
            }

    return results
