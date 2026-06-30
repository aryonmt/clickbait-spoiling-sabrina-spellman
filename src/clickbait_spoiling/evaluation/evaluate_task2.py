import argparse
import json
import logging
import os

import pandas as pd

from clickbait_spoiling.data.loader import load_split
from clickbait_spoiling.evaluation.metrics import bertscore_batch, bleu4, meteor
from clickbait_spoiling.inference.pipeline import run_pipeline
from clickbait_spoiling.logging_config import setup_logging
from clickbait_spoiling.nlp.enumeration import detect_enumeration_question

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 Spoiler Generation.")
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to raw validation.jsonl."
    )
    parser.add_argument(
        "--classifier_dir",
        type=str,
        default=None,
        help="Classifier path (If omitted, uses gold Task 1 tags).",
    )
    parser.add_argument(
        "--qa_phrase_dir",
        type=str,
        default=None,
        help="Specialized phrase QA checkpoint.",
    )
    parser.add_argument(
        "--qa_passage_dir",
        type=str,
        default=None,
        help="Specialized passage QA checkpoint.",
    )
    parser.add_argument(
        "--qa_multi_dir",
        type=str,
        default=None,
        help="Specialized multi QA checkpoint.",
    )
    parser.add_argument(
        "--qa_universal_dir",
        type=str,
        default=None,
        help="Universal QA model path (Fallback option).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save metric outputs."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device (cpu or cuda)."
    )
    args = parser.parse_args()

    setup_logging(args.output_dir, "evaluate_task2")
    logger.info("Initializing Task 2 Generation Evaluation...")

    posts = load_split(args.val_path)

    qa_model_dirs = {}
    if args.qa_universal_dir:
        qa_model_dirs["all"] = args.qa_universal_dir
    else:
        if args.qa_phrase_dir:
            qa_model_dirs["phrase"] = args.qa_phrase_dir
        if args.qa_passage_dir:
            qa_model_dirs["passage"] = args.qa_passage_dir
        if args.qa_multi_dir:
            qa_model_dirs["multi"] = args.qa_multi_dir

    if not qa_model_dirs:
        raise ValueError(
            "Either --qa_universal_dir or at least one sub-model directory must be supplied."
        )

    prediction_output = os.path.join(args.output_dir, "eval_predictions.jsonl")
    use_gold_tags = args.classifier_dir is None

    # Run full prediction pipeline
    run_pipeline(
        posts=posts,
        classifier_dir=args.classifier_dir,
        qa_model_dirs=qa_model_dirs,
        output_path=prediction_output,
        use_gold_tags=use_gold_tags,
        device=args.device,
    )

    # Read predictions back
    predictions = {}
    with open(prediction_output, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                predictions[record["uuid"]] = record

    refs = []
    hyps = []

    for post in posts:
        if post.uuid in predictions:
            # Join multiple gold target answers if present for evaluation compatibility
            ref = " ".join(post.spoiler) if post.spoiler else ""
            hyp = predictions[post.uuid]["spoiler"]
            refs.append(ref)
            hyps.append(hyp)

    logger.info("Computing sentence-level BLEU-4 and METEOR...")
    bleu_scores = [bleu4(r, h) for r, h in zip(refs, hyps)]
    meteor_scores = [meteor(r, h) for r, h in zip(refs, hyps)]

    logger.info("Computing BERTScore F1 metric (batched)...")
    bert_f1_scores = bertscore_batch(refs, hyps)

    # Build metric dataset for breakdown
    data = []
    for i, post in enumerate(posts):
        uuid = post.uuid
        if uuid not in predictions:
            continue
        tag = post.gold_tag()
        pred_tag = predictions[uuid]["spoilerType"]

        # Check sub-types to differentiate enumeration listicles from standard multi spans
        is_enum = detect_enumeration_question(post.joined_post_text()) is not None
        sub_type = tag
        if tag == "multi":
            sub_type = "multi-enumeration" if is_enum else "multi-iterative"

        data.append(
            {
                "uuid": uuid,
                "gold_tag": tag,
                "predicted_tag": pred_tag,
                "sub_type": sub_type,
                "bleu4": bleu_scores[i],
                "meteor": meteor_scores[i],
                "bert_score": bert_f1_scores[i] if i < len(bert_f1_scores) else 0.0,
            }
        )

    df = pd.DataFrame(data)

    # Save individual scores
    scores_csv = os.path.join(args.output_dir, "task2_individual_scores.csv")
    df.to_csv(scores_csv, index=False)

    # Compute aggregates
    summary_rows = []
    summary_rows.append(
        {
            "Group": "Overall",
            "Count": len(df),
            "BLEU-4": df["bleu4"].mean(),
            "METEOR": df["meteor"].mean(),
            "BERTScore": df["bert_score"].mean(),
        }
    )

    # Grouped aggregates
    for g_name, group_df in df.groupby("sub_type"):
        summary_rows.append(
            {
                "Group": f"Type: {g_name}",
                "Count": len(group_df),
                "BLEU-4": group_df["bleu4"].mean(),
                "METEOR": group_df["meteor"].mean(),
                "BERTScore": group_df["bert_score"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "task2_summary_report.csv")
    summary_df.to_csv(summary_path, index=False)

    # Output clean results representation to log stream
    logger.info(
        "\n"
        + "=" * 80
        + "\nTASK 2 GENERATION COMPREHENSIVE EVALUATION SUMMARY\n"
        + "=" * 80
    )
    logger.info(
        "\n"
        + (
            summary_df.to_markdown(index=False)
            if hasattr(summary_df, "to_markdown")
            else str(summary_df)
        )
    )
    logger.info(f"\nBreakdown logs exported successfully to: {summary_path}")


if __name__ == "__main__":
    main()
