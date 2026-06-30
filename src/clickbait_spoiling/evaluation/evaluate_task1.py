import argparse
import json
import logging
import os

import pandas as pd

from clickbait_spoiling.data.loader import load_split
from clickbait_spoiling.evaluation.metrics import classification_report_dict
from clickbait_spoiling.inference.predict_classifier import predict_spoiler_types
from clickbait_spoiling.logging_config import setup_logging
from clickbait_spoiling.models.classification_model import build_classifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Task 1 Spoiler Type Classifier."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained classifier checkpoint.",
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to raw gold validation.jsonl."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save metric files."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device (cpu or cuda)."
    )
    args = parser.parse_args()

    setup_logging(args.output_dir, "evaluate_task1")
    logger.info("Initializing Task 1 Evaluation...")

    posts = load_split(args.val_path)

    # Filter only posts containing valid gold labels
    valid_posts = [p for p in posts if p.gold_tag() is not None]
    y_true = [p.gold_tag() for p in valid_posts]

    model, tokenizer = build_classifier(args.model_dir, num_labels=3)
    predictions_dict = predict_spoiler_types(
        model, tokenizer, valid_posts, device=args.device
    )

    y_pred = [predictions_dict[p.uuid] for p in valid_posts]

    results = classification_report_dict(y_true, y_pred)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save raw JSON classification report
    report_path = os.path.join(args.output_dir, "task1_classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results["report"], f, indent=4)

    # Save Confusion Matrix as CSV
    cm_path = os.path.join(args.output_dir, "task1_confusion_matrix.csv")
    cm_df = pd.DataFrame(
        results["confusion_matrix"],
        index=["phrase", "passage", "multi"],
        columns=["phrase", "passage", "multi"],
    )
    cm_df.to_csv(cm_path)

    logger.info(
        f"Task 1 Eval Completed. Macro F1: {results['report']['macro avg']['f1-score']:.4f} | Accuracy: {results['report']['accuracy']:.4f}"
    )
    logger.info(f"Classification report exported to: {report_path}")


if __name__ == "__main__":
    main()
