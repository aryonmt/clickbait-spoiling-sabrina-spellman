import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clickbait_spoiling.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_command(command: list):
    """Safely run subprocess tracking output streams."""
    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"Command execution failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Chained High-Level Clickbait Spoiling Evaluation pipeline."
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/raw/validation.jsonl",
        help="Validation raw records.",
    )
    parser.add_argument(
        "--classifier_dir", type=str, default=None, help="Classifier model checkpoint."
    )
    parser.add_argument("--qa_phrase_dir", type=str, default=None)
    parser.add_argument("--qa_passage_dir", type=str, default=None)
    parser.add_argument("--qa_multi_dir", type=str, default=None)
    parser.add_argument("--qa_universal_dir", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pipeline_run",
        help="Metrics output target.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    setup_logging(args.output_dir, "run_full_pipeline")
    logger.info("Initializing full chained pipeline execution...")

    # 1. Run Task 1 Evaluation if classifier is provided
    if args.classifier_dir:
        logger.info("Step 1: Starting Task 1 Classifier Evaluation...")
        t1_command = [
            "python",
            "-m",
            "clickbait_spoiling.evaluation.evaluate_task1",
            "--model_dir",
            args.classifier_dir,
            "--val_path",
            args.val_path,
            "--output_dir",
            args.output_dir,
            "--device",
            args.device,
        ]
        run_command(t1_command)
    else:
        logger.warning(
            "Classifier directory omitted. Task 1 evaluation step skipped. Gold tags routing will be applied."
        )

    # 2. Run Task 2 / End-to-end Generation Pipeline Evaluation
    logger.info("Step 2: Starting Task 2 End-to-End Generation Evaluation...")
    t2_command = [
        "python",
        "-m",
        "clickbait_spoiling.evaluation.evaluate_task2",
        "--val_path",
        args.val_path,
        "--output_dir",
        args.output_dir,
        "--device",
        args.device,
    ]

    if args.classifier_dir:
        t2_command.extend(["--classifier_dir", args.classifier_dir])
    if args.qa_phrase_dir:
        t2_command.extend(["--qa_phrase_dir", args.qa_phrase_dir])
    if args.qa_passage_dir:
        t2_command.extend(["--qa_passage_dir", args.qa_passage_dir])
    if args.qa_multi_dir:
        t2_command.extend(["--qa_multi_dir", args.qa_multi_dir])
    if args.qa_universal_dir:
        t2_command.extend(["--qa_universal_dir", args.qa_universal_dir])

    run_command(t2_command)
    logger.info("Chained Clickbait Spoiling Pipeline finished successfully.")


if __name__ == "__main__":
    main()
