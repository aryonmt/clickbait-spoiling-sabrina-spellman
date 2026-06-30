import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clickbait_spoiling.data.loader import load_split, to_hf_dataset
from clickbait_spoiling.data.preprocessing import (
    build_classification_example,
    build_qa_example,
)
from clickbait_spoiling.logging_config import setup_logging
from clickbait_spoiling.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build and serialize Arrow formatted HF datasets from raw files."
    )
    parser.add_argument(
        "--train_raw",
        type=str,
        default="data/raw/training.jsonl",
        help="Path to training jsonl.",
    )
    parser.add_argument(
        "--val_raw",
        type=str,
        default="data/raw/validation.jsonl",
        help="Path to validation jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Target serialization directory.",
    )
    args = parser.parse_args()

    setup_logging("outputs/logs", "build_processed_dataset")
    logger.info("Starting processing and serialization of Hugging Face datasets...")

    train_path = Path(args.train_raw)
    val_path = Path(args.val_raw)

    if not train_path.exists() or not val_path.exists():
        logger.error(
            "Raw training/validation files not found. Run scripts/download_data.py first."
        )
        sys.exit(1)

    # 1. Load splits
    train_posts = load_split(train_path)
    val_posts = load_split(val_path)

    # 2. Build Task 1 Classification Datasets
    logger.info("Processing Task 1 Classification datasets...")
    train_task1_list = [build_classification_example(p) for p in train_posts]
    val_task1_list = [build_classification_example(p) for p in val_posts]

    train_task1_ds = to_hf_dataset(train_posts).map(
        lambda x, idx: train_task1_list[idx], with_indices=True
    )
    val_task1_ds = to_hf_dataset(val_posts).map(
        lambda x, idx: val_task1_list[idx], with_indices=True
    )

    # 3. Build Task 2 Extractive QA Datasets
    logger.info("Processing Task 2 Extractive QA datasets...")
    # Alignment-filtered records
    train_task2_processed = []
    val_task2_processed = []

    for p in train_posts:
        ex = build_qa_example(p)
        if ex is not None:
            train_task2_processed.append(ex)

    for p in val_posts:
        ex = build_qa_example(p)
        if ex is not None:
            val_task2_processed.append(ex)

    train_task2_ds = to_hf_dataset(train_posts).filter(
        lambda x, idx: build_qa_example(train_posts[idx]) is not None, with_indices=True
    )
    val_task2_ds = to_hf_dataset(val_posts).filter(
        lambda x, idx: build_qa_example(val_posts[idx]) is not None, with_indices=True
    )

    # Map the customized examples onto the datasets
    train_task2_ds = train_task2_ds.map(
        lambda x, idx: train_task2_processed[idx], with_indices=True
    )
    val_task2_ds = val_task2_ds.map(
        lambda x, idx: val_task2_processed[idx], with_indices=True
    )

    # 4. Serialize to Disk
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    task1_train_path = out_dir / "task1" / "train"
    task1_val_path = out_dir / "task1" / "validation"
    task2_train_path = out_dir / "task2" / "train"
    task2_val_path = out_dir / "task2" / "validation"

    train_task1_ds.save_to_disk(task1_train_path)
    val_task1_ds.save_to_disk(task1_val_path)
    train_task2_ds.save_to_disk(task2_train_path)
    val_task2_ds.save_to_disk(task2_val_path)

    logger.info(f"Task 1 datasets successfully saved to {out_dir / 'task1'}")
    logger.info(
        f"Task 2 datasets (aligned train count={len(train_task2_ds)}) saved to {out_dir / 'task2'}"
    )


if __name__ == "__main__":
    main()
