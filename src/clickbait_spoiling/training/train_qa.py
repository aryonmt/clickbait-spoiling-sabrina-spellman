import argparse
import logging
import os

import datasets
import yaml
from transformers import Trainer

from clickbait_spoiling.data.loader import load_split
from clickbait_spoiling.data.preprocessing import build_qa_example
from clickbait_spoiling.data.tokenization import prepare_train_features
from clickbait_spoiling.logging_config import setup_logging
from clickbait_spoiling.models.qa_model import build_qa_model
from clickbait_spoiling.training.callbacks import (
    CheckpointDiskBudgetCallback,
    GpuMemoryLoggingCallback,
)
from clickbait_spoiling.training.trainer_utils import (
    build_trainer_kwargs,
    get_training_args,
)
from clickbait_spoiling.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Task 2 Extractive QA Model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file."
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to raw training.jsonl."
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to raw validation.jsonl."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save checkpoints."
    )
    parser.add_argument(
        "--spoiler_type",
        type=str,
        default="all",
        choices=["phrase", "passage", "multi", "all"],
        help="Filter dataset to specific spoiler type to build specialized sub-models.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg.get("log_dir", "outputs/logs"), f"train_qa_{args.spoiler_type}")
    logger.info(
        f"Initializing Task 2 extractive QA model training (Mode: {args.spoiler_type})..."
    )

    set_global_seed(cfg.get("seed", 42))

    train_posts = load_split(args.train_path)
    val_posts = load_split(args.val_path)

    if args.spoiler_type != "all":
        train_posts = [p for p in train_posts if p.gold_tag() == args.spoiler_type]
        val_posts = [p for p in val_posts if p.gold_tag() == args.spoiler_type]
        logger.info(
            f"Filtered datasets. Train: {len(train_posts)} posts, Val: {len(val_posts)} posts."
        )
    else:
        logger.info(
            f"Using full dataset. Train: {len(train_posts)} posts, Val: {len(val_posts)} posts."
        )

    train_data = [build_qa_example(p) for p in train_posts]
    val_data = [build_qa_example(p) for p in val_posts]

    train_data = [ex for ex in train_data if ex is not None]
    val_data = [ex for ex in val_data if ex is not None]

    train_ds = datasets.Dataset.from_list(train_data)
    val_ds = datasets.Dataset.from_list(val_data)

    model, tokenizer = build_qa_model(cfg.get("model_name", "deberta-base"))

    max_len = cfg.get("max_seq_length", 384)
    stride = cfg.get("doc_stride", 128)

    train_tokenized = train_ds.map(
        lambda ex: prepare_train_features(ex, tokenizer, max_len, stride),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tokenized = val_ds.map(
        lambda ex: prepare_train_features(ex, tokenizer, max_len, stride),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    training_args = get_training_args(cfg, args.output_dir)

    # Build compatible trainer arguments
    base_kwargs = build_trainer_kwargs(
        model=model,
        training_args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        callbacks=[CheckpointDiskBudgetCallback(), GpuMemoryLoggingCallback()],
    )

    trainer = Trainer(**base_kwargs)

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    logger.info("Task 2 Extractive QA model training completed successfully.")


if __name__ == "__main__":
    main()
