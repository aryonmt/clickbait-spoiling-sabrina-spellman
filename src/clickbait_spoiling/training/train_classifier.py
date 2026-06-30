import argparse
import logging
import os

import datasets
import yaml
from transformers import Trainer

from clickbait_spoiling.data.loader import load_split
from clickbait_spoiling.data.preprocessing import build_classification_example
from clickbait_spoiling.logging_config import setup_logging
from clickbait_spoiling.models.classification_model import build_classifier
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
    parser = argparse.ArgumentParser(
        description="Train Task 1 Spoiler Type Classifier."
    )
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
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoint artifacts.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg.get("log_dir", "outputs/logs"), "train_classifier")
    logger.info("Initializing Task 1 Sequence Classifier Training...")

    set_global_seed(cfg.get("seed", 42))

    train_posts = load_split(args.train_path)
    val_posts = load_split(args.val_path)
    logger.info(
        f"Loaded {len(train_posts)} train posts and {len(val_posts)} validation posts."
    )

    train_data = [build_classification_example(p) for p in train_posts]
    val_data = [build_classification_example(p) for p in val_posts]

    train_ds = datasets.Dataset.from_list(train_data)
    val_ds = datasets.Dataset.from_list(val_data)

    model, tokenizer = build_classifier(
        cfg.get("model_name", "deberta-base"), num_labels=int(cfg.get("num_labels", 3))
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.get("max_length", 384),
            padding="max_length",
        )

    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    val_tokenized = val_ds.map(tokenize_fn, batched=True)

    training_args = get_training_args(cfg, args.output_dir)

    # Build compatible native trainer arguments
    base_kwargs = build_trainer_kwargs(
        model=model,
        training_args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        callbacks=[CheckpointDiskBudgetCallback(), GpuMemoryLoggingCallback()],
    )

    # Directly instantiate the highly optimized native Trainer
    trainer = Trainer(**base_kwargs)

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    logger.info("Task 1 Classifier training successfully completed.")


if __name__ == "__main__":
    main()
