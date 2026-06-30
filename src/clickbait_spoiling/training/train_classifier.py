import argparse
import logging
import os

import datasets
import torch
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
    compute_class_weights,
    get_training_args,
)
from clickbait_spoiling.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


class WeightedLossTrainer(Trainer):
    """Subclass of Trainer to support class weights for imbalanced sequence classification.
    Dynamically casts weights to match logits precision to prevent multi-GPU NCCL dtype conflicts.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Dynamically cast weights to the exact dtype of logits (Half or Float)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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

    labels = [ex["label"] for ex in train_data]
    class_weights = compute_class_weights(
        labels, num_labels=int(cfg.get("num_labels", 3))
    )
    logger.info(f"Computed inverse-frequency class weights: {class_weights}")

    training_args = get_training_args(cfg, args.output_dir)

    trainer_cls = WeightedLossTrainer if cfg.get("use_class_weights", True) else Trainer
    trainer_kwargs = (
        {"class_weights": class_weights} if cfg.get("use_class_weights", True) else {}
    )

    # Build compatible trainer arguments
    base_kwargs = build_trainer_kwargs(
        model=model,
        training_args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        callbacks=[CheckpointDiskBudgetCallback(), GpuMemoryLoggingCallback()],
    )

    trainer = trainer_cls(**base_kwargs, **trainer_kwargs)

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    logger.info("Task 1 Classifier training successfully completed.")


if __name__ == "__main__":
    main()
