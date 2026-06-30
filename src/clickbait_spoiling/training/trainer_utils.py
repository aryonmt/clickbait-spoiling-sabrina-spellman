import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import TrainingArguments


def get_training_args(cfg: dict, output_dir: str) -> TrainingArguments:
    """Build a transformers.TrainingArguments from a parsed YAML config dict."""
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=cfg.get("evaluation_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 16)),
        num_train_epochs=int(cfg.get("num_train_epochs", 3)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
        fp16=torch.cuda.is_available() and bool(cfg.get("fp16", True)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        logging_steps=10,
    )


def compute_class_weights(labels: list, num_labels: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for Task 1 class imbalance."""
    clean_labels = [l for l in labels if l >= 0]
    if not clean_labels:
        return torch.ones(num_labels, dtype=torch.float32)
    classes = np.arange(num_labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=clean_labels
    )
    return torch.tensor(weights, dtype=torch.float32)
