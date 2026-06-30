import torch
from transformers import TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import inspect


def get_training_args(cfg: dict, output_dir: str) -> TrainingArguments:
    """Build a transformers.TrainingArguments from a parsed YAML config dict.
    Supports both old (evaluation_strategy) and modern (eval_strategy) transformers APIs.
    Forces fp16 to False and uses adam_epsilon=1e-6 to guarantee absolute numerical stability for DeBERTa.
    """
    eval_strategy_val = cfg.get(
        "eval_strategy", cfg.get("evaluation_strategy", "epoch")
    )

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy_val,
        save_strategy=cfg.get("save_strategy", "epoch"),
        learning_rate=float(
            cfg.get("learning_rate", 1.0e-5)
        ),  # Extremely stable default for DeBERTa-v3
        per_device_train_batch_size=int(cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 16)),
        num_train_epochs=int(cfg.get("num_train_epochs", 3)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
        fp16=False,  # Hardcoded to False for absolute stability
        adam_epsilon=1e-6,  # CRITICAL FIX: prevents division-by-zero NaN in DeBERTa embeddings
        max_grad_norm=1.0,  # Explicitly clip exploded gradients
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


def build_trainer_kwargs(
    model, training_args, train_dataset, eval_dataset, tokenizer, callbacks
) -> dict:
    """Build kwargs for HF Trainer supporting both old (<4.46) and new (>=4.46) transformers versions."""
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": callbacks,
    }
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return kwargs
