import logging
import os
from pathlib import Path

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def get_directory_size_gb(directory: Path) -> float:
    """Calculate the directory size in Gigabytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)


class CheckpointDiskBudgetCallback(TrainerCallback):
    """Deletes excess checkpoints and logs cumulative directory size to avoid Kaggle's 20GB limit."""

    def __init__(self, max_size_gb: float = 15.0):
        self.max_size_gb = max_size_gb

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        output_dir = Path(args.output_dir)
        size_gb = get_directory_size_gb(output_dir)
        logger.info(
            f"Cumulative checkpoint directory size: {size_gb:.2f} GB / budget {self.max_size_gb:.2f} GB"
        )

        if size_gb > self.max_size_gb:
            logger.warning(
                f"Cumulative checkpoint directory size ({size_gb:.2f} GB) exceeded soft limit "
                f"({self.max_size_gb:.2f} GB). Trainer's save_total_limit will help purge older checkpoints."
            )
        return control


class GpuMemoryLoggingCallback(TrainerCallback):
    """Logs maximum GPU memory allocated per epoch."""

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024**2)
            logger.info(
                f"Epoch {state.epoch:.1f} ended. Max GPU memory allocated: {max_mem:.2f} MB"
            )
        return control
