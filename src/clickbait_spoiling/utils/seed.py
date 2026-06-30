import random

import numpy as np
import torch

from clickbait_spoiling.constants import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random, numpy, torch, and CUDA seeds for strict experimental reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
