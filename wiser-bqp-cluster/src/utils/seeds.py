"""Global seeding for reproducible runs."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: the integer seed.
        deterministic: if True, request deterministic cuDNN/torch algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
