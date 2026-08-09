"""Determinism helpers.

The projector, the split, and the calibration fits must be bit-reproducible on
the same hardware, and structurally reproducible across hardware. Anything that
cannot be made deterministic is not used.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch, and disable nondeterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng(seed: int) -> np.random.Generator:
    """A named, reproducible NumPy generator (PCG64, explicit seed)."""
    return np.random.default_rng(seed)


def torch_deterministic(warn_only: bool = True) -> None:
    """Ask Torch to error (or warn) on nondeterministic ops.

    ``warn_only`` is the default because some CLIP kernels have no deterministic
    implementation; the projector and all fitted parameters are deterministic
    regardless, and per-view logits are cached so downstream analysis is exact.
    """
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
