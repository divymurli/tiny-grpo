from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch random number generators.

    This makes DataLoader shuffling, model initialization, and sampling more
    reproducible for debugging. CUDA devices are seeded when available.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    """Resolve a CLI device string into a `torch.device`.

    `auto` chooses CUDA when available and CPU otherwise. Any other value is
    passed directly to `torch.device`, so values like `cpu`, `cuda`, or
    `cuda:0` work.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
