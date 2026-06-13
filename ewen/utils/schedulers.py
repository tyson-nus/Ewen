"""Learning-rate schedulers."""

from __future__ import annotations

import math

import numpy as np


def cosine_scheduler(base_value: float, final_value: float, epochs: int, niter_per_ep: int,
                     warmup_epochs: int = 0, start_warmup_value: float = 0.0,
                     warmup_steps: int = -1) -> np.ndarray:
    """Cosine LR schedule with linear warm-up, returning per-iter values."""
    total_iters = max(1, epochs * niter_per_ep)
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    else:
        warmup_iters = warmup_epochs * niter_per_ep
    warmup_iters = int(max(0, min(warmup_iters, total_iters)))

    warmup_schedule = (
        np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_iters > 0
        else np.array([], dtype=np.float64)
    )

    remaining = total_iters - warmup_iters
    if remaining > 0:
        iters = np.arange(remaining)
        cosine = np.array([
            final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / len(iters)))
            for i in iters
        ])
    else:
        cosine = np.array([], dtype=np.float64)
    return np.concatenate((warmup_schedule, cosine))
