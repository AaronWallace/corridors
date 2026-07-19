"""Shared training utilities used by both the classical (train.py) and
AlphaZero (az_train.py) pipelines.

Kept intentionally small — just the pieces that were duplicated between the
two: early stopping, epoch-target resolution, and batch/epoch info dataclasses
that both display layers can consume identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BatchInfo:
    """Per-batch progress hook data. `phase` is 'train' or 'val'."""
    epoch: int
    epochs: int
    phase: str
    batch: int
    batches: int
    elapsed: float
    loss: float


class ValidationEarlyStopper:
    """Stop after validation ceases making meaningful relative progress.

    Ignores wobble smaller than `min_delta` (relative to current best), only
    counts stale epochs after `min_epochs` warmup, then stops after `patience`
    consecutive non-improving epochs. Reason is returned for display.
    """

    def __init__(self, patience: int, min_epochs: int,
                 min_delta: float) -> None:
        self.patience = max(1, int(patience))
        self.min_epochs = max(1, int(min_epochs))
        self.min_delta = max(0.0, float(min_delta))
        self.best_meaningful = float("inf")
        self.stale_epochs = 0

    def update(self, epoch: int, val_loss: float) -> tuple[bool, str]:
        threshold = (abs(self.best_meaningful) * self.min_delta
                     if math.isfinite(self.best_meaningful) else 0.0)
        if val_loss < self.best_meaningful - threshold:
            self.best_meaningful = val_loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        should_stop = (epoch >= self.min_epochs
                       and self.stale_epochs >= self.patience)
        reason = (f"no >={self.min_delta:.2%} validation improvement for "
                  f"{self.stale_epochs} epochs") if should_stop else ""
        return should_stop, reason


def resolve_min_epochs(target_epochs: int, min_epochs: int) -> int:
    """min_epochs before early stopping is allowed to fire. 0 = auto (~35% of
    the configured target, clamped to at least 5 and never past the target)."""
    if min_epochs > 0:
        return min(target_epochs, min_epochs)
    return min(target_epochs, max(5, math.ceil(target_epochs * 0.35)))


def resolve_max_epochs(target_epochs: int, max_epochs: int,
                       extension_factor: float) -> int:
    """Hard cap on training length. 0 = auto (target x extension_factor,
    default 1.5x). Never less than the configured target."""
    if max_epochs > 0:
        return max(target_epochs, max_epochs)
    return max(target_epochs, math.ceil(target_epochs * float(extension_factor)))
