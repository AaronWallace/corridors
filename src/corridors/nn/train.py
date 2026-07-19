"""Training loop for ValueNet.

Loss: MSE(pred, outcome) + aux_weight * MSE(pred, tt_score_normalized).
Optimizer: AdamW, cosine-annealed LR. Saves best-only by validation MSE.

Adaptive epochs (shared with az_train): trains up to a max epoch cap that
extends past the configured target when validation is still improving, and
stops early once validation plateaus (see train_common.ValidationEarlyStopper).
Ctrl-C safe: finishes the current batch, saves nothing mid-epoch, keeps the
best checkpoint written so far.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import datasets as ds_mod
from . import model as model_mod
from .model import ValueNet
from .train_common import (
    BatchInfo,
    ValidationEarlyStopper,
    resolve_max_epochs,
    resolve_min_epochs,
)


@dataclass
class TrainConfig:
    dataset: str
    epochs: int = 30              # soft target; adaptive training may extend
    batch_size: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    aux_weight: float = 0.3       # weight on the tt-score auxiliary target
    val_frac: float = 0.1
    seed: int = 0
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    checkpoint_name: str = ""     # default: {dataset}__e{epochs}
    # Adaptive epoch controls (parity with AZTrainConfig).
    early_stopping: bool = True
    early_stop_patience: int = 3
    early_stop_min_epochs: int = 0   # 0 = auto (~35% of target, >= 5)
    early_stop_min_delta: float = 1e-3  # meaningful relative val gain
    max_epochs: int = 0              # 0 = auto (target * extension_factor)
    epoch_extension_factor: float = 1.5


@dataclass
class EpochInfo:
    epoch: int
    epochs: int                     # target (soft)
    train_loss: float
    val_mse: float
    val_sign_acc: float
    lr: float
    elapsed: float
    is_best: bool
    # Adaptive-training display data (parity with AZEpochInfo).
    will_stop: bool = False
    stop_reason: str = ""
    extension_started: bool = False
    target_epochs: int = 0


def default_checkpoint_name(cfg: TrainConfig) -> str:
    return cfg.checkpoint_name or f"{cfg.dataset}__e{cfg.epochs}"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolved_min_epochs(cfg: TrainConfig) -> int:
    return resolve_min_epochs(cfg.epochs, cfg.early_stop_min_epochs)


def _resolved_max_epochs(cfg: TrainConfig) -> int:
    return resolve_max_epochs(cfg.epochs, cfg.max_epochs,
                              cfg.epoch_extension_factor)


def train(
    cfg: TrainConfig,
    on_epoch: Optional[Callable[[EpochInfo], None]] = None,
    on_batch: Optional[Callable[[BatchInfo], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    """Train a ValueNet. Returns summary dict. Raises FileNotFoundError if the
    dataset has no shards.

    Trains for at most `_resolved_max_epochs(cfg)` epochs (>= cfg.epochs),
    but stops early once validation plateaus after a warmup of
    `_resolved_min_epochs(cfg)`. LR schedule is cosine-annealed over the full
    max horizon so extension epochs still get sensible LR values.
    """
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tensors, outcomes, tt_scores = ds_mod.load_dataset(cfg.dataset)
    n = len(tensors)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * cfg.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _to_tensors(ix: np.ndarray):
        return (
            torch.from_numpy(tensors[ix]),
            torch.from_numpy(outcomes[ix].astype(np.float32)),
            torch.from_numpy(tt_scores[ix]),
        )

    train_set = TensorDataset(*_to_tensors(train_idx))
    val_set = TensorDataset(*_to_tensors(val_idx))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=len(train_set) > cfg.batch_size)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    train_batches = max(1, len(train_loader))
    val_batches = max(1, len(val_loader))

    model = ValueNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    max_epochs = _resolved_max_epochs(cfg)
    # Anneal over the max horizon; if we stop earlier the schedule was still
    # sensible for the epochs we ran.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs,
                                                       eta_min=cfg.lr_min)
    mse = torch.nn.MSELoss()

    ckpt_name = default_checkpoint_name(cfg)
    best_val = float("inf")
    best_epoch = -1
    t0 = time.monotonic()
    stopped_early = False
    stop_reason = ""

    early_stopper = ValidationEarlyStopper(
        cfg.early_stop_patience,
        _resolved_min_epochs(cfg),
        cfg.early_stop_min_delta,
    ) if cfg.early_stopping else None

    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        extension_started = epoch == cfg.epochs + 1  # first epoch past target
        if stop_flag is not None and stop_flag():
            stopped_early = True
            stop_reason = "external stop signal"
            break
        model.train()
        total_loss = 0.0
        batches = 0
        for xb, yb, sb in train_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            pred = model(xb)
            loss = mse(pred, yb) + cfg.aux_weight * mse(pred, sb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            batches += 1
            if on_batch is not None and (batches == 1 or batches == train_batches
                                         or batches % 8 == 0):
                on_batch(BatchInfo(
                    epoch=epoch, epochs=max_epochs, phase="train",
                    batch=batches, batches=train_batches,
                    elapsed=time.monotonic() - t0,
                    loss=total_loss / batches,
                ))
        sched.step()
        train_loss = total_loss / max(batches, 1)

        model.eval()
        v_se = 0.0
        v_n = 0
        v_sign_ok = 0
        v_sign_n = 0
        val_batch = 0
        with torch.no_grad():
            for xb, yb, sb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                v_se += float(((pred - yb) ** 2).sum().item())
                v_n += len(yb)
                nz = yb != 0
                if bool(nz.any()):
                    v_sign_ok += int((torch.sign(pred[nz]) == torch.sign(yb[nz])).sum().item())
                    v_sign_n += int(nz.sum().item())
                val_batch += 1
                if on_batch is not None and (val_batch == 1
                                             or val_batch == val_batches
                                             or val_batch % 8 == 0):
                    on_batch(BatchInfo(
                        epoch=epoch, epochs=max_epochs, phase="val",
                        batch=val_batch, batches=val_batches,
                        elapsed=time.monotonic() - t0,
                        loss=v_se / max(v_n, 1),
                    ))
        val_mse = v_se / max(v_n, 1)
        val_sign_acc = v_sign_ok / v_sign_n if v_sign_n else float("nan")

        # Early-stopping decision AFTER we compute this epoch's val_mse. We
        # don't break yet — first report the epoch (with will_stop=True) so
        # the caller can display it.
        will_stop = False
        this_reason = ""
        if early_stopper is not None:
            will_stop, this_reason = early_stopper.update(epoch, val_mse)

        is_best = val_mse < best_val
        if is_best:
            best_val = val_mse
            best_epoch = epoch
            model_mod.save_checkpoint(model, ckpt_name, meta={
                "dataset": cfg.dataset,
                "epoch": epoch,
                "epochs": max_epochs,
                "target_epochs": cfg.epochs,
                "val_mse": round(val_mse, 6),
                "val_sign_acc": round(val_sign_acc, 4) if v_sign_n else None,
                "train_loss": round(train_loss, 6),
                "positions": n,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "aux_weight": cfg.aux_weight,
                "device": device,
            })

        if on_epoch is not None:
            on_epoch(EpochInfo(
                epoch=epoch, epochs=cfg.epochs,
                train_loss=train_loss, val_mse=val_mse, val_sign_acc=val_sign_acc,
                lr=sched.get_last_lr()[0], elapsed=time.monotonic() - t0,
                is_best=is_best,
                will_stop=will_stop, stop_reason=this_reason,
                extension_started=extension_started,
                target_epochs=cfg.epochs,
            ))

        if will_stop:
            stopped_early = True
            stop_reason = this_reason
            break

    return {
        "checkpoint": ckpt_name,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "positions": n,
        "device": device,
        "elapsed": time.monotonic() - t0,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "epochs_run": epoch,
        "target_epochs": cfg.epochs,
        "max_epochs": max_epochs,
    }
