"""Training loop for ValueNet.

Loss: MSE(pred, outcome) + aux_weight * MSE(pred, tt_score_normalized).
Optimizer: AdamW, cosine-annealed LR. Saves best-only by validation MSE.
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


@dataclass
class TrainConfig:
    dataset: str
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    aux_weight: float = 0.3      # weight on the tt-score auxiliary target
    val_frac: float = 0.1
    seed: int = 0
    device: str = "auto"         # "auto" | "cuda" | "cpu"
    checkpoint_name: str = ""    # default: {dataset}__e{epochs}


@dataclass
class EpochInfo:
    epoch: int
    epochs: int
    train_loss: float
    val_mse: float
    val_sign_acc: float
    lr: float
    elapsed: float
    is_best: bool


def default_checkpoint_name(cfg: TrainConfig) -> str:
    return cfg.checkpoint_name or f"{cfg.dataset}__e{cfg.epochs}"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(
    cfg: TrainConfig,
    on_epoch: Optional[Callable[[EpochInfo], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    """Train a ValueNet. Returns summary dict. Raises FileNotFoundError if the
    dataset has no shards."""
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

    model = ValueNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
    mse = torch.nn.MSELoss()

    ckpt_name = default_checkpoint_name(cfg)
    best_val = float("inf")
    best_epoch = -1
    t0 = time.monotonic()
    stopped = False

    for epoch in range(1, cfg.epochs + 1):
        if stop_flag is not None and stop_flag():
            stopped = True
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
        sched.step()
        train_loss = total_loss / max(batches, 1)

        model.eval()
        v_se = 0.0
        v_n = 0
        v_sign_ok = 0
        v_sign_n = 0
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
        val_mse = v_se / max(v_n, 1)
        val_sign_acc = v_sign_ok / v_sign_n if v_sign_n else float("nan")

        is_best = val_mse < best_val
        if is_best:
            best_val = val_mse
            best_epoch = epoch
            model_mod.save_checkpoint(model, ckpt_name, meta={
                "dataset": cfg.dataset,
                "epoch": epoch,
                "epochs": cfg.epochs,
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
            ))

    return {
        "checkpoint": ckpt_name,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "positions": n,
        "device": device,
        "elapsed": time.monotonic() - t0,
        "stopped_early": stopped,
    }
