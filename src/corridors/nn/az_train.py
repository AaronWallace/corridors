"""AlphaZero training loop.

Loss = value_MSE(v, z) + policy_CE(p, π) + c * L2_reg
where:
  v = network value prediction
  z = game outcome (+1/-1/0) from side-to-move perspective
  p = network policy logits (log-softmax)
  π = MCTS visit-count distribution (training target)
  c = weight_decay (handled by AdamW)

Supports iterative training: generate data → train → generate more → train again.
Each iteration uses a replay buffer of recent games.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import az_net
from .az_selfplay import AZ_DATA_ROOT  # single source of truth (torch-free module)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class AZTrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 2e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    val_frac: float = 0.1
    value_weight: float = 1.0
    policy_weight: float = 1.0
    seed: int = 0
    device: str = "auto"
    checkpoint_name: str = "az_latest"


@dataclass
class AZEpochInfo:
    epoch: int
    epochs: int
    train_loss: float
    policy_loss: float
    value_loss: float
    val_loss: float
    val_policy_loss: float
    val_value_loss: float
    lr: float
    elapsed: float
    is_best: bool


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_training_data(
    name: str,
    states: np.ndarray,
    policies: np.ndarray,
    outcomes: np.ndarray,
    iteration: int = 0,
) -> Path:
    """Save self-play data to disk."""
    d = AZ_DATA_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"iter_{iteration:04d}.npz"
    np.savez_compressed(path, states=states, policies=policies, outcomes=outcomes)
    return path


def _shard_files(name: str, max_iterations: int = 0):
    """The shard files that would be loaded for `name` (most recent N if capped)."""
    d = AZ_DATA_ROOT / name
    files = sorted(f for f in d.glob("*.npz") if not f.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no training data in {d}")
    if max_iterations > 0:
        files = files[-max_iterations:]
    return files


def load_training_data(
    name: str,
    max_iterations: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load self-play data. If max_iterations > 0, only load the most recent N."""
    ss, ps, os_ = [], [], []
    for f in _shard_files(name, max_iterations):
        with np.load(f) as z:
            ss.append(z["states"])
            ps.append(z["policies"])
            os_.append(z["outcomes"])
    return np.concatenate(ss), np.concatenate(ps), np.concatenate(os_)


def load_training_datasets(names) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate the shards of one or more runs (names: str or list)."""
    if isinstance(names, str):
        names = [names]
    ss, ps, os_ = [], [], []
    for name in names:
        s, p, o = load_training_data(name)
        ss.append(s); ps.append(p); os_.append(o)
    if not ss:
        raise FileNotFoundError("no runs selected")
    return np.concatenate(ss), np.concatenate(ps), np.concatenate(os_)


def dataset_provenance(names, max_iterations: int = 0) -> dict:
    """Fingerprint the dataset(s) that would be loaded, for stamping into a
    checkpoint. `names` is a run name or a list of run names.

    The hash is over each shard's (run/name, size, mtime) — cheap (no file reads)
    and changes if any shard is added, removed, or regenerated. Lets you later
    answer "was this data baked into this checkpoint?" by comparing hashes."""
    import hashlib
    if isinstance(names, str):
        names = [names]
    all_runs, shard_names, total = [], [], 0
    h = hashlib.sha256()
    for name in names:
        files = _shard_files(name, max_iterations)
        for f in files:
            st = f.stat()
            h.update(f"{name}/{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
            shard_names.append(f"{name}/{f.name}")
        total += len(files)
        all_runs.append(name)
    return {
        "data_run": "+".join(all_runs),
        "data_runs": all_runs,
        "data_shards": total,
        "data_shard_names": shard_names,
        "data_sha": h.hexdigest()[:16],
    }


def train_az(
    states: np.ndarray,
    policies: np.ndarray,
    outcomes: np.ndarray,
    config: AZTrainConfig,
    resume_from: str = "",
    on_epoch: Optional[Callable[[AZEpochInfo], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    data_meta: Optional[dict] = None,
) -> dict:
    """Train the AZNet. Returns summary dict.

    data_meta: optional dataset-provenance dict (see dataset_provenance) stamped
    into the checkpoint's meta so you can later tell which data trained it."""
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Weight lineage: the checkpoint this run continued from, plus the propagated
    # cross-lineage origin (e.g. a loop seeded from az_latest keeps seeded_from
    # through every iteration). Stamped into the meta on each save.
    lineage = {}
    if resume_from:
        lineage["resumed_from"] = resume_from
        prev = az_net.read_meta(resume_from)
        if prev.get("seeded_from"):
            lineage["seeded_from"] = prev["seeded_from"]

    n = len(states)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * config.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _to_tensors(ix):
        return (
            torch.from_numpy(states[ix]),
            torch.from_numpy(policies[ix]),
            torch.from_numpy(outcomes[ix]),
        )

    train_set = TensorDataset(*_to_tensors(train_idx))
    val_set = TensorDataset(*_to_tensors(val_idx))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              drop_last=len(train_set) > config.batch_size)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    if resume_from:
        model = az_net.load_checkpoint(resume_from, device=device)
        model.train()
    else:
        model = az_net.AZNet().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.epochs, eta_min=config.lr_min)

    best_val = float("inf")
    best_epoch = -1
    t0 = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        if stop_flag and stop_flag():
            break

        model.train()
        total_loss = total_ploss = total_vloss = 0.0
        batches = 0
        for xb, pb, ob in train_loader:
            xb = xb.to(device)
            pb = pb.to(device)
            ob = ob.to(device)

            policy_logits, value = model(xb)

            # Policy loss: cross-entropy with soft targets (MCTS distribution)
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(pb * log_probs).sum(dim=1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(value, ob)

            loss = config.policy_weight * policy_loss + config.value_weight * value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_ploss += policy_loss.item()
            total_vloss += value_loss.item()
            batches += 1

        sched.step()
        train_loss = total_loss / max(batches, 1)
        train_ploss = total_ploss / max(batches, 1)
        train_vloss = total_vloss / max(batches, 1)

        # Validation
        model.eval()
        v_loss = v_ploss = v_vloss = 0.0
        v_n = 0
        with torch.no_grad():
            for xb, pb, ob in val_loader:
                xb, pb, ob = xb.to(device), pb.to(device), ob.to(device)
                policy_logits, value = model(xb)
                log_probs = F.log_softmax(policy_logits, dim=1)
                pl = -(pb * log_probs).sum(dim=1).mean()
                vl = F.mse_loss(value, ob)
                v_ploss += pl.item() * len(xb)
                v_vloss += vl.item() * len(xb)
                v_loss += (config.policy_weight * pl + config.value_weight * vl).item() * len(xb)
                v_n += len(xb)

        val_loss = v_loss / max(v_n, 1)
        val_ploss = v_ploss / max(v_n, 1)
        val_vloss = v_vloss / max(v_n, 1)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_epoch = epoch
            az_net.save_checkpoint(model, config.checkpoint_name, meta={
                "epoch": epoch,
                "epochs": config.epochs,
                "val_loss": round(val_loss, 6),
                "val_policy_loss": round(val_ploss, 6),
                "val_value_loss": round(val_vloss, 6),
                "train_loss": round(train_loss, 6),
                "positions": n,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "device": device,
                **lineage,
                **(data_meta or {}),
            })

        if on_epoch:
            on_epoch(AZEpochInfo(
                epoch=epoch, epochs=config.epochs,
                train_loss=train_loss, policy_loss=train_ploss, value_loss=train_vloss,
                val_loss=val_loss, val_policy_loss=val_ploss, val_value_loss=val_vloss,
                lr=sched.get_last_lr()[0], elapsed=time.monotonic() - t0,
                is_best=is_best,
            ))

    return {
        "checkpoint": config.checkpoint_name,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "positions": n,
        "device": device,
        "elapsed": time.monotonic() - t0,
    }
