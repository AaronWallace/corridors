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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from . import az_net
from .actions import PAWN_ACTIONS
from .az_selfplay import AZ_DATA_ROOT  # single source of truth (torch-free module)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def balanced_policy_log_probs(policy_logits: torch.Tensor,
                              policy_targets: torch.Tensor) -> torch.Tensor:
    """Legal-action log probabilities with action-type count normalization.

    New self-play targets contain a negligible positive floor on every legal
    action, allowing the target itself to carry its legal mask. Subtracting the
    log legal count from each action means equal logits give pawn moves and wall
    moves equal aggregate probability rather than weighting types by branching
    factor. Legacy targets use their nonzero visit support as a safe fallback.
    """
    legal = policy_targets > 0
    pawn_count = legal[:, :PAWN_ACTIONS].sum(dim=1).clamp_min(1)
    wall_count = legal[:, PAWN_ACTIONS:].sum(dim=1).clamp_min(1)

    adjusted = policy_logits.clone()
    adjusted[:, :PAWN_ACTIONS] -= pawn_count.log().unsqueeze(1)
    adjusted[:, PAWN_ACTIONS:] -= wall_count.log().unsqueeze(1)
    adjusted = adjusted.masked_fill(~legal, -torch.inf)
    return F.log_softmax(adjusted, dim=1)


def balanced_policy_loss(policy_logits: torch.Tensor,
                         policy_targets: torch.Tensor) -> torch.Tensor:
    log_probs = balanced_policy_log_probs(policy_logits, policy_targets)
    terms = torch.where(policy_targets > 0, policy_targets * log_probs, 0.0)
    return -terms.sum(dim=1).mean()


@dataclass
class AZTrainConfig:
    epochs: int = 10  # soft target; productive training may extend beyond it
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
    reflection_prob: float = 0.5
    early_stopping: bool = True
    early_stop_patience: int = 3
    early_stop_min_epochs: int = 0  # 0 = auto (35% of configured epochs, at least 5)
    early_stop_min_delta: float = 1e-3  # meaningful relative validation gain (0.1%)
    max_epochs: int = 0  # 0 = auto from epoch_extension_factor
    epoch_extension_factor: float = 1.5


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
    will_stop: bool = False
    stop_reason: str = ""
    extension_started: bool = False
    target_epochs: int = 0


@dataclass
class AZBatchInfo:
    epoch: int
    epochs: int
    phase: str
    batch: int
    batches: int
    elapsed: float
    loss: float


class _ValidationEarlyStopper:
    """Stop after validation ceases making meaningful relative progress."""

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
        should_stop = epoch >= self.min_epochs and self.stale_epochs >= self.patience
        reason = (f"no ≥{self.min_delta:.2%} validation improvement for "
                  f"{self.stale_epochs} epochs") if should_stop else ""
        return should_stop, reason


def _resolved_early_stop_min_epochs(config: AZTrainConfig) -> int:
    if config.early_stop_min_epochs > 0:
        return min(config.epochs, config.early_stop_min_epochs)
    return min(config.epochs, max(5, math.ceil(config.epochs * 0.35)))


def _resolved_max_epochs(config: AZTrainConfig) -> int:
    if config.max_epochs > 0:
        return max(config.epochs, config.max_epochs)
    return max(config.epochs, math.ceil(config.epochs * config.epoch_extension_factor))


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


def _shard_positions(path: Path) -> int:
    """Number of positions in a shard, read from the .npy header only (no data
    load). Falls back to loading the array if the header format is unexpected."""
    try:
        import zipfile
        from numpy.lib import format as npf
        with zipfile.ZipFile(path) as zf:
            name = "states.npy" if "states.npy" in zf.namelist() else "states"
            with zf.open(name) as f:
                shape, _, _ = npf._read_array_header(f, npf.read_magic(f))
                return int(shape[0])
    except Exception:
        with np.load(path) as z:
            return int(z["states"].shape[0])


def _shard_files(name: str, max_positions: int = 0):
    """The shard files that would be loaded for `name`. If max_positions > 0, keep
    the most recent shards whose combined positions reach that many (a rolling
    replay window over the freshest self-play, measured in positions/samples)."""
    parts = Path(name).parts
    d = (AZ_DATA_ROOT.parent / Path(name)
         if parts[:1] == ("shared",) else AZ_DATA_ROOT / name)
    files = sorted(f for f in d.glob("*.npz") if not f.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no training data in {d}")
    if max_positions > 0:
        kept, total = [], 0
        for f in reversed(files):  # newest shard first
            kept.append(f)
            total += _shard_positions(f)
            if total >= max_positions:
                break
        files = list(reversed(kept))  # back to chronological order
    return files


def load_training_data(
    name: str,
    max_positions: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load self-play data. If max_positions > 0, only the most recent that many
    positions (rolling replay window)."""
    ss, ps, os_ = [], [], []
    for f in _shard_files(name, max_positions):
        with np.load(f) as z:
            ss.append(z["states"])
            ps.append(z["policies"])
            os_.append(z["outcomes"])
    return np.concatenate(ss), np.concatenate(ps), np.concatenate(os_)


def _combined_shard_files(names, max_positions: int = 0):
    """Chronological ``(run, shard)`` pairs with one cap across all runs.

    Run order matters: callers put one-time seed runs first and the current run
    last, ensuring the newest self-play survives a constrained replay window.
    """
    if isinstance(names, str):
        names = [names]
    files = [(name, path) for name in names for path in _shard_files(name)]
    if max_positions > 0:
        kept, total = [], 0
        for item in reversed(files):
            kept.append(item)
            total += _shard_positions(item[1])
            if total >= max_positions:
                break
        files = list(reversed(kept))
    return files


def load_training_datasets(names, on_progress: Optional[Callable] = None,
                           max_positions: int = 0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load runs with one optional rolling-position cap across the combination."""
    ss, ps, os_ = [], [], []
    files = _combined_shard_files(names, max_positions)
    positions = 0
    total = len(files)
    if on_progress:
        on_progress("loading", 0, total, "", "", positions)
    for done, (name, path) in enumerate(files, 1):
        with np.load(path) as z:
            s, p, o = z["states"], z["policies"], z["outcomes"]
            ss.append(s); ps.append(p); os_.append(o)
            positions += len(s)
        if on_progress:
            on_progress("loading", done, total, name, path.name, positions)
    if not ss:
        raise FileNotFoundError("no runs selected")
    if on_progress:
        on_progress("combining", total, total, "", "", positions)
    return np.concatenate(ss), np.concatenate(ps), np.concatenate(os_)


def dataset_provenance(names, max_positions: int = 0) -> dict:
    """Fingerprint the dataset(s) that would be loaded, for stamping into a
    checkpoint. `names` is a run name or a list of run names. max_positions mirrors
    load_training_data's rolling window so the provenance matches what was trained.

    The hash is over each shard's (run/name, size, mtime) — cheap (no file reads)
    and changes if any shard is added, removed, or regenerated. Lets you later
    answer "was this data baked into this checkpoint?" by comparing hashes."""
    import hashlib
    if isinstance(names, str):
        names = [names]
    names = list(names)
    shard_names = []
    h = hashlib.sha256()
    files = _combined_shard_files(names, max_positions)
    for name, f in files:
        st = f.stat()
        h.update(f"{name}/{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
        shard_names.append(f"{name}/{f.name}")
    total = len(files)
    all_runs = list(dict.fromkeys(name for name, _path in files))
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
    on_batch: Optional[Callable[[AZBatchInfo], None]] = None,
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

    # Keep zero-copy views of the loaded NumPy arrays. Subset stores only the
    # split indices; the previous advanced indexing duplicated every tensor.
    full_set = TensorDataset(
        torch.from_numpy(states), torch.from_numpy(policies), torch.from_numpy(outcomes))
    train_set = Subset(full_set, train_idx)
    val_set = Subset(full_set, val_idx)
    pin = device.startswith("cuda")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              drop_last=len(train_set) > config.batch_size,
                              pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, pin_memory=pin)
    reflection_permutation = None
    if config.reflection_prob > 0:
        from .actions import REFLECT_ACTION_INDEX
        reflection_permutation = torch.as_tensor(REFLECT_ACTION_INDEX, dtype=torch.long)

    if resume_from:
        model = az_net.load_checkpoint(resume_from, device=device)
        model.train()
    else:
        model = az_net.AZNet().to(device)

    max_epochs = _resolved_max_epochs(config)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=config.lr_min)

    best_val = float("inf")
    best_epoch = -1
    t0 = time.monotonic()
    early_stopper = _ValidationEarlyStopper(
        config.early_stop_patience,
        _resolved_early_stop_min_epochs(config),
        config.early_stop_min_delta,
    )
    epochs_completed = 0
    stopped_early = False
    stop_reason = ""

    for epoch in range(1, max_epochs + 1):
        if stop_flag and stop_flag():
            break

        model.train()
        total_loss = total_ploss = total_vloss = 0.0
        batches = 0
        phase_t0 = time.monotonic()
        train_batches = len(train_loader)
        for xb, pb, ob in train_loader:
            if config.reflection_prob > 0:
                reflect = torch.rand(len(xb)) < config.reflection_prob
                if reflect.any():
                    source_states = xb[reflect]
                    reflected_states = source_states.flip(-1)
                    reflected_states[:, 2:4, :, :] = 0
                    reflected_states[:, 2:4, :, :-1] = source_states[:, 2:4, :, :-1].flip(-1)
                    xb[reflect] = reflected_states
                    reflected_policy = torch.empty_like(pb[reflect])
                    reflected_policy[:, reflection_permutation] = pb[reflect]
                    pb[reflect] = reflected_policy
            xb = xb.to(device, non_blocking=pin)
            pb = pb.to(device, non_blocking=pin)
            ob = ob.to(device, non_blocking=pin)

            policy_logits, value = model(xb)

            # Policy loss: action-type-balanced cross-entropy with the MCTS
            # distribution. Legal support is encoded by tiny positive targets.
            policy_loss = balanced_policy_loss(policy_logits, pb)

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
            if on_batch and (batches == 1 or batches == train_batches
                             or batches % max(1, train_batches // 20) == 0):
                on_batch(AZBatchInfo(
                    epoch, max_epochs, "training", batches, train_batches,
                    time.monotonic() - phase_t0, total_loss / batches))

        sched.step()
        train_loss = total_loss / max(batches, 1)
        train_ploss = total_ploss / max(batches, 1)
        train_vloss = total_vloss / max(batches, 1)

        # Validation
        model.eval()
        v_loss = v_ploss = v_vloss = 0.0
        v_n = 0
        val_batches = len(val_loader)
        val_batch = 0
        phase_t0 = time.monotonic()
        with torch.inference_mode():
            for xb, pb, ob in val_loader:
                val_batch += 1
                xb = xb.to(device, non_blocking=pin)
                pb = pb.to(device, non_blocking=pin)
                ob = ob.to(device, non_blocking=pin)
                policy_logits, value = model(xb)
                pl = balanced_policy_loss(policy_logits, pb)
                vl = F.mse_loss(value, ob)
                v_ploss += pl.item() * len(xb)
                v_vloss += vl.item() * len(xb)
                v_loss += (config.policy_weight * pl + config.value_weight * vl).item() * len(xb)
                v_n += len(xb)
                if on_batch and (val_batch == 1 or val_batch == val_batches
                                 or val_batch % max(1, val_batches // 10) == 0):
                    on_batch(AZBatchInfo(
                        epoch, max_epochs, "validation", val_batch, val_batches,
                        time.monotonic() - phase_t0, v_loss / max(v_n, 1)))

        val_loss = v_loss / max(v_n, 1)
        val_ploss = v_ploss / max(v_n, 1)
        val_vloss = v_vloss / max(v_n, 1)
        epochs_completed = epoch

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_epoch = epoch
            az_net.save_checkpoint(model, config.checkpoint_name, meta={
                "epoch": epoch,
                "epochs": max_epochs,
                "epoch_target": config.epochs,
                "epoch_limit": max_epochs,
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

        should_stop, reason = early_stopper.update(epoch, val_loss)
        should_stop = bool(config.early_stopping and should_stop
                           and epoch < max_epochs)
        extension_started = bool(
            epoch == config.epochs and not should_stop and max_epochs > config.epochs)
        if on_epoch:
            on_epoch(AZEpochInfo(
                epoch=epoch, epochs=max_epochs,
                train_loss=train_loss, policy_loss=train_ploss, value_loss=train_vloss,
                val_loss=val_loss, val_policy_loss=val_ploss, val_value_loss=val_vloss,
                lr=sched.get_last_lr()[0], elapsed=time.monotonic() - t0,
                is_best=is_best,
                will_stop=should_stop,
                stop_reason=reason if should_stop else "",
                extension_started=extension_started,
                target_epochs=config.epochs,
            ))
        if should_stop:
            stopped_early = True
            stop_reason = reason
            break

    return {
        "checkpoint": config.checkpoint_name,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "target_epochs": config.epochs,
        "max_epochs": max_epochs,
        "extended": epochs_completed > config.epochs,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "positions": n,
        "device": device,
        "elapsed": time.monotonic() - t0,
    }
