"""ValueNet: convolutional value network for Corridors.

Architecture (~550k params): stem conv (9→64ch) → 6 residual blocks (64ch,
3×3, BN, ReLU) → global average pool → FC 64→64 → FC 64→1 → tanh.

Output is expected outcome in [-1, 1] from the side-to-move's perspective.

Designed so a policy head can be bolted on later for AlphaZero-style training
(the trunk is shared; add a conv+FC head next to the value head).

Checkpoints are safetensors + a sidecar .meta.json (epoch, val metrics, config,
tournament results).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .checkpoints import checkpoint_elo, load_elo_ratings, ranked_checkpoint_paths
from .encoding import NUM_PLANES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_ROOT = _PROJECT_ROOT / "nn_checkpoints"

CHANNELS = 64
BLOCKS = 6


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class ValueNet(nn.Module):
    def __init__(self, channels: int = CHANNELS, blocks: int = BLOCKS) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.trunk(self.stem(x))
        y = y.mean(dim=(2, 3))  # global average pool
        return self.head(y).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def checkpoint_path(name: str) -> Path:
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    return CHECKPOINT_ROOT / name


def meta_path(name: str) -> Path:
    return checkpoint_path(name).with_suffix(".meta.json")


def save_checkpoint(model: ValueNet, name: str, meta: Optional[Dict] = None) -> Path:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(name)
    tmp = path.with_name("." + path.name + ".tmp")
    save_file(model.state_dict(), str(tmp))
    tmp.replace(path)
    if meta is not None:
        m = dict(meta)
        m.setdefault("channels", model.channels)
        m.setdefault("blocks", model.blocks)
        mp = meta_path(name)
        mp_tmp = mp.with_name("." + mp.name + ".tmp")
        mp_tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
        mp_tmp.replace(mp)
    return path


def read_meta(name: str) -> Dict:
    p = meta_path(name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def update_meta(name: str, updates: Dict) -> None:
    m = read_meta(name)
    m.update(updates)
    p = meta_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("." + p.name + ".tmp")
    tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    tmp.replace(p)


def load_checkpoint(name: str, device: str = "cpu") -> ValueNet:
    meta = read_meta(name)
    model = ValueNet(
        channels=int(meta.get("channels", CHANNELS)),
        blocks=int(meta.get("blocks", BLOCKS)),
    )
    state = load_file(str(checkpoint_path(name)), device=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def list_checkpoints() -> list:
    if not CHECKPOINT_ROOT.exists():
        return []
    out = []
    ratings = load_elo_ratings(CHECKPOINT_ROOT)
    for f in ranked_checkpoint_paths(CHECKPOINT_ROOT):
        name = f.stem
        meta = read_meta(name)
        out.append({
            "name": name,
            "size_mb": f.stat().st_size / 1e6,
            "epoch": meta.get("epoch"),
            "val_mse": meta.get("val_mse"),
            "val_sign_acc": meta.get("val_sign_acc"),
            "dataset": meta.get("data_run") or meta.get("dataset"),
            "data_sha": meta.get("data_sha"),
            "seeded_from": meta.get("seeded_from"),
            "resumed_from": meta.get("resumed_from"),
            "elo": checkpoint_elo(f, ratings),
        })
    return out


def delete_checkpoint(name: str) -> bool:
    p = checkpoint_path(name)
    if not p.exists():
        return False
    p.unlink()
    mp = meta_path(name)
    if mp.exists():
        mp.unlink()
    return True


def rename_checkpoint(name: str, new_name: str) -> bool:
    """Rename checkpoint weights and metadata without overwriting anything."""
    new_name = new_name.strip()
    if new_name.endswith(".safetensors"):
        new_name = new_name[:-len(".safetensors")]
    if (not new_name or Path(new_name).name != new_name
            or new_name in {".", ".."}):
        raise ValueError("checkpoint name must be a plain file name")

    source = checkpoint_path(name)
    target = checkpoint_path(new_name)
    source_meta = meta_path(name)
    target_meta = meta_path(new_name)
    if not source.exists():
        return False
    if target.exists() or target_meta.exists():
        raise FileExistsError(f"checkpoint '{new_name}' already exists")

    source.replace(target)
    try:
        if source_meta.exists():
            source_meta.replace(target_meta)
    except Exception:
        target.replace(source)
        raise

    meta = read_meta(new_name)
    updates = {}
    for key in ("name", "checkpoint"):
        if meta.get(key) == name:
            updates[key] = new_name
    if meta.get("resumed_from") == name:
        updates["resumed_from"] = new_name
    if updates:
        update_meta(new_name, updates)
    return True
