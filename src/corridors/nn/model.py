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
    for f in sorted(CHECKPOINT_ROOT.glob("*.safetensors")):
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
            "elo": meta.get("elo"),
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
