"""AlphaZero dual-headed network for Corridors.

Architecture: stem conv (9→ch) → N residual blocks → two heads:
  - Policy head: conv 1×1 → BN → ReLU → flatten → FC → 227 logits
  - Value head:  conv 1×1 → BN → ReLU → flatten → FC → FC → tanh scalar

The policy head outputs raw logits; masking + softmax happen outside.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .actions import NUM_ACTIONS
from .encoding import NCOLS, NROWS, NUM_PLANES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_ROOT = _PROJECT_ROOT / "nn_checkpoints"

CHANNELS = 128
BLOCKS = 10


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class AZNet(nn.Module):
    def __init__(self, channels: int = CHANNELS, blocks: int = BLOCKS) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * NROWS * NCOLS, NUM_ACTIONS)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(NROWS * NCOLS, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits [B, 227], value [B])."""
        trunk = self.trunk(self.stem(x))

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(trunk)), inplace=True)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(trunk)), inplace=True)
        v = v.view(v.size(0), -1)  # flatten (B, 1, 11, 9) → (B, 99)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- Checkpoint I/O (mirrors model.py pattern) --------------------------------

def checkpoint_path(name: str) -> Path:
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    return CHECKPOINT_ROOT / name


def meta_path(name: str) -> Path:
    return checkpoint_path(name).with_suffix(".meta.json")


def save_checkpoint(model: AZNet, name: str, meta: Optional[Dict] = None) -> Path:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(name)
    tmp = path.with_name("." + path.name + ".tmp")
    save_file(model.state_dict(), str(tmp))
    tmp.replace(path)
    m = dict(meta or {})
    m["arch"] = "az"
    m.setdefault("channels", model.channels)
    m.setdefault("blocks", model.blocks_n)
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


def load_checkpoint(name: str, device: str = "cpu") -> AZNet:
    meta = read_meta(name)
    model = AZNet(
        channels=int(meta.get("channels", CHANNELS)),
        blocks=int(meta.get("blocks", BLOCKS)),
    )
    state = load_file(str(checkpoint_path(name)), device=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
