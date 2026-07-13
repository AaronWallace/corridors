"""Torch-free NumPy inference for AZNet.

Self-play workers only need a forward pass, not the whole DL framework. Loading
PyTorch in every worker costs ~0.5 GB RSS and import time for a 12 MB model; this
module runs the identical network in NumPy so workers stay lightweight and never
import torch.

It matches AZNet.forward numerically: same conv (cross-correlation, 'same'
padding), BatchNorm folded into the preceding bias-free conv at load time, ReLU,
and tanh on the value head. Weights load via safetensors' NumPy backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .actions import NUM_ACTIONS
from .checkpoints import resolve_checkpoint_path
from .encoding import NCOLS, NROWS, NUM_PLANES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_ROOT = _PROJECT_ROOT / "nn_checkpoints"

_BN_EPS = 1e-5  # torch BatchNorm2d default
CHANNELS = 128
BLOCKS = 10


def _fuse_conv_bn(w: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  mean: np.ndarray, var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm (inference affine) into a bias-free conv:
        bn(conv(x)) = (w*scale) * x + (beta - mean*scale),  scale = gamma/sqrt(var+eps)
    Returns (fused_weight, fused_bias)."""
    scale = (gamma / np.sqrt(var + _BN_EPS)).astype(np.float32)
    w_f = (w * scale[:, None, None, None]).astype(np.float32)
    b_f = (beta - mean * scale).astype(np.float32)
    return np.ascontiguousarray(w_f), np.ascontiguousarray(b_f)


def _conv(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Single-sample conv, stride 1, 'same' padding. x (Cin,H,W), w (Cout,Cin,kh,kw)."""
    Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    if kh == 1 and kw == 1:
        out = w.reshape(Cout, Cin) @ x.reshape(Cin, H * W)
        return (out + b[:, None]).reshape(Cout, H, W)
    pad = kh // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    # im2col: rows ordered (di, dj, cin) to match w.transpose(0,2,3,1).reshape below.
    cols = np.empty((kh * kw * Cin, H * W), dtype=np.float32)
    r = 0
    for di in range(kh):
        for dj in range(kw):
            cols[r:r + Cin] = xp[:, di:di + H, dj:dj + W].reshape(Cin, H * W)
            r += Cin
    wm = w.transpose(0, 2, 3, 1).reshape(Cout, kh * kw * Cin)
    out = wm @ cols
    return (out + b[:, None]).reshape(Cout, H, W)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0, out=x)


class NumpyAZNet:
    """AZNet forward pass in NumPy. Construct from a weights dict (numpy arrays)."""

    def __init__(self, weights: Dict[str, np.ndarray], channels: int = CHANNELS,
                 blocks: int = BLOCKS) -> None:
        def g(k: str) -> np.ndarray:
            return np.asarray(weights[k], dtype=np.float32)

        self.stem = _fuse_conv_bn(g("stem.0.weight"), g("stem.1.weight"),
                                  g("stem.1.bias"), g("stem.1.running_mean"),
                                  g("stem.1.running_var"))
        self.blocks = []
        for i in range(blocks):
            p = f"trunk.{i}."
            w1 = _fuse_conv_bn(g(p + "conv1.weight"), g(p + "bn1.weight"),
                               g(p + "bn1.bias"), g(p + "bn1.running_mean"),
                               g(p + "bn1.running_var"))
            w2 = _fuse_conv_bn(g(p + "conv2.weight"), g(p + "bn2.weight"),
                               g(p + "bn2.bias"), g(p + "bn2.running_mean"),
                               g(p + "bn2.running_var"))
            self.blocks.append((w1, w2))
        self.policy_conv = _fuse_conv_bn(g("policy_conv.weight"), g("policy_bn.weight"),
                                         g("policy_bn.bias"), g("policy_bn.running_mean"),
                                         g("policy_bn.running_var"))
        self.policy_fc = (g("policy_fc.weight"), g("policy_fc.bias"))
        self.value_conv = _fuse_conv_bn(g("value_conv.weight"), g("value_bn.weight"),
                                        g("value_bn.bias"), g("value_bn.running_mean"),
                                        g("value_bn.running_var"))
        self.value_fc1 = (g("value_fc1.weight"), g("value_fc1.bias"))
        self.value_fc2 = (g("value_fc2.weight"), g("value_fc2.bias"))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """x (NUM_PLANES, NROWS, NCOLS) → (policy_logits [NUM_ACTIONS], value scalar)."""
        h = _relu(_conv(np.ascontiguousarray(x, dtype=np.float32), *self.stem))
        for (w1, w2) in self.blocks:
            y = _relu(_conv(h, *w1))
            y = _conv(y, *w2)
            h = _relu(h + y)
        p = _relu(_conv(h, *self.policy_conv)).reshape(-1)
        p = self.policy_fc[0] @ p + self.policy_fc[1]
        v = _relu(_conv(h, *self.value_conv)).reshape(-1)
        v = _relu(self.value_fc1[0] @ v + self.value_fc1[1])
        v = np.tanh(self.value_fc2[0] @ v + self.value_fc2[1])
        return p.astype(np.float32), float(v[0])


def _read_meta(name: str) -> dict:
    p = resolve_checkpoint_path(CHECKPOINT_ROOT, name).with_suffix(".meta.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_np(name: str) -> NumpyAZNet:
    """Load an AZ checkpoint (by name) for torch-free inference."""
    from safetensors.numpy import load_file
    meta = _read_meta(name)
    weights = load_file(str(resolve_checkpoint_path(CHECKPOINT_ROOT, name)))
    return NumpyAZNet(weights, int(meta.get("channels", CHANNELS)),
                      int(meta.get("blocks", BLOCKS)))


def random_np(channels: int = CHANNELS, blocks: int = BLOCKS, seed: int = 0) -> NumpyAZNet:
    """A randomly-initialized net (for iteration-1 self-play with no checkpoint).
    Exact init doesn't matter — the net is random and MCTS adds Dirichlet noise."""
    rng = np.random.default_rng(seed)

    def conv(cout, cin, k):  # He-ish init
        return (rng.standard_normal((cout, cin, k, k)) * np.sqrt(2.0 / (cin * k * k))
                ).astype(np.float32)

    def lin(out, inp):
        return (rng.standard_normal((out, inp)) * np.sqrt(1.0 / inp)).astype(np.float32)

    def bn(n):  # identity BN
        return {"weight": np.ones(n, np.float32), "bias": np.zeros(n, np.float32),
                "running_mean": np.zeros(n, np.float32), "running_var": np.ones(n, np.float32)}

    w: Dict[str, np.ndarray] = {}
    w["stem.0.weight"] = conv(channels, NUM_PLANES, 3)
    for k, v in bn(channels).items():
        w[f"stem.1.{k}"] = v
    for i in range(blocks):
        w[f"trunk.{i}.conv1.weight"] = conv(channels, channels, 3)
        w[f"trunk.{i}.conv2.weight"] = conv(channels, channels, 3)
        for k, v in bn(channels).items():
            w[f"trunk.{i}.bn1.{k}"] = v
            w[f"trunk.{i}.bn2.{k}"] = v
    w["policy_conv.weight"] = conv(2, channels, 1)
    for k, v in bn(2).items():
        w[f"policy_bn.{k}"] = v
    w["policy_fc.weight"] = lin(NUM_ACTIONS, 2 * NROWS * NCOLS)
    w["policy_fc.bias"] = np.zeros(NUM_ACTIONS, np.float32)
    w["value_conv.weight"] = conv(1, channels, 1)
    for k, v in bn(1).items():
        w[f"value_bn.{k}"] = v
    w["value_fc1.weight"] = lin(channels, NROWS * NCOLS)
    w["value_fc1.bias"] = np.zeros(channels, np.float32)
    w["value_fc2.weight"] = lin(1, channels)
    w["value_fc2.bias"] = np.zeros(1, np.float32)
    return NumpyAZNet(w, channels, blocks)
