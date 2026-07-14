"""Introspection helpers for the local CNN explorer web interface."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..game import Board, State, legal_moves
from .actions import move_to_index
from .encoding import encode_state

INPUT_PLANES = (
    "P1 pawn", "P2 pawn", "Horizontal walls", "Vertical walls",
    "P1 walls remaining", "P2 walls remaining", "Side to move",
    "P1 goal", "P2 goal",
)


def _numbers(tensor) -> dict[str, float]:
    array = tensor.detach().float().cpu().numpy()
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "zeroFraction": float(np.count_nonzero(np.abs(array) < 1e-8) / array.size),
    }


def _spatial_map(tensor) -> list[list[float]] | None:
    if tensor.ndim != 4:
        return None
    values = tensor.detach().float().abs().mean(dim=1)[0].cpu().numpy()
    return values.tolist()


def _activation_record(name: str, label: str, tensor, module=None) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "shape": list(tensor.shape[1:]),
        "stats": _numbers(tensor),
        "map": _spatial_map(tensor),
        "parameters": sum(p.numel() for p in module.parameters()) if module else 0,
    }


def _weight_details(module, out_channel: int, in_channel: int) -> dict[str, Any]:
    import torch

    named = [(name, value) for name, value in module.named_parameters()
             if value.numel()]
    if not named:
        return {"count": 0, "shape": [], "histogram": [], "kernel": None}
    flattened = torch.cat([value.detach().float().cpu().flatten()
                           for _name, value in named]).numpy()
    counts, edges = np.histogram(flattened, bins=31)
    weight_name, weight = next(
        ((name, value.detach().float().cpu()) for name, value in named
         if value.ndim >= 2),
        (named[0][0], named[0][1].detach().float().cpu()),
    )
    kernel = None
    selected = None
    if weight.ndim == 4:
        oc = out_channel % weight.shape[0]
        ic = in_channel % weight.shape[1]
        selected = weight[oc, ic].numpy().tolist()
        kernel = {"out": oc, "in": ic, "values": selected}
    elif weight.ndim == 2:
        oc = out_channel % weight.shape[0]
        selected = weight[oc, :min(64, weight.shape[1])].numpy().tolist()
        kernel = {"out": oc, "in": None, "values": [selected]}
    return {
        "tensor": weight_name,
        "shape": list(weight.shape),
        "count": int(flattened.size),
        "stats": {
            "min": float(flattened.min()), "max": float(flattened.max()),
            "mean": float(flattened.mean()), "std": float(flattened.std()),
            "zeroFraction": float(np.count_nonzero(np.abs(flattened) < 1e-8)
                                  / flattened.size),
        },
        "histogram": [
            {"from": float(edges[i]), "to": float(edges[i + 1]),
             "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "kernel": kernel,
    }


def _selected_activation(records, tensors, selected_layer: str,
                         channel: int) -> dict[str, Any] | None:
    if selected_layer not in tensors:
        selected_layer = next((r["name"] for r in records if r["map"] is not None), "")
    tensor = tensors.get(selected_layer)
    if tensor is None or tensor.ndim != 4:
        return None
    channels = tensor.shape[1]
    selected = channel % channels
    values = tensor[0, selected].detach().float().cpu().numpy()
    energy = tensor[0].detach().float().abs().mean(dim=(1, 2)).cpu().numpy()
    top = np.argsort(energy)[-min(12, channels):][::-1]
    return {
        "layer": selected_layer,
        "channel": selected,
        "channels": channels,
        "map": values.tolist(),
        "stats": _numbers(tensor[:, selected:selected + 1]),
        "topChannels": [
            {"channel": int(index), "energy": float(energy[index])}
            for index in top
        ],
    }


def analyze_model(model, architecture: str, state: State, board: Board,
                  selected_layer: str = "stem", channel: int = 0,
                  weight_out: int = 0, weight_in: int = 0) -> dict[str, Any]:
    """Run one real position through a model and return JSON-ready internals."""
    import torch
    import torch.nn.functional as functional

    x = torch.from_numpy(encode_state(state, board)).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)
    records: list[dict[str, Any]] = []
    tensors = {"input": x}
    modules = {}
    records.append(_activation_record("input", "Encoded input", x))

    with torch.no_grad():
        stem = model.stem(x)
        tensors["stem"] = stem
        modules["stem"] = model.stem
        records.append(_activation_record("stem", "Stem convolution", stem, model.stem))
        trunk = stem
        for index, block in enumerate(model.trunk, 1):
            trunk = block(trunk)
            name = f"block_{index}"
            tensors[name] = trunk
            modules[name] = block
            records.append(_activation_record(name, f"Residual block {index}", trunk, block))

        if architecture == "az":
            policy_space = functional.relu(model.policy_bn(model.policy_conv(trunk)))
            value_space = functional.relu(model.value_bn(model.value_conv(trunk)))
            policy_logits = model.policy_fc(policy_space.flatten(1))[0]
            value_hidden = functional.relu(model.value_fc1(value_space.flatten(1)))
            value = torch.tanh(model.value_fc2(value_hidden)).squeeze()
            for name, label, tensor, module in (
                ("policy_space", "Policy spatial head", policy_space, model.policy_conv),
                ("value_space", "Value spatial head", value_space, model.value_conv),
                ("policy_logits", "Policy logits", policy_logits.unsqueeze(0), model.policy_fc),
                ("value_hidden", "Value hidden layer", value_hidden, model.value_fc1),
            ):
                tensors[name] = tensor
                modules[name] = module
                records.append(_activation_record(name, label, tensor, module))
            moves = legal_moves(state, board)
            indices = [move_to_index(move) for move in moves]
            legal_logits = policy_logits[indices]
            probabilities = torch.softmax(legal_logits, dim=0).cpu().numpy()
            order = np.argsort(probabilities)[::-1][:12]
            policy = [
                {
                    "move": {"kind": moves[i][0], "at": list(moves[i][1])},
                    "probability": float(probabilities[i]),
                    "logit": float(legal_logits[i]),
                }
                for i in order
            ]
        else:
            pooled = trunk.mean(dim=(2, 3))
            hidden = model.head[1](model.head[0](pooled))
            value = model.head[3](model.head[2](hidden)).squeeze()
            tensors["global_pool"] = pooled
            records.append(_activation_record(
                "global_pool", "Global average pool", pooled))
            tensors["value_hidden"] = hidden
            modules["value_hidden"] = model.head[0]
            records.append(_activation_record(
                "value_hidden", "Value hidden layer", hidden, model.head[0]))
            policy = []

    selected = _selected_activation(records, tensors, selected_layer, channel)
    selected_name = selected["layer"] if selected else selected_layer
    weight_module = modules.get(selected_name)
    weights = (_weight_details(weight_module, weight_out, weight_in)
               if weight_module else {"count": 0, "shape": [], "histogram": [], "kernel": None})
    return {
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "inputPlanes": list(INPUT_PLANES),
        "input": x[0].detach().float().cpu().numpy().tolist(),
        "layers": records,
        "selectedActivation": selected,
        "weights": weights,
        "value": float(value.detach().cpu()),
        "valuePerspective": f"Player {state.turn} to move",
        "policy": policy,
    }
