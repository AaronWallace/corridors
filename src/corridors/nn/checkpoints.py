"""Lightweight checkpoint discovery and Elo-based ordering."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _numeric_elo(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        elo = float(value)
    except (TypeError, ValueError):
        return None
    return elo if math.isfinite(elo) else None


def load_elo_ratings(root: Path) -> dict[str, float]:
    try:
        data = json.loads((root / "elo.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        str(name): elo
        for name, value in data.get("ratings", {}).items()
        if (elo := _numeric_elo(value)) is not None
    }


def checkpoint_elo(path: Path, ratings: dict[str, float] | None = None) -> float | None:
    ratings = ratings if ratings is not None else load_elo_ratings(path.parent)
    if path.stem in ratings:
        return ratings[path.stem]
    try:
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _numeric_elo(meta.get("elo"))


def ranked_checkpoint_paths(root: Path) -> list[Path]:
    """Return rated checkpoints by descending Elo, followed by unrated names."""
    if not root.exists():
        return []
    ratings = load_elo_ratings(root)
    paths = list(root.glob("*.safetensors"))
    elo_by_path = {path: checkpoint_elo(path, ratings) for path in paths}
    return sorted(
        paths,
        key=lambda path: (
            elo_by_path[path] is None,
            -(elo_by_path[path] or 0.0),
            path.stem.casefold(),
        ),
    )
