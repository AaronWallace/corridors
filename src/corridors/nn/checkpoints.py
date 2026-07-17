"""Lightweight checkpoint discovery and Elo-based ordering."""

from __future__ import annotations

import json
import math
from pathlib import Path


def local_checkpoint_path(root: Path, name: str) -> Path:
    """Writable machine-local path for a logical checkpoint name."""
    if name.endswith(".safetensors"):
        name = name[:-len(".safetensors")]
    return root / f"{name}.safetensors"


def curated_checkpoint_path(root: Path, name: str) -> Path:
    """Git-tracked curated path for a logical checkpoint name."""
    if name.endswith(".safetensors"):
        name = name[:-len(".safetensors")]
    return root / "best" / f"{name}.safetensors"


def resolve_checkpoint_path(root: Path, name: str) -> Path:
    """Prefer a local checkpoint, falling back transparently to best/."""
    local = local_checkpoint_path(root, name)
    if local.exists():
        return local
    curated = curated_checkpoint_path(root, name)
    return curated if curated.exists() else local


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


def stale_elo_checkpoints(root: Path) -> set[str]:
    """Rated checkpoints that sat out the latest round robin.

    Their Elo still comes from full-history recomputation, but it wasn't
    refreshed with new games, so surfaces flag it as stale. Empty when no
    tournament has recorded its participant list yet."""
    try:
        data = json.loads((root / "elo.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    participants = set((data.get("last_run") or {}).get("checkpoints") or [])
    if not participants:
        return set()
    anchor = data.get("anchor", "classical")
    return {
        str(name)
        for name in data.get("ratings", {})
        if name not in participants and name != anchor
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
    # A local checkpoint shadows a curated copy with the same logical name.
    by_name = {path.stem: path for path in (root / "best").glob("*.safetensors")}
    by_name.update({path.stem: path for path in root.glob("*.safetensors")})
    paths = list(by_name.values())
    elo_by_path = {path: checkpoint_elo(path, ratings) for path in paths}
    return sorted(
        paths,
        key=lambda path: (
            elo_by_path[path] is None,
            -(elo_by_path[path] or 0.0),
            path.stem.casefold(),
        ),
    )
