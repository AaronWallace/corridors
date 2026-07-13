"""Self-play dataset storage.

Layout (all under <project-root>/nn_data/):

    nn_data/<dataset-name>/
        manifest.json      config + shard list (atomic writes with .bak rollback)
        shard_000.npz      tensors (N,9,11,9) f32, outcomes (N,) i8, tt_scores (N,) f32
        shard_001.npz      ...

Shards are one-per-worker per generation run; appending a new run to an existing
dataset adds shards (a config mismatch warns but is allowed).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = _PROJECT_ROOT / "nn_data"


@dataclass
class DatasetConfig:
    depth: int
    time_limit: float
    tiebreak_epsilon: int
    max_plies: int
    starts: str


def dataset_dir(name: str) -> Path:
    return DATA_ROOT / name


def default_dataset_name(games: int, depth: int, eps: int) -> str:
    return f"{games}g_d{depth}_e{eps}"


def _manifest_path(name: str) -> Path:
    return dataset_dir(name) / "manifest.json"


def read_manifest(name: str) -> Optional[dict]:
    p = _manifest_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        bak = p.with_suffix(".json.bak")
        if bak.exists():
            try:
                return json.loads(bak.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        return None


def write_manifest(name: str, manifest: dict) -> None:
    d = dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    p = _manifest_path(name)
    if p.exists():
        shutil.copy2(p, p.with_suffix(".json.bak"))
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def next_shard_index(name: str) -> int:
    d = dataset_dir(name)
    if not d.exists():
        return 0
    best = -1
    for f in d.glob("shard_*.npz"):
        m = re.match(r"shard_(\d+)\.npz$", f.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def write_shard(name: str, shard_index: int,
                tensors: np.ndarray, outcomes: np.ndarray, tt_scores: np.ndarray) -> Path:
    d = dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"shard_{shard_index:03d}.npz"
    # np.savez appends ".npz" unless the name already ends with it, so the temp
    # file must carry the suffix; leading "." keeps it out of shard globs.
    tmp = d / f".tmp_shard_{shard_index:03d}.npz"
    np.savez_compressed(tmp, tensors=tensors, outcomes=outcomes, tt_scores=tt_scores)
    os.replace(tmp, path)
    return path


def register_run(name: str, config: DatasetConfig, new_shards: List[str],
                 games: int, positions: int) -> None:
    manifest = read_manifest(name)
    if manifest is None:
        manifest = {"config": asdict(config), "shards": [], "games": 0, "positions": 0}
    manifest["shards"].extend(new_shards)
    manifest["games"] = manifest.get("games", 0) + games
    manifest["positions"] = manifest.get("positions", 0) + positions
    write_manifest(name, manifest)


def config_mismatch(name: str, config: DatasetConfig) -> Optional[str]:
    manifest = read_manifest(name)
    if manifest is None:
        return None
    old = manifest.get("config", {})
    new = asdict(config)
    diffs = [f"{k}: {old.get(k)!r} -> {v!r}" for k, v in new.items() if old.get(k) != v]
    return ", ".join(diffs) if diffs else None


def _shard_files(d: Path) -> List[Path]:
    """Shard files for a dataset dir. Self-play writes ``shard_*.npz``; the legacy
    AlphaZero loop writes ``iter_*.npz``."""
    return sorted(d.glob("shard_*.npz")) + sorted(d.glob("iter_*.npz"))


def is_dataset_dir(d: Path) -> bool:
    """A directory holds a dataset iff it directly contains shard files or a
    metadata file. A pure *container* of nested runs (e.g. ``nn_data/alphazero``)
    holds neither directly, so it is never treated as a deletable dataset."""
    if not d.is_dir():
        return False
    return ((d / "manifest.json").exists() or (d / "run.json").exists()
            or bool(_shard_files(d)))


def _read_meta(d: Path) -> Optional[dict]:
    """Normalize a dataset dir's metadata (classical ``manifest.json`` or
    AlphaZero ``run.json``) into a common shape for listing."""
    for fname, cfg_key in (("manifest.json", "config"), ("run.json", "selfplay")):
        p = d / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return {
            "games": data.get("games", "?"),
            "positions": data.get("positions", "?"),
            "config": data.get(cfg_key, {}),
        }
    return None


def _dataset_entry(d: Path, name: str) -> Dict:
    m = _read_meta(d)
    shards = _shard_files(d)
    size = sum(f.stat().st_size for f in shards)
    return {
        "name": name,
        "games": m["games"] if m else "?",
        "positions": m["positions"] if m else "?",
        "shards": len(shards),
        "size_mb": size / 1e6,
        "config": (m or {}).get("config", {}),
    }


def list_datasets() -> List[Dict]:
    """List every manageable dataset. Leaf dataset dirs under ``nn_data/`` are
    listed directly; a container dir (one holding only nested run dirs, such as
    ``alphazero``) is expanded so each run is listed individually as
    ``<container>/<run>`` rather than as the whole folder."""
    out: List[Dict] = []
    if not DATA_ROOT.exists():
        return out
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if is_dataset_dir(d):
            out.append(_dataset_entry(d, d.name))
            continue
        # Container: surface its nested runs individually.
        for sub in sorted(d.iterdir()):
            if is_dataset_dir(sub):
                out.append(_dataset_entry(sub, f"{d.name}/{sub.name}"))
    return out


def delete_dataset(name: str) -> bool:
    """Delete a single dataset dir. Refuses anything that is not a leaf dataset
    dir strictly inside ``nn_data/`` — so a container like ``alphazero`` (or
    ``nn_data`` itself) can never be wiped, only the runs beneath it."""
    d = dataset_dir(name)
    if not d.exists() or not d.is_dir() or not is_dataset_dir(d):
        return False
    if DATA_ROOT.resolve() not in d.resolve().parents:
        return False
    shutil.rmtree(d)
    return True


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all shards. Returns (tensors, outcomes, tt_scores)."""
    d = dataset_dir(name)
    shards = sorted(d.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {d}")
    ts, os_, ss = [], [], []
    for s in shards:
        with np.load(s) as z:
            ts.append(z["tensors"])
            os_.append(z["outcomes"])
            ss.append(z["tt_scores"])
    return np.concatenate(ts), np.concatenate(os_), np.concatenate(ss)
