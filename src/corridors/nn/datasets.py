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
import math
import os
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = _PROJECT_ROOT / "nn_data"
ARCHIVE_DIRNAME = "archive"
SHARED_DIRNAME = "shared"
INDEX_FILENAME = ".dataset-index.json"
INDEX_VERSION = 1


@dataclass
class DatasetConfig:
    depth: int
    time_limit: float
    tiebreak_epsilon: int
    max_plies: int
    starts: str


def dataset_dir(name: str) -> Path:
    return DATA_ROOT / name


def archive_root() -> Path:
    return DATA_ROOT / ARCHIVE_DIRNAME


def shared_root() -> Path:
    return DATA_ROOT / SHARED_DIRNAME


def _index_path(d: Path) -> Path:
    return d / INDEX_FILENAME


def _read_index(d: Path) -> dict:
    path = _index_path(d)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == INDEX_VERSION and isinstance(data.get("shards"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"version": INDEX_VERSION, "shards": {}}


def _write_index(d: Path, data: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    path = _index_path(d)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _index_summary(records: Dict[str, dict]) -> dict:
    positions = [record.get("positions") for record in records.values()]
    games = [record.get("games") for record in records.values()]
    return {
        "shards": len(records),
        "size": sum(record.get("size", 0) for record in records.values()),
        "positions": (sum(positions) if positions
                      and all(isinstance(value, int) for value in positions) else None),
        "games": (sum(games) if games
                  and all(isinstance(value, int) for value in games) else None),
        "latest_mtime_ns": max(
            (record.get("mtime_ns", 0) for record in records.values()), default=0),
    }


def register_shard_metadata(path: Path, positions: int,
                            games: Optional[int] = None) -> None:
    """Record a newly-written shard without reopening its compressed arrays."""
    stat = path.stat()
    data = _read_index(path.parent)
    record = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "positions": int(positions),
        "games": int(games) if games is not None else None,
    }
    data["shards"][path.name] = record
    data["summary"] = _index_summary(data["shards"])
    _write_index(path.parent, data)


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
                tensors: np.ndarray, outcomes: np.ndarray, tt_scores: np.ndarray,
                moves: Optional[np.ndarray] = None) -> Path:
    """Write a classical-autoplay shard. `moves` (int16 action indices) is
    optional for backward compatibility; new runs record it so the shard can
    be converted to AlphaZero training format later."""
    d = dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"shard_{shard_index:03d}.npz"
    tmp = d / f".tmp_shard_{shard_index:03d}.npz"
    arrays = {"tensors": tensors, "outcomes": outcomes, "tt_scores": tt_scores}
    if moves is not None:
        arrays["moves"] = moves
    np.savez_compressed(tmp, **arrays)
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
    # Worker processes may have written several shards concurrently. Index them
    # once here in the parent rather than racing on one metadata file.
    d = dataset_dir(name)
    _refresh_shard_index(d, _shard_files(d))


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


def _npz_array_rows(path: Path) -> Optional[int]:
    """Read the leading dimension from an NPZ array header without loading it."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            member = "states.npy" if "states.npy" in names else "tensors.npy"
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                reader = (np.lib.format.read_array_header_1_0
                          if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _fortran, _dtype = reader(stream)
            return int(shape[0]) if shape else 0
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _shard_positions(shards: List[Path]) -> Optional[int]:
    counts = [_npz_array_rows(path) for path in shards]
    return sum(counts) if all(count is not None for count in counts) else None


def _npz_games(path: Path) -> Optional[int]:
    """Read the small per-shard game counter used by newer AlphaZero shards."""
    try:
        with zipfile.ZipFile(path) as archive:
            if "games.npy" not in archive.namelist():
                return None
            with archive.open("games.npy") as stream:
                return int(np.load(stream, allow_pickle=False))
    except (OSError, ValueError, zipfile.BadZipFile):
        return None


def _shard_games(shards: List[Path]) -> Optional[int]:
    counts = [_npz_games(path) for path in shards]
    return sum(counts) if all(count is not None for count in counts) else None


def _legacy_az_game_starts(path: Path) -> Optional[int]:
    """Count unmistakable initial positions in a legacy AZ shard."""
    try:
        with np.load(path) as archive:
            states = archive["states"]
            if states.ndim != 4 or states.shape[1] < 2 or states.shape[2] < 11:
                return None
            p1_home = states[:, 0, 10, :].sum(axis=1) > 0.5
            p2_home = states[:, 1, 0, :].sum(axis=1) > 0.5
            return int(np.count_nonzero(p1_home & p2_home))
    except (OSError, KeyError, ValueError):
        return None


def _refresh_shard_index(d: Path, shards: List[Path]) -> Dict[str, dict]:
    """Stat every shard but open only files absent from or changed in the index."""
    data = _read_index(d)
    records = data["shards"]
    current = {path.name for path in shards}
    changed = not _index_path(d).exists()

    # Only nameless, metadata-free legacy AZ runs need the expensive initial
    # position scan used to infer game starts. Cache that result too.
    needs_starts = (
        d.parent.name == "alphazero"
        and not (d / "run.json").exists()
        and re.search(r"_g\d+_s\d+", d.name) is None
    )
    for path in shards:
        stat = path.stat()
        record = records.get(path.name)
        valid = (
            isinstance(record, dict)
            and record.get("size") == stat.st_size
            and record.get("mtime_ns") == stat.st_mtime_ns
            and "positions" in record
            and "games" in record
            and (not needs_starts or "game_starts" in record)
        )
        if valid:
            continue
        record = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "positions": _npz_array_rows(path),
            "games": _npz_games(path),
        }
        if needs_starts:
            record["game_starts"] = _legacy_az_game_starts(path)
        records[path.name] = record
        changed = True

    for missing in set(records) - current:
        del records[missing]
        changed = True
    summary = _index_summary(records)
    if data.get("summary") != summary:
        data["summary"] = summary
        changed = True
    if changed:
        _write_index(d, data)
    return records


def _indexed_sum(records: Dict[str, dict], key: str) -> Optional[int]:
    values = [record.get(key) for record in records.values()]
    return sum(values) if values and all(isinstance(v, int) for v in values) else None


def _legacy_az_games(data: dict, shard_count: int) -> int | str:
    """Best available game count for run.json files created before counters."""
    config = data.get("selfplay", {})
    per_round = int(config.get("num_games", 0) or 0)
    if per_round <= 0 or shard_count <= 0:
        return "?"
    if data.get("mode") != "loop":
        return per_round
    shards_per_round = math.ceil(per_round / 25)
    full_rounds, partial_shards = divmod(shard_count, shards_per_round)
    configured_rounds = int(data.get("iterations", full_rounds) or full_rounds)
    full_rounds = min(full_rounds, configured_rounds)
    if partial_shards == 0:
        return full_rounds * per_round
    # Old shards did not record their game count. All but the final shard in a
    # partial round hold 25 games; the final one holds at least one.
    minimum = full_rounds * per_round + (partial_shards - 1) * 25 + 1
    return f">={minimum}"


def _legacy_name_meta(d: Path, shards: List[Path],
                      records: Dict[str, dict]) -> Optional[dict]:
    """Recover useful metadata from pre-run.json AlphaZero dataset names."""
    positions = _indexed_sum(records, "positions")
    is_az = d.parent.name == "alphazero"
    if not is_az:
        return {
            "games": "?",
            "positions": positions if positions is not None else "?",
            "config": {},
            "kind": "unknown",
        }
    match = re.search(r"_g(\d+)_s(\d+)(?:-(\d+))?(?:_p(\d+))?", d.name)
    config = {}
    games: int | str = "?"
    if match:
        per_round = int(match.group(1))
        minimum = int(match.group(2))
        maximum = int(match.group(3) or minimum)
        config["simulations"] = maximum
        if minimum != maximum:
            config["min_mcts"] = minimum
            config["max_mcts"] = maximum
        if match.group(4):
            config["max_plies"] = int(match.group(4))
        games = _legacy_az_games(
            {"mode": "loop", "selfplay": {"num_games": per_round}},
            len(shards),
        )
    else:
        starts = [records[path.name].get("game_starts") for path in shards]
        if all(count is not None for count in starts):
            games = sum(starts)
    return {
        "games": games,
        "positions": positions if positions is not None else "?",
        "config": config,
        "kind": "alphazero",
    }


def _read_meta(d: Path, shards: List[Path], records: Dict[str, dict]) -> Optional[dict]:
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
        positions = data.get("positions")
        indexed_positions = _indexed_sum(records, "positions")
        if indexed_positions is not None:
            positions = indexed_positions
        games = data.get("games")
        if fname == "run.json":
            indexed_games = _indexed_sum(records, "games")
            if indexed_games is not None:
                games = indexed_games
            elif games is None or (games == 0 and shards):
                games = _legacy_az_games(data, len(shards))
        else:
            expected = set(data.get("shards", []))
            actual = {path.name for path in shards}
            if expected and expected != actual:
                games = "?"
        return {
            "games": games if games is not None else "?",
            "positions": positions if positions is not None else "?",
            "config": data.get(cfg_key, {}),
            "kind": "alphazero" if fname == "run.json" else "classical",
        }
    return _legacy_name_meta(d, shards, records) if shards else None


def _dir_signature(d: Path) -> Optional[list]:
    """A fingerprint of the dataset dir's contents-of-interest — invariant to
    writes of our own _index.json (which would otherwise create a chicken-and-
    egg problem with dir-mtime-based caching). Uses a single scandir + N cheap
    stats; on modern filesystems this is far cheaper than opening any shard."""
    try:
        npz_count = 0
        npz_size = 0
        manifest_mtime = 0
        run_mtime = 0
        for e in os.scandir(d):
            n = e.name
            if n.endswith(".npz"):
                st = e.stat()
                npz_count += 1
                npz_size += st.st_size
            elif n == "manifest.json":
                manifest_mtime = e.stat().st_mtime_ns
            elif n == "run.json":
                run_mtime = e.stat().st_mtime_ns
    except OSError:
        return None
    return [npz_count, npz_size, manifest_mtime, run_mtime]


def _dataset_entry(d: Path, name: str) -> Dict:
    # Fast path: skip opening any shards / rebuilding the full index when the
    # dataset's fingerprint (npz count + total size + manifest/run mtimes) is
    # unchanged since we last cached the entry. Signature ignores our own
    # _index.json writes, so it's stable across repeated listings.
    sig = _dir_signature(d)
    if sig is not None:
        try:
            cached = _read_index(d).get("entry")
            if (isinstance(cached, dict) and cached.get("_v") == 1
                    and cached.get("sig") == sig):
                return {**cached["entry"], "name": name}
        except OSError:
            pass

    # Slow path — refresh the shard index and persist a cacheable entry.
    shards = _shard_files(d)
    records = _refresh_shard_index(d, shards)
    m = _read_meta(d, shards, records)
    size = sum(record["size"] for record in records.values())
    dated_files = [
        path for path in [*shards, d / "manifest.json", d / "run.json"]
        if path.exists()
    ]
    modified = max(
        (path.stat().st_mtime for path in dated_files),
        default=d.stat().st_mtime,
    )
    entry = {
        "name": name,
        "games": m["games"] if m else "?",
        "positions": m["positions"] if m else "?",
        "shards": len(shards),
        "size_mb": size / 1e6,
        "modified": modified,
        "config": (m or {}).get("config", {}),
        "kind": (m or {}).get("kind", "unknown"),
    }
    # Persist for the fast path next time — re-signature AFTER we've done any
    # index writes so the stored sig reflects the same state we return.
    try:
        data = _read_index(d)
        final_sig = _dir_signature(d)  # after _refresh_shard_index may have written
        data["entry"] = {
            "_v": 1,
            "sig": final_sig,
            "entry": {k: v for k, v in entry.items() if k != "name"},
        }
        _write_index(d, data)
    except OSError:
        pass
    return entry


def _iter_dataset_dirs(root: Path):
    """Yield subdirs under `root` using scandir (much faster than Path.rglob on
    big trees — scandir returns names + is_dir() from one syscall per parent)."""
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for e in entries:
        if not e.is_dir(follow_symlinks=False):
            continue
        yield Path(e.path)
        yield from _iter_dataset_dirs(Path(e.path))


def list_datasets() -> List[Dict]:
    """List every manageable dataset. Leaf dataset dirs under ``nn_data/`` are
    listed directly; a container dir (one holding only nested run dirs, such as
    ``alphazero``) is expanded so each run is listed individually as
    ``<container>/<run>`` rather than as the whole folder."""
    out: List[Dict] = []
    if not DATA_ROOT.exists():
        return out
    archive = archive_root()
    for d in _iter_dataset_dirs(DATA_ROOT):
        if d == archive or archive in d.parents:
            continue
        if is_dataset_dir(d):
            out.append(_dataset_entry(d, d.relative_to(DATA_ROOT).as_posix()))
    out.sort(key=lambda item: (-item["modified"], item["name"]))
    return out


def list_archived_datasets() -> List[Dict]:
    """Archived datasets, kept separate from every training-data selector."""
    root = archive_root()
    if not root.exists():
        return []
    out = []
    for d in root.rglob("*"):
        if d.is_dir() and is_dataset_dir(d):
            name = d.relative_to(root).as_posix()
            out.append({**_dataset_entry(d, name), "archived": True})
    out.sort(key=lambda item: (-item["modified"], item["name"]))
    return out


def _safe_named_path(root: Path, name: str) -> Optional[Path]:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    return resolved if resolved_root in resolved.parents else None


def _remove_empty_parents(path: Path, stop: Path) -> None:
    parent = path.parent
    stop = stop.resolve()
    while parent.resolve() != stop:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def archive_dataset(name: str) -> bool:
    if Path(name).parts[:1] == (ARCHIVE_DIRNAME,):
        return False
    source = _safe_named_path(DATA_ROOT, name)
    target = _safe_named_path(archive_root(), name)
    if source is None or target is None or not is_dataset_dir(source) or target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    _remove_empty_parents(source, DATA_ROOT)
    return True


def share_dataset(name: str) -> Optional[str]:
    """Move an active dataset into the Git-visible shared tree."""
    parts = Path(name).parts
    if not parts or parts[0] in (ARCHIVE_DIRNAME, SHARED_DIRNAME):
        return None
    source = _safe_named_path(DATA_ROOT, name)
    shared_name = f"{SHARED_DIRNAME}/{Path(name).as_posix()}"
    target = _safe_named_path(DATA_ROOT, shared_name)
    if source is None or target is None or not is_dataset_dir(source) or target.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    _remove_empty_parents(source, DATA_ROOT)
    return shared_name


def restore_dataset(name: str) -> bool:
    if Path(name).parts[:1] == (ARCHIVE_DIRNAME,):
        return False
    source = _safe_named_path(archive_root(), name)
    target = _safe_named_path(DATA_ROOT, name)
    if source is None or target is None or not is_dataset_dir(source) or target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    _remove_empty_parents(source, archive_root())
    return True


def delete_archived_dataset(name: str) -> bool:
    d = _safe_named_path(archive_root(), name)
    if d is None or not is_dataset_dir(d):
        return False
    shutil.rmtree(d)
    _remove_empty_parents(d, archive_root())
    return True


def delete_dataset(name: str) -> bool:
    """Delete a single dataset dir. Refuses anything that is not a leaf dataset
    dir strictly inside ``nn_data/`` — so a container like ``alphazero`` (or
    ``nn_data`` itself) can never be wiped, only the runs beneath it."""
    if Path(name).parts[:1] == (ARCHIVE_DIRNAME,):
        return False
    d = _safe_named_path(DATA_ROOT, name)
    if d is None:
        return False
    if not d.exists() or not d.is_dir() or not is_dataset_dir(d):
        return False
    shutil.rmtree(d)
    _remove_empty_parents(d, DATA_ROOT)
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
