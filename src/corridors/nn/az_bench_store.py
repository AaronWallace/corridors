"""Persistent AlphaZero benchmark history.

Full run history is appended per hardware fingerprint, one file per
fingerprint under az_benchmarks/ (next to corridors.json). The directory is
committed to the repo — records carry no machine-transient identity, so pods
of a known hardware class inherit tuned defaults without re-benchmarking —
and per-fingerprint files mean two hardware classes never touch the same
file, so concurrent pods cannot merge-conflict. A pre-directory monolithic
az_benchmarks.json is still read as a (machine-local) legacy fallback. The
latest winner is mirrored into the "az_tuning_profiles" settings entry by the
benchmark code, so detect_hardware() keeps working from settings alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_RECORDS_PER_FINGERPRINT = 50

# Stable hardware identity only: no hostnames, pod names, or point-in-time
# free-memory readings. vram_gb/ram_gb arrive pre-rounded so the same box
# always fingerprints identically.
FINGERPRINT_FIELDS = ("device", "gpu_name", "vram_gb", "gpu_count", "ncpu",
                      "ram_gb")


def _store_dir() -> Path:
    from .. import settings
    return settings.path().with_name("az_benchmarks")


def _legacy_path() -> Path:
    from .. import settings
    return settings.path().with_name("az_benchmarks.json")


def _file_for(hardware_key: str) -> Path:
    """Per-fingerprint file: readable slug plus a short hash for uniqueness
    (keys contain '|', spaces, and '=' which are not filesystem-safe)."""
    slug = re.sub(r"[^a-z0-9.]+", "-", hardware_key.lower()).strip("-")[:80]
    digest = hashlib.sha1(hardware_key.encode("utf-8")).hexdigest()[:8]
    return _store_dir() / f"{slug}-{digest}.json"


def hardware_fingerprint(hw: dict) -> dict:
    return {
        "device": str(hw.get("device", "cpu")),
        "gpu_name": str(hw.get("gpu_name", "")),
        "vram_gb": round(float(hw.get("vram_gb", 0.0)), 1),
        "gpu_count": int(hw.get("gpu_count", 0)),
        "ncpu": int(hw.get("ncpu", 0)),
        "ram_gb": int(hw.get("ram_gb", 0)),
    }


def load_store() -> Dict[str, List[dict]]:
    """All recorded benchmark history, keyed by hardware fingerprint key."""
    records: Dict[str, List[dict]] = {}
    legacy = _legacy_path()
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            old = data.get("records") if isinstance(data, dict) else None
            if isinstance(old, dict):
                records.update(old)
        except (OSError, ValueError):
            pass
    store_dir = _store_dir()
    if store_dir.is_dir():
        for path in sorted(store_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            key = data.get("hardware_key") if isinstance(data, dict) else None
            history = data.get("records") if isinstance(data, dict) else None
            if isinstance(key, str) and isinstance(history, list):
                # Per-fingerprint files supersede any legacy monolith copy.
                records[key] = history
    return records


def _save_key(hardware_key: str, history: List[dict]) -> None:
    try:
        _store_dir().mkdir(parents=True, exist_ok=True)
        _file_for(hardware_key).write_text(
            json.dumps({"version": 1, "hardware_key": hardware_key,
                        "records": history}, indent=2),
            encoding="utf-8")
    except OSError:
        pass


def append_record(hardware_key: str, record: dict) -> None:
    history = list(load_store().get(hardware_key, []))
    history.append(record)
    _save_key(hardware_key, history[-MAX_RECORDS_PER_FINGERPRINT:])


def records_for(hardware_key: str) -> List[dict]:
    """Oldest-first history for one fingerprint."""
    return list(load_store().get(hardware_key, []))


def latest_record(hardware_key: str) -> Optional[dict]:
    history = records_for(hardware_key)
    return history[-1] if history else None


def select_best(rows: List[dict]) -> Tuple[dict, bool]:
    """(best numerics-neutral row for the tuning profile, whether an fp16 row
    beat it).

    fp16 and torch.compile change network outputs, so they never win the
    auto-applied profile — they are recorded and surfaced as recommendations
    only. (The profile schema carries no fp16/compile keys either way, so
    neither can leak into a loop config implicitly.)
    """
    neutral = [row for row in rows
               if not row.get("fp16") and not row.get("compile")]
    pool = neutral or rows
    best = max(pool, key=lambda row: row.get("positions_per_s", 0.0))
    fp16_recommended = any(
        row.get("fp16")
        and row.get("positions_per_s", 0.0) > best.get("positions_per_s", 0.0)
        for row in rows)
    return best, fp16_recommended
