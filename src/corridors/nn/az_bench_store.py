"""Persistent AlphaZero benchmark history.

Full run history is appended per hardware fingerprint to az_benchmarks.json
(a sidecar next to corridors.json — the history is too bulky and append-only
for the settings whitelist), capped per fingerprint. The latest winner is
mirrored into the "az_tuning_profiles" settings entry by the benchmark code,
so detect_hardware() keeps working from settings alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_RECORDS_PER_FINGERPRINT = 50

# Stable hardware identity only: no hostnames, pod names, or point-in-time
# free-memory readings. vram_gb/ram_gb arrive pre-rounded so the same box
# always fingerprints identically.
FINGERPRINT_FIELDS = ("device", "gpu_name", "vram_gb", "gpu_count", "ncpu",
                      "ram_gb")


def _store_path() -> Path:
    from .. import settings
    return settings.path().with_name("az_benchmarks.json")


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
    path = _store_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    records = data.get("records") if isinstance(data, dict) else None
    return records if isinstance(records, dict) else {}


def _save_store(records: Dict[str, List[dict]]) -> None:
    try:
        _store_path().write_text(
            json.dumps({"version": 1, "records": records}, indent=2),
            encoding="utf-8")
    except OSError:
        pass


def append_record(hardware_key: str, record: dict) -> None:
    records = load_store()
    history = list(records.get(hardware_key, []))
    history.append(record)
    records[hardware_key] = history[-MAX_RECORDS_PER_FINGERPRINT:]
    _save_store(records)


def records_for(hardware_key: str) -> List[dict]:
    """Oldest-first history for one fingerprint."""
    return list(load_store().get(hardware_key, []))


def latest_record(hardware_key: str) -> Optional[dict]:
    history = records_for(hardware_key)
    return history[-1] if history else None


def select_best(rows: List[dict]) -> Tuple[dict, bool]:
    """(best fp32 row for the tuning profile, whether an fp16 row beat it).

    fp16 changes network outputs, so it never wins the auto-applied profile —
    it is recorded and surfaced as a recommendation only.
    """
    fp32 = [row for row in rows if not row.get("fp16")]
    pool = fp32 or rows
    best = max(pool, key=lambda row: row.get("positions_per_s", 0.0))
    fp16_recommended = any(
        row.get("fp16")
        and row.get("positions_per_s", 0.0) > best.get("positions_per_s", 0.0)
        for row in rows)
    return best, fp16_recommended
