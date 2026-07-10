"""Persisted user settings for corridors.

Stored as JSON in <project-root>/corridors.json. Missing/invalid file falls back
to defaults; save() is best-effort (silently ignores write failures).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PATH = _PROJECT_ROOT / "corridors.json"

DEFAULTS: Dict[str, Any] = {
    "display": "live",      # "live" (full dashboard) or "headless" (one summary line)
    "workers": 0,           # 0 = auto (cpu_count - 1)
    "num_games": 10,        # how many games in a session
    "starts": "random",     # "fixed" or "random" per-game starting columns
    "p1_col": 4,            # used when starts == "fixed"
    "p2_col": 4,
    "depth": 4,             # AI iterative-deepening cap
    "time_limit": 3.0,      # per-move soft cap in seconds
    "tiebreak_epsilon": 15, # root-move variety for self-play
    "max_plies": 160,       # draw cap per game
    "use_persistent_tt": True,
    "tt_path": str(_PROJECT_ROOT / "corridors_tt.sqlite3"),
}


def load() -> Dict[str, Any]:
    out = dict(DEFAULTS)
    if not _PATH.exists():
        return out
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for k, v in data.items():
        if k in DEFAULTS:
            out[k] = v
    return out


def save(**kwargs: Any) -> None:
    data = load()
    data.update({k: v for k, v in kwargs.items() if k in DEFAULTS})
    try:
        _PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def path() -> Path:
    return _PATH
