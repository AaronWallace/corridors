"""Candidate-versus-incumbent gating for iterative AlphaZero training."""

from __future__ import annotations

import json
import shutil
import time
from typing import Callable, Optional

from .az_net import CHECKPOINT_ROOT, checkpoint_path, meta_path
from .tournament import AgentSpec, _AGENT_CACHE, play_pair_game


def run_arena(incumbent: str, candidate: str, games: int = 20,
              max_plies: int = 150, device: str = "cpu",
              on_game: Optional[Callable[[int, int, float], None]] = None) -> dict:
    """Play color-balanced games and return the candidate's aggregate score."""
    # Both checkpoint names are overwritten across loop iterations; never reuse
    # model objects cached for a previous arena.
    _AGENT_CACHE.clear()
    incumbent_spec = AgentSpec("net", incumbent, epsilon_band=0.0)
    candidate_spec = AgentSpec("net", candidate, epsilon_band=0.0)
    score = wins = draws = losses = 0.0
    t0 = time.monotonic()
    for game in range(games):
        if game % 2 == 0:
            result = play_pair_game(candidate_spec, incumbent_spec, game,
                                    device=device, max_plies=max_plies)
        else:
            incumbent_score = play_pair_game(incumbent_spec, candidate_spec, game,
                                              device=device, max_plies=max_plies)
            result = 1.0 - incumbent_score
        score += result
        if result == 1.0:
            wins += 1
        elif result == 0.5:
            draws += 1
        else:
            losses += 1
        if on_game:
            on_game(game + 1, games, result)
    result = {
        "games": games, "wins": int(wins), "draws": int(draws),
        "losses": int(losses), "score": score / max(games, 1),
        "elapsed": time.monotonic() - t0,
    }
    _AGENT_CACHE.clear()
    return result


def promote_candidate(candidate: str, incumbent: str, arena: dict) -> None:
    """Atomically replace incumbent weights/meta with the accepted candidate."""
    dst_w = checkpoint_path(incumbent)
    dst_w.parent.mkdir(parents=True, exist_ok=True)
    tmp_w = dst_w.with_name("." + dst_w.name + ".promote.tmp")
    shutil.copyfile(checkpoint_path(candidate), tmp_w)
    tmp_w.replace(dst_w)

    meta = {}
    candidate_meta = meta_path(candidate)
    if candidate_meta.exists():
        try:
            meta = json.loads(candidate_meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    meta.update({
        "promoted_from": candidate,
        "arena": arena,
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    dst_m = meta_path(incumbent)
    tmp_m = dst_m.with_name("." + dst_m.name + ".promote.tmp")
    tmp_m.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp_m.replace(dst_m)
