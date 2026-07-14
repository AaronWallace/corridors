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
              on_game: Optional[Callable] = None) -> dict:
    """Play color-balanced games and return the candidate's aggregate score."""
    # Both checkpoint names are overwritten across loop iterations; never reuse
    # model objects cached for a previous arena.
    _AGENT_CACHE.clear()
    incumbent_spec = AgentSpec("net", incumbent, epsilon_band=0.0)
    candidate_spec = AgentSpec("net", candidate, epsilon_band=0.0)
    score = wins = draws = losses = 0.0
    total_plies = 0
    game_details = []
    by_side = {
        "P1": {"wins": 0, "draws": 0, "losses": 0},
        "P2": {"wins": 0, "draws": 0, "losses": 0},
    }
    terminations = {}
    t0 = time.monotonic()
    for game in range(games):
        game_t0 = time.monotonic()
        if game % 2 == 0:
            raw = play_pair_game(
                candidate_spec, incumbent_spec, game, device=device,
                max_plies=max_plies, return_details=True)
            side = "P1"
            result = raw["score"] if isinstance(raw, dict) else raw
        else:
            raw = play_pair_game(
                incumbent_spec, candidate_spec, game, device=device,
                max_plies=max_plies, return_details=True)
            side = "P2"
            incumbent_score = raw["score"] if isinstance(raw, dict) else raw
            result = 1.0 - incumbent_score
        details = dict(raw) if isinstance(raw, dict) else {}
        plies = int(details.get("plies", 0))
        elapsed = float(details.get("elapsed", time.monotonic() - game_t0))
        termination = str(details.get("termination", "unknown"))
        total_plies += plies
        terminations[termination] = terminations.get(termination, 0) + 1
        score += result
        if result == 1.0:
            wins += 1
            outcome = "win"
        elif result == 0.5:
            draws += 1
            outcome = "draw"
        else:
            losses += 1
            outcome = "loss"
        by_side[side][{"win": "wins", "draw": "draws", "loss": "losses"}[outcome]] += 1
        info = {
            **details,
            "candidate_score": result,
            "candidate_side": side,
            "plies": plies,
            "elapsed": elapsed,
            "termination": termination,
            "running_wins": int(wins),
            "running_draws": int(draws),
            "running_losses": int(losses),
            "running_score": score / (game + 1),
        }
        game_details.append(info)
        if on_game:
            on_game(game + 1, games, result, info)
    result = {
        "games": games, "wins": int(wins), "draws": int(draws),
        "losses": int(losses), "score": score / max(games, 1),
        "elapsed": time.monotonic() - t0,
        "total_plies": total_plies,
        "avg_plies": total_plies / max(games, 1),
        "min_plies": min((item["plies"] for item in game_details), default=0),
        "max_plies_played": max((item["plies"] for item in game_details), default=0),
        "by_side": by_side,
        "terminations": terminations,
        "game_details": game_details,
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
