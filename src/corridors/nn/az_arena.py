"""Candidate-versus-incumbent gating for iterative AlphaZero training."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

from .az_net import CHECKPOINT_ROOT, checkpoint_path, meta_path
from .az_selfplay import _THREAD_ENV_VARS
from .tournament import (AgentSpec, _AGENT_CACHE, _worker_init,
                         auto_tournament_workers, play_pair_game)


def _arena_game_task(incumbent: str, candidate: str, game: int,
                     device: str, max_plies: int):
    """One color-balanced arena game (candidate is P1 on even game indices).
    Returns (game, candidate_side, candidate_score, details)."""
    if game % 2 == 0:
        raw = play_pair_game(
            AgentSpec("net", candidate, epsilon_band=0.0),
            AgentSpec("net", incumbent, epsilon_band=0.0),
            game, device=device, max_plies=max_plies, return_details=True)
        side = "P1"
        result = raw["score"] if isinstance(raw, dict) else raw
    else:
        raw = play_pair_game(
            AgentSpec("net", incumbent, epsilon_band=0.0),
            AgentSpec("net", candidate, epsilon_band=0.0),
            game, device=device, max_plies=max_plies, return_details=True)
        side = "P2"
        incumbent_score = raw["score"] if isinstance(raw, dict) else raw
        result = 1.0 - incumbent_score
    return game, side, result, (dict(raw) if isinstance(raw, dict) else {})


def run_arena(incumbent: str, candidate: str, games: int = 20,
              max_plies: int = 150, device: str = "cpu",
              on_game: Optional[Callable] = None, workers: int = 0) -> dict:
    """Play color-balanced games and return the candidate's aggregate score.

    workers: 0 = auto (tournament-style, one thread-pinned process per core);
    1 = sequential in this process. Every game is seeded by its index, so the
    aggregate result is identical either way; with a pool, on_game fires in
    completion order rather than index order."""
    # Both checkpoint names are overwritten across loop iterations; never reuse
    # model objects cached for a previous arena. Pool workers are fresh
    # processes per call, so only the in-process path can hold stale models.
    _AGENT_CACHE.clear()
    if workers <= 0:
        workers = auto_tournament_workers(device)
    workers = max(1, min(workers, games))

    score = 0.0
    wins = draws = losses = 0
    total_plies = 0
    done = 0
    game_details = []
    by_side = {
        "P1": {"wins": 0, "draws": 0, "losses": 0},
        "P2": {"wins": 0, "draws": 0, "losses": 0},
    }
    terminations = {}
    t0 = time.monotonic()

    def record(side: str, result: float, details: dict,
               fallback_elapsed: float) -> None:
        nonlocal score, wins, draws, losses, total_plies, done
        done += 1
        plies = int(details.get("plies", 0))
        elapsed = float(details.get("elapsed", fallback_elapsed))
        termination = str(details.get("termination", "unknown"))
        total_plies += plies
        terminations[termination] = terminations.get(termination, 0) + 1
        score += result
        if result == 1.0:
            wins += 1
            outcome = "wins"
        elif result == 0.5:
            draws += 1
            outcome = "draws"
        else:
            losses += 1
            outcome = "losses"
        by_side[side][outcome] += 1
        info = {
            **details,
            "candidate_score": result,
            "candidate_side": side,
            "plies": plies,
            "elapsed": elapsed,
            "termination": termination,
            "running_wins": wins,
            "running_draws": draws,
            "running_losses": losses,
            "running_score": score / done,
        }
        game_details.append(info)
        if on_game:
            on_game(done, games, result, info)

    if workers == 1:
        for game in range(games):
            game_t0 = time.monotonic()
            _g, side, result, details = _arena_game_task(
                incumbent, candidate, game, device, max_plies)
            record(side, result, details, time.monotonic() - game_t0)
    else:
        # Same worker setup as the tournament: children are thread-pinned via
        # env inherited at spawn/fork time, and each caches both nets once.
        if device == "cpu" and sys.platform != "win32":
            ctx = mp.get_context("fork")
        else:
            ctx = mp.get_context("spawn")
        saved_env = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}
        for v in _THREAD_ENV_VARS:
            os.environ[v] = "1"
        try:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                       initializer=_worker_init)
            futures = []
            try:
                futures = [pool.submit(_arena_game_task, incumbent, candidate,
                                       game, device, max_plies)
                           for game in range(games)]
                for fut in as_completed(futures):
                    _g, side, result, details = fut.result()
                    record(side, result, details, 0.0)
            except BaseException:
                for fut in futures:
                    fut.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        finally:
            for v, val in saved_env.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val

    result = {
        "games": games, "wins": wins, "draws": draws,
        "losses": losses, "score": score / max(games, 1),
        "elapsed": time.monotonic() - t0,
        "workers": workers,
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
