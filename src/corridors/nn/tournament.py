"""Round-robin tournament: selected checkpoints vs each other + the classical
engine (the menu defaults to all checkpoints). Rated checkpoints left out of the
latest round robin are surfaced with a stale Elo flag (see
checkpoints.stale_elo_checkpoints).

- Every pair plays games_per_pair games, colors alternating.
- Games are distributed across worker processes (full CPU utilization).
- Elo: iterated K-factor updates (K=20, 400 shuffled passes), draws count 0.5,
  classical engine pinned at Elo 0 as the anchor.
- Results persist to nn_checkpoints/elo.json and each checkpoint's .meta.json.

Workers are thread-pinned (no core oversubscription), auto-sized to the host,
and cache each net across games instead of reloading per game. Inference runs
on the detected device (device="auto"); CPU is usually fastest here since the
net is tiny (~450k params) and the bottleneck is CPU game logic + the classical
solver, but GPU is supported for larger nets.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .. import solver
from ..game import (
    NCOLS, State, WALLS_PER_PLAYER, apply_move, is_threefold_repetition,
    legal_moves,
)
from .az_selfplay import (_THREAD_ENV_VARS, _limit_blas_threads, auto_workers,
                          resolve_device)

CLASSICAL = "classical"
ELO_PATH = Path(__file__).resolve().parent.parent.parent.parent / "nn_checkpoints" / "elo.json"

K_FACTOR = 20.0
ELO_PASSES = 400
MAX_PLIES = 120

# Per-worker LRU cache of constructed agents, keyed by (checkpoint, device).
# Avoids reloading a net from disk every game while bounding how many models a
# worker holds at once (a many-checkpoint tournament otherwise loads them all).
from collections import OrderedDict

_AGENT_CACHE: "OrderedDict[Tuple[str, str], object]" = OrderedDict()
_AGENT_CACHE_MAX = 6  # models kept loaded per worker; evicted ones are freed


def _worker_init() -> None:
    """ProcessPoolExecutor initializer: pin native math threads to 1 (so N
    workers don't each grab every core) and ignore SIGINT (parent handles it)."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for v in _THREAD_ENV_VARS:
        os.environ.setdefault(v, "1")
    _limit_blas_threads(1)  # bulletproof BLAS cap (env alone can miss after import)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


@dataclass(frozen=True)
class AgentSpec:
    kind: str            # "classical" | "net"
    name: str            # display / checkpoint name
    depth: int = 2       # classical only
    time_limit: float = 0.5
    epsilon_band: float = 0.02  # net only


def _get_mover(spec: AgentSpec, seed: int, device: str):
    """Return pick(state, board) -> Move. Nets are cached per (name, device) in
    the worker process and reseeded per game; the classical mover is stateless.

    Round-robin only pairs distinct specs, so the two movers in a game never
    alias the same cached agent."""
    if spec.kind == "classical":
        def pick_classical(state, board):
            mv, _s, _st, _pv = solver.best_move(
                state, board, max_depth=spec.depth,
                time_limit=spec.time_limit if spec.time_limit > 0 else None,
                tiebreak_epsilon=10, verbose=False,
            )
            return mv
        return pick_classical

    key = (spec.name, device)
    agent = _AGENT_CACHE.get(key)
    if agent is None:
        from .agent import NetworkAgent
        agent = NetworkAgent(spec.name, device=device,
                             epsilon_band=spec.epsilon_band, seed=seed)
        _AGENT_CACHE[key] = agent
        # LRU: keep only the few most-recently-used models loaded; drop the rest
        # so a many-checkpoint tournament doesn't hold every model in every worker.
        while len(_AGENT_CACHE) > _AGENT_CACHE_MAX:
            _AGENT_CACHE.popitem(last=False)  # evict oldest -> its model is freed
    else:
        _AGENT_CACHE.move_to_end(key)  # mark most-recently-used
        agent.reseed(seed)
    # The caller holds this agent for the duration of the game via the closure,
    # so eviction from the cache never frees a model that's currently in play.
    return lambda state, board: agent.pick_move(state, board)


def play_pair_game(a: AgentSpec, b: AgentSpec, game_idx: int,
                   device: str = "cpu", max_plies: int = MAX_PLIES,
                   return_details: bool = False):
    """Play one game; 'a' moves first as P1. Returns score for a: 1 / 0.5 / 0.
    Threefold repetition or reaching `max_plies` half-moves scores 0.5.
    ``return_details`` adds plies, duration, colors, and termination reason."""
    started = time.monotonic()
    # Process-stable seed (str hash() is salted per process): the same pairing
    # plays the same game whether it runs here or in any pool worker.
    seed = zlib.crc32(f"{a.name}|{b.name}|{game_idx}".encode()) & 0x7FFFFFFF
    rng = random.Random(seed)
    p1_col = rng.randint(0, NCOLS - 1)
    p2_col = rng.randint(0, NCOLS - 1)
    board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)

    movers = {1: _get_mover(a, seed, device), 2: _get_mover(b, seed ^ 0x5A5A5A, device)}
    states_seen: List[State] = [state]
    plies = 0

    def finish(score: float, reason: str):
        if not return_details:
            return score
        return {
            "score": score,
            "plies": plies,
            "elapsed": time.monotonic() - started,
            "termination": reason,
            "p1": a.name,
            "p2": b.name,
            "p1_col": p1_col,
            "p2_col": p2_col,
        }

    while True:
        w = state.winner(board)
        if w is not None:
            return finish(1.0 if w == 1 else 0.0, "goal")
        if plies >= max_plies:
            return finish(0.5, "max plies")
        if is_threefold_repetition(states_seen):
            return finish(0.5, "threefold")
        try:
            mv = movers[state.turn](state, board)
        except RuntimeError as exc:
            # Defensive fallback for malformed/legacy states. Legal play now
            # preserves pawn mobility, but one bad game must never abort a full
            # tournament. Both built-in agents use this exact exception when no
            # action exists; verify the state before adjudicating the loss.
            if str(exc) != "no legal moves" or legal_moves(state, board):
                raise
            return finish(0.0 if state.turn == 1 else 1.0, "no legal moves")
        state = apply_move(state, mv)
        states_seen.append(state)
        plies += 1


def _pair_game_task(a: AgentSpec, b: AgentSpec, game_idx: int,
                    swap: bool, device: str, max_plies: int) -> Tuple[str, str, float, str]:
    """Worker task. When swap, b plays P1; score is still reported for (a, b).
    Fourth element is the termination reason (from play_pair_game details)."""
    if swap:
        details = play_pair_game(b, a, game_idx, device, max_plies,
                                 return_details=True)
        return (a.name, b.name, 1.0 - details["score"], details["termination"])
    details = play_pair_game(a, b, game_idx, device, max_plies,
                             return_details=True)
    return (a.name, b.name, details["score"], details["termination"])


def compute_elo(results: List[Tuple[str, str, float]],
                anchor: str = CLASSICAL) -> Dict[str, float]:
    """Iterated Elo: K=20, ELO_PASSES shuffled passes, anchor pinned at 0."""
    ratings: Dict[str, float] = {}
    for a, b, _ in results:
        ratings.setdefault(a, 0.0)
        ratings.setdefault(b, 0.0)
    rng = random.Random(42)
    games = list(results)
    for _ in range(ELO_PASSES):
        rng.shuffle(games)
        for a, b, score in games:
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            delta = K_FACTOR * (score - ea) / ELO_PASSES * 40  # scaled per-pass K
            ratings[a] += delta
            ratings[b] -= delta
        # Re-anchor once per pass: the update only reads rating differences,
        # which a uniform shift never changes, so this matches per-game
        # re-anchoring while keeping the inner loop O(1) per game.
        shift = ratings.get(anchor, 0.0)
        if shift:
            for k in ratings:
                ratings[k] -= shift
    return {k: round(v, 1) for k, v in ratings.items()}


def load_elo() -> dict:
    if not ELO_PATH.exists():
        return {"anchor": CLASSICAL, "ratings": {}, "games": [], "updated": None}
    try:
        return json.loads(ELO_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"anchor": CLASSICAL, "ratings": {}, "games": [], "updated": None}


def save_elo(data: dict) -> None:
    ELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ELO_PATH.with_name("." + ELO_PATH.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(ELO_PATH)


def rename_elo_checkpoint(old_name: str, new_name: str) -> bool:
    """Rename a checkpoint throughout saved tournament history."""
    data = load_elo()
    changed = False

    ratings = data.get("ratings", {})
    if old_name in ratings:
        ratings[new_name] = ratings.pop(old_name)
        changed = True

    def rename_results(results) -> None:
        nonlocal changed
        for result in results or []:
            for index in (0, 1):
                if len(result) > index and result[index] == old_name:
                    result[index] = new_name
                    changed = True

    rename_results(data.get("games"))
    last_run = data.get("last_run", {})
    rename_results(last_run.get("results"))
    checkpoints = last_run.get("checkpoints", [])
    for index, checkpoint in enumerate(checkpoints):
        if checkpoint == old_name:
            checkpoints[index] = new_name
            changed = True

    if changed:
        save_elo(data)
    return changed


def auto_tournament_workers(device: str) -> int:
    """Default worker count. CPU: one per core (workers are single-threaded).
    GPU: capped, since each worker holds its own CUDA context + model copies."""
    ncpu = os.cpu_count() or 4
    if device == "cpu":
        return max(1, ncpu - 1)
    return max(1, min(ncpu - 1, 16))


def run_tournament(
    checkpoints: List[str],
    games_per_pair: int = 4,
    classical_depth: int = 2,
    classical_time: float = 0.5,
    workers: int = 0,
    device: str = "auto",
    max_plies: int = MAX_PLIES,
    on_progress: Optional[Callable[[int, int, Tuple[str, str, float]], None]] = None,
) -> dict:
    """Round-robin over checkpoints + classical anchor. Returns elo.json payload.

    device: "auto"/"cpu"/"cuda" — where nets run inference. workers: 0 = auto.
    Workers are thread-pinned (no oversubscription) and cache their models across
    games. GPU mode uses spawn (fork can't re-init CUDA)."""
    dev = resolve_device(device)
    if workers <= 0:
        workers = auto_tournament_workers(dev)

    specs = [AgentSpec(kind="classical", name=CLASSICAL,
                       depth=classical_depth, time_limit=classical_time)]
    specs.extend(AgentSpec(kind="net", name=c) for c in checkpoints)
    if len(specs) < 2:
        raise ValueError("need at least one trained checkpoint")

    tasks = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            for g in range(games_per_pair):
                tasks.append((specs[i], specs[j], g, g % 2 == 1))

    # GPU inference can't be forked (CUDA re-init); CPU can use fast fork on POSIX.
    if dev == "cpu" and sys.platform != "win32":
        ctx = mp.get_context("fork")
    else:
        ctx = mp.get_context("spawn")

    # Pin native math threads to 1 in every child (inherited at spawn/fork), so
    # N inference workers don't collectively spawn N*ncpu threads and thrash.
    saved_env = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}
    for v in _THREAD_ENV_VARS:
        os.environ[v] = "1"

    results: List[Tuple[str, str, float]] = []
    total = len(tasks)
    done = 0
    try:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                   initializer=_worker_init)
        futures = []
        try:
            futures = [pool.submit(_pair_game_task, a, b, g, swap, dev, max_plies)
                       for a, b, g, swap in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                done += 1
                if on_progress is not None:
                    on_progress(done, total, res)
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

    # Tally termination breakdown for the summary. Results are 4-tuples
    # (a, b, score, termination); elo.json's `games` stays 3-tuples for
    # backward compat with compute_elo and older records.
    terminations = {"goal": 0, "no legal moves": 0, "threefold": 0, "max plies": 0}
    for _, _, _, term in results:
        terminations[term] = terminations.get(term, 0) + 1

    data = load_elo()
    prior_games = data.get("games", [])
    # Persist only the (a, b, score) tuple in cross-run history.
    new_games = [[a, b, score] for (a, b, score, _term) in results]
    all_games = prior_games + new_games
    ratings = compute_elo([tuple(g) for g in all_games])
    data.update({
        "anchor": CLASSICAL,
        "ratings": ratings,
        "games": all_games,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_run": {
            "checkpoints": checkpoints,
            "games_per_pair": games_per_pair,
            "classical_depth": classical_depth,
            "results": [list(r) for r in results],  # includes termination
            "terminations": terminations,
        },
    })
    save_elo(data)

    from . import model as model_mod
    for c in checkpoints:
        if c in ratings:
            model_mod.update_meta(c, {"elo": ratings[c]})
    return data
