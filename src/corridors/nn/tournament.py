"""Round-robin tournament: all checkpoints vs each other + the classical engine.

- Every pair plays games_per_pair games, colors alternating.
- Games are distributed across worker processes (full CPU utilization).
- Elo: iterated K-factor updates (K=20, 400 shuffled passes), draws count 0.5,
  classical engine pinned at Elo 0 as the anchor.
- Results persist to nn_checkpoints/elo.json and each checkpoint's .meta.json.

Nets run inference on CPU inside workers: a ~450k-param net is fast on CPU and
this avoids N processes fighting over one GPU.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .. import solver
from ..game import NCOLS, State, WALLS_PER_PLAYER, apply_move
from ..parallel import _draw_by_no_progress

CLASSICAL = "classical"
ELO_PATH = Path(__file__).resolve().parent.parent.parent.parent / "nn_checkpoints" / "elo.json"

K_FACTOR = 20.0
ELO_PASSES = 400
MAX_PLIES = 120


@dataclass(frozen=True)
class AgentSpec:
    kind: str            # "classical" | "net"
    name: str            # display / checkpoint name
    depth: int = 2       # classical only
    time_limit: float = 0.5
    epsilon_band: float = 0.02  # net only


def _make_mover(spec: AgentSpec, seed: int):
    """Returns pick(state, board) -> Move. Lazily constructed inside workers."""
    if spec.kind == "classical":
        tiebreak = 10

        def pick_classical(state, board):
            mv, _s, _st, _pv = solver.best_move(
                state, board, max_depth=spec.depth,
                time_limit=spec.time_limit if spec.time_limit > 0 else None,
                tiebreak_epsilon=tiebreak, verbose=False,
            )
            return mv
        return pick_classical

    from .agent import NetworkAgent
    agent = NetworkAgent(spec.name, device="cpu",
                         epsilon_band=spec.epsilon_band, seed=seed)
    return lambda state, board: agent.pick_move(state, board)


def play_pair_game(a: AgentSpec, b: AgentSpec, game_idx: int) -> float:
    """Play one game; 'a' moves first as P1. Returns score for a: 1 / 0.5 / 0."""
    seed = hash((a.name, b.name, game_idx)) & 0x7FFFFFFF
    rng = random.Random(seed)
    p1_col = rng.randint(0, NCOLS - 1)
    p2_col = rng.randint(0, NCOLS - 1)
    board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)

    movers = {1: _make_mover(a, seed), 2: _make_mover(b, seed ^ 0x5A5A5A)}
    states_seen: List[State] = [state]
    plies = 0
    while True:
        w = state.winner(board)
        if w is not None:
            return 1.0 if w == 1 else 0.0
        if plies >= MAX_PLIES or _draw_by_no_progress(states_seen, board):
            return 0.5
        mv = movers[state.turn](state, board)
        state = apply_move(state, mv)
        states_seen.append(state)
        plies += 1


def _pair_game_task(a: AgentSpec, b: AgentSpec, game_idx: int,
                    swap: bool) -> Tuple[str, str, float]:
    """Worker task. When swap, b plays P1; score is still reported for (a, b)."""
    if swap:
        score_b = play_pair_game(b, a, game_idx)
        return (a.name, b.name, 1.0 - score_b)
    return (a.name, b.name, play_pair_game(a, b, game_idx))


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
            if anchor in ratings:
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


def run_tournament(
    checkpoints: List[str],
    games_per_pair: int = 4,
    classical_depth: int = 2,
    classical_time: float = 0.5,
    workers: int = 4,
    on_progress: Optional[Callable[[int, int, Tuple[str, str, float]], None]] = None,
) -> dict:
    """Round-robin over checkpoints + classical anchor. Returns elo.json payload."""
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

    results: List[Tuple[str, str, float]] = []
    total = len(tasks)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_pair_game_task, a, b, g, swap) for a, b, g, swap in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if on_progress is not None:
                on_progress(done, total, res)

    ratings = compute_elo(results)

    data = load_elo()
    prior_games = data.get("games", [])
    all_games = prior_games + [list(r) for r in results]
    # Recompute over full history so ratings stabilize across tournaments.
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
            "results": [list(r) for r in results],
        },
    })
    save_elo(data)

    from . import model as model_mod
    for c in checkpoints:
        if c in ratings:
            model_mod.update_meta(c, {"elo": ratings[c]})
    return data
