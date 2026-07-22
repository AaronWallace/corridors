"""Multiprocess self-play workers for Corridors.

Each worker process plays complete games independently and reports progress to
the parent over a multiprocessing Queue. Workers share the persistent SQLite TT
(WAL mode + busy_timeout make concurrent writers safe).

Message protocol (tuples, first element is the kind):

    ("move", wid, game_num, state, board, move, turn, score, elapsed, nodes, depth)
    ("game_start", wid, game_num, p1_col, p2_col)
    ("game_end", wid, game_num, winner, plies, game_time, game_nodes, think_time)
    ("done", wid)
    ("error", wid, repr_of_exception)

All payloads are plain picklable values (State/Board are frozen dataclasses of
tuples/frozensets).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import solver
from .game import NCOLS, State, WALLS_PER_PLAYER, apply_move, is_threefold_repetition


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: int
    num_games: int
    starts: str          # "fixed" | "random"
    p1_col: int
    p2_col: int
    depth: int
    time_limit: float
    tiebreak_epsilon: int
    max_plies: int
    tt_path: Optional[str]
    seed: int
    report_moves: bool   # False lets high-worker-count runs skip per-move traffic
    record_dataset: Optional[str] = None   # dataset name; records training data when set
    record_shard_index: int = 0            # this worker's shard number
    p1_agent: str = "classical"
    p2_agent: str = "classical"


def run_worker(cfg: WorkerConfig, queue) -> None:
    """Entry point for a worker process. Plays cfg.num_games games.

    When cfg.record_dataset is set, every non-terminal position is recorded as
    (encoded_state, outcome_for_mover, normalized_tt_score) and written to this
    worker's own shard NPZ at the end (also flushed on interrupt).
    """
    # Coordinator's p.terminate() sends SIGTERM whose default action is to exit
    # immediately, skipping the finally block that flushes our shard. Install a
    # handler that raises SystemExit so unwinding happens and _flush_shard() runs
    # — otherwise a Ctrl-C on the coordinator loses all in-progress worker data.
    import signal

    def _on_sigterm(_sig, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    rng = random.Random(cfg.seed)
    tt = None
    rec_tensors: list = []
    rec_turns: list = []       # side to move at each recorded position
    rec_scores: list = []      # normalized search score (mover's perspective)
    rec_outcomes: list = []    # filled at game end
    rec_moves: list = []       # action index of the move actually played (0..226)
    games_recorded = 0
    agents = {}
    if cfg.p1_agent != "classical" or cfg.p2_agent != "classical":
        from .nn.agent import NetworkAgent
        for player, name in ((1, cfg.p1_agent), (2, cfg.p2_agent)):
            if name != "classical":
                agents[player] = NetworkAgent(name, device="cpu",
                                              seed=cfg.seed ^ player)
    if cfg.record_dataset:
        import numpy as np
        from .nn import encoding

    shard_flushed = False

    def _flush_shard() -> None:
        nonlocal shard_flushed
        if shard_flushed or not cfg.record_dataset or not rec_tensors:
            return
        import numpy as np
        from .nn import datasets
        # Trim to the shortest completed list — an interrupted game may leave
        # a state recorded without an outcome/move.
        n = min(len(rec_tensors), len(rec_outcomes), len(rec_scores),
                len(rec_moves))
        datasets.write_shard(
            cfg.record_dataset, cfg.record_shard_index,
            np.stack(rec_tensors[:n]),
            np.asarray(rec_outcomes[:n], dtype=np.int8),
            np.asarray(rec_scores[:n], dtype=np.float32),
            moves=np.asarray(rec_moves[:n], dtype=np.int16),
        )
        shard_flushed = True
        queue.put(("shard", cfg.worker_id, cfg.record_shard_index,
                   len(rec_tensors), games_recorded))

    try:
        if cfg.tt_path:
            tt = solver.TT(sqlite_path=cfg.tt_path)
        for game_num in range(1, cfg.num_games + 1):
            if cfg.starts == "random":
                p1_col = rng.randint(0, NCOLS - 1)
                p2_col = rng.randint(0, NCOLS - 1)
            else:
                p1_col = cfg.p1_col
                p2_col = cfg.p2_col
            board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)
            queue.put(("game_start", cfg.worker_id, game_num, p1_col, p2_col))

            states_seen: List[State] = [state]
            # Position-occurrence counts (zobrist includes side to move).
            # Positions already seen twice are fed to the solver as
            # avoid_child_hashes so the root steers away from the threefold
            # repetition instead of shuffling into a draw.
            seen_hashes: Dict[int, int] = {solver.zobrist(state, board): 1}
            game_t0 = time.monotonic()
            winner: Optional[int] = None
            plies = 0
            game_nodes = 0
            think_time = 0.0
            while True:
                w = state.winner(board)
                if w is not None:
                    winner = w
                    break
                if plies >= cfg.max_plies or is_threefold_repetition(states_seen):
                    winner = None
                    break

                mover = state.turn
                agent_name = cfg.p1_agent if mover == 1 else cfg.p2_agent
                if agent_name == "classical":
                    tl = cfg.time_limit if cfg.time_limit > 0 else None
                    avoid = {h for h, c in seen_hashes.items() if c >= 2}
                    mv, score, stats, _pv = solver.best_move(
                        state, board,
                        max_depth=cfg.depth, time_limit=tl,
                        tiebreak_epsilon=cfg.tiebreak_epsilon,
                        avoid_child_hashes=avoid or None,
                        tt=tt, verbose=False,
                        flush_on_exit=False,
                    )
                    elapsed, nodes, depth_reached = stats.elapsed, stats.nodes, stats.max_depth
                else:
                    think_t0 = time.monotonic()
                    mv = agents[mover].pick_move(state, board)
                    elapsed, nodes, depth_reached, score = time.monotonic() - think_t0, 0, 0, 0
                if cfg.record_dataset:
                    from .nn.actions import move_to_index
                    rec_tensors.append(encoding.encode_state(state, board))
                    rec_turns.append(mover)
                    rec_scores.append(encoding.normalize_score(score))
                    # Record the move as an action index so the AZ converter can
                    # turn it into a one-hot policy target without needing the
                    # original Move object (also fixed-width, small).
                    rec_moves.append(move_to_index(mv))
                state = apply_move(state, mv)
                states_seen.append(state)
                h = solver.zobrist(state, board)
                seen_hashes[h] = seen_hashes.get(h, 0) + 1
                plies += 1
                game_nodes += nodes
                think_time += elapsed
                if cfg.report_moves:
                    queue.put((
                        "move", cfg.worker_id, game_num, state, board, mv,
                        mover, score, elapsed, nodes, depth_reached,
                    ))

            if tt is not None:
                try:
                    tt.flush()
                except Exception:
                    pass
            if cfg.record_dataset:
                # Back-fill outcomes for this game's positions (mover perspective).
                new_positions = len(rec_turns) - len(rec_outcomes)
                for t in rec_turns[-new_positions:] if new_positions else []:
                    rec_outcomes.append(encoding.outcome_for_mover(winner, t))
                games_recorded += 1
            game_time = time.monotonic() - game_t0
            queue.put(("game_end", cfg.worker_id, game_num, winner, plies, game_time,
                       game_nodes, think_time))
        _flush_shard()
        queue.put(("done", cfg.worker_id))
    except (KeyboardInterrupt, SystemExit):
        # Drop the in-flight game's unlabelled positions, keep completed games.
        # SystemExit fires when the coordinator SIGTERMs us; treat identically.
        del rec_tensors[len(rec_outcomes):]
        del rec_scores[len(rec_outcomes):]
    except Exception as e:  # surface crashes to the parent instead of dying silently
        del rec_tensors[len(rec_outcomes):]
        del rec_scores[len(rec_outcomes):]
        try:
            queue.put(("error", cfg.worker_id, repr(e)))
        except Exception:
            pass
    finally:
        try:
            _flush_shard()
        except Exception:
            pass
        if tt is not None:
            try:
                tt.close()
            except Exception:
                pass


def split_games(total: int, workers: int) -> List[int]:
    """Distribute total games across workers as evenly as possible."""
    base, extra = divmod(total, workers)
    return [base + (1 if i < extra else 0) for i in range(workers)]
