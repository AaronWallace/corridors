"""Measure classical persistent-TT throughput at various worker counts.

There's no baked-in threshold for "when does the shared sqlite TT stop being
worth it" because it depends on the workload, hardware, DB size, filesystem,
and worker count. This module answers the question empirically: it spawns N
worker processes running the classical solver on a stream of positions, times
how many searches complete in a fixed budget, and reports aggregate nodes-per-
second. Runs twice (persistent TT on vs off) so you can see the actual
crossover on YOUR host and act on it.

Each worker plays through a short deterministic sequence of positions using
solver.best_move, which is exactly the code path that hits the TT — no full-
game overhead confounding the measurement.
"""

from __future__ import annotations

import multiprocessing
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .. import settings, solver
from ..game import NCOLS, State, WALLS_PER_PLAYER, apply_move, legal_moves


@dataclass
class BenchResult:
    workers: int
    tt_shared: bool
    duration_s: float
    total_searches: int
    total_nodes: int

    @property
    def searches_per_sec(self) -> float:
        return self.total_searches / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def nps_per_worker(self) -> float:
        """Nodes/sec/worker — the metric most sensitive to TT contention.
        Compare on/off at the same worker count to see the impact."""
        if self.duration_s <= 0 or self.workers <= 0:
            return 0.0
        return self.total_nodes / self.duration_s / self.workers


def _bench_worker(worker_id: int, tt_path: Optional[str], depth: int,
                  time_limit: float, deadline: float, seed: int,
                  results_q) -> None:
    """One worker: search positions from a deterministic random walk until
    time budget is exhausted. Reports total searches and total nodes."""
    import signal

    def _on_sigterm(_sig, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)
    rng = random.Random(seed)
    tt = solver.TT(sqlite_path=tt_path) if tt_path else solver.TT()
    total_searches = 0
    total_nodes = 0
    try:
        # Fresh random game each worker; if the game finishes, start another.
        while time.monotonic() < deadline:
            p1_col = rng.randint(0, NCOLS - 1)
            p2_col = rng.randint(0, NCOLS - 1)
            board, state = State.start(p1_col=p1_col, p2_col=p2_col,
                                       walls=WALLS_PER_PLAYER)
            while time.monotonic() < deadline:
                if state.winner(board) is not None:
                    break
                _mv, _s, stats, _pv = solver.best_move(
                    state, board, max_depth=depth,
                    time_limit=time_limit, tt=tt,
                    verbose=False, flush_on_exit=False,
                )
                total_searches += 1
                total_nodes += stats.nodes
                moves = legal_moves(state, board)
                if not moves:
                    break
                # Just play a random legal move to keep the game moving —
                # we're measuring solver throughput, not play quality.
                state = apply_move(state, rng.choice(moves))
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if tt is not None:
            try:
                tt.close()
            except Exception:
                pass
        results_q.put((worker_id, total_searches, total_nodes))


def run_bench(workers: int, tt_shared: bool, duration_s: float = 20.0,
              depth: int = 2, time_limit: float = 0.1) -> BenchResult:
    """Spawn `workers` processes, each hammering solver.best_move for
    `duration_s` seconds. `tt_shared=True` uses the persistent sqlite TT
    (contention test); `False` gives each worker a private in-memory TT."""
    cfg = settings.load()
    tt_path = str(cfg["tt_path"]) if tt_shared else None
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    deadline = time.monotonic() + duration_s
    t0 = time.monotonic()
    procs = [
        ctx.Process(
            target=_bench_worker,
            args=(i, tt_path, depth, time_limit, deadline, 12345 + i, q),
            daemon=True,
        )
        for i in range(workers)
    ]
    for p in procs:
        p.start()
    # Wait for all worker results, then measure elapsed. Small grace window.
    reports: Dict[int, tuple] = {}
    end_wait = deadline + max(5.0, duration_s * 0.25)
    while len(reports) < workers and time.monotonic() < end_wait:
        try:
            wid, searches, nodes = q.get(timeout=1.0)
            reports[wid] = (searches, nodes)
        except Exception:
            pass
    elapsed = time.monotonic() - t0
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=2.0)
    total_searches = sum(s for s, _ in reports.values())
    total_nodes = sum(n for _, n in reports.values())
    return BenchResult(
        workers=workers, tt_shared=tt_shared, duration_s=elapsed,
        total_searches=total_searches, total_nodes=total_nodes,
    )


def sweep(worker_counts: List[int], duration_s: float = 20.0,
          depth: int = 2, time_limit: float = 0.1,
          on_config=None) -> List[BenchResult]:
    """Run the benchmark at each worker count, both TT settings. Returns the
    list of BenchResult in the order run. Optional callback fires per config
    with (worker_count, tt_shared, config_index, total_configs) for progress."""
    total = len(worker_counts) * 2
    results: List[BenchResult] = []
    idx = 0
    for w in worker_counts:
        for shared in (True, False):
            idx += 1
            if on_config is not None:
                on_config(w, shared, idx, total)
            results.append(run_bench(
                workers=w, tt_shared=shared,
                duration_s=duration_s, depth=depth, time_limit=time_limit,
            ))
    return results
