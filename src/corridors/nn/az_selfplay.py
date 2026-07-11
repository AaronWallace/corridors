"""AlphaZero self-play pipeline.

Architecture:
  - 1 inference server process: holds the model on GPU, batches evaluation
    requests from workers, returns (policy, value) pairs.
  - N game worker processes: each plays games using MCTS, sending leaf
    evaluations to the server and collecting training data.
  - 1 coordinator (the caller): collects completed game data from workers.

Dynamic scaling: auto-detects GPU and CPU count, allocates workers accordingly.
Designed to saturate a single GPU (RTX 3070 to H200) with batched inference.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..game import NCOLS, WALLS_PER_PLAYER, State, apply_move
from .actions import NUM_ACTIONS, move_to_index
from .encoding import NROWS, NUM_PLANES, encode_state

# Sentinel values for the inference queue
_SHUTDOWN = None
_BATCH_TIMEOUT = 0.005  # 5ms — wait this long to fill a batch
SHARD_EVERY = 25  # flush to disk every N games


class _ShardWriter:
    """Accumulates game data and flushes shards to disk periodically."""

    def __init__(self, save_dir: Optional[str], flush_every: int = SHARD_EVERY) -> None:
        self.save_dir = save_dir
        self.flush_every = flush_every
        self._states: List[np.ndarray] = []
        self._policies: List[np.ndarray] = []
        self._outcomes: List[np.ndarray] = []
        self._games_since_flush = 0
        self._shard_idx = 0
        self._total_positions = 0

        if save_dir:
            from pathlib import Path
            d = Path(save_dir)
            d.mkdir(parents=True, exist_ok=True)
            existing = sorted(d.glob("shard_*.npz"))
            if existing:
                import re
                nums = [int(re.search(r"shard_(\d+)", f.stem).group(1))
                        for f in existing if re.search(r"shard_(\d+)", f.stem)]
                self._shard_idx = max(nums) + 1 if nums else 0

    def add_game(self, states: np.ndarray, policies: np.ndarray,
                 outcomes: np.ndarray) -> None:
        self._states.append(states)
        self._policies.append(policies)
        self._outcomes.append(outcomes)
        self._total_positions += len(states)
        self._games_since_flush += 1
        if self.save_dir and self._games_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.save_dir or not self._states:
            return
        from pathlib import Path
        s = np.concatenate(self._states)
        p = np.concatenate(self._policies)
        o = np.concatenate(self._outcomes)
        path = Path(self.save_dir) / f"shard_{self._shard_idx:04d}.npz"
        tmp = path.with_name(f".tmp_{path.name}")
        np.savez_compressed(tmp, states=s, policies=p, outcomes=o)
        import os as _os
        _os.replace(tmp, path)
        self._shard_idx += 1
        self._states.clear()
        self._policies.clear()
        self._outcomes.clear()
        self._games_since_flush = 0

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return any unflushed data (plus empty arrays if everything was flushed)."""
        if not self._states:
            return (np.zeros((0, NUM_PLANES, NROWS, NCOLS), dtype=np.float32),
                    np.zeros((0, NUM_ACTIONS), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))
        return (np.concatenate(self._states),
                np.concatenate(self._policies),
                np.concatenate(self._outcomes))

    @property
    def total_positions(self) -> int:
        return self._total_positions


@dataclass
class SelfPlayConfig:
    num_games: int = 100
    simulations: int = 200
    workers: int = 0           # 0 = auto-detect
    batch_size: int = 64       # max batch for GPU inference
    temperature_moves: int = 20  # use temp=1 for first N moves, then temp→0.1
    temp_high: float = 1.0
    temp_low: float = 0.1
    max_plies: int = 150
    checkpoint: str = ""       # empty = random init
    device: str = "auto"
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


@dataclass
class GameRecord:
    states: List[np.ndarray] = field(default_factory=list)   # encoded states
    policies: List[np.ndarray] = field(default_factory=list)  # MCTS visit distributions
    turns: List[int] = field(default_factory=list)            # side to move
    winner: Optional[int] = None                               # 1, 2, or None (draw)


# ---------------------------------------------------------------------------
# Inference server — runs on GPU
# ---------------------------------------------------------------------------

def _inference_server(
    request_queue: mp.Queue,
    response_queues: Dict[int, mp.Queue],
    checkpoint: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Batched inference process. Reads (worker_id, tensor) from request_queue,
    runs forward pass, sends (policy, value) back via per-worker response queues."""
    import torch
    from .az_net import AZNet, load_checkpoint as load_az

    if checkpoint:
        model = load_az(checkpoint, device=device)
    else:
        model = AZNet().to(device)
        model.eval()

    workers_done = 0
    pending: List[Tuple[int, int, np.ndarray]] = []  # (worker_id, req_id, tensor)

    def _flush_batch():
        if not pending:
            return
        batch_np = np.stack([t for _, _, t in pending])
        with torch.no_grad():
            x = torch.from_numpy(batch_np).to(device)
            policy_logits, values = model(x)
            policy_np = policy_logits.cpu().numpy()
            values_np = values.cpu().numpy()
        for i, (wid, rid, _) in enumerate(pending):
            response_queues[wid].put((rid, policy_np[i], float(values_np[i])))
        pending.clear()

    while workers_done < num_workers:
        try:
            item = request_queue.get(timeout=_BATCH_TIMEOUT)
        except Exception:
            _flush_batch()
            continue

        if item is _SHUTDOWN:
            workers_done += 1
            continue

        pending.append(item)
        if len(pending) >= batch_size:
            _flush_batch()

    _flush_batch()


# ---------------------------------------------------------------------------
# Game worker — runs on CPU
# ---------------------------------------------------------------------------

def _game_worker(
    worker_id: int,
    num_games: int,
    config: SelfPlayConfig,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    """Plays games using MCTS, sending leaf evaluations to the inference server."""
    rng = random.Random(seed)
    np.random.seed(seed & 0x7FFFFFFF)
    req_counter = 0

    # Cache for pending inference responses
    pending_responses: Dict[int, Tuple[np.ndarray, float]] = {}

    def evaluate_fn(state: State, board) -> Tuple[np.ndarray, float]:
        nonlocal req_counter
        tensor = encode_state(state, board)
        rid = req_counter
        req_counter += 1
        request_queue.put((worker_id, rid, tensor))
        # Wait for our response
        while rid not in pending_responses:
            resp_rid, policy, value = response_queue.get()
            pending_responses[resp_rid] = (policy, value)
        policy, value = pending_responses.pop(rid)
        return policy, value

    from .mcts import run_mcts
    import time as _time

    for game_num in range(num_games):
        p1_col = rng.randint(0, NCOLS - 1)
        p2_col = rng.randint(0, NCOLS - 1)
        board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)

        record = GameRecord()
        ply = 0
        last_heartbeat = _time.monotonic()
        while True:
            w = state.winner(board)
            if w is not None:
                record.winner = w
                break
            if ply >= config.max_plies:
                record.winner = None
                break

            temp = config.temp_high if ply < config.temperature_moves else config.temp_low

            pi, root_val, move = run_mcts(
                state, board,
                evaluate_fn=evaluate_fn,
                num_simulations=config.simulations,
                temperature=temp,
                add_noise=True,
            )

            if move is None:
                record.winner = 2 if state.turn == 1 else 1
                break

            record.states.append(encode_state(state, board))
            record.policies.append(pi)
            record.turns.append(state.turn)

            state = apply_move(state, move)
            ply += 1

            now = _time.monotonic()
            if now - last_heartbeat >= 5.0:
                result_queue.put(("heartbeat", worker_id, game_num, ply))
                last_heartbeat = now

        # Back-fill outcomes
        outcomes = []
        for t in record.turns:
            if record.winner is None:
                outcomes.append(0.0)
            elif record.winner == t:
                outcomes.append(1.0)
            else:
                outcomes.append(-1.0)

        if record.states:
            result_queue.put((
                worker_id,
                game_num,
                np.stack(record.states),
                np.stack(record.policies),
                np.array(outcomes, dtype=np.float32),
                record.winner,
                ply,
            ))

    # Signal done
    request_queue.put(_SHUTDOWN)
    result_queue.put(("done", worker_id))


def _game_worker_local(
    worker_id: int,
    num_games: int,
    config: SelfPlayConfig,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    """Self-contained worker: loads model locally, no inference server needed.
    Each process does its own inference on CPU — ideal for CPU-only clusters."""
    import torch
    from .az_net import AZNet, load_checkpoint as load_az
    from .mcts import run_mcts

    rng = random.Random(seed)
    np.random.seed(seed & 0x7FFFFFFF)

    if config.checkpoint:
        model = load_az(config.checkpoint, device="cpu")
    else:
        model = AZNet().to("cpu")
        model.eval()

    @torch.no_grad()
    def evaluate_fn(state: State, board) -> Tuple[np.ndarray, float]:
        tensor = encode_state(state, board)
        x = torch.from_numpy(tensor).unsqueeze(0)
        p, v = model(x)
        return p[0].numpy(), float(v[0])

    import time as _time

    for game_num in range(num_games):
        p1_col = rng.randint(0, NCOLS - 1)
        p2_col = rng.randint(0, NCOLS - 1)
        board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)

        record = GameRecord()
        ply = 0
        last_heartbeat = _time.monotonic()
        while True:
            w = state.winner(board)
            if w is not None:
                record.winner = w
                break
            if ply >= config.max_plies:
                record.winner = None
                break

            temp = config.temp_high if ply < config.temperature_moves else config.temp_low
            pi, root_val, move = run_mcts(
                state, board,
                evaluate_fn=evaluate_fn,
                num_simulations=config.simulations,
                temperature=temp,
                add_noise=True,
            )
            if move is None:
                record.winner = 2 if state.turn == 1 else 1
                break

            record.states.append(encode_state(state, board))
            record.policies.append(pi)
            record.turns.append(state.turn)
            state = apply_move(state, move)
            ply += 1

            now = _time.monotonic()
            if now - last_heartbeat >= 5.0:
                result_queue.put(("heartbeat", worker_id, game_num, ply))
                last_heartbeat = now

        outcomes = []
        for t in record.turns:
            if record.winner is None:
                outcomes.append(0.0)
            elif record.winner == t:
                outcomes.append(1.0)
            else:
                outcomes.append(-1.0)

        if record.states:
            result_queue.put((
                worker_id,
                game_num,
                np.stack(record.states),
                np.stack(record.policies),
                np.array(outcomes, dtype=np.float32),
                record.winner,
                ply,
            ))

    result_queue.put(("done", worker_id))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    import torch
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def auto_workers() -> int:
    ncpu = os.cpu_count() or 4
    return max(2, ncpu - 2)


def run_selfplay(
    config: SelfPlayConfig,
    on_game: Optional[Callable[[int, int, Optional[int], int, int], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_heartbeat: Optional[Callable[[int, int, int], None]] = None,
    save_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run self-play games. Returns (states, policies, outcomes) arrays.

    on_game(done, total, winner, ply, positions): called after each game completes.
    on_status(msg): called with status updates.
    save_dir: if set, flush shards to disk every 25 games (crash-safe).

    Returns concatenated arrays ready for training:
      states:   (N, 9, 11, 9) float32
      policies: (N, 227) float32
      outcomes: (N,) float32  — from side-to-move perspective
    """
    ctx = mp.get_context("spawn")
    device = resolve_device(config.device)
    num_workers = config.workers if config.workers > 0 else auto_workers()

    if on_status:
        mode = "local-inference" if device == "cpu" else "gpu-server"
        on_status(f"device={device}, workers={num_workers}, mode={mode}, "
                  f"sims={config.simulations}, games={config.num_games}")

    # Distribute games
    base, extra = divmod(config.num_games, num_workers)
    games_per = [base + (1 if i < extra else 0) for i in range(num_workers)]

    result_queue = ctx.Queue()

    if device == "cpu":
        # CPU mode: each worker loads its own model — true parallelism
        workers = []
        for i in range(num_workers):
            if games_per[i] == 0:
                continue
            seed = random.randint(0, 2**31 - 1)
            w = ctx.Process(
                target=_game_worker_local,
                args=(i, games_per[i], config, result_queue, seed),
                daemon=True,
            )
            w.start()
            workers.append(w)
        server = None
    else:
        # GPU mode: single inference server batches requests from workers
        request_queue = ctx.Queue()
        response_queues = {i: ctx.Queue() for i in range(num_workers)}

        server = ctx.Process(
            target=_inference_server,
            args=(request_queue, response_queues, config.checkpoint, device,
                  config.batch_size, num_workers),
            daemon=True,
        )
        server.start()

        workers = []
        for i in range(num_workers):
            if games_per[i] == 0:
                request_queue.put(_SHUTDOWN)
                continue
            seed = random.randint(0, 2**31 - 1)
            w = ctx.Process(
                target=_game_worker,
                args=(i, games_per[i], config, request_queue,
                      response_queues[i], result_queue, seed),
                daemon=True,
            )
            w.start()
            workers.append(w)

    # Collect results
    writer = _ShardWriter(save_dir)
    done_count = 0
    workers_done = 0
    active_workers = sum(1 for g in games_per if g > 0)

    try:
        while workers_done < active_workers:
            item = result_queue.get()
            if item[0] == "done":
                workers_done += 1
                continue
            if item[0] == "heartbeat":
                if on_heartbeat:
                    _, wid, game_num, ply = item
                    on_heartbeat(wid, game_num, ply)
                continue

            wid, game_num, states, policies, outcomes, winner, ply = item
            writer.add_game(states, policies, outcomes)
            done_count += 1

            if on_game:
                on_game(done_count, config.num_games, winner, ply, writer.total_positions)
    finally:
        writer.flush()

    # Wait for processes
    if server is not None:
        server.join(timeout=10)
    for w in workers:
        w.join(timeout=10)

    return writer.get_all()


def run_selfplay_single(
    config: SelfPlayConfig,
    on_game: Optional[Callable] = None,
    on_status: Optional[Callable] = None,
    save_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-process self-play (no multiprocessing). Good for debugging or
    when running inside a container with limited resources."""
    import torch
    from .az_net import AZNet, load_checkpoint as load_az
    from .mcts import run_mcts

    device = resolve_device(config.device)
    if config.checkpoint:
        model = load_az(config.checkpoint, device=device)
    else:
        model = AZNet().to(device)
        model.eval()

    if on_status:
        on_status(f"single-process mode, device={device}, sims={config.simulations}")

    @torch.no_grad()
    def evaluate_fn(state: State, board) -> Tuple[np.ndarray, float]:
        tensor = encode_state(state, board)
        x = torch.from_numpy(tensor).unsqueeze(0).to(device)
        p, v = model(x)
        return p[0].cpu().numpy(), float(v[0])

    writer = _ShardWriter(save_dir)

    try:
        for game_num in range(config.num_games):
            p1_col = random.randint(0, NCOLS - 1)
            p2_col = random.randint(0, NCOLS - 1)
            board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)

            states, policies, turns = [], [], []
            ply = 0
            winner = None
            while True:
                w = state.winner(board)
                if w is not None:
                    winner = w
                    break
                if ply >= config.max_plies:
                    break

                temp = config.temp_high if ply < config.temperature_moves else config.temp_low
                pi, _, move = run_mcts(state, board, evaluate_fn, config.simulations,
                                       temp, add_noise=True)
                if move is None:
                    winner = 2 if state.turn == 1 else 1
                    break

                states.append(encode_state(state, board))
                policies.append(pi)
                turns.append(state.turn)
                state = apply_move(state, move)
                ply += 1

            outcomes = []
            for t in turns:
                if winner is None:
                    outcomes.append(0.0)
                elif winner == t:
                    outcomes.append(1.0)
                else:
                    outcomes.append(-1.0)

            if states:
                writer.add_game(np.stack(states), np.stack(policies),
                                np.array(outcomes, dtype=np.float32))

            if on_game:
                on_game(game_num + 1, config.num_games, winner, ply,
                        writer.total_positions)
    finally:
        writer.flush()

    return writer.get_all()
