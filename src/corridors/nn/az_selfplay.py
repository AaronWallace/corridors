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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..game import NCOLS, WALLS_PER_PLAYER, State, apply_move
from .actions import NUM_ACTIONS, move_to_index
from .encoding import NROWS, NUM_PLANES, encode_state

# Sentinel values for the inference queue
_SHUTDOWN = None
_BATCH_TIMEOUT = 0.005  # 5ms — wait this long to fill a batch
SHARD_EVERY = 25  # flush to disk every N games
_EVAL_CACHE_MAX = 20_000  # per-game NN-eval cache cap (~22 MB/worker worst case)

# Where self-play shards are written / training reads them. Defined here (a
# torch-free module) so CPU self-play never imports az_train (and thus torch).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AZ_DATA_ROOT = _PROJECT_ROOT / "nn_data" / "alphazero"

# Env vars that native math libraries (OpenMP/MKL/OpenBLAS) read at import time to
# size their thread pools. We pin these to 1 in each worker so N single-threaded
# workers don't collectively spawn N*ncpu threads and thrash the CPU.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pin_torch_threads(n: int = 1) -> None:
    """Pin the current process's PyTorch thread pools. Call at worker start,
    after `import torch`. Safe to call once per process."""
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            # Can only be set before any inter-op work has started; ignore if late.
            pass
    except Exception:
        pass


_blas_limiter = None  # kept alive for the process lifetime once set


def _limit_blas_threads(n: int = 1) -> None:
    """Force this process's BLAS/OpenMP pools to n threads at runtime. The
    inherited *_NUM_THREADS env vars usually suffice, but if the BLAS initialized
    before they were set it ignores them — then hundreds of workers each spin up a
    full multi-threaded BLAS and thrash. threadpoolctl caps it regardless."""
    global _blas_limiter
    try:
        from threadpoolctl import threadpool_limits
        _blas_limiter = threadpool_limits(limits=n)
    except Exception:
        pass


def _raise_fd_limit() -> None:
    """Raise the soft open-file limit toward the hard limit (POSIX only). Each
    worker gets its own command queue (a pipe + semaphores) and a fork pipe, so
    hundreds of workers blow past the default 1024-fd soft limit."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 1_048_576
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass


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
    concurrent_games: int = 0  # GPU mode: games in flight per worker (0 = auto)
    batch_size: int = 64       # max batch for GPU inference
    temperature_moves: int = 20  # use temp=1 for first N moves, then temp→0.1
    temp_high: float = 1.0
    temp_low: float = 0.1
    max_plies: int = 150
    checkpoint: str = ""       # empty = random init
    device: str = "auto"
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    c_puct: float = 1.5


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
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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

def _play_one_game(
    game_num: int,
    worker_id: int,
    config: SelfPlayConfig,
    evaluate_fn: Callable[[State, object], Tuple[np.ndarray, float]],
    rng: "random.Random",
    result_queue: mp.Queue,
) -> None:
    """Play a single game to completion and push its record to result_queue.
    `evaluate_fn` may block on the inference server; that's fine — when many of
    these run as threads, the blocking overlaps and the server sees real batches."""
    from .mcts import run_mcts
    import time as _time

    p1_col = rng.randint(0, NCOLS - 1)
    p2_col = rng.randint(0, NCOLS - 1)
    board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)

    # Per-game NN-eval cache: board (goals) is fixed within a game, so key by
    # state. Deduplicates transpositions within a search and positions recurring
    # across moves. Scoped to this game — starts randomize the goals, so the same
    # position rarely recurs across games, and a cross-process cache would just
    # reintroduce IPC contention.
    eval_cache = {}

    def cached_eval(st: State, bd) -> Tuple[np.ndarray, float]:
        hit = eval_cache.get(st)
        if hit is None:
            hit = evaluate_fn(st, bd)
            if len(eval_cache) < _EVAL_CACHE_MAX:
                eval_cache[st] = hit
        return hit

    record = GameRecord()
    ply = 0
    reuse = None  # tree reuse: carry the chosen subtree to the next move
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
        pi, root_val, move, reuse = run_mcts(
            state, board,
            evaluate_fn=cached_eval,
            num_simulations=config.simulations,
            temperature=temp,
            add_noise=True,
            reuse_root=reuse,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_frac=config.dirichlet_frac,
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


def _game_worker(
    worker_id: int,
    num_games: int,
    config: SelfPlayConfig,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
    concurrency: int,
) -> None:
    """Plays games using MCTS, sending leaf evaluations to the inference server.

    Runs `concurrency` games at once as threads. Each game's evaluate_fn blocks
    on the server, but because the threads block independently there are up to
    `concurrency` requests in flight per worker — that's what lets the server
    assemble batches instead of servicing one leaf per round-trip.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import itertools
    import threading

    np.random.seed(seed & 0x7FFFFFFF)

    # Per-request routing. A single reader thread drains the (shared) response
    # queue and hands each result to the waiting game thread via its Event slot.
    req_ids = itertools.count()  # next() is atomic in CPython — safe across threads
    slots: Dict[int, list] = {}          # rid -> [Event, result]
    slots_lock = threading.Lock()
    stop_reader = threading.Event()

    def _reader() -> None:
        while not stop_reader.is_set():
            try:
                rid, policy, value = response_queue.get(timeout=0.2)
            except Exception:
                continue
            with slots_lock:
                slot = slots.get(rid)
            if slot is not None:
                slot[1] = (policy, value)
                slot[0].set()

    def evaluate_fn(state: State, board) -> Tuple[np.ndarray, float]:
        tensor = encode_state(state, board)
        rid = next(req_ids)
        ev = threading.Event()
        slot = [ev, None]
        with slots_lock:
            slots[rid] = slot
        request_queue.put((worker_id, rid, tensor))
        ev.wait()
        with slots_lock:
            del slots[rid]
        return slot[1]

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    def _run_range(thread_idx: int, game_nums: List[int]) -> None:
        rng = random.Random(seed + 1 + thread_idx)
        for game_num in game_nums:
            _play_one_game(game_num, worker_id, config, evaluate_fn, rng, result_queue)

    # Split this worker's games round-robin across `concurrency` threads.
    concurrency = max(1, min(concurrency, num_games))
    buckets: List[List[int]] = [[] for _ in range(concurrency)]
    for g in range(num_games):
        buckets[g % concurrency].append(g)

    threads = [
        threading.Thread(target=_run_range, args=(i, buckets[i]), daemon=True)
        for i in range(concurrency) if buckets[i]
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stop_reader.set()
    reader.join(timeout=1)

    # Signal done — exactly one shutdown token per worker process.
    request_queue.put(_SHUTDOWN)
    result_queue.put(("done", worker_id))


def _game_worker_local(
    worker_id: int,
    num_games: int,
    config: SelfPlayConfig,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    """Self-contained worker: runs its own NumPy inference, no server and no
    torch — ideal for CPU clusters (each process stays lightweight)."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _limit_blas_threads(1)  # one BLAS thread per worker — parallelism is per-process

    from .az_infer_np import load_np, random_np

    rng = random.Random(seed)
    np.random.seed(seed & 0x7FFFFFFF)

    net = load_np(config.checkpoint) if config.checkpoint else random_np()

    def evaluate_fn(state: State, board) -> Tuple[np.ndarray, float]:
        return net.forward(encode_state(state, board))

    # Shares the single game loop (NN-eval cache + tree reuse live there).
    for game_num in range(num_games):
        _play_one_game(game_num, worker_id, config, evaluate_fn, rng, result_queue)

    result_queue.put(("done", worker_id))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    # Only touch torch when we actually have to decide (device="auto"). An explicit
    # "cpu"/"cuda" resolves without importing torch, so CPU self-play needs no torch.
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_WORKER_MEM_MB = 300  # conservative per-worker RSS estimate for the memory cap


def _read_int_file(path: str):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _cgroup_available_gb() -> float:
    """Available RAM within this process's cgroup (containers/pods report the
    HOST's memory in /proc/meminfo, so the cgroup limit is what actually applies).
    0.0 if there's no cgroup limit."""
    _NO_LIMIT = 1 << 62
    # cgroup v2
    limit = _read_int_file("/sys/fs/cgroup/memory.max")  # None if 'max' (no limit)
    if limit is not None and 0 < limit < _NO_LIMIT:
        cur = _read_int_file("/sys/fs/cgroup/memory.current") or 0
        return max(0.0, (limit - cur) / (1024 ** 3))
    # cgroup v1
    limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None and 0 < limit < _NO_LIMIT:
        usage = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0
        return max(0.0, (limit - usage) / (1024 ** 3))
    return 0.0


def available_memory_gb() -> float:
    """Best-effort available RAM in GB (0.0 if it can't be determined). Takes the
    min of host-available and the cgroup limit so it's correct inside containers."""
    candidates = []
    try:
        with open("/proc/meminfo") as f:  # Linux host view
            for line in f:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) / (1024 * 1024))  # kB -> GB
                    break
    except Exception:
        pass
    cg = _cgroup_available_gb()  # container limit (0 if none)
    if cg > 0:
        candidates.append(cg)
    if candidates:
        return min(candidates)
    try:
        import ctypes  # Windows

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return ms.ullAvailPhys / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def memory_worker_cap(per_worker_mb: int = _WORKER_MEM_MB) -> Optional[int]:
    """Max workers that fit in ~75% of available RAM (None if RAM is unknown)."""
    avail = available_memory_gb()
    if avail <= 0:
        return None
    return max(1, int(avail * 1024 * 0.75 / per_worker_mb))


def auto_workers(mode: str = "gpu") -> int:
    ncpu = os.cpu_count() or 4
    if mode == "cpu":
        # Each worker is pinned to a single thread, so one worker per core is
        # optimal; leave one core for the collecting coordinator.
        n = max(2, ncpu - 1)
    else:
        # GPU mode: workers drive MCTS (CPU-bound; ~1 core each due to the GIL) to
        # feed a single inference server. Scale with cores but cap it — past ~a
        # couple hundred, the single request queue becomes the bottleneck.
        n = max(2, min(ncpu - 2, 128))
    # Also cap by RAM: each worker holds a model/cache/tree, so on memory-light
    # hosts core count alone will OOM (each worker is ~300 MB).
    mem_cap = memory_worker_cap()
    if mem_cap is not None:
        n = min(n, mem_cap)
    return max(1, n)


def _auto_concurrency(batch_size: int, num_workers: int) -> int:
    """Games per worker needed to keep about two inference batches in flight.

    One batch worth can leave the GPU idle while responses travel back and
    workers perform CPU search. Two batches provide portable overlap across
    separate CPU and GPU processes while retaining a conservative host cap.
    """
    target = 2 * batch_size
    return min(16, max(1, -(-target // max(num_workers, 1))))


def hardware_tuning_key(device: str, ncpu: int, gpu_name: str = "",
                        vram_gb: float = 0.0, gpu_count: int = 0) -> str:
    """Stable hardware-class key; intentionally excludes volatile host names."""
    if device == "cuda":
        return f"cuda|cpu={ncpu}|gpu={gpu_name}|vram={vram_gb:.1f}|count={gpu_count}"
    return f"cpu|cpu={ncpu}"


def save_tuning_profile(hardware_key: str, values: Dict[str, object]) -> None:
    """Persist a benchmark winner without disturbing profiles for other hosts."""
    from .. import settings
    current = settings.load().get("az_tuning_profiles", {})
    profiles = dict(current) if isinstance(current, dict) else {}
    profiles[hardware_key] = dict(values)
    settings.save(az_tuning_profiles=profiles)


def _load_tuning_profile(hardware_key: str) -> dict:
    from .. import settings
    profiles = settings.load().get("az_tuning_profiles", {})
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(hardware_key, {})
    return profile if isinstance(profile, dict) else {}


def _apply_tuning_profile(result: dict) -> dict:
    profile = _load_tuning_profile(result["hardware_key"])
    if not profile:
        return result
    allowed = {"workers", "inference_batch", "concurrency", "games_per_iter"}
    for key in allowed:
        value = profile.get(key)
        if isinstance(value, int) and value > 0:
            result[key] = value
    result["benchmark_tuned"] = True
    return result


def detect_hardware() -> dict:
    """Probe the CPU/GPU and recommend self-play + training defaults tuned to it.

    Returns a dict with: device, gpu_name, vram_gb, ncpu, workers,
    inference_batch, concurrency, games_per_iter, train_batch. The self-play
    knobs are sized so the GPU inference batches actually fill (games in flight =
    workers*concurrency), which is the main throughput lever. Falls back to safe
    CPU values if torch or a GPU is unavailable.
    """
    ncpu = os.cpu_count() or 4
    avail_gb = available_memory_gb()
    device, gpu_name, vram_gb, gpu_count = "cpu", "", 0.0, 0
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = props.total_memory / (1024 ** 3)
    except Exception:
        pass

    mode = "gpu" if device == "cuda" else "cpu"
    workers = auto_workers(mode)
    cpu_workers = auto_workers("cpu")
    cpu_profile = _load_tuning_profile(hardware_tuning_key("cpu", ncpu))
    if isinstance(cpu_profile.get("workers"), int) and cpu_profile["workers"] > 0:
        cpu_workers = cpu_profile["workers"]

    mem_cap = memory_worker_cap()
    mem_capped = mem_cap is not None and mem_cap < (ncpu - 1 if mode == "cpu"
                                                    else min(ncpu - 2, 128))
    if mode == "cpu":
        result = {
            "device": device, "gpu_name": gpu_name, "vram_gb": vram_gb,
            "gpu_count": gpu_count, "ncpu": ncpu, "avail_gb": avail_gb,
            "mem_capped": mem_capped, "workers": workers,
            "cpu_workers": cpu_workers,
            "inference_batch": 64, "concurrency": 1,
            "games_per_iter": max(64, workers * 4), "train_batch": 256,
            "hardware_key": hardware_tuning_key(device, ncpu),
            "benchmark_tuned": False,
        }
        return _apply_tuning_profile(result)

    # GPU: the net is small, so batch size is about keeping the card busy, not a
    # hard memory limit — scale it (and the training batch) by VRAM tier.
    if vram_gb >= 20:
        inference_batch, train_batch = 512, 1024
    elif vram_gb >= 12:
        inference_batch, train_batch = 256, 512
    elif vram_gb >= 10:
        inference_batch, train_batch = 128, 256
    else:
        # A conservative batch for GPUs below the higher VRAM tiers. The
        # benchmark menu can tune this for the host's CPU/GPU balance.
        inference_batch, train_batch = 64, 128

    concurrency = _auto_concurrency(inference_batch, workers)
    in_flight = workers * concurrency
    # Self-play is CPU-bound, so iteration time scales with total games. Match the
    # in-flight count (all concurrency slots busy) without a multiplier that would
    # just make each iteration longer for no GPU benefit.
    games_per_iter = int(min(512, max(128, in_flight)))
    result = {
        "device": device, "gpu_name": gpu_name, "vram_gb": vram_gb,
        "gpu_count": gpu_count, "ncpu": ncpu, "avail_gb": avail_gb,
        "mem_capped": mem_capped, "workers": workers,
        "cpu_workers": cpu_workers,
        "inference_batch": inference_batch, "concurrency": concurrency,
        "games_per_iter": games_per_iter, "train_batch": train_batch,
        "hardware_key": hardware_tuning_key(device, ncpu, gpu_name, vram_gb, gpu_count),
        "benchmark_tuned": False,
    }
    return _apply_tuning_profile(result)


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
    import sys
    _raise_fd_limit()  # per-worker queues + spawn pipes can exceed the 1024 default
    device = resolve_device(config.device)
    mode = "cpu" if device == "cpu" else "gpu"
    # Always spawn: GPU can't re-init CUDA in a fork, and CPU workers re-import
    # NumPy fresh so its BLAS honors the pinned 1-thread env (a forked child would
    # inherit the parent's already-initialized multi-threaded BLAS → oversubscribe).
    ctx = mp.get_context("spawn")
    num_workers = config.workers if config.workers > 0 else auto_workers(mode)
    num_workers = min(num_workers, config.num_games)

    # GPU mode: how many games each worker keeps in flight so the server can batch.
    # Aim for the whole batch to be fillable across all workers.
    concurrency = config.concurrent_games
    if concurrency <= 0:
        concurrency = _auto_concurrency(config.batch_size, num_workers)

    if on_status:
        label = "local-inference" if device == "cpu" else "gpu-server"
        extra_info = f", concurrency={concurrency}" if device != "cpu" else ""
        on_status(f"device={device}, workers={num_workers}, mode={label}{extra_info}, "
                  f"sims={config.simulations}, games={config.num_games}")

    # Distribute games
    base, extra = divmod(config.num_games, num_workers)
    games_per = [base + (1 if i < extra else 0) for i in range(num_workers)]

    # Pin native math-library thread pools to 1 in every child. Child processes
    # inherit this env at spawn time; N single-threaded workers then scale across
    # cores instead of each grabbing every core and thrashing. Restore afterwards
    # so we don't throttle anything the caller runs later (e.g. GPU training).
    _saved_env = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}
    for v in _THREAD_ENV_VARS:
        os.environ[v] = "1"

    result_queue = ctx.Queue()

    try:
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
                          response_queues[i], result_queue, seed, concurrency),
                    daemon=True,
                )
                w.start()
                workers.append(w)
    finally:
        for v, val in _saved_env.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val

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
    except KeyboardInterrupt:
        pass
    finally:
        writer.flush()
        for w in workers:
            w.terminate()
        if server is not None:
            server.terminate()
        for w in workers:
            w.join(timeout=5)
        if server is not None:
            server.join(timeout=5)

    return writer.get_all()


def run_selfplay_single(
    config: SelfPlayConfig,
    on_game: Optional[Callable] = None,
    on_status: Optional[Callable] = None,
    save_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-process self-play (no multiprocessing). Good for debugging or
    when running inside a container with limited resources."""
    from .mcts import run_mcts

    device = resolve_device(config.device)
    if on_status:
        on_status(f"single-process mode, device={device}, sims={config.simulations}")

    if device == "cpu":
        # Torch-free NumPy inference — no torch needed for CPU self-play.
        from .az_infer_np import load_np, random_np
        net = load_np(config.checkpoint) if config.checkpoint else random_np()

        def evaluate_fn(state: State, board) -> Tuple[np.ndarray, float]:
            return net.forward(encode_state(state, board))
    else:
        import torch
        from .az_net import AZNet, load_checkpoint as load_az
        if config.checkpoint:
            model = load_az(config.checkpoint, device=device)
        else:
            model = AZNet().to(device)
            model.eval()

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
            reuse = None
            eval_cache = {}

            def cached_eval(st, bd, _c=eval_cache):
                hit = _c.get(st)
                if hit is None:
                    hit = evaluate_fn(st, bd)
                    if len(_c) < _EVAL_CACHE_MAX:
                        _c[st] = hit
                return hit

            while True:
                w = state.winner(board)
                if w is not None:
                    winner = w
                    break
                if ply >= config.max_plies:
                    break

                temp = config.temp_high if ply < config.temperature_moves else config.temp_low
                pi, _, move, reuse = run_mcts(state, board, cached_eval,
                                              config.simulations, temp, add_noise=True,
                                              reuse_root=reuse,
                                              c_puct=config.c_puct,
                                              dirichlet_alpha=config.dirichlet_alpha,
                                              dirichlet_frac=config.dirichlet_frac)
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


# ---------------------------------------------------------------------------
# Persistent pool — reuse server + workers across many rounds (iterations)
#
# The one-shot run_selfplay above spawns everything on every call, which costs
# process-spawn + CUDA init on each iteration of a training loop. SelfPlayPool
# spawns once and plays repeated rounds; between rounds the GPU server hot-swaps
# the model checkpoint (CPU workers reload their own local model on change).
# ---------------------------------------------------------------------------

# Control message tags sent through the (otherwise eval-request) request_queue.
_CMD_TAG = "__cmd__"


def _inference_server_persistent(
    request_queue: mp.Queue,
    response_queues: Dict[int, mp.Queue],
    ack_queue: mp.Queue,
    device: str,
    batch_size: int,
) -> None:
    """Long-lived batched inference server. Services eval requests and, between
    rounds, handles control messages: ("__cmd__","reload",ckpt) reloads the model
    and acks; ("__cmd__","stop",_) exits. Blocks (no busy-spin) while idle."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import torch
    from .az_net import AZNet, load_checkpoint as load_az

    model = None
    pending: List[Tuple[int, int, np.ndarray]] = []
    pending_since = 0.0
    metrics = {}

    def _reset_metrics() -> None:
        metrics.clear()
        metrics.update({
            "batches": 0, "positions": 0, "full_batches": 0,
            "timeout_batches": 0, "max_batch": 0,
            "batch_wait_s": 0.0, "inference_s": 0.0,
        })

    _reset_metrics()

    def _flush_batch(reason: str = "control"):
        nonlocal pending_since
        if not pending or model is None:
            pending.clear()
            pending_since = 0.0
            return
        now = time.monotonic()
        count = len(pending)
        metrics["batches"] += 1
        metrics["positions"] += count
        metrics["max_batch"] = max(metrics["max_batch"], count)
        metrics["batch_wait_s"] += now - pending_since
        if reason == "full":
            metrics["full_batches"] += 1
        elif reason == "timeout":
            metrics["timeout_batches"] += 1
        batch_np = np.stack([t for _, _, t in pending])
        infer_t0 = time.monotonic()
        with torch.inference_mode():
            x = torch.from_numpy(batch_np).to(device)
            policy_logits, values = model(x)
            policy_np = policy_logits.cpu().numpy()
            values_np = values.cpu().numpy()
        metrics["inference_s"] += time.monotonic() - infer_t0
        for i, (wid, rid, _) in enumerate(pending):
            response_queues[wid].put((rid, policy_np[i], float(values_np[i])))
        pending.clear()
        pending_since = 0.0

    while True:
        # Use a short timeout only when a partial batch is waiting to be flushed;
        # otherwise block so an idle server (e.g. during training) doesn't spin.
        if pending:
            try:
                item = request_queue.get(timeout=_BATCH_TIMEOUT)
            except Exception:
                _flush_batch("timeout")
                continue
        else:
            item = request_queue.get()

        if item[0] == _CMD_TAG:
            _flush_batch()
            _, sub, arg = item
            if sub == "reload":
                if arg:
                    model = load_az(arg, device=device)
                else:
                    model = AZNet().to(device)
                    model.eval()
                _reset_metrics()
                ack_queue.put("ready")
                continue
            if sub == "stats":
                result = dict(metrics)
                result["avg_batch"] = (
                    result["positions"] / result["batches"]
                    if result["batches"] else 0.0
                )
                ack_queue.put(result)
                continue
            if sub == "stop":
                break
            continue

        if not pending:
            pending_since = time.monotonic()
        pending.append(item)
        if len(pending) >= batch_size:
            _flush_batch("full")


def _game_worker_persistent(
    worker_id: int,
    config: SelfPlayConfig,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    cmd_queue: mp.Queue,
) -> None:
    """Long-lived GPU-mode worker. Waits for ("play", n, seed, concurrency, base)
    commands, plays that many games (threaded for concurrency, feeding the shared
    inference server), signals ("round_done", wid), then waits for the next round.
    ("stop",) exits."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import itertools
    import threading

    req_ids = itertools.count()
    slots: Dict[int, list] = {}
    slots_lock = threading.Lock()
    stop_reader = threading.Event()
    metric_lock = threading.Lock()
    eval_requests = 0
    inference_wait_s = 0.0

    def _reader() -> None:
        while not stop_reader.is_set():
            try:
                rid, policy, value = response_queue.get(timeout=0.2)
            except Exception:
                continue
            with slots_lock:
                slot = slots.get(rid)
            if slot is not None:
                slot[1] = (policy, value)
                slot[0].set()

    def evaluate_fn(state: State, board) -> Tuple[np.ndarray, float]:
        nonlocal eval_requests, inference_wait_s
        tensor = encode_state(state, board)
        rid = next(req_ids)
        ev = threading.Event()
        slot = [ev, None]
        with slots_lock:
            slots[rid] = slot
        wait_t0 = time.monotonic()
        request_queue.put((worker_id, rid, tensor))
        ev.wait()
        waited = time.monotonic() - wait_t0
        with metric_lock:
            eval_requests += 1
            inference_wait_s += waited
        with slots_lock:
            del slots[rid]
        return slot[1]

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        while True:
            cmd = cmd_queue.get()
            if cmd[0] == "stop":
                break
            _, num_games, seed, concurrency, base_game = cmd
            with metric_lock:
                eval_requests = 0
                inference_wait_s = 0.0
            np.random.seed(seed & 0x7FFFFFFF)
            concurrency = max(1, min(concurrency, num_games))
            buckets: List[List[int]] = [[] for _ in range(concurrency)]
            for g in range(num_games):
                buckets[g % concurrency].append(base_game + g)

            def _run_range(thread_idx: int, game_nums: List[int]) -> None:
                rng = random.Random(seed + 1 + thread_idx)
                for game_num in game_nums:
                    _play_one_game(game_num, worker_id, config, evaluate_fn,
                                   rng, result_queue)

            threads = [
                threading.Thread(target=_run_range, args=(i, buckets[i]), daemon=True)
                for i in range(concurrency) if buckets[i]
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            with metric_lock:
                worker_metrics = {
                    "eval_requests": eval_requests,
                    "inference_wait_s": inference_wait_s,
                }
            result_queue.put(("round_done", worker_id, worker_metrics))
    finally:
        stop_reader.set()
        reader.join(timeout=1)


def _game_worker_local_persistent(
    worker_id: int,
    config: SelfPlayConfig,
    result_queue: mp.Queue,
    cmd_queue: mp.Queue,
) -> None:
    """Long-lived CPU-mode worker: runs its own NumPy inference (no torch) and
    reloads weights when the checkpoint changes between rounds. Command:
    ("play", n, seed, ckpt, base)."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _limit_blas_threads(1)  # one BLAS thread per worker — parallelism is per-process

    from .az_infer_np import load_np, random_np

    net = None
    current_ckpt = None

    while True:
        cmd = cmd_queue.get()
        if cmd[0] == "stop":
            break
        _, num_games, seed, checkpoint, base_game = cmd

        if net is None or checkpoint != current_ckpt:
            net = load_np(checkpoint) if checkpoint else random_np()
            current_ckpt = checkpoint

        def evaluate_fn(state: State, board, _net=net) -> Tuple[np.ndarray, float]:
            return _net.forward(encode_state(state, board))

        np.random.seed(seed & 0x7FFFFFFF)
        rng = random.Random(seed)
        for i in range(num_games):
            _play_one_game(base_game + i, worker_id, config, evaluate_fn,
                           rng, result_queue)
        result_queue.put(("round_done", worker_id))


class SelfPlayPool:
    """Persistent self-play worker pool. Spawn once, play many rounds.

        pool = SelfPlayPool(config)
        states, policies, outcomes = pool.run(num_games, checkpoint, save_dir, ...)
        ...                                    # repeat for each iteration
        pool.close()

    Use as a context manager to close automatically. The GPU inference server (or
    each CPU worker's local model) reloads `checkpoint` at the start of each run.
    """

    def __init__(self, config: SelfPlayConfig, on_status: Optional[Callable[[str], None]] = None) -> None:
        import sys
        _raise_fd_limit()  # one command queue + fork pipe per worker adds up fast
        self.config = config
        self.device = resolve_device(config.device)
        self.mode = "cpu" if self.device == "cpu" else "gpu"
        nw = config.workers if config.workers > 0 else auto_workers(self.mode)
        self.num_workers = max(1, min(nw, config.num_games))
        self.concurrency = config.concurrent_games
        if self.concurrency <= 0:
            self.concurrency = _auto_concurrency(config.batch_size, self.num_workers)

        # Always spawn: GPU can't re-init CUDA in a fork, and CPU workers must
        # re-import NumPy fresh so its BLAS honors the pinned 1-thread env.
        ctx = mp.get_context("spawn")
        self.ctx = ctx

        self.result_queue = ctx.Queue()
        self.cmd_queues = [ctx.Queue() for _ in range(self.num_workers)]
        self.workers: List[mp.Process] = []
        self.server: Optional[mp.Process] = None
        self.request_queue = None
        self.response_queues = None
        self.ack_queue = None
        self.last_metrics: Dict[str, object] = {}
        self._closed = False

        # Pin native math threads to 1 in every child (inherited at spawn/fork).
        saved_env = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}
        for v in _THREAD_ENV_VARS:
            os.environ[v] = "1"
        try:
            if self.mode == "cpu":
                for i in range(self.num_workers):
                    w = ctx.Process(
                        target=_game_worker_local_persistent,
                        args=(i, config, self.result_queue, self.cmd_queues[i]),
                        daemon=True,
                    )
                    w.start()
                    self.workers.append(w)
            else:
                self.request_queue = ctx.Queue()
                self.response_queues = {i: ctx.Queue() for i in range(self.num_workers)}
                self.ack_queue = ctx.Queue()
                self.server = ctx.Process(
                    target=_inference_server_persistent,
                    args=(self.request_queue, self.response_queues, self.ack_queue,
                          self.device, config.batch_size),
                    daemon=True,
                )
                self.server.start()
                for i in range(self.num_workers):
                    w = ctx.Process(
                        target=_game_worker_persistent,
                        args=(i, config, self.request_queue, self.response_queues[i],
                              self.result_queue, self.cmd_queues[i]),
                        daemon=True,
                    )
                    w.start()
                    self.workers.append(w)
        finally:
            for v, val in saved_env.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val

        if on_status:
            on_status(f"pool: device={self.device}, workers={self.num_workers}, "
                      f"mode={self.mode}, concurrency={self.concurrency} (persistent)")

    def run(
        self,
        num_games: int,
        checkpoint: str,
        save_dir: Optional[str] = None,
        on_game: Optional[Callable] = None,
        on_heartbeat: Optional[Callable] = None,
        base_game: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Play one round of `num_games` games using `checkpoint`. Returns any
        unflushed (states, policies, outcomes)."""
        if self._closed:
            raise RuntimeError("pool is closed")

        # Load the model for this round before any worker starts playing.
        if self.mode == "gpu":
            self.request_queue.put((_CMD_TAG, "reload", checkpoint))
            self.ack_queue.get()

        base, extra = divmod(num_games, self.num_workers)
        games_per = [base + (1 if i < extra else 0) for i in range(self.num_workers)]
        active = 0
        gbase = base_game
        for i in range(self.num_workers):
            if games_per[i] == 0:
                continue
            seed = random.randint(0, 2**31 - 1)
            if self.mode == "cpu":
                self.cmd_queues[i].put(("play", games_per[i], seed, checkpoint, gbase))
            else:
                self.cmd_queues[i].put(("play", games_per[i], seed, self.concurrency, gbase))
            gbase += games_per[i]
            active += 1

        writer = _ShardWriter(save_dir)
        done_workers = 0
        done_count = 0
        worker_metrics = []
        round_t0 = time.monotonic()
        try:
            while done_workers < active:
                item = self.result_queue.get()
                if item[0] == "round_done":
                    done_workers += 1
                    if len(item) > 2:
                        worker_metrics.append(item[2])
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
                    on_game(done_count, num_games, winner, ply, writer.total_positions)
        finally:
            writer.flush()
        elapsed = time.monotonic() - round_t0
        inference = {}
        if self.mode == "gpu":
            self.request_queue.put((_CMD_TAG, "stats", ""))
            inference = self.ack_queue.get()
        requests = sum(m["eval_requests"] for m in worker_metrics)
        wait_s = sum(m["inference_wait_s"] for m in worker_metrics)
        self.last_metrics = {
            "elapsed_s": elapsed,
            "games": done_count,
            "positions": writer.total_positions,
            "eval_requests": requests,
            "worker_inference_wait_s": wait_s,
            "avg_request_wait_ms": 1000.0 * wait_s / requests if requests else 0.0,
            "inference": inference,
        }
        return writer.get_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for q in self.cmd_queues:
            try:
                q.put(("stop",))
            except Exception:
                pass
        if self.mode == "gpu" and self.request_queue is not None:
            try:
                self.request_queue.put((_CMD_TAG, "stop", ""))
            except Exception:
                pass
        for w in self.workers:
            w.join(timeout=5)
        if self.server is not None:
            self.server.join(timeout=5)
        for w in self.workers:
            if w.is_alive():
                w.terminate()
        if self.server is not None and self.server.is_alive():
            self.server.terminate()

    def __enter__(self) -> "SelfPlayPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
