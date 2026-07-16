"""Repeatable full-pipeline AlphaZero self-play throughput benchmark.

The benchmark writes no training shards or checkpoints. It supports GPU systems
by comparing inference batch/concurrency pairs and CPU systems by comparing
worker counts.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .az_selfplay import (
    SelfPlayConfig,
    SelfPlayPool,
    _auto_concurrency,
    detect_hardware,
    hardware_tuning_key,
    save_tuning_profile,
)


def gpu_configurations(batch_size: int, workers: int) -> List[Tuple[int, int]]:
    """Portable comparison around the detected VRAM-based batch recommendation.

    Sweeps batch size at auto concurrency, plus higher-concurrency variants at
    the recommended batch: on wide GPUs throughput is usually bound by games
    in flight (workers x concurrency), not batch size, and the auto value is
    conservatively capped.
    """
    batches = sorted({max(8, batch_size // 2), batch_size, batch_size * 2})
    auto = _auto_concurrency(batch_size, workers)
    configs = [(batch, _auto_concurrency(batch, workers)) for batch in batches]
    for scale in (2, 3):
        candidate = (batch_size, min(64, auto * scale))
        if candidate not in configs:
            configs.append(candidate)
    return configs


def cpu_configurations(workers: int) -> List[int]:
    """Compare a half-host allocation with the detected full allocation."""
    return sorted({max(1, workers // 2), max(1, workers)})


def benchmark_configuration(
    *,
    device: str,
    games: int,
    simulations: int,
    max_plies: int,
    workers: int,
    batch_size: int = 1,
    concurrency: int = 1,
    checkpoint: str = "",
    inference_servers: int = 1,
    fp16: bool = False,
    batch_timeout_ms: float = 0.0,
    on_game: Optional[Callable] = None,
    on_heartbeat: Optional[Callable] = None,
) -> Dict[str, object]:
    config = SelfPlayConfig(
        num_games=games,
        simulations=simulations,
        max_plies=max_plies,
        workers=workers,
        concurrent_games=concurrency,
        batch_size=batch_size,
        device=device,
        checkpoint=checkpoint,
        inference_servers=inference_servers,
        inference_fp16=fp16,
        batch_timeout_ms=batch_timeout_ms,
    )
    with SelfPlayPool(config) as pool:
        pool.run(games, checkpoint=checkpoint, save_dir=None,
                 on_game=on_game, on_heartbeat=on_heartbeat)
        metrics = dict(pool.last_metrics)
    metrics.update({
        "device": device,
        "workers": workers,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "inference_servers": inference_servers,
        "fp16": fp16,
        "batch_timeout_ms": batch_timeout_ms,
    })
    elapsed = max(float(metrics["elapsed_s"]), 1e-9)
    metrics["games_per_s"] = metrics["games"] / elapsed
    metrics["positions_per_s"] = metrics["positions"] / elapsed
    metrics["evals_per_s"] = metrics["eval_requests"] / elapsed
    return metrics


def row_from_metrics(m: Dict[str, object]) -> Dict[str, object]:
    """Normalize one benchmark_configuration result into a stored history row."""
    inf = m.get("inference") or {}
    batches = int(inf.get("batches", 0))
    nsrv = max(1, int(inf.get("num_servers", m.get("inference_servers", 1))))
    elapsed = max(float(m["elapsed_s"]), 1e-9)
    return {
        "workers": int(m["workers"]),
        "batch": int(m["batch_size"]),
        "concurrency": int(m["concurrency"]),
        "servers": int(m.get("inference_servers", 1)),
        "fp16": bool(m.get("fp16", False)),
        "batch_timeout_ms": float(m.get("batch_timeout_ms", 0.0)),
        "evals_per_s": float(m["evals_per_s"]),
        "positions_per_s": float(m["positions_per_s"]),
        "games_per_s": float(m["games_per_s"]),
        "avg_batch": float(inf.get("avg_batch", 0.0)),
        "fill_pct": (100.0 * int(inf.get("full_batches", 0)) / batches
                     if batches else 0.0),
        "request_wait_ms": float(m.get("avg_request_wait_ms", 0.0)),
        # inference_s sums wall-time across parallel servers, so busy fraction
        # is per-server-average.
        "gpu_busy_pct": 100.0 * float(inf.get("inference_s", 0.0)) / (elapsed * nsrv),
    }


def record_benchmark(hw: dict, *, device: str, simulations: int, max_plies: int,
                     games: int, checkpoint: str,
                     results: List[Dict[str, object]]) -> Dict[str, object]:
    """Append a history record and persist the winner as the tuned default.

    The saved profile is always the fastest fp32 row: fp16 changes network
    outputs, so it is recorded (and flagged via "fp16_recommended") but never
    auto-applied. Returns the stored record.
    """
    from .az_bench_store import append_record, hardware_fingerprint, select_best

    ram_gb = int(hw.get("ram_gb", 0))
    rows = [row_from_metrics(m) for m in results]
    best, fp16_recommended = select_best(rows)
    if device == "cuda":
        key = hardware_tuning_key(
            "cuda", hw["ncpu"], hw["gpu_name"], hw["vram_gb"], hw["gpu_count"],
            ram_gb=ram_gb)
        fingerprint = hardware_fingerprint({**hw, "device": "cuda"})
        profile = {
            "workers": best["workers"],
            "inference_batch": best["batch"],
            "concurrency": best["concurrency"],
            "inference_servers": best["servers"],
            "batch_timeout_ms": best["batch_timeout_ms"],
        }
    else:
        key = hardware_tuning_key("cpu", hw["ncpu"], ram_gb=ram_gb)
        # CPU-local self-play has no inference server; only workers matter.
        fingerprint = hardware_fingerprint({
            "device": "cpu", "ncpu": hw["ncpu"], "ram_gb": ram_gb})
        profile = {"workers": best["workers"]}
    save_tuning_profile(key, profile)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fingerprint": fingerprint,
        "params": {"simulations": simulations, "max_plies": max_plies,
                   "games": games, "checkpoint": checkpoint},
        "rows": rows,
        "best": profile,
        "fp16_recommended": fp16_recommended,
    }
    append_record(key, record)
    return record


def _parse_gpu_configs(value: str) -> Iterable[Tuple[int, int]]:
    for item in value.split(","):
        batch, concurrency = item.split(":", 1)
        yield int(batch), int(concurrency)


def main() -> None:
    hw = detect_hardware()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--games", type=int, default=0,
                        help="0 chooses enough games to fill the tested configuration")
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 uses the saved or detected default for the selected device")
    parser.add_argument(
        "--configs", default="",
        help="optional comma-separated GPU batch:games-per-worker pairs",
    )
    parser.add_argument("--checkpoint", default="",
                        help="checkpoint name for realistic tree shapes "
                             "(empty = random init)")
    parser.add_argument("--servers", default="",
                        help="comma-separated inference-server counts to sweep "
                             "on GPU (default: the detected recommendation)")
    parser.add_argument("--fp16", action="store_true",
                        help="also run every GPU configuration with "
                             "inference_fp16 enabled, for comparison")
    parser.add_argument("--batch-timeout-ms", type=float, default=0.0,
                        help="partial-batch flush wait (0 = 5ms default)")
    args = parser.parse_args()
    device = hw["device"] if args.device == "auto" else args.device
    workers = args.workers or (hw.get("cpu_workers", hw["workers"])
                               if device == "cpu" else hw["workers"])

    if device == "cuda":
        configs = (list(_parse_gpu_configs(args.configs)) if args.configs else
                   gpu_configurations(hw["inference_batch"], workers))
        server_counts = ([int(s) for s in args.servers.split(",")]
                         if args.servers else [hw.get("inference_servers", 1)])
        fp16_modes = [False, True] if args.fp16 else [False]
        games = args.games or max(32, max(workers * c for _, c in configs))
        print(f"device=cuda gpu={hw['gpu_name']} vram={hw['vram_gb']:.1f}GB "
              f"ncpu={hw['ncpu']}")
        print(f"games={games} simulations={args.simulations} "
              f"max_plies={args.max_plies} workers={workers} "
              f"checkpoint={args.checkpoint or '(random)'}")
        print("batch conc srv fp16 elapsed games/s pos/s eval/s avg_batch "
              "fill% request_ms gpu_busy%")
        results = []
        for servers in server_counts:
            for fp16 in fp16_modes:
                for batch, concurrency in configs:
                    m = benchmark_configuration(
                        device=device, games=games,
                        simulations=args.simulations,
                        max_plies=args.max_plies, workers=workers,
                        batch_size=batch, concurrency=concurrency,
                        checkpoint=args.checkpoint,
                        inference_servers=servers, fp16=fp16,
                        batch_timeout_ms=args.batch_timeout_ms,
                    )
                    results.append(m)
                    inf = m["inference"]
                    batches = int(inf.get("batches", 0))
                    full = int(inf.get("full_batches", 0))
                    nsrv = max(1, int(inf.get("num_servers", 1)))
                    # inference_s sums wall-time across parallel servers, so
                    # busy fraction is per-server-average.
                    busy = 100 * inf.get("inference_s", 0) / (
                        m["elapsed_s"] * nsrv)
                    print(
                        f"{batch:>5} {concurrency:>4} {nsrv:>3} "
                        f"{'y' if fp16 else 'n':>4} {m['elapsed_s']:>7.2f} "
                        f"{m['games_per_s']:>7.2f} {m['positions_per_s']:>7.1f} "
                        f"{m['evals_per_s']:>7.1f} {inf.get('avg_batch', 0):>9.1f} "
                        f"{(100 * full / batches if batches else 0):>5.1f} "
                        f"{m['avg_request_wait_ms']:>10.2f} "
                        f"{busy:>8.1f}",
                        flush=True,
                    )
        record = record_benchmark(
            hw, device="cuda", simulations=args.simulations,
            max_plies=args.max_plies, games=games, checkpoint=args.checkpoint,
            results=results)
        best = record["best"]
        print(f"saved recommendation: workers={best['workers']} "
              f"batch={best['inference_batch']} "
              f"concurrency={best['concurrency']} "
              f"servers={best['inference_servers']} "
              f"batch_timeout_ms={best['batch_timeout_ms']:g}")
        if record["fp16_recommended"]:
            print("note: an fp16 configuration was fastest — fp16 is never "
                  "auto-applied (it changes network outputs); enable "
                  "inference_fp16 explicitly to use it")
    else:
        configs = cpu_configurations(workers)
        games = args.games or max(32, max(configs) * 2)
        print(f"device=cpu games={games} simulations={args.simulations} "
              f"max_plies={args.max_plies}")
        print("workers elapsed games/s pos/s")
        results = []
        for workers in configs:
            m = benchmark_configuration(
                device="cpu", games=games, simulations=args.simulations,
                max_plies=args.max_plies, workers=workers,
                checkpoint=args.checkpoint,
            )
            results.append(m)
            print(f"{workers:>7} {m['elapsed_s']:>7.2f} "
                  f"{m['games_per_s']:>7.2f} {m['positions_per_s']:>7.1f}")
        record = record_benchmark(
            hw, device="cpu", simulations=args.simulations,
            max_plies=args.max_plies, games=games, checkpoint=args.checkpoint,
            results=results)
        print(f"saved recommendation: workers={record['best']['workers']}")


if __name__ == "__main__":
    main()
