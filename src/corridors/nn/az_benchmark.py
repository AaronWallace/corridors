"""Repeatable full-pipeline AlphaZero self-play throughput benchmark.

The benchmark writes no training shards or checkpoints. It supports GPU systems
by comparing inference batch/concurrency pairs and CPU systems by comparing
worker counts.
"""

from __future__ import annotations

import argparse
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
    """Portable comparison around the detected VRAM-based batch recommendation."""
    batches = sorted({max(8, batch_size // 2), batch_size, batch_size * 2})
    return [(batch, _auto_concurrency(batch, workers)) for batch in batches]


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
    })
    elapsed = max(float(metrics["elapsed_s"]), 1e-9)
    metrics["games_per_s"] = metrics["games"] / elapsed
    metrics["positions_per_s"] = metrics["positions"] / elapsed
    metrics["evals_per_s"] = metrics["eval_requests"] / elapsed
    return metrics


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
        best = max(results, key=lambda result: result["positions_per_s"])
        key = hardware_tuning_key(
            "cuda", hw["ncpu"], hw["gpu_name"], hw["vram_gb"], hw["gpu_count"])
        save_tuning_profile(key, {
            "workers": int(best["workers"]),
            "inference_batch": int(best["batch_size"]),
            "concurrency": int(best["concurrency"]),
        })
        print(f"saved recommendation: workers={best['workers']} "
              f"batch={best['batch_size']} concurrency={best['concurrency']}")
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
            )
            results.append(m)
            print(f"{workers:>7} {m['elapsed_s']:>7.2f} "
                  f"{m['games_per_s']:>7.2f} {m['positions_per_s']:>7.1f}")
        best = max(results, key=lambda result: result["positions_per_s"])
        save_tuning_profile(hardware_tuning_key("cpu", hw["ncpu"]), {
            "workers": int(best["workers"]),
        })
        print(f"saved recommendation: workers={best['workers']}")


if __name__ == "__main__":
    main()
