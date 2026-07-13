"""AlphaZero training pipeline menu.

Full loop: self-play → train → repeat. Or run each step individually.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from rich import box
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .checkpoints import ranked_checkpoint_paths, resolve_checkpoint_path

STYLE_GRID = "grey35"
STYLE_HINT = "bold green"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_ROOT = _PROJECT_ROOT / "nn_checkpoints"


def _read_meta(name: str) -> dict:
    p = resolve_checkpoint_path(CHECKPOINT_ROOT, name).with_suffix(".meta.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _console():
    from ..play import console
    return console


def _prompt_int(label: str, default: int, lo: int, hi: int) -> int:
    console = _console()
    while True:
        raw = Prompt.ask(f"[dim]{label}[/dim]", default=str(default))
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        console.print(f"[red]  must be an integer {lo}..{hi}[/red]")


def _prompt_float(label: str, default: float, lo: float, hi: float) -> float:
    console = _console()
    while True:
        raw = Prompt.ask(f"[dim]{label}[/dim]", default=f"{default:g}")
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        console.print(f"[red]  must be a number {lo}..{hi}[/red]")


# ---------------------------------------------------------------------------
# Shared self-play runner with a single self-refreshing status line
# ---------------------------------------------------------------------------

def _detect_and_show_hw() -> dict:
    """Probe the CPU/GPU and print the detected hardware plus the tuned defaults."""
    from .az_selfplay import detect_hardware
    console = _console()
    hw = detect_hardware()
    source = "benchmark" if hw.get("benchmark_tuned") else "hardware heuristic"
    ram = hw.get("avail_gb", 0.0)
    ram_tag = f" · {ram:.0f} GB free RAM" if ram > 0 else ""
    if hw["device"] == "cuda":
        gpu_n = hw.get("gpu_count", 1)
        gpu_tag = f"{gpu_n}× {hw['gpu_name']}" if gpu_n > 1 else hw['gpu_name']
        console.print(f"\n[dim]detected:[/dim] [white]{gpu_tag}[/white] "
                      f"[dim]({hw['vram_gb']:.0f} GB VRAM each) · {hw['ncpu']} CPUs{ram_tag}[/dim]")
        if gpu_n > 1:
            console.print(f"[yellow]note:[/yellow] [dim]only cuda:0 is used. To use the "
                          f"other {gpu_n - 1}, run separate jobs pinned with "
                          f"CUDA_VISIBLE_DEVICES=<n> (see below).[/dim]")
        console.print(f"[dim]defaults ({source}): workers {hw['workers']} · inference batch "
                      f"{hw['inference_batch']} · games/iter {hw['games_per_iter']} · "
                      f"train batch {hw['train_batch']}[/dim]")
    else:
        console.print(f"\n[dim]detected: CPU only ({hw['ncpu']} cores{ram_tag}), no CUDA GPU[/dim]")
        console.print(f"[dim]defaults ({source}): workers {hw['workers']} · "
                      f"games/iter {hw['games_per_iter']} · train batch {hw['train_batch']}[/dim]")
    if hw.get("mem_capped"):
        console.print(f"[yellow]note:[/yellow] [dim]workers capped to {hw['workers']} to fit "
                      f"~{ram:.0f} GB RAM (each worker ~0.3 GB); raise only if you have headroom.[/dim]")
    return hw


def _prompt_selfplay_params(num_games: int, hw: dict):
    """Prompt for the self-play device and parallelism. Returns
    (device, workers, batch_size, concurrency).

    CPU-local self-play (each worker plays full games with its own model, no
    inference server) scales with cores, starts instantly, and streams completed
    games — usually fastest on many-core hosts, where funneling a tiny net's evals
    through one GPU server queue becomes the bottleneck. GPU batching wins on
    few-core hosts. So default to CPU when there are plenty of cores."""
    console = _console()
    ncpu = hw["ncpu"]
    if hw["device"] == "cpu":
        device = "cpu"
    else:
        default_dev = "cpu" if ncpu >= 32 else "cuda"
        console.print("[dim]CPU-local self-play scales with cores + instant start; "
                      "GPU batches evals through one server (better on few cores).[/dim]")
        device = Prompt.ask("[dim]Self-play device[/dim]",
                            choices=["cpu", "cuda"], default=default_dev)

    hi = min(ncpu, num_games) if num_games > 0 else ncpu
    if device == "cpu":
        default_workers = min(hw.get("cpu_workers", max(2, ncpu - 1)), hi)
    else:
        default_workers = min(hw["workers"], hi)
    workers = _prompt_int(f"Workers [1-{hi}]", default_workers, 1, hi)

    batch_size = 64
    concurrency = 0
    inference_servers = 1
    if device != "cpu":
        batch_size = _prompt_int("Inference batch size", hw["inference_batch"], 1, 8192)
        concurrency = _prompt_int("Concurrent games per worker (0=auto)",
                                  hw.get("concurrency", 0), 0, 256)
        inference_servers = _prompt_int("Inference servers (parallel, share the GPU)",
                                        hw.get("inference_servers", 1), 1, 64)
    return device, workers, batch_size, concurrency, inference_servers


def _prompt_search_params() -> dict:
    """Optional AlphaZero exploration controls; defaults remain conservative."""
    if not Confirm.ask("[dim]Adjust advanced MCTS exploration settings?[/dim]", default=False):
        return {}
    return {
        "c_puct": _prompt_float("PUCT exploration constant", 1.5, 0.01, 20.0),
        "dirichlet_alpha": _prompt_float("Root Dirichlet alpha", 0.3, 0.001, 10.0),
        "dirichlet_frac": _prompt_float("Root noise fraction", 0.25, 0.0, 1.0),
        "temperature_moves": _prompt_int("High-temperature opening plies", 20, 0, 1000),
        "temp_high": _prompt_float("Opening temperature", 1.0, 0.001, 10.0),
        "temp_low": _prompt_float("Later temperature", 0.1, 0.001, 10.0),
    }


def _auto_dataset_name(*, prefix: str, games: int, simulations: int,
                       max_plies: int, device: str, workers: int,
                       batch_size: int, concurrency: int,
                       search_params: dict, timestamp: Optional[str] = None) -> str:
    """Build a short recognizable name; full settings live in run metadata."""
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}_g{games}_s{simulations}"


def _checkpoint_table(checkpoints, title: str = "AZ checkpoints") -> Table:
    """Build the ranked checkpoint table shown by AlphaZero setup prompts."""
    from . import model as model_mod

    details = {item["name"]: item for item in model_mod.list_checkpoints()}
    table = Table(box=box.SIMPLE, header_style="dim", title=title, title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("checkpoint", max_width=48, overflow="ellipsis", no_wrap=True)
    table.add_column("Elo", justify="right")
    table.add_column("epoch", justify="right")
    table.add_column("training data", style="dim", max_width=28,
                     overflow="ellipsis", no_wrap=True)
    table.add_column("ancestry", style="dim", max_width=30,
                     overflow="ellipsis", no_wrap=True)
    table.add_column("MB", justify="right")
    for index, name in enumerate(checkpoints, 1):
        item = details.get(name, {})
        elo = item.get("elo")
        if item.get("seeded_from"):
            ancestry = f"seed: {item['seeded_from']}"
        elif item.get("resumed_from") and item["resumed_from"] != name:
            ancestry = f"resume: {item['resumed_from']}"
        else:
            ancestry = "-"
        size = item.get("size_mb")
        table.add_row(
            str(index),
            name,
            f"{elo:+.0f}" if isinstance(elo, (int, float)) else "-",
            str(item.get("epoch") or "-"),
            str(item.get("dataset") or "-"),
            ancestry,
            f"{size:.1f}" if isinstance(size, (int, float)) else "-",
        )
    return table


def _select_checkpoint(checkpoints, prompt: str,
                       title: str = "AZ checkpoints") -> str:
    """Show ranked checkpoint details and accept either an index or exact name."""
    console = _console()
    console.print(_checkpoint_table(checkpoints, title))
    raw = Prompt.ask(f"[dim]{prompt} (#, name, or Enter)[/dim]", default="").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(checkpoints):
        return checkpoints[int(raw) - 1]
    return raw if raw in checkpoints else ""


def _selfplay_live(*, effective_workers: int, num_games: int, sims: int,
                   max_plies: int, run_fn):
    """Drive `run_fn(on_game, on_status, on_heartbeat)` behind one self-refreshing
    status line (no per-game or per-heartbeat spam). `run_fn` returns the
    (states, policies, outcomes) arrays. Returns ((states, policies, outcomes),
    stats); stats holds done/positions/plies_sum plus wins={1,2} and draws for the
    caller's summary. Propagates KeyboardInterrupt/EOFError (completed games are
    saved by the runner)."""
    console = _console()
    from rich.live import Live

    t0 = time.monotonic()
    stats = {
        "done": 0, "total": num_games, "positions": 0,
        "wins": {1: 0, 2: 0}, "draws": 0, "plies_sum": 0,
        "workers": effective_workers, "online": set(), "t0": None,
        "hb": {}, "play_t0": None,  # heartbeat plies + when play started
    }

    def _hcount(n: float) -> str:
        if n < 1000:
            return f"{int(n):,}"
        if n < 1_000_000:
            return f"{n / 1e3:.1f}K"
        if n < 1_000_000_000:
            return f"{n / 1e6:.2f}M"
        return f"{n / 1e9:.2f}B"

    def _hdur(secs: float) -> str:
        s = int(secs)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m{s % 60:02d}s"
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"

    def _render() -> Text:
        sep = ("  ·  ", STYLE_GRID)
        # Phase 1 — workers still spawning (no game has reported a heartbeat yet).
        if not stats["online"]:
            return Text.assemble(
                ("  starting workers  ", "cyan"),
                (f"{len(stats['online'])}/{stats['workers']}", "bold white"),
                sep, (_hdur(time.monotonic() - t0), "dim"),
            )
        # Phase 2 — playing, but no game has finished yet. With many games in
        # flight they all complete near the end, so show live ply/sim progress
        # from heartbeats instead of a frozen "starting workers" line.
        if stats["t0"] is None:
            plies = sum(stats["hb"].values())
            n = len(stats["hb"])
            avg = plies / n if n else 0.0
            elapsed = time.monotonic() - (stats["play_t0"] or t0)
            # Games mostly run to the ply cap early in training, so estimate when
            # the average game reaches it. First completions arrive a bit sooner.
            if avg > 0 and max_plies > avg:
                eta = f"~{_hdur(elapsed * (max_plies / avg - 1))} to first games"
            else:
                eta = "estimating…"
            plies_ps = plies / elapsed if elapsed > 0 else 0.0
            return Text.assemble(
                ("  playing  ", "cyan"),
                (f"{len(stats['online'])}/{stats['workers']} workers", "bold white"),
                sep, (f"~{_hcount(plies)} plies", "white"),
                sep, (f"{_hcount(plies_ps)} plies/s", "cyan"),
                sep, (f"{_hcount(plies_ps * sims)} sims/s", "cyan"),
                sep, (f"avg {avg:.0f}/{max_plies} ply", "white"),
                sep, (_hdur(elapsed), "dim"),
                sep, (eta, "green"),
            )
        # Phase 3 — games are completing; full stats with throughput and ETA.
        done, total = stats["done"], stats["total"]
        elapsed = time.monotonic() - stats["t0"]
        gps = done / elapsed if elapsed > 0 else 0.0
        pos = stats["positions"]
        remaining = total - done
        in_flight = max(0, min(stats["workers"], remaining))
        eta = _hdur(remaining / gps) if gps > 0 and remaining > 0 else "—"
        avg_ply = stats["plies_sum"] / done if done else 0.0
        wins = stats["wins"][1] + stats["wins"][2]
        # Rates over time-since-play-started. plies (live, incl. in-flight) vs pos
        # (committed to disk) are distinct: the gap is work in games not yet done.
        elapsed_play = time.monotonic() - (stats["play_t0"] or t0)
        total_plies = sum(stats["hb"].values())
        plies_ps = total_plies / elapsed_play if elapsed_play > 0 else 0.0
        pos_ps = pos / elapsed_play if elapsed_play > 0 else 0.0
        return Text.assemble(
            ("  ", ""),
            (f"{done}/{total} games", "bold white"),
            (f" {done / total * 100:.0f}%" if total else "", "dim"),
            sep, (f"{in_flight} in flight", "white"),
            sep, (f"{_hcount(pos)} pos", "white"),
            sep, (f"{gps:.1f} g/s", "cyan"),
            sep, (f"{_hcount(plies_ps)} plies/s", "cyan"),
            sep, (f"{_hcount(plies_ps * sims)} sims/s", "cyan"),
            sep, (f"{_hcount(pos_ps)} pos/s", "cyan"),
            sep, (f"W{wins} D{stats['draws']}", "dim"),
            sep, (f"avg {avg_ply:.0f} ply", "dim"),
            sep, (f"ETA {eta}", "green"),
        )

    class _StatusView:
        def __rich__(self) -> Text:
            return _render()

    def on_status(msg):
        console.print(f"[dim]{msg}[/dim]")

    def on_game(done, total, winner, ply, positions):
        if stats["t0"] is None:
            stats["t0"] = time.monotonic()  # first completion = end of warmup
        stats["done"] = done
        stats["positions"] = positions
        stats["plies_sum"] += ply
        if winner is None:
            stats["draws"] += 1
        else:
            stats["wins"][winner] = stats["wins"].get(winner, 0) + 1

    def on_heartbeat(wid, game_num, ply):
        if stats["play_t0"] is None:
            stats["play_t0"] = time.monotonic()
        stats["online"].add(wid)
        # Key by (worker, game) — each worker runs many concurrent games, so
        # keying by worker alone would only track one game's plies per worker.
        stats["hb"][(wid, game_num)] = ply

    with Live(_StatusView(), console=console, refresh_per_second=4,
              transient=False):
        states, policies, outcomes = run_fn(on_game, on_status, on_heartbeat)

    return (states, policies, outcomes), stats


def _run_selfplay_live(config, *, workers: int, num_games: int, sims: int,
                       save_dir: str):
    """One-shot self-play (spawns fresh) with the live status line.
    Returns (states, policies, outcomes, stats)."""
    # run_selfplay clamps workers to the game count; mirror that for the display.
    effective = min(workers, num_games) if num_games > 0 else workers

    def run_fn(on_game, on_status, on_heartbeat):
        if workers <= 1:
            from .az_selfplay import run_selfplay_single
            return run_selfplay_single(
                config, on_game=on_game, on_status=on_status, save_dir=save_dir)
        from .az_selfplay import run_selfplay
        return run_selfplay(
            config, on_game=on_game, on_status=on_status,
            on_heartbeat=on_heartbeat, save_dir=save_dir)

    (s, p, o), stats = _selfplay_live(
        effective_workers=effective, num_games=num_games, sims=sims,
        max_plies=config.max_plies, run_fn=run_fn)
    return s, p, o, stats


def _run_pool_round_live(pool, *, num_games: int, checkpoint: str, sims: int,
                         save_dir: str, base_game: int):
    """One round on a persistent SelfPlayPool with the live status line.
    Returns (states, policies, outcomes, stats)."""
    def run_fn(on_game, on_status, on_heartbeat):
        return pool.run(num_games, checkpoint, save_dir=save_dir,
                        on_game=on_game, on_heartbeat=on_heartbeat,
                        base_game=base_game)

    (s, p, o), stats = _selfplay_live(
        effective_workers=pool.num_workers, num_games=num_games, sims=sims,
        max_plies=pool.config.max_plies, run_fn=run_fn)
    stats["pipeline"] = pool.last_metrics
    return s, p, o, stats


def _benchmark_selfplay() -> None:
    """Benchmark this host without writing self-play data or checkpoints."""
    from .az_benchmark import (
        benchmark_configuration,
        cpu_configurations,
        gpu_configurations,
    )
    from .az_selfplay import hardware_tuning_key, save_tuning_profile
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    console = _console()
    console.print("\n[bold]AlphaZero self-play benchmark[/bold]")
    hw = _detect_and_show_hw()
    if hw["device"] == "cuda":
        device = Prompt.ask("[dim]Benchmark device[/dim]",
                            choices=["cpu", "cuda"], default="cuda")
    else:
        device = "cpu"

    default_workers = hw.get("cpu_workers", hw["workers"]) if device == "cpu" else hw["workers"]
    workers = _prompt_int("Workers", default_workers, 1, max(1, hw["ncpu"]))
    simulations = _prompt_int("MCTS simulations per move", 50, 5, 5000)
    max_plies = _prompt_int("Maximum plies per benchmark game", 12, 2, 1000)

    if device == "cuda":
        configs = gpu_configurations(hw["inference_batch"], workers)
        default_games = max(32, max(workers * c for _, c in configs))
    else:
        configs = cpu_configurations(workers)
        default_games = max(32, max(configs) * 2)
    games = _prompt_int("Games per configuration", default_games, 1, 100_000)

    results = []

    def run_with_progress(label: str, **kwargs):
        active_plies = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed:.0f}/{task.total:.0f} games"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task(label, total=games)

            def on_game(done, total, winner, ply, positions):
                progress.update(task, completed=done,
                                description=f"{label} · {positions:,} positions")

            def on_heartbeat(worker, game, ply):
                active_plies[(worker, game)] = ply
                progress.update(
                    task,
                    description=(f"{label} · ~{sum(active_plies.values()):,} "
                                 "active plies"),
                )

            return benchmark_configuration(
                **kwargs, on_game=on_game, on_heartbeat=on_heartbeat)

    try:
        if device == "cuda":
            for index, (batch, concurrency) in enumerate(configs, 1):
                label = f"[{index}/{len(configs)}] batch {batch}, concurrency {concurrency}"
                results.append(run_with_progress(
                    label,
                    device=device, games=games, simulations=simulations,
                    max_plies=max_plies, workers=workers,
                    batch_size=batch, concurrency=concurrency,
                ))
        else:
            for index, worker_count in enumerate(configs, 1):
                label = f"[{index}/{len(configs)}] {worker_count} workers"
                results.append(run_with_progress(
                    label,
                    device=device, games=games, simulations=simulations,
                    max_plies=max_plies, workers=worker_count,
                ))
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]benchmark interrupted[/dim]")
        return

    table = Table(title="Self-play benchmark", box=box.SIMPLE_HEAVY)
    if device == "cuda":
        table.add_column("Batch", justify="right")
        table.add_column("Concurrency", justify="right")
        table.add_column("Avg batch", justify="right")
        table.add_column("Request", justify="right")
        table.add_column("Eval/s", justify="right")
    else:
        table.add_column("Workers", justify="right")
    table.add_column("Games/s", justify="right")
    table.add_column("Positions/s", justify="right")

    for result in results:
        common = [f"{result['games_per_s']:.2f}", f"{result['positions_per_s']:.1f}"]
        if device == "cuda":
            inference = result["inference"]
            row = [
                str(result["batch_size"]), str(result["concurrency"]),
                f"{inference.get('avg_batch', 0):.1f}",
                f"{result['avg_request_wait_ms']:.1f} ms",
                f"{result['evals_per_s']:.0f}", *common,
            ]
        else:
            row = [str(result["workers"]), *common]
        table.add_row(*row)
    console.print(table)

    best = max(results, key=lambda result: result["positions_per_s"])
    if device == "cuda":
        recommendation = (
            f"batch {best['batch_size']}, concurrency {best['concurrency']}, "
            f"workers {best['workers']}"
        )
    else:
        recommendation = f"{best['workers']} workers"
    if device == "cuda":
        key = hardware_tuning_key(
            "cuda", hw["ncpu"], hw["gpu_name"], hw["vram_gb"], hw["gpu_count"])
        save_tuning_profile(key, {
            "workers": int(best["workers"]),
            "inference_batch": int(best["batch_size"]),
            "concurrency": int(best["concurrency"]),
        })
    else:
        key = hardware_tuning_key("cpu", hw["ncpu"])
        save_tuning_profile(key, {"workers": int(best["workers"])})
    console.print(f"[green]Recommended for this host:[/green] {recommendation}")
    console.print("[green]Saved as the default for matching hardware.[/green]")
    console.print("[dim]No training data or checkpoints were written.[/dim]")


# ---------------------------------------------------------------------------
# 1. Self-play
# ---------------------------------------------------------------------------

def _selfplay() -> None:
    console = _console()
    from .az_selfplay import SelfPlayConfig


    console.print("\n[bold]AlphaZero self-play[/bold]")
    hw = _detect_and_show_hw()
    num_games = _prompt_int("Number of games", hw["games_per_iter"], 1, 100_000)
    sims = _prompt_int("MCTS simulations per move", 200, 10, 5000)
    max_plies = _prompt_int("Max plies per game", 150, 20, 1000)
    sp_device, workers, batch_size, concurrency, inference_servers = \
        _prompt_selfplay_params(num_games, hw)
    search_params = _prompt_search_params()

    # Check for existing checkpoint
    ckpts = []
    if CHECKPOINT_ROOT.exists():
        ckpts = [f.stem for f in ranked_checkpoint_paths(CHECKPOINT_ROOT)
                 if _is_az_checkpoint(f.stem) and not f.stem.endswith("_candidate")]
    checkpoint = ""
    if ckpts:
        checkpoint = _select_checkpoint(
            ckpts, "Checkpoint; blank = random init", "Available AZ checkpoints")

    auto_name = _auto_dataset_name(
        prefix="az", games=num_games, simulations=sims, max_plies=max_plies,
        device=sp_device, workers=workers, batch_size=batch_size,
        concurrency=concurrency, search_params=search_params,
    )
    run_name = Prompt.ask("[dim]Dataset name[/dim]", default=auto_name).strip() or auto_name

    config = SelfPlayConfig(
        num_games=num_games,
        simulations=sims,
        max_plies=max_plies,
        workers=workers,
        batch_size=batch_size,
        concurrent_games=concurrency,
        inference_servers=inference_servers,
        checkpoint=checkpoint,
        device=sp_device,
        **search_params,
    )

    from .az_selfplay import AZ_DATA_ROOT, save_run_config
    save_dir = str(AZ_DATA_ROOT / run_name)
    save_run_config(run_name, config, mode="standalone")
    t0 = time.monotonic()

    try:
        states, policies, outcomes, stats = _run_selfplay_live(
            config, workers=workers, num_games=num_games, sims=sims,
            save_dir=save_dir)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]interrupted — completed games were saved.[/dim]")
        return

    elapsed = time.monotonic() - t0
    wins = stats["wins"][1] + stats["wins"][2]
    console.print(Panel(
        Text.assemble(
            ("games ", "dim"), (f"{stats['done']}", "white"),
            ("   positions ", "dim"), (f"{stats['positions']:,}", "white"),
            ("   wins ", "dim"), (f"{wins}", "white"),
            ("   draws ", "dim"), (f"{stats['draws']}", "white"),
            ("   ", "dim"), (f"{elapsed:.0f}s", "white"),
            ("   saved to ", "dim"), (run_name, STYLE_HINT),
        ),
        title="[bold]Self-play complete[/bold]", border_style=STYLE_GRID,
    ))


def _is_az_checkpoint(name: str) -> bool:
    meta = _read_meta(name)
    return meta.get("arch") == "az"


def _batch_progress_text(info) -> Text:
    percent = 100.0 * info.batch / max(info.batches, 1)
    rate = info.batch / max(info.elapsed, 1e-9)
    remaining = (info.batches - info.batch) / rate if rate > 0 else 0
    return Text.assemble(
        (f"  epoch {info.epoch}/{info.epochs}  ", "bold"),
        (f"{info.phase} {percent:5.1f}%", "cyan"),
        (f"  {info.batch:,}/{info.batches:,} batches", "white"),
        (f"  loss {info.loss:.4f}", "white"),
        (f"  {rate:.1f} batch/s", "dim"),
        (f"  ETA {remaining:.0f}s", "green"),
    )


# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

def _train() -> None:
    console = _console()
    from .az_train import (AZ_DATA_ROOT, AZTrainConfig, dataset_provenance,
                           load_training_datasets, train_az)

    if not AZ_DATA_ROOT.exists():
        console.print("[yellow]no AZ training data — run self-play first.[/yellow]")
        return

    runs = sorted(d.name for d in AZ_DATA_ROOT.iterdir() if d.is_dir())
    if not runs:
        console.print("[yellow]no AZ training data — run self-play first.[/yellow]")
        return

    for i, r in enumerate(runs, 1):
        d = AZ_DATA_ROOT / r
        files = [f for f in d.glob("*.npz") if not f.name.startswith(".")]
        console.print(f"  [dim]{i}.[/dim] {r} [dim]({len(files)} data file(s))[/dim]")

    # Select one, several, or all runs — train on the combined data.
    raw = Prompt.ask("[dim]Runs to train on (#, names, comma-separated, or 'all')[/dim]",
                     default="all").strip()
    if raw.lower() == "all":
        selected = list(runs)
    else:
        selected = []
        for tok in raw.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(runs):
                selected.append(runs[int(tok) - 1])
            elif tok in runs:
                selected.append(tok)
            else:
                console.print(f"[yellow]skipping unknown run '{tok}'[/yellow]")
        seen = set()
        selected = [r for r in selected if not (r in seen or seen.add(r))]
    if not selected:
        console.print("[red]no valid runs selected[/red]")
        return

    console.print(f"[dim]loading {len(selected)} selected run(s)...[/dim]")
    try:
        from rich.live import Live
        with Live(Text("  preparing replay shards…", style="dim"), console=console,
                  refresh_per_second=4, transient=True) as live:
            def on_load(phase, done, total, run, shard, positions):
                if phase == "combining":
                    message = f"  combining {positions:,} positions in memory…"
                else:
                    pct = 100 * done / max(total, 1)
                    message = (f"  loading shards {done:,}/{total:,} ({pct:.0f}%) · "
                               f"{positions:,} positions" + (f" · {run}" if run else ""))
                live.update(Text(message, style="cyan"))

            states, policies, outcomes = load_training_datasets(
                selected, on_progress=on_load)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print(f"[dim]{len(states):,} positions from {len(selected)} run(s)[/dim]")

    epochs = _prompt_int("Epochs", 10, 1, 1000)
    batch_size = _prompt_int("Batch size", 256, 8, 65536)
    lr = _prompt_float("Learning rate", 2e-3, 1e-6, 1.0)
    ckpt_name = Prompt.ask("[dim]Checkpoint name[/dim]", default="az_latest").strip()

    # Resume from existing?
    resume = ""

    if resolve_checkpoint_path(CHECKPOINT_ROOT, ckpt_name).exists():
        if Confirm.ask(f"[dim]resume from existing '{ckpt_name}'?[/dim]", default=True):
            resume = ckpt_name

    config = AZTrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        checkpoint_name=ckpt_name,
    )

    from .az_train import resolve_device
    console.print(f"[dim]device: {resolve_device(config.device)}[/dim]")

    def on_epoch(e):
        star = " [green]*best*[/green]" if e.is_best else ""
        console.print(
            f"  [dim]epoch[/dim] {e.epoch:>3}/{e.epochs}  "
            f"[dim]loss[/dim] {e.train_loss:.4f}  "
            f"[dim]π[/dim] {e.policy_loss:.4f}  "
            f"[dim]v[/dim] {e.value_loss:.4f}  "
            f"[dim]val[/dim] {e.val_loss:.4f}  "
            f"[dim]lr[/dim] {e.lr:.2e}  "
            f"[dim]{e.elapsed:.0f}s[/dim]{star}"
        )

    try:
        from rich.live import Live
        with Live(Text("  preparing training split…", style="dim"), console=console,
                  refresh_per_second=4, transient=True) as live:
            res = train_az(
                states, policies, outcomes, config,
                resume_from=resume, on_epoch=on_epoch,
                on_batch=lambda info: live.update(_batch_progress_text(info)),
                data_meta=dataset_provenance(selected))
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]training interrupted — best checkpoint saved.[/dim]")
        return

    console.print(Panel(
        Text.assemble(
            ("checkpoint ", "dim"), (res["checkpoint"], STYLE_HINT),
            ("   best val ", "dim"), (f"{res['best_val_loss']:.4f}", "white"),
            (" @ epoch ", "dim"), (f"{res['best_epoch']}", "white"),
            ("   positions ", "dim"), (f"{res['positions']:,}", "white"),
            ("   ", "dim"), (f"{res['elapsed']:.0f}s on {res['device']}", "white"),
        ),
        title="[bold]Training complete[/bold]", border_style=STYLE_GRID,
    ))


# ---------------------------------------------------------------------------
# 3. Full loop: self-play → train → repeat
# ---------------------------------------------------------------------------

def _seed_loop_checkpoint(src: str, dst: str) -> bool:
    """Copy an existing checkpoint (weights + meta) to the loop's best-checkpoint
    name so the loop bootstraps from it, and stamp seeded_from=src into the copy's
    meta so the weight lineage survives (train_az propagates it each iteration).
    Returns True on success."""
    import json
    import shutil
    src_w = resolve_checkpoint_path(CHECKPOINT_ROOT, src)
    if not src_w.exists():
        return False
    shutil.copy2(src_w, CHECKPOINT_ROOT / f"{dst}.safetensors")
    dst_m = CHECKPOINT_ROOT / f"{dst}.meta.json"
    src_m = src_w.with_suffix(".meta.json")
    meta = {}
    if src_m.exists():
        try:
            meta = json.loads(src_m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    meta["seeded_from"] = src  # root of this weight lineage
    dst_m.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True


def _full_loop() -> None:
    console = _console()
    from .az_selfplay import SelfPlayConfig, SelfPlayPool, save_run_config
    from .az_train import (AZ_DATA_ROOT, AZTrainConfig, dataset_provenance,
                           load_training_data, train_az)
    from .az_arena import promote_candidate, run_arena

    console.print("\n[bold]AlphaZero training loop[/bold]")
    console.print("[dim]Alternates: self-play → train → self-play → train → ...[/dim]")

    hw = _detect_and_show_hw()
    iterations = _prompt_int("Number of iterations", 5, 1, 1000)
    games_per_iter = _prompt_int("Games per iteration", hw["games_per_iter"], 1, 100_000)
    sims = _prompt_int("MCTS simulations per move", 200, 10, 5000)
    max_plies = _prompt_int("Max plies per game", 150, 20, 1000)
    epochs_per_iter = _prompt_int("Training epochs per iteration", 10, 1, 1000)
    train_batch_size = _prompt_int("Training batch size", hw["train_batch"], 8, 65536)
    lr = _prompt_float("Learning rate", 2e-3, 1e-6, 1.0)
    device, workers, sp_batch_size, concurrency, inference_servers = \
        _prompt_selfplay_params(games_per_iter, hw)
    search_params = _prompt_search_params()
    max_positions = _prompt_int(
        "Replay buffer: keep last N positions (0=all; e.g. 300000 ≈ recent games)",
        0, 0, 1_000_000_000)
    arena_games = _prompt_int("Arena games before promotion", 20, 2, 1000)
    promotion_score = _prompt_float("Candidate promotion score", 0.55, 0.5, 1.0)

    auto_name = _auto_dataset_name(
        prefix="azloop", games=games_per_iter, simulations=sims, max_plies=max_plies,
        device=device, workers=workers, batch_size=sp_batch_size,
        concurrency=concurrency, search_params=search_params,
    )
    run_name = Prompt.ask("[dim]Dataset/run name[/dim]", default=auto_name).strip() or auto_name
    ckpt_name = f"{run_name}_best"
    candidate_name = f"{run_name}_candidate"
    ckpt_path = resolve_checkpoint_path(CHECKPOINT_ROOT, ckpt_name)

    # Optionally seed the loop from an existing checkpoint (e.g. az_latest). The
    # loop otherwise bootstraps from {run_name}_best if it exists, else random init.
    seed_choices = [f.stem for f in ranked_checkpoint_paths(CHECKPOINT_ROOT)
                    if (_is_az_checkpoint(f.stem)
                        and not f.stem.endswith("_candidate")
                        and f.stem not in (ckpt_name, candidate_name))]
    if seed_choices:
        default_hint = f"continue {ckpt_name}" if ckpt_path.exists() else "random init"
        seed_from = _select_checkpoint(
            seed_choices, f"Seed checkpoint; blank = {default_hint}")
        if seed_from:
            overwrite = (not ckpt_path.exists()) or Confirm.ask(
                f"[yellow]{ckpt_name} exists — overwrite it with {seed_from}?[/yellow]",
                default=True)
            if overwrite and _seed_loop_checkpoint(seed_from, ckpt_name):
                console.print(f"[dim]seeded {ckpt_name} from {seed_from}[/dim]")
            else:
                console.print(f"[dim]keeping existing {ckpt_name}[/dim]")

    sp_extra = f" · inference batch {sp_batch_size}" if device != "cpu" else ""
    console.print(f"\n[dim]device: {device} · workers: {min(workers, games_per_iter)}"
                  f"{sp_extra} · persistent pool[/dim]")

    # One persistent pool for the whole loop — spawn/CUDA-init cost is paid once,
    # not per iteration. The model checkpoint is reloaded at the start of each
    # round; other self-play params are fixed for the pool's lifetime.
    sp_config = SelfPlayConfig(
        num_games=games_per_iter, simulations=sims, max_plies=max_plies,
        workers=workers, batch_size=sp_batch_size,
        concurrent_games=concurrency, inference_servers=inference_servers,
        device=device,
        **search_params,
    )
    save_run_config(
        run_name, sp_config, mode="loop", iterations=iterations,
        epochs_per_iteration=epochs_per_iter, training_batch=train_batch_size,
        learning_rate=lr, replay_positions=max_positions,
        arena_games=arena_games, promotion_score=promotion_score,
    )
    pool = SelfPlayPool(sp_config)

    loop_t0 = time.monotonic()
    cumulative_games = 0
    cumulative_positions = 0
    cumulative_wins = {1: 0, 2: 0}
    cumulative_draws = 0

    try:
        for it in range(1, iterations + 1):
            console.print(f"\n[bold]═══ Iteration {it}/{iterations} ═══[/bold]")

            # --- Self-play ---
            checkpoint = ckpt_name if resolve_checkpoint_path(
                CHECKPOINT_ROOT, ckpt_name).exists() else ""
            using = checkpoint or "random init"
            console.print(f"\n  [bold]Self-play[/bold] [dim]· {games_per_iter} games "
                          f"· {sims} sims · {max_plies} max plies · net: {using}[/dim]")

            t0 = time.monotonic()
            # Self-play streams data to shards on disk; the round returns empty
            # arrays (everything was flushed), so rely on stats for counts and
            # load from disk (below) for training.
            _, _, _, stats = _run_pool_round_live(
                pool, num_games=games_per_iter, checkpoint=checkpoint, sims=sims,
                save_dir=str(AZ_DATA_ROOT / run_name),
                base_game=(it - 1) * games_per_iter)

            sp_time = time.monotonic() - t0
            positions = stats["positions"]
            decisive = stats["wins"][1] + stats["wins"][2]
            avg_ply = stats["plies_sum"] / stats["done"] if stats["done"] else 0
            console.print(
                f"  [dim]→ {positions:,} positions in {sp_time:.0f}s  "
                f"({decisive} decisive, {stats['draws']} draws, avg {avg_ply:.0f} ply)[/dim]"
            )
            pipeline = stats.get("pipeline", {})
            inference = pipeline.get("inference", {})
            if inference:
                batches = inference.get("batches", 0)
                full = inference.get("full_batches", 0)
                fill_pct = 100.0 * full / batches if batches else 0.0
                elapsed = max(pipeline.get("elapsed_s", 0), 1e-9)
                n_srv = inference.get("num_servers", 1)
                # inference_s is summed over parallel servers; divide by their count
                # for a 0-100% per-server GPU-busy figure.
                gpu_pct = 100 * inference.get("inference_s", 0) / (elapsed * n_srv)
                srv_tag = f" · {n_srv} servers" if n_srv > 1 else ""
                console.print(
                    f"  [dim]  inference: avg batch {inference.get('avg_batch', 0):.1f} "
                    f"({fill_pct:.0f}% full) · request wait "
                    f"{pipeline.get('avg_request_wait_ms', 0):.1f} ms · "
                    f"GPU inference {gpu_pct:.0f}%{srv_tag}[/dim]"
                )

            cumulative_games += games_per_iter
            cumulative_positions += positions
            cumulative_wins[1] += stats["wins"][1]
            cumulative_wins[2] += stats["wins"][2]
            cumulative_draws += stats["draws"]

            if positions == 0:
                console.print("[yellow]no data generated, stopping.[/yellow]")
                break

            # --- Train ---
            all_s, all_p, all_o = load_training_data(
                run_name, max_positions=max_positions)
            console.print(
                f"\n  [bold]Training[/bold] [dim]· {epochs_per_iter} epochs "
                f"· batch {train_batch_size} · replay buffer: {len(all_s):,} positions[/dim]"
            )

            train_config = AZTrainConfig(
                epochs=epochs_per_iter, batch_size=train_batch_size, lr=lr,
                checkpoint_name=candidate_name,
            )
            resume = ckpt_name if resolve_checkpoint_path(
                CHECKPOINT_ROOT, ckpt_name).exists() else ""

            def on_epoch(e):
                star = " [green]*best*[/green]" if e.is_best else ""
                console.print(
                    f"    [dim]epoch {e.epoch:>3}/{e.epochs}  "
                    f"loss {e.train_loss:.4f}  "
                    f"π {e.policy_loss:.4f}  v {e.value_loss:.4f}  "
                    f"val {e.val_loss:.4f}  "
                    f"lr {e.lr:.2e}  "
                    f"{e.elapsed:.0f}s[/dim]{star}"
                )

            from rich.live import Live
            with Live(Text("    preparing training split…", style="dim"), console=console,
                      refresh_per_second=4, transient=True) as live:
                res = train_az(
                    all_s, all_p, all_o, train_config,
                    resume_from=resume, on_epoch=on_epoch,
                    on_batch=lambda info: live.update(_batch_progress_text(info)),
                    data_meta=dataset_provenance(run_name, max_positions=max_positions))
            console.print(
                f"  [dim]→ best val {res['best_val_loss']:.4f} @ epoch {res['best_epoch']}  "
                f"({res['elapsed']:.0f}s on {res['device']})[/dim]"
            )

            # --- Arena gate ---
            if not ckpt_path.exists():
                arena = {"games": 0, "score": 1.0, "bootstrap": True}
                promote_candidate(candidate_name, ckpt_name, arena)
                console.print("  [green]→ bootstrap candidate promoted[/green]")
            else:
                console.print(
                    f"\n  [bold]Arena[/bold] [dim]· {arena_games} games · candidate must score "
                    f"{promotion_score:.0%}[/dim]"
                )

                def on_arena_game(done, total, result):
                    symbol = "W" if result == 1.0 else "D" if result == 0.5 else "L"
                    console.print(f"    [dim]{done:>3}/{total} {symbol}[/dim]")

                arena = run_arena(
                    ckpt_name, candidate_name, games=arena_games,
                    max_plies=max_plies, device="cpu", on_game=on_arena_game,
                )
                if arena["score"] >= promotion_score:
                    promote_candidate(candidate_name, ckpt_name, arena)
                    decision = "[green]promoted[/green]"
                else:
                    decision = "[yellow]rejected; incumbent retained[/yellow]"
                console.print(
                    f"  [dim]→ candidate W{arena['wins']} D{arena['draws']} "
                    f"L{arena['losses']} · score {arena['score']:.1%} · [/dim]{decision}"
                )

            # Iteration summary
            loop_elapsed = time.monotonic() - loop_t0
            eta = loop_elapsed / it * (iterations - it) if it < iterations else 0
            console.print(
                f"\n  [dim]cumulative: {cumulative_games} games, "
                f"{cumulative_positions:,} positions, "
                f"{cumulative_wins[1]+cumulative_wins[2]} wins, "
                f"{cumulative_draws} draws  ·  "
                f"elapsed {loop_elapsed:.0f}s  ·  "
                f"ETA {eta:.0f}s[/dim]"
            )

    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]loop interrupted — last best checkpoint is saved.[/dim]")
        return
    finally:
        pool.close()

    loop_elapsed = time.monotonic() - loop_t0
    console.print(Panel(
        Text.assemble(
            ("checkpoint ", "dim"), (ckpt_name, STYLE_HINT),
            ("   iterations ", "dim"), (f"{iterations}", "white"),
            ("   games ", "dim"), (f"{cumulative_games}", "white"),
            ("   positions ", "dim"), (f"{cumulative_positions:,}", "white"),
            ("   ", "dim"), (f"{loop_elapsed:.0f}s total", "white"),
        ),
        title="[bold]Training loop complete[/bold]", border_style=STYLE_GRID,
    ))


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def az_menu() -> None:
    console = _console()
    while True:
        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("1", "Self-play (generate training data)")
        table.add_row("2", "Train network")
        table.add_row("3", "Full loop (self-play → train → repeat)")
        table.add_row("4", "Benchmark and tune self-play")
        table.add_row("5", "Back")
        console.print(Panel(table, title="[bold]AlphaZero pipeline[/bold]",
                            border_style=STYLE_GRID))
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4", "5"], default="3")
        if choice == "1":
            _selfplay()
        elif choice == "2":
            _train()
        elif choice == "3":
            _full_loop()
        elif choice == "4":
            _benchmark_selfplay()
        else:
            return
