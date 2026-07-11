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

STYLE_GRID = "grey35"
STYLE_HINT = "bold green"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_ROOT = _PROJECT_ROOT / "nn_checkpoints"


def _read_meta(name: str) -> dict:
    p = CHECKPOINT_ROOT / (name + ".meta.json") if not name.endswith(".meta.json") else CHECKPOINT_ROOT / name
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
    if hw["device"] == "cuda":
        console.print(f"\n[dim]detected:[/dim] [white]{hw['gpu_name']}[/white] "
                      f"[dim]({hw['vram_gb']:.0f} GB VRAM) · {hw['ncpu']} CPUs[/dim]")
        console.print(f"[dim]tuned defaults: workers {hw['workers']} · inference batch "
                      f"{hw['inference_batch']} · games/iter {hw['games_per_iter']} · "
                      f"train batch {hw['train_batch']}[/dim]")
    else:
        console.print(f"\n[dim]detected: CPU only ({hw['ncpu']} cores), no CUDA GPU[/dim]")
        console.print(f"[dim]tuned defaults: workers {hw['workers']} · "
                      f"games/iter {hw['games_per_iter']} · train batch {hw['train_batch']}[/dim]")
    return hw


def _prompt_selfplay_params(num_games: int, hw: dict):
    """Prompt for self-play parallelism using hardware-tuned defaults from `hw`.
    GPU mode additionally prompts for inference batch size and per-worker
    concurrency (the levers that feed the GPU). Returns
    (device, workers, batch_size, concurrency)."""
    device = hw["device"]
    ncpu = hw["ncpu"]
    hi = min(ncpu, num_games) if num_games > 0 else ncpu
    default_workers = min(hw["workers"], hi)
    workers = _prompt_int(f"Workers [1-{hi}]", default_workers, 1, hi)

    batch_size = 64
    concurrency = 0
    if device != "cpu":
        batch_size = _prompt_int("Inference batch size", hw["inference_batch"], 1, 8192)
        concurrency = _prompt_int("Concurrent games per worker (0=auto)", 0, 0, 256)
    return device, workers, batch_size, concurrency


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
            return Text.assemble(
                ("  playing  ", "cyan"),
                (f"{len(stats['online'])}/{stats['workers']} workers", "bold white"),
                sep, (f"~{_hcount(plies)} plies", "white"),
                sep, (f"~{_hcount(plies * sims)} sims", "white"),
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
        return Text.assemble(
            ("  ", ""),
            (f"{done}/{total} games", "bold white"),
            (f" {done / total * 100:.0f}%" if total else "", "dim"),
            sep, (f"{in_flight} in flight", "white"),
            sep, (f"{_hcount(pos)} pos", "white"),
            sep, (f"{_hcount(pos * sims)} sims", "white"),
            sep, (f"{gps:.1f} g/s", "cyan"),
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
    return s, p, o, stats


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
    _device, workers, batch_size, concurrency = _prompt_selfplay_params(num_games, hw)

    # Check for existing checkpoint
    ckpts = []
    if CHECKPOINT_ROOT.exists():
        ckpts = [f.stem for f in sorted(CHECKPOINT_ROOT.glob("*.safetensors"))
                 if _is_az_checkpoint(f.stem)]
    checkpoint = ""
    if ckpts:
        console.print(f"[dim]Available AZ checkpoints: {', '.join(ckpts)}[/dim]")
        raw = Prompt.ask("[dim]Checkpoint (name or Enter for random init)[/dim]",
                         default="").strip()
        if raw in ckpts:
            checkpoint = raw
        elif raw:
            console.print(f"[yellow]'{raw}' not found — using random init[/yellow]")

    run_name = Prompt.ask("[dim]Run name[/dim]", default="az_run").strip() or "az_run"

    config = SelfPlayConfig(
        num_games=num_games,
        simulations=sims,
        max_plies=max_plies,
        workers=workers,
        batch_size=batch_size,
        concurrent_games=concurrency,
        checkpoint=checkpoint,
        device="auto",
    )

    from .az_train import AZ_DATA_ROOT
    save_dir = str(AZ_DATA_ROOT / run_name)
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


# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

def _train() -> None:
    console = _console()
    from .az_train import AZ_DATA_ROOT, AZTrainConfig, load_training_data, train_az

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

    raw = Prompt.ask("[dim]Run name or #[/dim]", default=runs[-1]).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(runs):
        run_name = runs[int(raw) - 1]
    elif raw in runs:
        run_name = raw
    else:
        console.print("[red]unknown run[/red]")
        return

    console.print(f"[dim]loading data from '{run_name}'...[/dim]")
    try:
        states, policies, outcomes = load_training_data(run_name)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print(f"[dim]{len(states):,} positions loaded[/dim]")

    epochs = _prompt_int("Epochs", 10, 1, 1000)
    batch_size = _prompt_int("Batch size", 256, 8, 65536)
    lr = _prompt_float("Learning rate", 2e-3, 1e-6, 1.0)
    ckpt_name = Prompt.ask("[dim]Checkpoint name[/dim]", default="az_latest").strip()

    # Resume from existing?
    resume = ""

    if (CHECKPOINT_ROOT / f"{ckpt_name}.safetensors").exists():
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
        res = train_az(states, policies, outcomes, config,
                       resume_from=resume, on_epoch=on_epoch)
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

def _full_loop() -> None:
    console = _console()
    from .az_selfplay import SelfPlayConfig, SelfPlayPool
    from .az_train import AZ_DATA_ROOT, AZTrainConfig, load_training_data, train_az

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
    device, workers, sp_batch_size, concurrency = _prompt_selfplay_params(games_per_iter, hw)
    max_data_iters = _prompt_int("Replay buffer (keep last N iterations, 0=all)", 0, 0, 1000)

    run_name = Prompt.ask("[dim]Run name[/dim]", default="az_loop").strip() or "az_loop"
    ckpt_name = f"{run_name}_best"

    sp_extra = f" · inference batch {sp_batch_size}" if device != "cpu" else ""
    console.print(f"\n[dim]device: {device} · workers: {min(workers, games_per_iter)}"
                  f"{sp_extra} · persistent pool[/dim]")

    # One persistent pool for the whole loop — spawn/CUDA-init cost is paid once,
    # not per iteration. The model checkpoint is reloaded at the start of each
    # round; other self-play params are fixed for the pool's lifetime.
    sp_config = SelfPlayConfig(
        num_games=games_per_iter, simulations=sims, max_plies=max_plies,
        workers=workers, batch_size=sp_batch_size,
        concurrent_games=concurrency, device="auto",
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
            checkpoint = ckpt_name if (
                CHECKPOINT_ROOT / f"{ckpt_name}.safetensors").exists() else ""
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
                run_name, max_iterations=max_data_iters if max_data_iters > 0 else 0)
            console.print(
                f"\n  [bold]Training[/bold] [dim]· {epochs_per_iter} epochs "
                f"· batch {train_batch_size} · replay buffer: {len(all_s):,} positions[/dim]"
            )

            train_config = AZTrainConfig(
                epochs=epochs_per_iter, batch_size=train_batch_size, lr=lr,
                checkpoint_name=ckpt_name,
            )
            resume = ckpt_name if (
                CHECKPOINT_ROOT / f"{ckpt_name}.safetensors").exists() else ""

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

            res = train_az(all_s, all_p, all_o, train_config,
                           resume_from=resume, on_epoch=on_epoch)
            console.print(
                f"  [dim]→ best val {res['best_val_loss']:.4f} @ epoch {res['best_epoch']}  "
                f"({res['elapsed']:.0f}s on {res['device']})[/dim]"
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
        table.add_row("4", "Back")
        console.print(Panel(table, title="[bold]AlphaZero pipeline[/bold]",
                            border_style=STYLE_GRID))
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="3")
        if choice == "1":
            _selfplay()
        elif choice == "2":
            _train()
        elif choice == "3":
            _full_loop()
        else:
            return
