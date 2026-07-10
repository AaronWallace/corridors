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
# 1. Self-play
# ---------------------------------------------------------------------------

def _selfplay() -> None:
    console = _console()
    from .az_selfplay import SelfPlayConfig, auto_workers, resolve_device


    console.print("\n[bold]AlphaZero self-play[/bold]")
    num_games = _prompt_int("Number of games", 50, 1, 100_000)
    sims = _prompt_int("MCTS simulations per move", 200, 10, 5000)
    max_plies = _prompt_int("Max plies per game", 150, 20, 1000)
    ncpu = os.cpu_count() or 4
    default_workers = max(2, ncpu - 2)
    workers = _prompt_int(f"Workers [1-{ncpu}]", default_workers, 1, ncpu)

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
        checkpoint=checkpoint,
        device="auto",
    )

    t0 = time.monotonic()
    sp_wins = [0]
    sp_draws = [0]
    sp_plies = []

    def on_game(done, total, winner, ply, positions):
        sp_plies.append(ply)
        if winner is None:
            sp_draws[0] += 1
        else:
            sp_wins[0] += 1
        tag = f"P{winner}" if winner else "draw"
        elapsed = time.monotonic() - t0
        gps = done / elapsed if elapsed > 0 else 0
        avg_ply = sum(sp_plies) / len(sp_plies)
        console.print(
            f"  [dim]game[/dim] {done:>5}/{total}  "
            f"{tag:<5} "
            f"[dim]ply[/dim] {ply:>3}  "
            f"[dim]W[/dim] {sp_wins[0]} [dim]D[/dim] {sp_draws[0]}  "
            f"[dim]avg ply[/dim] {avg_ply:.0f}  "
            f"[dim]pos[/dim] {positions:>8,}  "
            f"[dim]{gps:.1f} g/s[/dim]"
        )

    def on_status(msg):
        console.print(f"[dim]{msg}[/dim]")

    try:
        if workers <= 1:
            from .az_selfplay import run_selfplay_single
            states, policies, outcomes = run_selfplay_single(
                config, on_game=on_game, on_status=on_status)
        else:
            from .az_selfplay import run_selfplay
            states, policies, outcomes = run_selfplay(
                config, on_game=on_game, on_status=on_status)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]interrupted.[/dim]")
        return

    if len(states) == 0:
        console.print("[yellow]no positions recorded.[/yellow]")
        return

    # Save
    from .az_train import save_training_data, AZ_DATA_ROOT
    iteration = len(list((AZ_DATA_ROOT / run_name).glob("iter_*.npz"))) if (
        AZ_DATA_ROOT / run_name).exists() else 0
    save_training_data(run_name, states, policies, outcomes, iteration)

    elapsed = time.monotonic() - t0
    console.print(Panel(
        Text.assemble(
            ("positions ", "dim"), (f"{len(states):,}", "white"),
            ("   games ", "dim"), (f"{num_games}", "white"),
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
        files = list(d.glob("iter_*.npz"))
        console.print(f"  [dim]{i}.[/dim] {r} [dim]({len(files)} iteration(s))[/dim]")

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
    from .az_selfplay import SelfPlayConfig, auto_workers, resolve_device
    from .az_train import AZ_DATA_ROOT, AZTrainConfig, load_training_data, train_az, save_training_data

    console.print("\n[bold]AlphaZero training loop[/bold]")
    console.print("[dim]Alternates: self-play → train → self-play → train → ...[/dim]\n")

    iterations = _prompt_int("Number of iterations", 5, 1, 1000)
    games_per_iter = _prompt_int("Games per iteration", 50, 1, 100_000)
    sims = _prompt_int("MCTS simulations per move", 200, 10, 5000)
    max_plies = _prompt_int("Max plies per game", 150, 20, 1000)
    epochs_per_iter = _prompt_int("Training epochs per iteration", 10, 1, 1000)
    batch_size = _prompt_int("Batch size", 256, 8, 65536)
    lr = _prompt_float("Learning rate", 2e-3, 1e-6, 1.0)
    ncpu = os.cpu_count() or 4
    workers = _prompt_int(f"Workers [1-{ncpu}]", max(2, ncpu - 2), 1, ncpu)
    max_data_iters = _prompt_int("Replay buffer (keep last N iterations, 0=all)", 0, 0, 1000)

    run_name = Prompt.ask("[dim]Run name[/dim]", default="az_loop").strip() or "az_loop"
    ckpt_name = f"{run_name}_best"

    device = resolve_device("auto")
    console.print(f"\n[dim]device: {device}, workers: {workers}[/dim]")

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

            sp_config = SelfPlayConfig(
                num_games=games_per_iter, simulations=sims, max_plies=max_plies,
                workers=workers, checkpoint=checkpoint, device="auto",
            )

            t0 = time.monotonic()
            sp_wins = {1: 0, 2: 0}
            sp_draws = [0]
            sp_plies = []

            def on_game(done, total, winner, ply, positions, _t0=t0,
                        _wins=sp_wins, _draws=sp_draws, _plies=sp_plies):
                _plies.append(ply)
                if winner is None:
                    _draws[0] += 1
                else:
                    _wins[winner] = _wins.get(winner, 0) + 1
                if done % max(1, total // 10) == 0 or done == total:
                    elapsed = time.monotonic() - _t0
                    gps = done / elapsed if elapsed > 0 else 0
                    avg_ply = sum(_plies) / len(_plies) if _plies else 0
                    w = _wins.get(1, 0) + _wins.get(2, 0)
                    console.print(
                        f"    [dim]{done:>5}/{total}  "
                        f"W {w} D {_draws[0]}  "
                        f"avg ply {avg_ply:.0f}  "
                        f"{positions:>8,} pos  "
                        f"{gps:.1f} g/s[/dim]"
                    )

            if workers <= 1:
                from .az_selfplay import run_selfplay_single
                states, policies, outcomes = run_selfplay_single(sp_config, on_game=on_game)
            else:
                from .az_selfplay import run_selfplay
                states, policies, outcomes = run_selfplay(sp_config, on_game=on_game)

            sp_time = time.monotonic() - t0
            decisive = sp_wins.get(1, 0) + sp_wins.get(2, 0)
            avg_ply = sum(sp_plies) / len(sp_plies) if sp_plies else 0
            console.print(
                f"  [dim]→ {len(states):,} positions in {sp_time:.0f}s  "
                f"({decisive} decisive, {sp_draws[0]} draws, avg {avg_ply:.0f} ply)[/dim]"
            )

            cumulative_games += games_per_iter
            cumulative_positions += len(states)
            cumulative_wins[1] += sp_wins.get(1, 0)
            cumulative_wins[2] += sp_wins.get(2, 0)
            cumulative_draws += sp_draws[0]

            if len(states) == 0:
                console.print("[yellow]no data generated, stopping.[/yellow]")
                break

            # Save this iteration's data
            existing = len(list((AZ_DATA_ROOT / run_name).glob("iter_*.npz"))) if (
                AZ_DATA_ROOT / run_name).exists() else 0
            save_training_data(run_name, states, policies, outcomes, existing)

            # --- Train ---
            all_s, all_p, all_o = load_training_data(
                run_name, max_iterations=max_data_iters if max_data_iters > 0 else 0)
            console.print(
                f"\n  [bold]Training[/bold] [dim]· {epochs_per_iter} epochs "
                f"· batch {batch_size} · replay buffer: {len(all_s):,} positions[/dim]"
            )

            train_config = AZTrainConfig(
                epochs=epochs_per_iter, batch_size=batch_size, lr=lr,
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
