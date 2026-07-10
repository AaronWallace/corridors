"""Neural-network training menu.

Options:
  1. Generate self-play data  (runs the autoplay wizard, records a dataset)
  2. Train a model            (choose dataset + hyperparameters)
  3. Round-robin tournament   (all checkpoints + classical anchor, Elo)
  4. List / manage datasets
  5. List / manage checkpoints
  6. Back
"""

from __future__ import annotations

import os
from typing import List, Optional

from rich import box
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .. import settings
from . import datasets as ds_mod

STYLE_GRID = "grey35"
STYLE_HINT = "bold green"


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
# 1. Generate self-play data
# ---------------------------------------------------------------------------

def _generate_data() -> None:
    from ..play import _setup, autoplay_headless, autoplay_parallel
    console = _console()
    cfg = settings.load()
    params = _setup(cfg)
    default_name = ds_mod.default_dataset_name(
        params.num_games, params.depth, params.tiebreak_epsilon
    )
    name = Prompt.ask("[dim]Dataset name[/dim]", default=default_name).strip()
    if not name:
        name = default_name
    existing = ds_mod.read_manifest(name)
    if existing is not None:
        console.print(
            f"[yellow]dataset '{name}' exists "
            f"({existing.get('games', '?')} games) — new shards will be appended[/yellow]"
        )
    try:
        if params.headless:
            autoplay_headless(params, cfg, record_dataset=name)
        else:
            autoplay_parallel(params, cfg, record_dataset=name)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]interrupted — completed games were saved.[/dim]")


# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

def _choose_dataset() -> Optional[str]:
    console = _console()
    items = ds_mod.list_datasets()
    if not items:
        console.print("[yellow]no datasets yet — generate self-play data first.[/yellow]")
        return None
    _print_datasets(items)
    names = [d["name"] for d in items]
    raw = Prompt.ask("[dim]Dataset (name or #)[/dim]", default=names[0])
    raw = raw.strip()
    if raw.isdigit() and 1 <= int(raw) <= len(names):
        return names[int(raw) - 1]
    if raw in names:
        return raw
    console.print("[red]unknown dataset[/red]")
    return None


def _train_model() -> None:
    console = _console()
    ds = _choose_dataset()
    if ds is None:
        return
    from .train import TrainConfig, train, default_checkpoint_name

    epochs = _prompt_int("Epochs", 30, 1, 10000)
    batch = _prompt_int("Batch size", 256, 8, 65536)
    lr = _prompt_float("Learning rate", 1e-3, 1e-6, 1.0)
    aux = _prompt_float("Aux weight (tt-score loss)", 0.3, 0.0, 10.0)
    cfg = TrainConfig(dataset=ds, epochs=epochs, batch_size=batch, lr=lr, aux_weight=aux)
    default_name = default_checkpoint_name(cfg)
    name = Prompt.ask("[dim]Checkpoint name[/dim]", default=default_name).strip()
    cfg.checkpoint_name = name or default_name

    from .train import resolve_device
    console.print(f"[dim]device: {resolve_device('auto')}[/dim]")

    def on_epoch(e) -> None:
        star = " [green]*best*[/green]" if e.is_best else ""
        console.print(
            f"  [dim]epoch[/dim] {e.epoch:>3}/{e.epochs}  "
            f"[dim]train[/dim] {e.train_loss:.4f}  "
            f"[dim]val[/dim] {e.val_mse:.4f}  "
            f"[dim]sign[/dim] {e.val_sign_acc:.3f}  "
            f"[dim]lr[/dim] {e.lr:.2e}  "
            f"[dim]{e.elapsed:.0f}s[/dim]{star}"
        )

    try:
        res = train(cfg, on_epoch=on_epoch)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]training interrupted — best checkpoint so far is saved.[/dim]")
        return
    console.print(Panel(
        Text.assemble(
            ("checkpoint ", "dim"), (res["checkpoint"], STYLE_HINT),
            ("   best val ", "dim"), (f"{res['best_val_mse']:.4f}", "white"),
            (" @ epoch ", "dim"), (f"{res['best_epoch']}", "white"),
            ("   positions ", "dim"), (f"{res['positions']:,}", "white"),
            ("   ", "dim"), (f"{res['elapsed']:.0f}s on {res['device']}", "white"),
        ),
        title="[bold]Training complete[/bold]", border_style=STYLE_GRID,
    ))


# ---------------------------------------------------------------------------
# 3. Tournament
# ---------------------------------------------------------------------------

def _tournament() -> None:
    console = _console()
    from . import model as model_mod
    from .tournament import run_tournament

    ckpts = [c["name"] for c in model_mod.list_checkpoints()]
    if not ckpts:
        console.print("[yellow]no checkpoints yet — train a model first.[/yellow]")
        return
    console.print(f"[dim]{len(ckpts)} checkpoint(s) + classical anchor[/dim]")
    games_per_pair = _prompt_int("Games per pair (even)", 4, 2, 200)
    if games_per_pair % 2:
        games_per_pair += 1
    depth = _prompt_int("Classical depth", 2, 1, 8)
    ctime = _prompt_float("Classical time limit (s)", 0.5, 0.0, 60.0)
    ncpu = os.cpu_count() or 2
    workers = _prompt_int(f"Workers [1-{ncpu}]", max(1, ncpu - 1), 1, ncpu)

    n_pairs = (len(ckpts) + 1) * len(ckpts) // 2
    console.print(f"[dim]{n_pairs} pairs × {games_per_pair} games = "
                  f"{n_pairs * games_per_pair} games[/dim]")

    def on_progress(done: int, total: int, res) -> None:
        a, b, score = res
        tag = "1-0" if score == 1.0 else ("0-1" if score == 0.0 else "½-½")
        console.print(f"  [dim]{done:>4}/{total}[/dim]  {a} vs {b}  {tag}")

    try:
        data = run_tournament(
            ckpts, games_per_pair=games_per_pair,
            classical_depth=depth, classical_time=ctime,
            workers=workers, on_progress=on_progress,
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]tournament interrupted — no ratings written.[/dim]")
        return

    t = Table(box=box.SIMPLE, header_style="dim", title="Elo standings", title_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("model")
    t.add_column("Elo", justify="right")
    ratings = sorted(data["ratings"].items(), key=lambda kv: -kv[1])
    for i, (name, elo) in enumerate(ratings, 1):
        style = "bold white" if name == "classical" else STYLE_HINT
        t.add_row(str(i), Text(name, style=style), f"{elo:+.0f}")
    console.print(t)


# ---------------------------------------------------------------------------
# 4/5. Manage datasets & checkpoints
# ---------------------------------------------------------------------------

def _print_datasets(items: List[dict]) -> None:
    console = _console()
    t = Table(box=box.SIMPLE, header_style="dim", title="Datasets", title_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("name")
    t.add_column("games", justify="right")
    t.add_column("positions", justify="right")
    t.add_column("shards", justify="right")
    t.add_column("MB", justify="right")
    t.add_column("config", style="dim")
    for i, d in enumerate(items, 1):
        c = d["config"]
        cfg_str = (f"d{c.get('depth', '?')} e{c.get('tiebreak_epsilon', '?')} "
                   f"{c.get('starts', '?')}") if c else "-"
        t.add_row(str(i), d["name"], str(d["games"]), f"{d['positions']:,}" if
                  isinstance(d["positions"], int) else str(d["positions"]),
                  str(d["shards"]), f"{d['size_mb']:.1f}", cfg_str)
    console.print(t)


def _manage_datasets() -> None:
    console = _console()
    items = ds_mod.list_datasets()
    if not items:
        console.print("[yellow]no datasets.[/yellow]")
        return
    _print_datasets(items)
    raw = Prompt.ask("[dim]delete <name>, or Enter to go back[/dim]", default="").strip()
    if raw.startswith("delete "):
        name = raw[len("delete "):].strip()
        if Confirm.ask(f"[red]really delete dataset '{name}'?[/red]", default=False):
            if ds_mod.delete_dataset(name):
                console.print(f"[dim]deleted {name}[/dim]")
            else:
                console.print("[red]not found[/red]")


def _manage_checkpoints() -> None:
    console = _console()
    from . import model as model_mod
    from .tournament import load_elo

    items = model_mod.list_checkpoints()
    if not items:
        console.print("[yellow]no checkpoints.[/yellow]")
        return
    elo = load_elo().get("ratings", {})
    t = Table(box=box.SIMPLE, header_style="dim", title="Checkpoints", title_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("name")
    t.add_column("Elo", justify="right")
    t.add_column("val MSE", justify="right")
    t.add_column("sign acc", justify="right")
    t.add_column("epoch", justify="right")
    t.add_column("dataset", style="dim")
    t.add_column("MB", justify="right")
    for i, c in enumerate(items, 1):
        e = elo.get(c["name"], c.get("elo"))
        t.add_row(
            str(i), c["name"],
            f"{e:+.0f}" if isinstance(e, (int, float)) else "-",
            f"{c['val_mse']:.4f}" if c.get("val_mse") is not None else "-",
            f"{c['val_sign_acc']:.3f}" if c.get("val_sign_acc") is not None else "-",
            str(c.get("epoch") or "-"),
            str(c.get("dataset") or "-"),
            f"{c['size_mb']:.1f}",
        )
    console.print(t)
    raw = Prompt.ask("[dim]delete <name>, or Enter to go back[/dim]", default="").strip()
    if raw.startswith("delete "):
        name = raw[len("delete "):].strip()
        if Confirm.ask(f"[red]really delete checkpoint '{name}'?[/red]", default=False):
            if model_mod.delete_checkpoint(name):
                console.print(f"[dim]deleted {name}[/dim]")
            else:
                console.print("[red]not found[/red]")


# ---------------------------------------------------------------------------
# Menu loop
# ---------------------------------------------------------------------------

def nn_menu() -> None:
    console = _console()
    while True:
        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("1", "AlphaZero pipeline")
        table.add_row("2", "Generate classical self-play data")
        table.add_row("3", "Train classical value net")
        table.add_row("4", "Round-robin tournament (Elo)")
        table.add_row("5", "List / manage datasets")
        table.add_row("6", "List / manage checkpoints")
        table.add_row("7", "Back")
        console.print(Panel(table, title="[bold]Neural network training[/bold]",
                            border_style=STYLE_GRID))
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")
        if choice == "1":
            from .az_menu import az_menu
            az_menu()
        elif choice == "2":
            _generate_data()
        elif choice == "3":
            _train_model()
        elif choice == "4":
            _tournament()
        elif choice == "5":
            _manage_datasets()
        elif choice == "6":
            _manage_checkpoints()
        else:
            return
