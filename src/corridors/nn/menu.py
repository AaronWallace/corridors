"""Neural-network training menu.

Includes data generation, training, tournaments, dataset/checkpoint management,
and promotion of a machine-local checkpoint into the Git-shared best folder.
"""

from __future__ import annotations

import os
import time
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


def _format_modified(value) -> str:
    """Compact local date/time used by every dataset/checkpoint table."""
    if not isinstance(value, (int, float)):
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(value))


# ---------------------------------------------------------------------------
# 1. Generate self-play data
# ---------------------------------------------------------------------------

def _generate_data() -> None:
    from ..play import _setup, autoplay_headless, autoplay_parallel
    console = _console()
    cfg = settings.load()
    params = _setup(cfg, allow_neural=False)
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

def _ranked_ratings(ratings, limit: Optional[int] = None):
    """Highest Elo entries with deterministic alphabetical tie-breaking."""
    ranked = sorted(ratings.items(), key=lambda item: (-item[1], item[0]))
    return ranked if limit is None else ranked[:limit]

def _print_head_to_head(console, results, ratings, modified=None) -> None:
    """Crosstable of each model's W-L-D vs every other model (row's perspective).
    Rows/columns are ordered by Elo; columns are numbered to keep it compact."""
    from collections import defaultdict
    modified = modified or {}
    if not results:
        return
    rec = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # rec[a][b] = [W,L,D]
    names = set()
    for a, b, score in results:
        names.add(a); names.add(b)
        if score == 1.0:
            rec[a][b][0] += 1; rec[b][a][1] += 1      # a wins, b loses
        elif score == 0.0:
            rec[a][b][1] += 1; rec[b][a][0] += 1      # a loses, b wins
        else:
            rec[a][b][2] += 1; rec[b][a][2] += 1      # draw

    order = [name for name, _ in _ranked_ratings(
        {name: ratings.get(name, 0.0) for name in names})]
    idx = {n: i + 1 for i, n in enumerate(order)}

    t = Table(
        box=box.SIMPLE,
        header_style="dim",
        title="Round robin (W-L-D, row vs column)",
        title_style="bold",
    )
    t.add_column("#", justify="right", style="dim")
    t.add_column("Elo", justify="right")
    t.add_column("model")
    t.add_column("Modified", no_wrap=True)
    for n in order:
        t.add_column(str(idx[n]), justify="center")
    for r in order:
        cells = []
        for c in order:
            if r == c:
                cells.append("[dim]·[/dim]")
                continue
            w, l, d = rec[r][c]
            if w + l + d == 0:
                cells.append("[dim]-[/dim]")
            else:
                colour = "green" if w > l else ("red" if l > w else "yellow")
                cells.append(f"[{colour}]{w}-{l}-{d}[/{colour}]")
        style = "bold white" if r == "classical" else STYLE_HINT
        t.add_row(
            str(idx[r]), f"{ratings.get(r, 0.0):+.0f}",
            Text(r, style=style), _format_modified(modified.get(r)), *cells,
        )
    console.print(t)


def _tournament() -> None:
    console = _console()
    from . import model as model_mod
    from .tournament import run_tournament

    from .tournament import auto_tournament_workers
    from .az_selfplay import resolve_device

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
    max_plies = _prompt_int("Max plies per game (draw if reached)", 120, 20, 2000)

    # Device: CPU is usually fastest for tournaments (tiny net, 1 forward/move,
    # CPU-bound game logic + classical solver) and can use one worker per core.
    # GPU caps workers to bound CUDA contexts/VRAM, so it parallelizes far less.
    if resolve_device("auto") == "cpu":
        device = "cpu"
    else:
        console.print("[dim]CPU is usually faster here (tiny net; CPU-bound game "
                      "logic). GPU caps workers to fit CUDA contexts in VRAM.[/dim]")
        device = Prompt.ask("[dim]Inference device[/dim]",
                            choices=["cpu", "cuda"], default="cpu")
    ncpu = os.cpu_count() or 2
    workers = _prompt_int(f"Workers [1-{ncpu}]", auto_tournament_workers(device), 1, ncpu)

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
            workers=workers, device=device, max_plies=max_plies,
            on_progress=on_progress,
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]tournament interrupted — no ratings written.[/dim]")
        return

    checkpoint_modified = {
        item["name"]: item.get("modified")
        for item in model_mod.list_checkpoints()
    }
    _print_head_to_head(
        console, data.get("last_run", {}).get("results", []),
        data["ratings"], checkpoint_modified,
    )


# ---------------------------------------------------------------------------
# 4/5. Manage datasets & checkpoints
# ---------------------------------------------------------------------------

def _numbered_selections(items: List[dict], raw: str) -> List[dict]:
    """Return unique, valid numbered selections in the order entered."""
    selected = []
    seen = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part) - 1
        except ValueError:
            continue
        if 0 <= index < len(items) and index not in seen:
            selected.append(items[index])
            seen.add(index)
    return selected

def _print_datasets(items: List[dict], title: str = "Datasets") -> None:
    console = _console()
    t = Table(box=box.SIMPLE, header_style="dim", title=title, title_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("name")
    t.add_column("games", justify="right")
    t.add_column("positions", justify="right")
    t.add_column("shards", justify="right")
    t.add_column("MB", justify="right")
    t.add_column("Modified", no_wrap=True)
    t.add_column("config", style="dim")
    for i, d in enumerate(items, 1):
        c = d["config"]
        if d.get("kind") == "alphazero":
            useful = []
            min_mcts = c.get("min_mcts", 0)
            max_mcts = c.get("max_mcts", 0)
            if min_mcts and max_mcts and min_mcts != max_mcts:
                bias = max(0.01, float(c.get("mcts_bias", 3.0)))
                expected = min_mcts + bias / (bias + 1.0) * (max_mcts - min_mcts)
                useful.append(
                    f"{min_mcts}-{max_mcts} sims (~{expected:.0f} avg)"
                )
            elif c.get("simulations") is not None:
                useful.append(f"{c['simulations']} sims")
            if c.get("max_plies") is not None:
                useful.append(f"{c['max_plies']} max plies")
            cfg_str = " · ".join(useful) or "-"
        else:
            cfg_str = (f"d{c.get('depth', '?')} e{c.get('tiebreak_epsilon', '?')} "
                       f"{c.get('starts', '?')}") if c else "-"
        t.add_row(str(i), d["name"], str(d["games"]), f"{d['positions']:,}" if
                  isinstance(d["positions"], int) else str(d["positions"]),
                  str(d["shards"]), f"{d['size_mb']:.1f}",
                  _format_modified(d.get("modified")), cfg_str)
    console.print(t)


def _delete_datasets(items: List[dict], *, archived: bool = False) -> None:
    console = _console()
    total_mb = sum(d["size_mb"] for d in items)
    location = "archived " if archived else ""
    if not Confirm.ask(
            f"[bold red]Permanently delete {len(items)} {location}dataset(s), "
            f"{total_mb:.1f} MB?[/bold red]", default=False):
        console.print("[dim]cancelled[/dim]")
        return
    delete = ds_mod.delete_archived_dataset if archived else ds_mod.delete_dataset
    for item in items:
        name = item["name"]
        if delete(name):
            console.print(f"[dim]deleted {name}[/dim]")
        else:
            console.print(f"[yellow]skipped {name}[/yellow]")


def _manage_archived_datasets() -> None:
    console = _console()
    items = ds_mod.list_archived_datasets()
    if not items:
        console.print("[yellow]no archived datasets.[/yellow]")
        return
    _print_datasets(items, "Archived datasets")
    raw = Prompt.ask(
        "[dim]r # to restore, d # to permanently delete, or q to go back[/dim]",
        default="q",
    ).strip()
    if raw.lower() == "q":
        return
    action, _, selection = raw.partition(" ")
    selected = _numbered_selections(items, selection)
    if not selected:
        return
    if action.lower() == "d":
        _delete_datasets(selected, archived=True)
        return
    if action.lower() == "r":
        for item in selected:
            name = item["name"]
            if ds_mod.restore_dataset(name):
                console.print(f"[dim]restored {name}[/dim]")
            else:
                console.print(f"[yellow]could not restore {name}; active name may exist[/yellow]")


def _manage_datasets() -> None:
    console = _console()
    items = ds_mod.list_datasets()
    archived = ds_mod.list_archived_datasets()
    if not items:
        if archived:
            console.print("[dim]no active datasets; opening archive[/dim]")
            _manage_archived_datasets()
        else:
            console.print("[yellow]no datasets.[/yellow]")
        return
    _print_datasets(items)
    raw = Prompt.ask(
        f"[dim]a # to archive, d # to delete, v to view archive ({len(archived)}), "
        "or q to go back[/dim]",
        default="q",
    ).strip()
    if raw.lower() == "q":
        return
    if raw.lower() == "v":
        _manage_archived_datasets()
        return
    action, separator, selection = raw.partition(" ")
    # Retain the old bare-number delete shorthand.
    if not separator:
        action, selection = "d", raw
    selected = _numbered_selections(items, selection)
    if not selected:
        return
    if action.lower() == "d":
        _delete_datasets(selected)
        return
    if action.lower() == "a":
        for item in selected:
            name = item["name"]
            if ds_mod.archive_dataset(name):
                console.print(f"[dim]archived {name}[/dim]")
            else:
                console.print(f"[yellow]could not archive {name}; archived name may exist[/yellow]")


def _manage_checkpoints(copy_only: bool = False) -> None:
    console = _console()
    from . import model as model_mod
    from .tournament import load_elo, rename_elo_checkpoint

    items = model_mod.list_checkpoints()
    if not items:
        console.print("[yellow]no checkpoints.[/yellow]")
        return
    elo = load_elo().get("ratings", {})
    t = Table(box=box.SIMPLE, header_style="dim", title="Checkpoints", title_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("name")
    t.add_column("Elo", justify="right")
    t.add_column("epoch", justify="right")
    t.add_column("dataset", style="dim")
    t.add_column("lineage", style="dim")
    t.add_column("best", justify="center")
    t.add_column("MB", justify="right")
    t.add_column("Modified", no_wrap=True)
    for i, c in enumerate(items, 1):
        e = elo.get(c["name"], c.get("elo"))
        # Weight lineage: seed origin (cross-lineage) beats a self-resume.
        if c.get("seeded_from"):
            lineage = f"seed:{c['seeded_from']}"
        elif c.get("resumed_from") and c["resumed_from"] != c["name"]:
            lineage = c["resumed_from"]
        else:
            lineage = "-"
        t.add_row(
            str(i), c["name"],
            f"{e:+.0f}" if isinstance(e, (int, float)) else "-",
            str(c.get("epoch") or "-"),
            str(c.get("dataset") or "-"),
            lineage,
            "✓" if c.get("in_best") else "-",
            f"{c['size_mb']:.1f}",
            _format_modified(c.get("modified")),
        )
    console.print(t)
    if copy_only:
        raw = Prompt.ask(
            "[dim]Checkpoint # to copy to shared best, or q to go back[/dim]",
            default="q",
        ).strip()
        if raw.lower() == "q":
            return
        selected = _numbered_selections(items, raw)
        if not selected:
            return
        if len(selected) != 1:
            console.print("[yellow]choose one checkpoint number to copy[/yellow]")
            return
        name = selected[0]["name"]
        target = model_mod.copy_checkpoint_to_best(name)
        console.print(f"[dim]copied {name} to {target.parent.name}/[/dim]")
        return
    raw = Prompt.ask(
        "[dim]Delete # (comma-separated), r # to rename, c # to copy to best, "
        "or q to go back[/dim]",
        default="q",
    ).strip()
    if raw.lower() == "q":
        return
    if raw.lower().startswith("c "):
        selected = _numbered_selections(items, raw[2:])
        if len(selected) != 1:
            console.print("[yellow]choose one checkpoint number to copy[/yellow]")
            return
        name = selected[0]["name"]
        target = model_mod.copy_checkpoint_to_best(name)
        console.print(f"[dim]copied {name} to {target.parent.name}/[/dim]")
        return
    if raw.lower().startswith("r "):
        selected = _numbered_selections(items, raw[2:])
        if len(selected) != 1:
            console.print("[yellow]choose one checkpoint number to rename[/yellow]")
            return
        old_name = selected[0]["name"]
        new_name = Prompt.ask("[dim]New checkpoint name[/dim]").strip()
        try:
            if model_mod.rename_checkpoint(old_name, new_name):
                rename_elo_checkpoint(old_name, new_name)
                console.print(f"[dim]renamed {old_name} to {new_name}[/dim]")
        except (ValueError, FileExistsError) as exc:
            console.print(f"[red]{exc}[/red]")
        return
    for item in _numbered_selections(items, raw):
        name = item["name"]
        if model_mod.delete_checkpoint(name):
            console.print(f"[dim]deleted {name}[/dim]")


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
        table.add_row("7", "Copy checkpoint to shared best")
        table.add_row("q", "Back")
        console.print(Panel(table, title="[bold]Neural network training[/bold]",
                            border_style=STYLE_GRID))
        choice = Prompt.ask(
            "Choose", choices=["1", "2", "3", "4", "5", "6", "7", "q"],
            default="1",
        )
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
        elif choice == "7":
            _manage_checkpoints(copy_only=True)
        elif choice == "q":
            return
