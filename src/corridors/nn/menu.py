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


class SetupCancelled(Exception):
    """Raised by prompt helpers when the user types 'q' during a setup flow.
    Setup entry points (self-play, train, full loop, tournament, etc.) catch
    this and return cleanly to their menu — so users can bail out of any long
    prompt sequence without Ctrl-C'ing the whole app."""


def _prompt_int(label: str, default: int, lo: int, hi: int) -> int:
    console = _console()
    while True:
        raw = Prompt.ask(f"[dim]{label} (q to cancel)[/dim]", default=str(default))
        if raw.strip().lower() == "q":
            raise SetupCancelled
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
        raw = Prompt.ask(f"[dim]{label} (q to cancel)[/dim]", default=f"{default:g}")
        if raw.strip().lower() == "q":
            raise SetupCancelled
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        console.print(f"[red]  must be a number {lo}..{hi}[/red]")


def _prompt_str(label: str, default: str = "") -> str:
    """Text prompt that respects the 'q'-cancels-setup convention. Use for
    names/paths so users can back out mid-setup instead of typing through."""
    raw = Prompt.ask(f"[dim]{label} (q to cancel)[/dim]", default=default).strip()
    if raw.lower() == "q":
        raise SetupCancelled
    return raw


def _format_modified(value) -> str:
    """Compact local date/time used by every dataset/checkpoint table."""
    if not isinstance(value, (int, float)):
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(value))


def _format_elo(elo, stale: bool = False) -> str:
    """Elo table cell; rated-but-absent-from-latest-round-robin shows as stale."""
    if not isinstance(elo, (int, float)):
        return "-"
    return f"{elo:+.0f}" + (" [yellow]stale[/yellow]" if stale else "")


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
    from .train import (TrainConfig, train, default_checkpoint_name,
                        _resolved_max_epochs, _resolved_min_epochs,
                        resolve_device)

    epochs = _prompt_int("Epochs (soft target — may extend if val is still improving)",
                         30, 1, 10000)
    batch = _prompt_int("Batch size", 256, 8, 65536)
    lr = _prompt_float("Learning rate", 1e-3, 1e-6, 1.0)
    aux = _prompt_float("Aux weight (tt-score loss)", 0.3, 0.0, 10.0)
    cfg = TrainConfig(dataset=ds, epochs=epochs, batch_size=batch, lr=lr,
                      aux_weight=aux)
    default_name = default_checkpoint_name(cfg)
    name = Prompt.ask("[dim]Checkpoint name[/dim]", default=default_name).strip()
    cfg.checkpoint_name = name or default_name

    console.print(f"[dim]device: {resolve_device('auto')}[/dim]")
    console.print(
        f"[dim]adaptive epochs: {_resolved_min_epochs(cfg)}-"
        f"{_resolved_max_epochs(cfg)} · stop patience {cfg.early_stop_patience} · "
        f"meaningful validation gain >={cfg.early_stop_min_delta:.2%}[/dim]"
    )
    console.print(f"[dim]{_FIT_LEGEND}[/dim]")

    fit_trend = _EpochFitTrend()

    def on_epoch(e) -> None:
        star = " [green]*best*[/green]" if e.is_best else ""
        stop = (f"  [yellow]-> stop: {e.stop_reason}[/yellow]"
                if e.will_stop else "")
        extend = (f"  [cyan]-> extending beyond target {e.target_epochs}[/cyan]"
                  if e.extension_started else "")
        # ValueNet's val metric is MSE (bounded), not the AZ policy+value sum,
        # so we feed a synthetic "train_loss" to the fit trend that keeps the
        # same relationship (val - train). Use train_loss as-is.
        try:
            fit = fit_trend.update(type("E", (), {
                "train_loss": e.train_loss, "val_loss": e.val_mse})())
        except Exception:
            fit = ""
        console.print(
            f"  [dim]epoch[/dim] {e.epoch:>3}/{e.epochs}  "
            f"[dim]train[/dim] {e.train_loss:.4f}  "
            f"[dim]val[/dim] {e.val_mse:.4f}  "
            f"[dim]sign[/dim] {e.val_sign_acc:.3f}  "
            f"[dim]lr[/dim] {e.lr:.2e}  "
            f"[dim]{e.elapsed:.0f}s[/dim]  {fit}{star}{extend}{stop}"
        )

    try:
        from rich.live import Live
        from .az_menu import _batch_progress_text
        with Live(Text("  preparing training split...", style="dim"),
                  console=console, refresh_per_second=4, transient=True) as live:
            res = train(cfg, on_epoch=on_epoch,
                        on_batch=lambda info: live.update(_batch_progress_text(info)))
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]training interrupted — best checkpoint so far is saved.[/dim]")
        return

    stop_note = (f"   [dim]stopped early: {res['stop_reason']}[/dim]"
                 if res.get("stopped_early") else "")
    console.print(Panel(
        Text.assemble(
            ("checkpoint ", "dim"), (res["checkpoint"], STYLE_HINT),
            ("   best val ", "dim"), (f"{res['best_val_mse']:.4f}", "white"),
            (" @ epoch ", "dim"), (f"{res['best_epoch']}", "white"),
            (f"/{res.get('epochs_run', res['best_epoch'])}", "dim"),
            ("   positions ", "dim"), (f"{res['positions']:,}", "white"),
            ("   ", "dim"), (f"{res['elapsed']:.0f}s on {res['device']}", "white"),
        ),
        title="[bold]Training complete[/bold]", border_style=STYLE_GRID,
    ))
    if stop_note:
        console.print(stop_note)


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
    for res in results:
        # results are (a, b, score) or (a, b, score, termination) — accept both
        a, b, score = res[0], res[1], res[2]
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


def _print_termination_summary(console, terminations: dict) -> None:
    """How did the tournament's games end? Wins (goal reached), losses (opponent
    boxed in with no legal moves — rare in Corridors), and the two draw kinds:
    threefold repetition and hitting the max-plies cap. Read the draw split as a
    training-health signal: lots of max-plies = wandering/underlearned; lots of
    threefolds = shuffling loops the net doesn't break out of."""
    if not terminations:
        return
    # Terminations use the reason keys from play_pair_game: goal / no legal
    # moves / threefold / max plies. "goal" is a decisive result for whichever
    # side reached its goal first; we count it once as a decisive game (not
    # once per side).
    decisive = int(terminations.get("goal", 0))
    blocked = int(terminations.get("no legal moves", 0))
    threefold = int(terminations.get("threefold", 0))
    max_plies = int(terminations.get("max plies", 0))
    total = decisive + blocked + threefold + max_plies
    if total <= 0:
        return

    def _pct(n: int) -> str:
        return f"{100.0 * n / total:5.1f}%"

    t = Table(box=box.SIMPLE, header_style="dim",
              title="How games ended", title_style="bold")
    t.add_column("outcome")
    t.add_column("count", justify="right")
    t.add_column("share", justify="right")
    t.add_row("decisive (goal reached)", str(decisive), _pct(decisive))
    if blocked:
        t.add_row("decisive (no legal moves)", str(blocked), _pct(blocked))
    t.add_row(Text("draw — threefold repetition", style="yellow"),
              str(threefold), _pct(threefold))
    t.add_row(Text("draw — max plies reached", style="yellow"),
              str(max_plies), _pct(max_plies))
    t.add_row(Text("total", style="dim"), str(total), "100.0%")
    console.print(t)


def _tournament() -> None:
    console = _console()
    from . import model as model_mod
    from .tournament import run_tournament

    from .tournament import auto_tournament_workers
    from .az_selfplay import resolve_device

    items = model_mod.list_checkpoints()
    if not items:
        console.print("[yellow]no checkpoints yet — train a model first.[/yellow]")
        return

    t = Table(box=box.SIMPLE, header_style="dim", title="Checkpoints",
              title_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("name")
    t.add_column("Elo", justify="right")
    t.add_column("Modified", no_wrap=True)
    for i, c in enumerate(items, 1):
        t.add_row(str(i), c["name"], _format_elo(c.get("elo"), c.get("elo_stale")),
                  _format_modified(c.get("modified")))
    console.print(t)

    raw = Prompt.ask(
        "[dim]Checkpoints to include (comma-separated #s, Enter = all)[/dim]",
        default="",
    ).strip()
    if raw:
        selected = _numbered_selections(items, raw)
        if not selected:
            console.print("[yellow]no valid checkpoint numbers selected.[/yellow]")
            return
        items = selected
    ckpts = [c["name"] for c in items]
    skipped = len(model_mod.list_checkpoints()) - len(ckpts)
    console.print(f"[dim]{len(ckpts)} checkpoint(s) + classical anchor"
                  + (f" ({skipped} excluded — their Elo will show as stale)"
                     if skipped else "") + "[/dim]")
    games_per_pair = _prompt_int("Games per pair (even)", 4, 2, 200)
    if games_per_pair % 2:
        games_per_pair += 1
    depth = _prompt_int("Classical depth (1-30; time_limit usually caps first past ~6)",
                        2, 1, 30)
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
        # res may be (a, b, score) or (a, b, score, termination). Accept both.
        a, b, score = res[0], res[1], res[2]
        term = res[3] if len(res) > 3 else ""
        if score == 1.0:
            tag = "1-0"
        elif score == 0.0:
            tag = "0-1"
        else:
            # ASCII draw label; distinguish threefold from max-plies for insight.
            tag = "3fold" if term == "threefold" else ("maxpl" if term == "max plies" else "draw ")
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
    _print_termination_summary(console, data.get("last_run", {}).get("terminations", {}))


# ---------------------------------------------------------------------------
# 4/5. Manage datasets & checkpoints
# ---------------------------------------------------------------------------

def _numbered_selections(items: List[dict], raw: str) -> List[dict]:
    """Return unique, valid numbered selections in the order entered.

    Accepts individual numbers, ranges, comma- or space-separated lists, and
    the literal 'all' — so bulk operations don't need `d 12` typed repeatedly.
    Examples: '1', '1,3,5', '1 3 5', '1-5', '1-3,7,9-11', 'all'.
    """
    if raw.strip().lower() == "all":
        return list(items)
    selected = []
    seen = set()
    # Comma-, space-, or mixed-separated tokens all work.
    for part in raw.replace(",", " ").split():
        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            lo_s, hi_s = part.split("-", 1)
            try:
                lo, hi = int(lo_s) - 1, int(hi_s) - 1
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                if 0 <= i < len(items) and i not in seen:
                    selected.append(items[i])
                    seen.add(i)
        else:
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
        "[dim]r <sel> restore · d <sel> permanently delete "
        "(sel: #, 1-5, 1,3,7, or 'all') · q back[/dim]",
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
        f"[dim]s <sel> share · a <sel> archive · d <sel> delete "
        f"(sel: #, 1-5, 1,3,7, or 'all') · "
        f"v view archive ({len(archived)}) · q back[/dim]",
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
        return
    if action.lower() == "s":
        for item in selected:
            name = item["name"]
            shared_name = ds_mod.share_dataset(name)
            if shared_name:
                console.print(f"[dim]moved {name} to {shared_name} (Git-shared)[/dim]")
            else:
                console.print(f"[yellow]could not share {name}; it may already be shared[/yellow]")


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
            _format_elo(e, c.get("elo_stale")),
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
        "[dim]Delete <sel> · r <sel> rename · c <sel> copy to best "
        "(sel: #, 1-5, 1,3,7, or 'all') · q back[/dim]",
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
    selected = _numbered_selections(items, raw)
    if not selected:
        return
    for item in selected:
        console.print(f"  {item['name']}" + (" [dim](has a best/ copy)[/dim]"
                                             if item.get("in_best") else ""))
    if not Confirm.ask(
            f"[bold red]Permanently delete {len(selected)} checkpoint(s)?[/bold red]",
            default=False):
        console.print("[dim]cancelled[/dim]")
        return
    for item in selected:
        name = item["name"]
        if model_mod.delete_checkpoint(name):
            note = (" (including its best/ copy — commit to remove it from Git)"
                    if item.get("in_best") else "")
            console.print(f"[dim]deleted {name}{note}[/dim]")


# ---------------------------------------------------------------------------
# Menu loop
# ---------------------------------------------------------------------------

def _benchmark_persistent_tt() -> None:
    """Measure classical-solver throughput at several worker counts, with the
    persistent sqlite TT on vs off, and print a comparison. Directly answers
    "should I turn the shared TT off on this host?" for a given worker count —
    no baked-in threshold, real data.

    Total runtime ≈ len(worker_counts) × 2 × duration + startup overhead.
    """
    console = _console()
    from .tt_benchmark import sweep
    ncpu = os.cpu_count() or 4
    console.print("\n[bold]Classical persistent-TT benchmark[/bold]")
    console.print(
        "[dim]Times classical solver throughput at each worker count with the "
        "shared sqlite TT on vs off. Higher searches/s and nps/worker are "
        "better; a big drop with TT on = contention wins over cache hit rate "
        "at that scale.[/dim]\n"
    )
    # Reasonable sweep across the useful range for this host.
    default_counts = sorted({1, 4, min(16, ncpu), min(64, ncpu),
                             max(1, ncpu - 1)})
    counts_raw = Prompt.ask(
        "[dim]Worker counts to test (comma-separated) "
        f"(q to cancel)[/dim]",
        default=",".join(str(c) for c in default_counts),
    ).strip()
    if counts_raw.lower() == "q":
        raise SetupCancelled
    try:
        counts = sorted({max(1, min(ncpu, int(x.strip())))
                         for x in counts_raw.split(",") if x.strip()})
    except ValueError:
        console.print("[red]invalid worker list[/red]")
        return
    if not counts:
        console.print("[red]no valid worker counts[/red]")
        return
    duration = _prompt_int("Seconds per config (higher = more stable)",
                           20, 5, 300)

    total_configs = len(counts) * 2
    est_seconds = total_configs * duration + total_configs * 3  # + spawn overhead
    console.print(f"[dim]{total_configs} configs × {duration}s ≈ "
                  f"{est_seconds}s total ({est_seconds // 60}m{est_seconds % 60}s). "
                  "Ctrl-C to abort.[/dim]\n")

    def on_config(workers, shared, idx, total):
        tag = "shared" if shared else "per-worker"
        console.print(f"  [dim][{idx}/{total}][/dim] running: "
                      f"workers={workers}, TT={tag}…")

    try:
        results = sweep(counts, duration_s=float(duration),
                        on_config=on_config)
    except (KeyboardInterrupt, EOFError):
        console.print("[dim]benchmark interrupted[/dim]")
        return

    # Group results by worker count for a clean side-by-side table.
    grouped: Dict[int, Dict[bool, "BenchResult"]] = {}
    for r in results:
        grouped.setdefault(r.workers, {})[r.tt_shared] = r

    t = Table(box=box.SIMPLE, header_style="dim",
              title="Classical solver throughput (higher = better)",
              title_style="bold")
    t.add_column("workers", justify="right")
    t.add_column("TT shared", justify="right")
    t.add_column("TT off", justify="right")
    t.add_column("winner", justify="center")
    t.add_column("speedup off/on", justify="right")
    for w in sorted(grouped):
        pair = grouped[w]
        on = pair.get(True)
        off = pair.get(False)
        on_sps = on.searches_per_sec if on else 0
        off_sps = off.searches_per_sec if off else 0
        if on_sps <= 0 and off_sps <= 0:
            continue
        winner = "off" if off_sps > on_sps else ("on" if on_sps > off_sps else "tie")
        color = "green" if winner == "off" else ("yellow" if winner == "on" else "dim")
        speedup = (off_sps / on_sps) if on_sps > 0 else float("inf")
        speedup_str = f"{speedup:.2f}×" if speedup < 100 else "∞"
        t.add_row(
            str(w),
            f"{on_sps:.1f}/s ({on.nps_per_worker:,.0f} nps)" if on else "-",
            f"{off_sps:.1f}/s ({off.nps_per_worker:,.0f} nps)" if off else "-",
            Text(winner, style=color),
            speedup_str,
        )
    console.print(t)

    # Actionable recommendation based on the largest tested worker count.
    largest = max(grouped)
    pair = grouped[largest]
    if True in pair and False in pair:
        on_r, off_r = pair[True], pair[False]
        rec = "off" if off_r.searches_per_sec > on_r.searches_per_sec else "on"
        console.print(f"\n[bold]At {largest} workers on this host, "
                      f"persistent TT [/bold][{'green' if rec == 'off' else 'yellow'}]"
                      f"{rec}[/{'green' if rec == 'off' else 'yellow'}]"
                      f"[bold] wins.[/bold]")
        save = Confirm.ask(
            "[dim]Save this as the default for future runs?[/dim]",
            default=(rec == "off"),
        )
        if save:
            settings.save(use_persistent_tt=(rec == "on"))
            console.print(f"[dim]saved use_persistent_tt={rec == 'on'}[/dim]")


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
        table.add_row("8", "Benchmark classical persistent-TT (measure on this host)")
        table.add_row("q", "Back")
        console.print(Panel(table, title="[bold]Neural network training[/bold]",
                            border_style=STYLE_GRID))
        choice = Prompt.ask(
            "Choose", choices=["1", "2", "3", "4", "5", "6", "7", "8", "q"],
            default="1",
        )
        if choice == "q":
            return
        # SetupCancelled bubbles up from any 'q' typed inside a setup prompt
        # sequence — catch it here so we return to this menu instead of
        # falling out of the app or requiring Ctrl-C.
        try:
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
            elif choice == "8":
                _benchmark_persistent_tt()
        except SetupCancelled:
            console.print("[dim]cancelled — back to menu[/dim]")
