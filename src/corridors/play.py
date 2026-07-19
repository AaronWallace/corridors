"""Autoplay console for Corridors.

Runs N AI-vs-AI games in a row with a single live-rendered panel that stays put
(no scrolling / bouncing). Configure once at startup (games, random-or-fixed
starts, depth, time limit, tiebreak epsilon, max plies) and let it rip.
"""

from __future__ import annotations

import io
import multiprocessing
import os
import queue as queue_mod
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, io.UnsupportedOperation):
        pass

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from . import game, settings, solver
from .game import (
    Board,
    Move,
    NCOLS,
    NROWS,
    P1_END_ROW,
    P2_END_ROW,
    Pos,
    State,
    Wall,
    WALLS_PER_PLAYER,
    apply_move,
    is_threefold_repetition,
)

COL_LETTERS = "ABCDEFGHI"

STYLE_P1 = "bold cyan"
STYLE_P2 = "bold magenta"
STYLE_GOAL_P1 = "cyan"
STYLE_GOAL_P2 = "magenta"
STYLE_ENDZONE = "grey37"
STYLE_EMPTY = "grey58"
STYLE_GRID = "grey35"
STYLE_WALL = "bold yellow"
STYLE_HINT = "bold green"
STYLE_ERR = "bold red"
STYLE_DRAW = "bold yellow"
STYLE_LABEL = "grey58"

MAX_HISTORY_PLIES = 20

console = Console(legacy_windows=False)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def parse_col(token: str) -> Optional[int]:
    token = token.strip().upper()
    if not token:
        return None
    if token[0].isalpha():
        idx = COL_LETTERS.find(token[0])
        return idx if idx >= 0 else None
    try:
        c = int(token)
        return c if 0 <= c < NCOLS else None
    except ValueError:
        return None


def format_pos(p: Pos) -> str:
    return f"{COL_LETTERS[p[1]]}{p[0]}"


def format_wall(w: Wall) -> str:
    r, c, o = w
    return f"{COL_LETTERS[c]}{r}{o}"


def format_move(m: Move) -> str:
    if m[0] == "m":
        return f"m {format_pos(m[1])}"
    return f"w {format_wall(m[1])}"


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PlayerStats:
    turns: int = 0
    wins: int = 0
    total_time: float = 0.0
    total_nodes: int = 0
    total_score: int = 0
    best_score_seen: int = -10 ** 9
    depth_sum: int = 0
    max_depth: int = 0

    def record_turn(self, stats, score: int) -> None:
        self.turns += 1
        self.total_time += stats.elapsed
        self.total_nodes += stats.nodes
        # Clamp mate scores so a single forced win doesn't swamp the average.
        self.total_score += max(-2000, min(2000, score))
        if score > self.best_score_seen:
            self.best_score_seen = score
        self.depth_sum += stats.max_depth
        if stats.max_depth > self.max_depth:
            self.max_depth = stats.max_depth

    def avg_time(self) -> float:
        return self.total_time / self.turns if self.turns else 0.0

    def avg_score(self) -> float:
        return self.total_score / self.turns if self.turns else 0.0

    def avg_nodes(self) -> float:
        return self.total_nodes / self.turns if self.turns else 0.0

    def avg_depth(self) -> float:
        return self.depth_sum / self.turns if self.turns else 0.0


@dataclass
class SessionStats:
    target_games: int
    games_played: int = 0
    draws: int = 0
    total_plies: int = 0
    total_time: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    p1: PlayerStats = field(default_factory=PlayerStats)
    p2: PlayerStats = field(default_factory=PlayerStats)

    def record_game(self, winner: Optional[int], plies: int, game_time: float) -> None:
        self.games_played += 1
        self.total_plies += plies
        self.total_time += game_time
        if winner == 1:
            self.p1.wins += 1
        elif winner == 2:
            self.p2.wins += 1
        else:
            self.draws += 1

    def avg_game_time(self) -> float:
        return self.total_time / self.games_played if self.games_played else 0.0

    def avg_plies(self) -> float:
        return self.total_plies / self.games_played if self.games_played else 0.0

    def eta_seconds(self) -> float:
        if self.games_played == 0:
            return 0.0
        remaining = self.target_games - self.games_played
        # Wall-clock rate, so ETA stays honest regardless of worker count.
        wall_per_game = (time.monotonic() - self.started_at) / self.games_played
        return remaining * wall_per_game


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def _cell_text(state: Optional[State], board: Optional[Board], r: int, c: int) -> Text:
    pos = (r, c)
    if state is not None:
        if pos == state.p1:
            return Text(" 1 ", style=STYLE_P1)
        if pos == state.p2:
            return Text(" 2 ", style=STYLE_P2)
    if board is not None:
        if pos == board.p1_goal:
            return Text(" ✦ ", style=STYLE_GOAL_P1)
        if pos == board.p2_goal:
            return Text(" ✦ ", style=STYLE_GOAL_P2)
    if r == P1_END_ROW or r == P2_END_ROW:
        return Text(" · ", style=STYLE_ENDZONE)
    return Text(" · ", style=STYLE_EMPTY)


def _board_renderable(state: Optional[State], board: Optional[Board]) -> Group:
    h_blocked = set()
    v_blocked = set()
    if state is not None:
        for (wr, wc, o) in state.walls:
            if o == "H":
                h_blocked.add((wr, wc))
                h_blocked.add((wr, wc + 1))
            else:
                v_blocked.add((wr, wc))
                v_blocked.add((wr + 1, wc))

    lines: List[Text] = []
    header = Text("     ", style=STYLE_GRID)
    for i, ch in enumerate(COL_LETTERS[:NCOLS]):
        header.append(f" {ch} ", style=STYLE_GRID)
        if i < NCOLS - 1:
            header.append(" ", style=STYLE_GRID)
    lines.append(header)

    top = Text("   ┌", style=STYLE_GRID)
    for c in range(NCOLS):
        top.append("───", style=STYLE_GRID)
        top.append("┬" if c < NCOLS - 1 else "┐", style=STYLE_GRID)
    lines.append(top)

    for r in range(NROWS):
        line = Text(f"{r:2d} ", style=STYLE_GRID)
        line.append("│", style=STYLE_GRID)
        for c in range(NCOLS):
            line.append_text(_cell_text(state, board, r, c))
            if c < NCOLS - 1:
                sep = "┃" if (r, c) in v_blocked else "│"
                line.append(sep, style=STYLE_WALL if sep == "┃" else STYLE_GRID)
        line.append("│", style=STYLE_GRID)
        lines.append(line)
        if r < NROWS - 1:
            wall_line = Text("   ├", style=STYLE_GRID)
            for c in range(NCOLS):
                if (r, c) in h_blocked:
                    wall_line.append("━━━", style=STYLE_WALL)
                else:
                    wall_line.append("───", style=STYLE_GRID)
                wall_line.append("┼" if c < NCOLS - 1 else "┤", style=STYLE_GRID)
            lines.append(wall_line)

    bot = Text("   └", style=STYLE_GRID)
    for c in range(NCOLS):
        bot.append("───", style=STYLE_GRID)
        bot.append("┴" if c < NCOLS - 1 else "┘", style=STYLE_GRID)
    lines.append(bot)
    return Group(*lines)


def _history_renderable(moves: List[Move]) -> Table:
    table = Table(
        box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1),
        title="History", title_style="dim",
    )
    table.add_column("#", justify="right", style="grey50", no_wrap=True, width=4)
    table.add_column("P1", style=STYLE_P1, no_wrap=True, width=7)
    table.add_column("P2", style=STYLE_P2, no_wrap=True, width=7)
    n = len(moves)
    if n == 0:
        for _ in range(MAX_HISTORY_PLIES // 2):
            table.add_row("", "", "")
        return table
    start_ply = max(0, n - MAX_HISTORY_PLIES)
    if start_ply % 2 == 1:
        start_ply -= 1
    rows_added = 0
    if start_ply > 0:
        table.add_row("…", "", "")
        rows_added += 1
    for i in range(start_ply, n, 2):
        round_num = i // 2 + 1
        p1_mv = format_move(moves[i]) if i < n else ""
        p2_mv = format_move(moves[i + 1]) if i + 1 < n else ""
        table.add_row(f"{round_num}.", p1_mv, p2_mv)
        rows_added += 1
    # Pad to a stable height so the panel doesn't jitter early in the game.
    while rows_added < MAX_HISTORY_PLIES // 2:
        table.add_row("", "", "")
        rows_added += 1
    return table


# ---------------------------------------------------------------------------
# Stat tables (fixed layout — no jitter)
# ---------------------------------------------------------------------------

def _player_stats_table(ss: SessionStats) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1),
              title_style="dim", expand=False, collapse_padding=False)
    t.add_column("",   style="dim",           no_wrap=True)
    t.add_column("W",  justify="right",       no_wrap=True)
    t.add_column("N",  justify="right",       no_wrap=True)
    t.add_column("avg score",  justify="right", no_wrap=True)
    t.add_column("best",       justify="right", no_wrap=True)
    t.add_column("avg move",   justify="right", no_wrap=True)
    t.add_column("nodes/turn", justify="right", no_wrap=True)
    t.add_column("avg d",      justify="right", no_wrap=True)

    def _row(name: str, style: str, p: PlayerStats) -> None:
        avg_s = p.avg_score()
        best = p.best_score_seen if p.turns else 0
        best_str = "WIN" if solver.is_mate_score(best) else f"{best:+d}"
        t.add_row(
            Text(name, style=style),
            f"{p.wins}",
            f"{p.turns}",
            f"{avg_s:+.0f}" if p.turns else "-",
            best_str if p.turns else "-",
            f"{p.avg_time():.2f}s" if p.turns else "-",
            f"{p.avg_nodes():,.0f}" if p.turns else "-",
            f"{p.avg_depth():.1f}" if p.turns else "-",
        )
    _row("P1", STYLE_P1, ss.p1)
    _row("P2", STYLE_P2, ss.p2)
    return t


def _session_lines(ss: SessionStats) -> Group:
    line1 = Text()
    line1.append("Session  ", style="dim")
    line1.append(f"{ss.games_played}/{ss.target_games}", style="bold white")
    line1.append(" games", style="dim")
    line1.append("   draws ", style="dim"); line1.append(f"{ss.draws}", style=STYLE_DRAW)
    line1.append("   avg game ", style="dim")
    line1.append(_fmt_secs(ss.avg_game_time()), style="white")
    line1.append("   avg plies ", style="dim")
    line1.append(f"{ss.avg_plies():.1f}", style="white")
    line2 = Text()
    line2.append("elapsed ", style="dim")
    line2.append(_fmt_secs(time.monotonic() - ss.started_at), style="white")
    line2.append("   ETA ", style="dim")
    line2.append(_fmt_secs(ss.eta_seconds()), style="white")
    return Group(line1, line2)


def _status_line(state: Optional[State], search: Optional[Dict[str, Any]]) -> Text:
    t = Text()
    if state is not None:
        turn_style = STYLE_P1 if state.turn == 1 else STYLE_P2
        t.append("Turn ", style="dim"); t.append(f"P{state.turn}", style=turn_style)
        t.append("   Walls ", style="dim")
        t.append(f"P1={state.p1_walls_left}", style=STYLE_P1)
        t.append("  ", style="dim")
        t.append(f"P2={state.p2_walls_left}", style=STYLE_P2)
    if search is not None:
        t.append("   │  ", style="dim")
        who = search.get("who")
        who_style = STYLE_P1 if who == 1 else STYLE_P2
        t.append("thinking ", style="dim"); t.append(f"P{who}", style=who_style)
        d = search.get("depth")
        if d is not None:
            t.append("  d", style="dim"); t.append(str(d), style="white")
        score = search.get("score")
        if score is not None:
            t.append("  score=", style="dim"); t.append(f"{score:+d}", style="white")
        best = search.get("best")
        if best is not None:
            t.append("  best=", style="dim"); t.append(best, style="white")
        nodes = search.get("nodes")
        if nodes is not None:
            t.append("  nodes=", style="dim"); t.append(f"{nodes:,}", style="white")
        elapsed = search.get("elapsed")
        if elapsed is not None:
            t.append("  ", style="dim"); t.append(f"{elapsed:.2f}s", style="white")
    return t


# ---------------------------------------------------------------------------
# Compose the single Panel
# ---------------------------------------------------------------------------

def _build_view(
    state: Optional[State],
    board: Optional[Board],
    moves: List[Move],
    session: SessionStats,
    game_num: int,
    search: Optional[Dict[str, Any]],
    banner: Optional[Text],
) -> Panel:
    board_render = _board_renderable(state, board)
    grid = Table.grid(padding=(0, 3))
    grid.add_column(vertical="top", width=42)
    grid.add_column(vertical="top", width=28)
    grid.add_row(board_render, _history_renderable(moves))

    parts: List[Any] = [
        Align.center(grid),
        Text(""),
        Align.center(_status_line(state, search)),
        Text(""),
        Align.center(_player_stats_table(session)),
        Align.center(_session_lines(session)),
    ]
    if banner is not None:
        parts.append(Text(""))
        parts.append(Align.center(banner))

    title = f"[bold]Corridors — Game {game_num} of {session.target_games}[/bold]"
    return Panel(Group(*parts), title=title, border_style=STYLE_GRID, padding=(1, 2))


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------

@dataclass
class AutoplayParams:
    num_games: int
    starts: str
    p1_col: int
    p2_col: int
    depth: int
    time_limit: float
    tiebreak_epsilon: int
    max_plies: int
    workers: int
    headless: bool
    p1_agent: str = "classical"
    p2_agent: str = "classical"


def _auto_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _prompt_int(label: Text, default: int, lo: int, hi: int) -> int:
    from .nn.menu import SetupCancelled
    while True:
        raw = Prompt.ask(label, default=str(default))
        if raw.strip().lower() == "q":
            raise SetupCancelled
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        console.print(f"[red]  must be an integer {lo}..{hi} (or 'q' to cancel)[/red]")


def _prompt_float(label: Text, default: float, lo: float, hi: float) -> float:
    from .nn.menu import SetupCancelled
    while True:
        raw = Prompt.ask(label, default=f"{default:g}")
        if raw.strip().lower() == "q":
            raise SetupCancelled
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        console.print(f"[red]  must be a number {lo}..{hi} (or 'q' to cancel)[/red]")


def _prompt_col(player: int, endzone_row: int, default: int) -> int:
    style = STYLE_P1 if player == 1 else STYLE_P2
    default_letter = COL_LETTERS[default]
    while True:
        raw = Prompt.ask(
            Text.assemble(
                ("Player ", "dim"), (f"{player}", style),
                (f" starting column in row {endzone_row} [A-I]", "dim"),
            ),
            default=default_letter,
        )
        col = parse_col(raw)
        if col is None:
            console.print("[red]  bad column[/red]")
            continue
        return col


def _setup(cfg: dict, allow_neural: bool = True) -> AutoplayParams:
    console.clear()
    console.print(Panel(
        Text.assemble(
            ("Corridors — Autoplay\n", "bold"),
            ("Self-play mode. Configure a batch of AI-vs-AI games and watch them run.\n", "dim"),
            ("Ctrl-C at any time to stop.", "dim"),
        ),
        border_style=STYLE_GRID,
    ))

    num_games = _prompt_int(
        Text.assemble(("Number of games ", "dim"), ("[1-10000]", "dim")),
        default=int(cfg["num_games"]), lo=1, hi=10000,
    )

    starts_default = str(cfg.get("starts", "random")).lower()
    if starts_default not in ("fixed", "random"):
        starts_default = "random"
    starts = Prompt.ask(
        Text.assemble(("Starting columns ", "dim"), ("[fixed/random]", "dim")),
        choices=["fixed", "random"], default=starts_default,
    )

    if starts == "fixed":
        p1_col = _prompt_col(1, P1_END_ROW, default=int(cfg["p1_col"]))
        p2_col = _prompt_col(2, P2_END_ROW, default=int(cfg["p2_col"]))
    else:
        p1_col = int(cfg["p1_col"])
        p2_col = int(cfg["p2_col"])

    agent_choices = ["classical"]
    agent_modified = {}
    if allow_neural:
        from .nn.checkpoints import ranked_checkpoint_paths
        checkpoint_root = Path(__file__).resolve().parent.parent.parent / "nn_checkpoints"
        checkpoint_paths = ranked_checkpoint_paths(checkpoint_root)
        agent_choices.extend(f.stem for f in checkpoint_paths)
        agent_modified = {
            f.stem: time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)
            )
            for f in checkpoint_paths
        }

    def choose_agent(player: int) -> str:
        saved = str(cfg.get(f"p{player}_agent", "classical"))
        default = saved if saved in agent_choices else "classical"
        if len(agent_choices) == 1:
            return "classical"
        displayed = [
            choice if choice == "classical"
            else f"{choice} (modified {agent_modified[choice]})"
            for choice in agent_choices
        ]
        console.print(f"[dim]P{player} agents:[/dim] " + ", ".join(displayed))
        return Prompt.ask(f"[dim]Player {player} AI[/dim]",
                          choices=agent_choices, default=default)

    p1_agent = choose_agent(1)
    p2_agent = choose_agent(2)

    depth = int(cfg["depth"])
    time_limit = float(cfg["time_limit"])
    tiebreak = int(cfg["tiebreak_epsilon"])
    if "classical" in (p1_agent, p2_agent):
        depth = _prompt_int(
            Text.assemble(("Classical AI ", "dim"), ("depth", "bold"), (" [1-8]", "dim")),
            default=depth, lo=1, hi=8,
        )
        time_limit = _prompt_float(
            Text.assemble(("Classical AI ", "dim"), ("time limit", "bold"),
                          (" (s, 0 = none)", "dim")),
            default=time_limit, lo=0.0, hi=600.0,
        )
        tiebreak = _prompt_int(
            Text.assemble(("Classical tiebreak ε ", "dim"),
                          ("(0 = deterministic, higher = more variety)", "dim")),
            default=tiebreak, lo=0, hi=500,
        )
    max_plies = _prompt_int(
        Text.assemble(("Max plies per game ", "dim"), ("(draw threshold)", "dim")),
        default=int(cfg["max_plies"]), lo=20, hi=2000,
    )

    ncpu = os.cpu_count() or 2
    saved_workers = int(cfg.get("workers", 0))
    workers_default = saved_workers if saved_workers > 0 else _auto_workers()
    workers = _prompt_int(
        Text.assemble(
            ("Worker processes ", "dim"),
            (f"[1-{ncpu}] ({ncpu} cores detected)", "dim"),
        ),
        default=min(workers_default, ncpu), lo=1, hi=ncpu,
    )
    workers = min(workers, num_games)

    display_default = str(cfg.get("display", "live")).lower()
    if display_default not in ("live", "headless"):
        display_default = "live"
    display = Prompt.ask(
        Text.assemble(("Display ", "dim"),
                      ("[live = full dashboard, headless = one summary line]", "dim")),
        choices=["live", "headless"], default=display_default,
    )

    settings.save(
        num_games=num_games, starts=starts,
        p1_col=p1_col, p2_col=p2_col,
        depth=depth, time_limit=time_limit,
        tiebreak_epsilon=tiebreak, max_plies=max_plies,
        workers=workers, display=display,
        p1_agent=p1_agent, p2_agent=p2_agent,
    )
    return AutoplayParams(
        num_games=num_games, starts=starts, p1_col=p1_col, p2_col=p2_col,
        depth=depth, time_limit=time_limit,
        tiebreak_epsilon=tiebreak, max_plies=max_plies,
        workers=workers, headless=(display == "headless"),
        p1_agent=p1_agent, p2_agent=p2_agent,
    )


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _run_one_game(
    params: AutoplayParams,
    session: SessionStats,
    game_num: int,
    tt: Optional[solver.TT],
    live: Live,
    agents: Dict[int, object],
) -> None:
    if params.starts == "random":
        p1_col = random.randint(0, NCOLS - 1)
        p2_col = random.randint(0, NCOLS - 1)
    else:
        p1_col = params.p1_col
        p2_col = params.p2_col
    board, state = State.start(p1_col=p1_col, p2_col=p2_col, walls=WALLS_PER_PLAYER)
    moves: List[Move] = []
    states_seen: List[State] = [state]
    game_t0 = time.monotonic()

    banner = Text.assemble(
        ("start ", "dim"),
        (format_pos((P1_END_ROW, p1_col)), STYLE_P1),
        (" vs ", "dim"),
        (format_pos((P2_END_ROW, p2_col)), STYLE_P2),
    )
    live.update(_build_view(state, board, moves, session, game_num, None, banner))

    winner: Optional[int] = None
    plies = 0
    while True:
        w = state.winner(board)
        if w is not None:
            winner = w
            break
        if plies >= params.max_plies:
            winner = None
            break
        if is_threefold_repetition(states_seen):
            winner = None
            break

        search_state: Dict[str, Any] = {"who": state.turn}

        def on_iter(info: solver.IterInfo) -> None:
            search_state["depth"] = info.depth
            search_state["score"] = info.score
            search_state["best"] = format_move(info.best_move) if info.best_move else "-"
            search_state["nodes"] = info.nodes
            search_state["elapsed"] = info.elapsed
            live.update(_build_view(state, board, moves, session, game_num, search_state, banner))

        player_stats = session.p1 if state.turn == 1 else session.p2
        agent = params.p1_agent if state.turn == 1 else params.p2_agent
        if agent == "classical":
            tl = params.time_limit if params.time_limit > 0 else None
            mv, score, stats, _pv = solver.best_move(
                state, board,
                max_depth=params.depth, time_limit=tl,
                tiebreak_epsilon=params.tiebreak_epsilon,
                tt=tt, on_iteration=on_iter, verbose=False,
            )
            player_stats.record_turn(stats, score)
        else:
            think_t0 = time.monotonic()
            mv = agents[state.turn].pick_move(state, board)
            elapsed = time.monotonic() - think_t0
            score = 0
            player_stats.turns += 1
            player_stats.total_time += elapsed
        state = apply_move(state, mv)
        moves.append(mv)
        states_seen.append(state)
        plies += 1
        banner = Text.assemble(
            ("last ", "dim"),
            (f"P{3 - state.turn}",
             STYLE_P1 if 3 - state.turn == 1 else STYLE_P2),
            (" ", "dim"),
            (format_move(mv), STYLE_P1 if 3 - state.turn == 1 else STYLE_P2),
        )
        live.update(_build_view(state, board, moves, session, game_num, None, banner))

    game_time = time.monotonic() - game_t0
    session.record_game(winner, plies, game_time)

    if winner is None:
        finish_banner = Text.assemble(
            ("★ DRAW ★  ", STYLE_DRAW),
            ("plies=", "dim"), (f"{plies}", "white"),
            ("  time=", "dim"), (_fmt_secs(game_time), "white"),
        )
    else:
        style = STYLE_P1 if winner == 1 else STYLE_P2
        finish_banner = Text.assemble(
            ("★ P", style), (f"{winner}", style), (" wins ★  ", style),
            ("plies=", "dim"), (f"{plies}", "white"),
            ("  time=", "dim"), (_fmt_secs(game_time), "white"),
        )
    live.update(_build_view(state, board, moves, session, game_num, None, finish_banner))


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------

def _final_summary(session: SessionStats) -> Panel:
    body = Group(
        Text.assemble(
            ("Games played ", "dim"), (f"{session.games_played}", "bold white"),
            ("     P1 wins ", "dim"), (f"{session.p1.wins}", STYLE_P1),
            ("     P2 wins ", "dim"), (f"{session.p2.wins}", STYLE_P2),
            ("     draws ", "dim"), (f"{session.draws}", STYLE_DRAW),
        ),
        Text.assemble(
            ("Total time ", "dim"), (_fmt_secs(session.total_time), "white"),
            ("     avg game ", "dim"), (_fmt_secs(session.avg_game_time()), "white"),
            ("     avg plies ", "dim"), (f"{session.avg_plies():.1f}", "white"),
        ),
        Align.center(_player_stats_table(session)),
    )
    return Panel(body, title="[bold]Session complete[/bold]", border_style=STYLE_GRID)


def autoplay(params: AutoplayParams, tt: Optional[solver.TT]) -> None:
    session = SessionStats(target_games=params.num_games)
    agents = {}
    NetworkAgent = None
    for player, name in ((1, params.p1_agent), (2, params.p2_agent)):
        if name != "classical":
            if NetworkAgent is None:
                from .nn.agent import NetworkAgent
            agents[player] = NetworkAgent(name, device="cpu", seed=player)
    console.clear()
    initial = _build_view(None, None, [], session, 1, None, None)
    with Live(initial, console=console, refresh_per_second=10, screen=False, transient=False) as live:
        for i in range(params.num_games):
            _run_one_game(params, session, i + 1, tt, live, agents)
    console.print(_final_summary(session))


# ---------------------------------------------------------------------------
# Parallel session driver
# ---------------------------------------------------------------------------

@dataclass
class WorkerView:
    """Parent-side snapshot of one worker's progress."""
    games_assigned: int
    game_num: int = 0
    plies: int = 0
    state: Optional[State] = None
    board: Optional[Board] = None
    last_move: str = ""
    games_done: int = 0
    status: str = "starting"


_STATUS_GLYPH = {"starting": ("…", "yellow"), "playing": ("▶", "green"),
                 "done": ("✓", "dim"), "error": ("✗", STYLE_ERR)}


def _worker_table(views: Dict[int, WorkerView]) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim", padding=(0, 1),
              title="Workers", title_style="dim")
    t.add_column("", justify="right", style="dim", no_wrap=True)
    t.add_column("game", justify="right", no_wrap=True)
    t.add_column("ply", justify="right", no_wrap=True)
    t.add_column("last", no_wrap=True)
    t.add_column("", no_wrap=True)
    for wid in sorted(views):
        v = views[wid]
        glyph, style = _STATUS_GLYPH.get(v.status, ("?", "yellow"))
        cur = min(v.games_done + (1 if v.status == "playing" else 0), v.games_assigned)
        t.add_row(
            f"W{wid}",
            f"{cur}/{v.games_assigned}",
            f"{v.plies}",
            v.last_move,
            Text(glyph, style=style),
        )
    return t


def _build_parallel_view(
    views: Dict[int, WorkerView],
    featured_wid: Optional[int],
    session: SessionStats,
) -> Panel:
    fv = views.get(featured_wid) if featured_wid is not None else None
    state = fv.state if fv else None
    board = fv.board if fv else None

    grid = Table.grid(padding=(0, 3))
    grid.add_column(vertical="top", width=42)
    grid.add_column(vertical="top", width=30)
    board_title = Text(
        f"featured: W{featured_wid}" if featured_wid is not None else "featured: —",
        style="dim",
    )
    grid.add_row(Group(board_title, _board_renderable(state, board)), _worker_table(views))

    parts: List[Any] = [
        Align.center(grid),
        Text(""),
        Align.center(_player_stats_table(session)),
        Align.center(_session_lines(session)),
    ]
    title = (
        f"[bold]Corridors — Parallel autoplay — "
        f"{session.games_played}/{session.target_games} games — "
        f"{len(views)} workers[/bold]"
    )
    return Panel(Group(*parts), title=title, border_style=STYLE_GRID, padding=(1, 2))


def _spawn_workers(params: AutoplayParams, cfg: dict, report_moves: bool,
                   record_dataset: Optional[str] = None):
    from . import parallel

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    counts = parallel.split_games(params.num_games, params.workers)
    uses_classical = "classical" in (params.p1_agent, params.p2_agent)
    tt_path = (str(cfg["tt_path"])
               if uses_classical and bool(cfg.get("use_persistent_tt", True)) else None)

    shard_base = 0
    if record_dataset is not None:
        from .nn import datasets
        shard_base = datasets.next_shard_index(record_dataset)

    views: Dict[int, WorkerView] = {}
    procs: List[multiprocessing.Process] = []
    base_seed = random.randrange(1 << 30)
    for wid, n in enumerate(counts):
        if n == 0:
            continue
        wcfg = parallel.WorkerConfig(
            worker_id=wid, num_games=n,
            starts=params.starts, p1_col=params.p1_col, p2_col=params.p2_col,
            depth=params.depth, time_limit=params.time_limit,
            tiebreak_epsilon=params.tiebreak_epsilon, max_plies=params.max_plies,
            tt_path=tt_path, seed=base_seed + wid * 7919,
            report_moves=report_moves,
            record_dataset=record_dataset,
            record_shard_index=shard_base + wid,
            p1_agent=params.p1_agent, p2_agent=params.p2_agent,
        )
        p = ctx.Process(target=parallel.run_worker, args=(wcfg, q), daemon=True)
        p.start()
        procs.append(p)
        views[wid] = WorkerView(games_assigned=n)
    return q, views, procs


def autoplay_parallel(params: AutoplayParams, cfg: dict,
                      record_dataset: Optional[str] = None) -> None:
    q, views, procs = _spawn_workers(params, cfg, report_moves=True,
                                     record_dataset=record_dataset)
    shards: List[str] = []
    shard_positions = 0
    shard_games = 0

    session = SessionStats(target_games=params.num_games)
    active = set(views.keys())
    featured: Optional[int] = min(active) if active else None

    console.clear()
    interrupted = False
    try:
        with Live(_build_parallel_view(views, featured, session),
                  console=console, refresh_per_second=10, transient=False) as live:
            last_render = 0.0
            while active:
                try:
                    msg = q.get(timeout=0.25)
                except queue_mod.Empty:
                    msg = None
                if msg is not None:
                    kind = msg[0]
                    wid = msg[1]
                    v = views[wid]
                    if kind == "game_start":
                        _, _, game_num, _p1c, _p2c = msg
                        v.game_num = game_num
                        v.plies = 0
                        v.state = None
                        v.board = None
                        v.last_move = ""
                        v.status = "playing"
                    elif kind == "move":
                        (_, _, game_num, state, board, mv,
                         mover, score, elapsed, nodes, depth_reached) = msg
                        v.state = state
                        v.board = board
                        v.plies += 1
                        v.last_move = format_move(mv)
                        pstats = session.p1 if mover == 1 else session.p2
                        pstats.turns += 1
                        pstats.total_time += elapsed
                        pstats.total_nodes += nodes
                        pstats.total_score += max(-2000, min(2000, score))
                        if score > pstats.best_score_seen:
                            pstats.best_score_seen = score
                        pstats.depth_sum += depth_reached
                        if depth_reached > pstats.max_depth:
                            pstats.max_depth = depth_reached
                    elif kind == "game_end":
                        _, _, game_num, winner, plies, game_time, _gn, _tt = msg
                        v.games_done += 1
                        session.record_game(winner, plies, game_time)
                    elif kind == "shard":
                        _, _, shard_idx, positions, games = msg
                        shards.append(f"shard_{shard_idx:03d}.npz")
                        shard_positions += positions
                        shard_games += games
                    elif kind == "done":
                        v.status = "done"
                        active.discard(wid)
                        if featured == wid:
                            playing = [w for w in active if views[w].status == "playing"]
                            featured = min(playing) if playing else (min(active) if active else None)
                    elif kind == "error":
                        v.status = "error"
                        v.last_move = msg[2][:24]
                        active.discard(wid)
                        if featured == wid:
                            featured = min(active) if active else None
                now = time.monotonic()
                if now - last_render >= 0.1:
                    live.update(_build_parallel_view(views, featured, session))
                    last_render = now
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[yellow]interrupted — saving partial run…[/yellow]")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        # Drain any late shard/game_end messages the workers pushed before dying.
        while True:
            try:
                msg = q.get_nowait()
            except (queue_mod.Empty, OSError):
                break
            if msg[0] == "shard":
                _, _, shard_idx, positions, games = msg
                name = f"shard_{shard_idx:03d}.npz"
                if name not in shards:
                    shards.append(name)
                    shard_positions += positions
                    shard_games += games
            elif msg[0] == "game_end":
                _, _, _gnum, winner, plies, game_time, *_rest = msg
                session.record_game(winner, plies, game_time)
        # Save a manifest even on interrupt so the run shows up in the menus
        # with real counts instead of "?" (was the cause of "why don't my
        # classical runs appear" — Ctrl-C skipped register_run entirely).
        if record_dataset is not None and shards:
            from .nn import datasets
            dcfg = datasets.DatasetConfig(
                depth=params.depth, time_limit=params.time_limit,
                tiebreak_epsilon=params.tiebreak_epsilon,
                max_plies=params.max_plies, starts=params.starts,
            )
            mismatch = datasets.config_mismatch(record_dataset, dcfg)
            if mismatch:
                console.print(f"[yellow]warning: dataset config changed ({mismatch})[/yellow]")
            datasets.register_run(record_dataset, dcfg, shards, shard_games, shard_positions)
            console.print(
                f"[green]dataset '{record_dataset}': +{shard_positions:,} positions "
                f"from {shard_games} games in {len(shards)} shards"
                + (" (partial — run was interrupted)" if interrupted else "") + "[/green]"
            )

    if not interrupted:
        console.print(_final_summary(session))


# ---------------------------------------------------------------------------
# Headless driver (SSH-friendly: one summary line, no board)
# ---------------------------------------------------------------------------

def _headless_line(
    views: Dict[int, WorkerView],
    session: SessionStats,
    total_nodes: int,
    total_think: float,
) -> str:
    playing = sum(1 for v in views.values() if v.status == "playing")
    done = sum(1 for v in views.values() if v.status == "done")
    errors = sum(1 for v in views.values() if v.status == "error")
    elapsed = time.monotonic() - session.started_at
    games_per_hour = session.games_played / elapsed * 3600 if elapsed > 0 else 0.0
    nps = int(total_nodes / total_think) if total_think > 0 else 0
    parts = [
        f"games {session.games_played}/{session.target_games}",
        f"P1 {session.p1.wins} P2 {session.p2.wins} D {session.draws}",
        f"workers {playing}>" + (f" {errors}!" if errors else ""),
        f"{games_per_hour:.0f} games/h",
        f"avg {session.avg_game_time():.1f}s/game",
        f"plies {session.avg_plies():.0f}",
        f"{nps:,} nps/worker",
        f"elapsed {_fmt_secs(elapsed)}",
        f"ETA {_fmt_secs(session.eta_seconds())}",
    ]
    _ = done
    return "  |  ".join(parts)


def autoplay_headless(params: AutoplayParams, cfg: dict,
                      record_dataset: Optional[str] = None) -> None:
    q, views, procs = _spawn_workers(params, cfg, report_moves=False,
                                     record_dataset=record_dataset)
    session = SessionStats(target_games=params.num_games)
    active = set(views.keys())
    total_nodes = 0
    total_think = 0.0
    shards: List[str] = []
    shard_positions = 0
    shard_games = 0
    is_tty = console.is_terminal
    REPORT_EVERY = 1.0 if is_tty else 10.0
    last_report = 0.0

    def _emit(final: bool = False) -> None:
        line = _headless_line(views, session, total_nodes, total_think)
        if is_tty and not final:
            # \r-overwrite a single status line; pad to clear leftovers.
            sys.stdout.write("\r" + line.ljust(158)[:158])
            sys.stdout.flush()
        else:
            if is_tty:
                sys.stdout.write("\r" + " " * 158 + "\r")
            print(line, flush=True)

    print(
        f"corridors autoplay: {params.num_games} games, {len(views)} workers, "
        f"P1 {params.p1_agent}, P2 {params.p2_agent}, "
        f"classical depth {params.depth}, time {params.time_limit:g}s, "
        f"starts {params.starts}, eps {params.tiebreak_epsilon}",
        flush=True,
    )
    interrupted = False
    try:
        while active:
            try:
                msg = q.get(timeout=0.5)
            except queue_mod.Empty:
                msg = None
            if msg is not None:
                kind = msg[0]
                wid = msg[1]
                v = views[wid]
                if kind == "game_start":
                    v.status = "playing"
                    v.game_num = msg[2]
                elif kind == "game_end":
                    _, _, _gnum, winner, plies, game_time, game_nodes, think_time = msg
                    v.games_done += 1
                    session.record_game(winner, plies, game_time)
                    total_nodes += game_nodes
                    total_think += think_time
                elif kind == "shard":
                    _, _, shard_idx, positions, games = msg
                    shards.append(f"shard_{shard_idx:03d}.npz")
                    shard_positions += positions
                    shard_games += games
                elif kind == "done":
                    v.status = "done"
                    active.discard(wid)
                elif kind == "error":
                    v.status = "error"
                    active.discard(wid)
                    print(f"\nworker {wid} error: {msg[2]}", flush=True)
            now = time.monotonic()
            if now - last_report >= REPORT_EVERY:
                _emit()
                last_report = now
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted — saving partial run…", flush=True)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        # Drain any late shard/game_end messages the workers pushed before dying,
        # so an interrupted run still records what actually landed on disk.
        while True:
            try:
                msg = q.get_nowait()
            except (queue_mod.Empty, OSError):
                break
            if msg[0] == "shard":
                _, _, shard_idx, positions, games = msg
                name = f"shard_{shard_idx:03d}.npz"
                if name not in shards:
                    shards.append(name)
                    shard_positions += positions
                    shard_games += games
            elif msg[0] == "game_end":
                _, _, _gnum, winner, plies, game_time, *_rest = msg
                session.record_game(winner, plies, game_time)
        # Register the run whether we finished cleanly OR were interrupted —
        # writes a manifest so the dataset shows up in Train/View Datasets and
        # its games/positions counts are real, not "?".
        if record_dataset is not None and shards:
            from .nn import datasets
            dcfg = datasets.DatasetConfig(
                depth=params.depth, time_limit=params.time_limit,
                tiebreak_epsilon=params.tiebreak_epsilon,
                max_plies=params.max_plies, starts=params.starts,
            )
            mismatch = datasets.config_mismatch(record_dataset, dcfg)
            if mismatch:
                print(f"warning: dataset config changed ({mismatch})", flush=True)
            datasets.register_run(record_dataset, dcfg, shards, shard_games, shard_positions)
            print(f"dataset '{record_dataset}': +{shard_positions:,} positions "
                  f"from {shard_games} games in {len(shards)} shards"
                  + (" (partial — run was interrupted)" if interrupted else ""),
                  flush=True)

    _emit(final=True)
    if not interrupted:
        print(
            f"complete: {session.games_played} games in "
            f"{_fmt_secs(time.monotonic() - session.started_at)}"
            f"  (aggregate game time {_fmt_secs(session.total_time)})",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_autoplay() -> None:
    cfg = settings.load()
    params = _setup(cfg)
    _dispatch_autoplay(params, cfg)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        while True:
            table = Table(box=None, show_header=False, pad_edge=False)
            table.add_column(style="bold")
            table.add_column()
            table.add_row("1", "Autoplay (AI vs AI)")
            table.add_row("2", "Neural network training")
            table.add_row("q", "Quit")
            console.print(Panel(table, title="[bold]Corridors[/bold]", border_style=STYLE_GRID))
            choice = Prompt.ask("Choose", choices=["1", "2", "q"], default="1")
            if choice == "1":
                try:
                    _run_autoplay()
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[dim]interrupted.[/dim]")
            elif choice == "2":
                from .nn.menu import nn_menu
                nn_menu()
            elif choice == "q":
                return 0
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye.[/dim]")
        return 0


def _dispatch_autoplay(params: AutoplayParams, cfg: dict) -> None:
    if params.headless:
        # Headless always uses the worker-process driver, even for one worker.
        try:
            autoplay_headless(params, cfg)
        except (KeyboardInterrupt, EOFError):
            print("\ninterrupted.", flush=True)
        return

    if params.workers > 1:
        try:
            autoplay_parallel(params, cfg)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]interrupted.[/dim]")
        return

    tt: Optional[solver.TT] = None
    uses_classical = "classical" in (params.p1_agent, params.p2_agent)
    if uses_classical and bool(cfg.get("use_persistent_tt", True)):
        try:
            tt = solver.TT(sqlite_path=str(cfg["tt_path"]))
        except Exception as e:
            console.print(f"[dim](TT disabled: {e})[/dim]")
            tt = None

    try:
        autoplay(params, tt)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]interrupted.[/dim]")
    finally:
        if tt is not None:
            tt.close()


if __name__ == "__main__":
    raise SystemExit(main())
