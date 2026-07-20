"""Alpha-beta solver for Corridors.

Iterative deepening + principal variation search (PVS) + aspiration windows,
Zobrist-hashed transposition table, TT/PV/killer/history move ordering with
strategic bias, selective extensions on jump moves, anti-loop child-hash
tracking, tiebreak-epsilon at the root for self-play variety.
"""

from __future__ import annotations

import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from . import game
from .game import (
    ALL_WALLS,
    Board,
    Move,
    Pos,
    State,
    Wall,
    apply_move,
    blocked_mask_for,
    legal_moves,
)

WIN = 1_000_000
LOSE = -WIN
INF = WIN * 2
MATE_MARGIN = 10_000  # scores with |s| >= WIN - MATE_MARGIN are mate scores

EXACT, LOWER, UPPER = 0, 1, 2

TT_NAMESPACE = "corridors-tt-v3"


# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------

_RNG = random.Random(0xC0FF33_C0DE)
_Z_P1 = [_RNG.getrandbits(64) for _ in range(game.NROWS * game.NCOLS)]
_Z_P2 = [_RNG.getrandbits(64) for _ in range(game.NROWS * game.NCOLS)]
_Z_WALL = {w: _RNG.getrandbits(64) for w in ALL_WALLS}
_Z_WALLS_LEFT = [
    [_RNG.getrandbits(64) for _ in range(game.WALLS_PER_PLAYER + 1)] for _ in range(2)
]
_Z_TURN = _RNG.getrandbits(64)
_Z_GOAL_P1 = [_RNG.getrandbits(64) for _ in range(game.NCOLS)]
_Z_GOAL_P2 = [_RNG.getrandbits(64) for _ in range(game.NCOLS)]


def _pos_idx(p: Pos) -> int:
    return p[0] * game.NCOLS + p[1]


def zobrist(state: State, board: Board) -> int:
    h = _Z_P1[_pos_idx(state.p1)] ^ _Z_P2[_pos_idx(state.p2)]
    for w in state.walls:
        h ^= _Z_WALL[w]
    h ^= _Z_WALLS_LEFT[0][state.p1_walls_left]
    h ^= _Z_WALLS_LEFT[1][state.p2_walls_left]
    if state.turn == 2:
        h ^= _Z_TURN
    h ^= _Z_GOAL_P1[board.p1_goal[1]]
    h ^= _Z_GOAL_P2[board.p2_goal[1]]
    return h


# ---------------------------------------------------------------------------
# Transposition table (memory + optional SQLite backing)
# ---------------------------------------------------------------------------

@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int
    best: Optional[Move]


_SQLITE_INT_MASK = 0x7FFFFFFFFFFFFFFF  # 63 bits — fits sqlite signed INTEGER


def _serialize_move(mv: Optional[Move]) -> str:
    if mv is None:
        return ""
    kind, arg = mv
    if kind == "m":
        return f"m{arg[0]},{arg[1]}"
    return f"w{arg[0]},{arg[1]},{arg[2]}"


def _deserialize_move(s: str) -> Optional[Move]:
    if not s:
        return None
    head = s[0]
    body = s[1:]
    if head == "m":
        r, c = body.split(",", 1)
        return ("m", (int(r), int(c)))
    if head == "w":
        r, c, o = body.split(",", 2)
        return ("w", (int(r), int(c), o))
    return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tt_entries (
    namespace TEXT NOT NULL,
    zkey INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    score INTEGER NOT NULL,
    flag INTEGER NOT NULL,
    best_move TEXT,
    PRIMARY KEY (namespace, zkey)
) WITHOUT ROWID;
"""

_UPSERT = """
INSERT INTO tt_entries (namespace, zkey, depth, score, flag, best_move)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(namespace, zkey) DO UPDATE SET
    depth=excluded.depth, score=excluded.score,
    flag=excluded.flag, best_move=excluded.best_move
WHERE excluded.depth >= tt_entries.depth
"""


class TT:
    """Two-tier transposition table.

    In-memory dict is authoritative during a search. Optional SQLite backing
    stores entries across game sessions, batched at flush() time. Probes fall
    back to disk on miss and promote to memory.
    """

    def __init__(
        self,
        namespace: str = TT_NAMESPACE,
        sqlite_path: Optional[str] = None,
    ) -> None:
        self.namespace = namespace
        self.table: Dict[int, TTEntry] = {}
        self._pending: List[TTEntry] = []
        self._pending_keys: Set[int] = set()
        self.conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self.probes = 0
        self.mem_hits = 0
        self.disk_promotions = 0
        if sqlite_path:
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(
                sqlite_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit; we control transactions via BEGIN/COMMIT
            )
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            # Multiple worker processes share this file; wait instead of erroring
            # when another writer holds the lock.
            self.conn.execute("PRAGMA busy_timeout=5000")
            # Memory-map the DB. Each worker's read hits kernel-mapped pages, so
            # the OS shares one copy of the file across ALL worker processes via
            # the page cache — instead of every worker duplicating the hot set
            # in its own Python dict as it warms up. Cap at 2 GB (well above
            # current DB size) — the kernel maps lazily, so we only pay for
            # pages actually touched. Cheap; wins big with many workers.
            self.conn.execute("PRAGMA mmap_size=2147483648")
            self.conn.executescript(_SCHEMA)

    def probe(self, key: int) -> Optional[TTEntry]:
        self.probes += 1
        e = self.table.get(key)
        if e is not None:
            self.mem_hits += 1
            return e
        if self.conn is None:
            return None
        row = self.conn.execute(
            "SELECT depth, score, flag, best_move FROM tt_entries WHERE namespace=? AND zkey=?",
            (self.namespace, key & _SQLITE_INT_MASK),
        ).fetchone()
        if row is None:
            return None
        depth, score, flag, bm_s = row
        entry = TTEntry(key=key, depth=depth, score=score, flag=flag, best=_deserialize_move(bm_s))
        self.table[key] = entry
        self.disk_promotions += 1
        return entry

    def store(self, entry: TTEntry) -> None:
        old = self.table.get(entry.key)
        if old is not None and entry.depth < old.depth:
            return
        self.table[entry.key] = entry
        if self.conn is not None:
            # Coalesce: keep only the deepest pending entry per key.
            if entry.key in self._pending_keys:
                for i, p in enumerate(self._pending):
                    if p.key == entry.key:
                        self._pending[i] = entry
                        return
            self._pending.append(entry)
            self._pending_keys.add(entry.key)

    def flush(self) -> int:
        """Write pending entries to sqlite. Returns number of rows attempted."""
        if self.conn is None or not self._pending:
            return 0
        rows = [
            (self.namespace, e.key & _SQLITE_INT_MASK, e.depth, e.score, e.flag,
             _serialize_move(e.best))
            for e in self._pending
        ]
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.executemany(_UPSERT, rows)
                self.conn.execute("COMMIT")
            except sqlite3.DatabaseError:
                try:
                    self.conn.execute("ROLLBACK")
                except sqlite3.DatabaseError:
                    pass
                # Cross-process contention is non-fatal: keep entries pending and
                # let a later flush retry.
                return 0
        n = len(self._pending)
        self._pending.clear()
        self._pending_keys.clear()
        return n

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self.conn is not None:
                self.conn.close()
                self.conn = None

    def stats_snapshot(self) -> Dict[str, int]:
        return {
            "mem_entries": len(self.table),
            "probes": self.probes,
            "mem_hits": self.mem_hits,
            "disk_promotions": self.disk_promotions,
            "pending_writes": len(self._pending),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

WEIGHT_DIST = 100
WEIGHT_WALLS = 6
WEIGHT_ALT = 2


def evaluate(state: State, board: Board) -> int:
    mask = blocked_mask_for(state.walls)
    d1, alt1 = game.dist_and_alt(state.p1, board.p1_goal, mask)
    d2, alt2 = game.dist_and_alt(state.p2, board.p2_goal, mask)
    walls_diff = state.p1_walls_left - state.p2_walls_left
    # from P1's perspective: P1 wants small d1, opp large d2
    score = WEIGHT_DIST * (d2 - d1) + WEIGHT_WALLS * walls_diff + WEIGHT_ALT * (alt1 - alt2)
    return score if state.turn == 1 else -score


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

def _order_moves(
    state: State,
    board: Board,
    moves: List[Move],
    tt_move: Optional[Move],
    killers: Tuple[Optional[Move], Optional[Move]],
    history: Dict[Move, int],
) -> List[Move]:
    mask = blocked_mask_for(state.walls)
    me = state.p1 if state.turn == 1 else state.p2
    opp = state.p2 if state.turn == 1 else state.p1
    my_goal = board.p1_goal if state.turn == 1 else board.p2_goal
    opp_goal = board.p2_goal if state.turn == 1 else board.p1_goal
    my_dist = game.dist_reader(my_goal, mask)
    my_dist_now = my_dist(me)
    opp_path_mask = game.shortest_path_mask(opp, opp_goal, mask)

    def score(mv: Move) -> int:
        if mv == tt_move:
            return 10 ** 9
        if mv == killers[0]:
            return 10 ** 8
        if mv == killers[1]:
            return 10 ** 8 - 1
        s = history.get(mv, 0)
        if mv[0] == "m":
            # Pawn moves: prefer advancing along shortest path.
            new_pos = mv[1]
            new_d = my_dist(new_pos)
            s += 500 * (my_dist_now - new_d)
            # Jumps get a modest bonus (selective-extension candidates).
            dr = abs(new_pos[0] - me[0])
            dc = abs(new_pos[1] - me[1])
            if dr + dc >= 2:
                s += 200
        else:
            # Walls: prefer walls that touch opponent's shortest path.
            w = mv[1]
            if game._WALL_BITMASK[w] & opp_path_mask:
                s += 300
        return s

    return sorted(moves, key=score, reverse=True)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class SearchStats:
    nodes: int = 0
    tt_hits: int = 0
    cutoffs: int = 0
    max_depth: int = 0
    elapsed: float = 0.0


class _TimeUp(Exception):
    pass


class Searcher:
    def __init__(self, board: Board, tt: Optional[TT] = None) -> None:
        self.board = board
        self.tt = tt or TT()
        self.killers: Dict[int, List[Optional[Move]]] = {}
        self.history: Dict[Move, int] = {}
        self.stats = SearchStats()
        self.deadline: Optional[float] = None
        self.avoid_child_hashes: Set[int] = set()

    def _check_time(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise _TimeUp

    def _killers_at(self, ply: int) -> Tuple[Optional[Move], Optional[Move]]:
        ks = self.killers.get(ply)
        if not ks:
            return (None, None)
        return (ks[0] if len(ks) > 0 else None, ks[1] if len(ks) > 1 else None)

    def _push_killer(self, ply: int, mv: Move) -> None:
        ks = self.killers.setdefault(ply, [None, None])
        if ks[0] == mv:
            return
        ks[1] = ks[0]
        ks[0] = mv

    def negamax(self, state: State, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.stats.nodes += 1
        if self.stats.nodes % 1024 == 0:
            self._check_time()

        win = state.winner(self.board)
        if win is not None:
            # Side-to-move sees a lost position (opponent just moved to their goal).
            return LOSE + ply

        if depth <= 0:
            return evaluate(state, self.board)

        key = zobrist(state, self.board)
        tt_move: Optional[Move] = None
        entry = self.tt.probe(key)
        if entry is not None and entry.key == key:
            tt_move = entry.best
            if entry.depth >= depth:
                self.stats.tt_hits += 1
                s = entry.score
                if entry.flag == EXACT:
                    return s
                if entry.flag == LOWER and s >= beta:
                    return s
                if entry.flag == UPPER and s <= alpha:
                    return s

        moves = legal_moves(state, self.board)
        if not moves:
            return LOSE + ply

        moves = _order_moves(state, self.board, moves, tt_move, self._killers_at(ply), self.history)

        orig_alpha = alpha
        best = -INF
        best_move: Optional[Move] = None
        first = True
        for mv in moves:
            child = apply_move(state, mv)

            # Selective extension: jump moves probe one ply deeper.
            ext = 0
            if mv[0] == "m":
                me = state.p1 if state.turn == 1 else state.p2
                dr = abs(mv[1][0] - me[0])
                dc = abs(mv[1][1] - me[1])
                if dr + dc >= 2:
                    ext = 1

            if first:
                score = -self.negamax(child, depth - 1 + ext, -beta, -alpha, ply + 1)
            else:
                score = -self.negamax(child, depth - 1 + ext, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self.negamax(child, depth - 1 + ext, -beta, -score, ply + 1)
            first = False

            if score > best:
                best = score
                best_move = mv
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self.stats.cutoffs += 1
                self._push_killer(ply, mv)
                self.history[mv] = self.history.get(mv, 0) + depth * depth
                break

        flag = EXACT
        if best <= orig_alpha:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        self.tt.store(TTEntry(key, depth, best, flag, best_move))
        return best

    def root_search(
        self,
        state: State,
        depth: int,
        prev_score: int,
        aspiration: int = 40,
    ) -> Tuple[int, Optional[Move], List[Tuple[Move, int]]]:
        """One iteration of iterative deepening with aspiration windows.
        Returns (best_score, best_move, [(move, score), ...] for all root moves)."""
        moves = legal_moves(state, self.board)
        if not moves:
            return LOSE, None, []
        key = zobrist(state, self.board)
        entry = self.tt.probe(key)
        tt_move = entry.best if entry else None
        moves = _order_moves(state, self.board, moves, tt_move, self._killers_at(0), self.history)

        alpha = prev_score - aspiration
        beta = prev_score + aspiration
        while True:
            root_scores: List[Tuple[Move, int]] = []
            best = -INF
            best_move: Optional[Move] = None
            a = alpha
            first = True
            for mv in moves:
                child = apply_move(state, mv)
                # Anti-loop: avoid_child_hashes are positions seen >=2 times in self-play.
                if zobrist(child, self.board) in self.avoid_child_hashes and len(moves) > 1:
                    root_scores.append((mv, LOSE + 1))
                    continue
                if first:
                    s = -self.negamax(child, depth - 1, -beta, -a, 1)
                else:
                    s = -self.negamax(child, depth - 1, -a - 1, -a, 1)
                    if a < s < beta:
                        s = -self.negamax(child, depth - 1, -beta, -s, 1)
                first = False
                root_scores.append((mv, s))
                if s > best:
                    best = s
                    best_move = mv
                if best > a:
                    a = best
                # No beta cutoff at root: we want scores for every root move.
            if best <= alpha:
                if alpha <= -INF + 1:
                    self.tt.store(TTEntry(key, depth, best, UPPER, best_move))
                    return best, best_move, root_scores
                alpha -= aspiration * 4
                aspiration *= 3
                continue
            if best >= beta:
                if beta >= INF - 1:
                    self.tt.store(TTEntry(key, depth, best, LOWER, best_move))
                    return best, best_move, root_scores
                beta += aspiration * 4
                aspiration *= 3
                continue
            self.tt.store(TTEntry(key, depth, best, EXACT, best_move))
            return best, best_move, root_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_mate_score(s: int) -> bool:
    return abs(s) >= WIN - MATE_MARGIN


def pv_from_tt(state: State, board: Board, tt: TT, max_len: int = 8) -> List[Move]:
    """Follow TT best-moves from state to reconstruct the principal variation."""
    pv: List[Move] = []
    seen: Set[int] = set()
    cur = state
    while len(pv) < max_len:
        if cur.winner(board) is not None:
            break
        key = zobrist(cur, board)
        if key in seen:
            break
        seen.add(key)
        entry = tt.probe(key)
        if entry is None or entry.best is None:
            break
        if entry.best not in legal_moves(cur, board):
            break
        pv.append(entry.best)
        cur = apply_move(cur, entry.best)
    return pv


@dataclass
class IterInfo:
    depth: int
    score: int
    best_move: Optional[Move]
    pv: List[Move]
    root_scores: List[Tuple[Move, int]]
    nodes: int
    tt_hits: int
    cutoffs: int
    tt_size: int
    elapsed: float


def best_move(
    state: State,
    board: Board,
    max_depth: int = 4,
    time_limit: Optional[float] = None,
    tiebreak_epsilon: int = 15,
    tt: Optional[TT] = None,
    sqlite_path: Optional[str] = None,
    avoid_child_hashes: Optional[Set[int]] = None,
    verbose: bool = False,
    on_iteration: Optional[Callable[[IterInfo], None]] = None,
    flush_on_exit: bool = True,
) -> Tuple[Move, int, SearchStats, List[Move]]:
    """Search state and return (move, score, stats, pv). Score is side-to-move perspective.

    If `tt` is None and `sqlite_path` is provided, a new TT backed by that db is
    created for this call. `flush_on_exit` writes pending TT entries to disk on
    return; pass False when the caller intends to flush later.
    """
    if tt is None and sqlite_path is not None:
        tt = TT(sqlite_path=sqlite_path)
    searcher = Searcher(board, tt=tt)
    if avoid_child_hashes:
        searcher.avoid_child_hashes = avoid_child_hashes
    if time_limit is not None:
        searcher.deadline = time.monotonic() + time_limit

    prev_score = 0
    best_mv: Optional[Move] = None
    last_root_scores: List[Tuple[Move, int]] = []
    last_pv: List[Move] = []
    t0 = time.monotonic()

    try:
        for d in range(1, max_depth + 1):
            score, mv, root_scores = searcher.root_search(state, d, prev_score)
            if mv is not None:
                best_mv = mv
                prev_score = score
                last_root_scores = root_scores
                searcher.stats.max_depth = d
            elapsed = time.monotonic() - t0
            pv = pv_from_tt(state, board, searcher.tt)
            last_pv = pv
            if verbose:
                print(f"  depth {d}: score={score:+d} move={_pretty_move(mv)} nodes={searcher.stats.nodes}")
            if on_iteration is not None:
                on_iteration(IterInfo(
                    depth=d,
                    score=score,
                    best_move=mv,
                    pv=pv,
                    root_scores=root_scores,
                    nodes=searcher.stats.nodes,
                    tt_hits=searcher.stats.tt_hits,
                    cutoffs=searcher.stats.cutoffs,
                    tt_size=len(searcher.tt.table),
                    elapsed=elapsed,
                ))
            if is_mate_score(score):
                break
    except _TimeUp:
        if verbose:
            print("  (time up)")

    searcher.stats.elapsed = time.monotonic() - t0

    if best_mv is None:
        moves = legal_moves(state, board)
        if not moves:
            raise RuntimeError("no legal moves")
        best_mv = moves[0]
        prev_score = evaluate(state, board)

    if last_root_scores and tiebreak_epsilon > 0:
        top = max(s for _, s in last_root_scores)
        band = [m for m, s in last_root_scores if s >= top - tiebreak_epsilon]
        if band:
            best_mv = random.choice(band)

    if flush_on_exit and searcher.tt.conn is not None:
        try:
            searcher.tt.flush()
        except sqlite3.DatabaseError:
            pass

    return best_mv, prev_score, searcher.stats, last_pv


def _pretty_move(mv: Optional[Move]) -> str:
    if mv is None:
        return "-"
    if mv[0] == "m":
        r, c = mv[1]
        return f"m {r},{c}"
    r, c, o = mv[1]
    return f"w {r},{c},{o}"
