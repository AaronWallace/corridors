"""Corridors game engine.

Board is 11 rows x 9 cols. Rows 1-9 form the 9x9 playable area; row 0 and row 10
are end-zone strips. Each end-zone cell connects only to the playable cell in
the same column (single edge); end-zone cells never connect to each other.

Each player picks a starting cell in their own end zone; that cell becomes the
opponent's goal. A pawn may leave its own end zone only once, may never return
to it, and may enter the opponent's end zone only through that exact goal cell.

Walls occupy an interior 8x8 grid of slots (rows 1..8, cols 0..7) with H or V
orientation, each blocking two edges. Walls never sit on the goal-entry edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

NROWS = 11
NCOLS = 9
P1_END_ROW = 10
P2_END_ROW = 0
PLAY_MIN = 1
PLAY_MAX = 9
WALLS_PER_PLAYER = 9
REPETITIONS_FOR_DRAW = 3

Pos = Tuple[int, int]
Wall = Tuple[int, int, str]
Move = Tuple[str, object]

def valid(r: int, c: int) -> bool:
    return 0 <= r < NROWS and 0 <= c < NCOLS

def is_playable(r: int, c: int) -> bool:
    return PLAY_MIN <= r <= PLAY_MAX and 0 <= c < NCOLS

def is_endzone(r: int, c: int) -> bool:
    return (r == P1_END_ROW or r == P2_END_ROW) and 0 <= c < NCOLS


def _edge_key(a: Pos, b: Pos) -> Tuple[Pos, Pos]:
    return (a, b) if a <= b else (b, a)


_EDGE_BIT: Dict[Tuple[Pos, Pos], int] = {}
_ADJ: Dict[Pos, Tuple[Pos, ...]] = {}


def _init_edges() -> None:
    adj: Dict[Pos, List[Pos]] = {}
    for r in range(NROWS):
        for c in range(NCOLS):
            adj[(r, c)] = []
    bit = 0
    # Playable-area edges (blockable by walls).
    for r in range(PLAY_MIN, PLAY_MAX + 1):
        for c in range(NCOLS):
            a = (r, c)
            if r + 1 <= PLAY_MAX:
                b = (r + 1, c)
                _EDGE_BIT[_edge_key(a, b)] = bit
                bit += 1
                adj[a].append(b)
                adj[b].append(a)
            if c + 1 < NCOLS:
                b = (r, c + 1)
                _EDGE_BIT[_edge_key(a, b)] = bit
                bit += 1
                adj[a].append(b)
                adj[b].append(a)
    # End-zone entry edges (unblockable, no bit assigned).
    for c in range(NCOLS):
        for endrow, playrow in ((P2_END_ROW, PLAY_MIN), (P1_END_ROW, PLAY_MAX)):
            adj[(endrow, c)].append((playrow, c))
            adj[(playrow, c)].append((endrow, c))
    for k in adj:
        _ADJ[k] = tuple(adj[k])


_init_edges()
NUM_EDGE_BITS = len(_EDGE_BIT)


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# (a, b) -> bitmask of the blockable edge between adjacent cells a and b
# (0 for end-zone entry edges, which walls can never block). Both orderings
# are present, so a single dict get replaces valid() + _edge_open().
_EDGE_MASK: Dict[Tuple[Pos, Pos], int] = {}
# cell -> OR of the blockable-edge masks of all edges touching that cell.
_CELL_EDGE_MASK: Dict[Pos, int] = {}


def _init_edge_masks() -> None:
    for a, neighbors in _ADJ.items():
        cell_mask = 0
        for b in neighbors:
            bit = _EDGE_BIT.get(_edge_key(a, b))
            mask = 0 if bit is None else 1 << bit
            _EDGE_MASK[(a, b)] = mask
            cell_mask |= mask
        _CELL_EDGE_MASK[a] = cell_mask


_init_edge_masks()


def _wall_edges(w: Wall) -> Tuple[Tuple[Pos, Pos], Tuple[Pos, Pos]]:
    r, c, o = w
    if o == "H":
        return _edge_key((r, c), (r + 1, c)), _edge_key((r, c + 1), (r + 1, c + 1))
    if o == "V":
        return _edge_key((r, c), (r, c + 1)), _edge_key((r + 1, c), (r + 1, c + 1))
    raise ValueError(f"bad orient: {o}")


_WALL_BITMASK: Dict[Wall, int] = {}
_WALL_CONFLICTS: Dict[Wall, FrozenSet[Wall]] = {}
_ALL_WALLS: List[Wall] = []


def _init_walls() -> None:
    for r in range(PLAY_MIN, PLAY_MAX):
        for c in range(NCOLS - 1):
            for o in ("H", "V"):
                w = (r, c, o)
                e1, e2 = _wall_edges(w)
                _WALL_BITMASK[w] = (1 << _EDGE_BIT[e1]) | (1 << _EDGE_BIT[e2])
                _ALL_WALLS.append(w)
    for w1 in _ALL_WALLS:
        b1 = _WALL_BITMASK[w1]
        r1, c1, o1 = w1
        conflicts = set()
        for w2 in _ALL_WALLS:
            if w1 == w2:
                continue
            r2, c2, o2 = w2
            if _WALL_BITMASK[w2] & b1:
                conflicts.add(w2)
            elif r1 == r2 and c1 == c2 and o1 != o2:
                conflicts.add(w2)
        _WALL_CONFLICTS[w1] = frozenset(conflicts)


_init_walls()
ALL_WALLS: Tuple[Wall, ...] = tuple(_ALL_WALLS)
# (wall, edge-bitmask) pairs in ALL_WALLS order — avoids a dict lookup per
# candidate in the legal_wall_moves hot loop.
_WALL_MASK_ITEMS: Tuple[Tuple[Wall, int], ...] = tuple(
    (w, _WALL_BITMASK[w]) for w in ALL_WALLS)


def blocked_mask_for(walls) -> int:
    m = 0
    for w in walls:
        m |= _WALL_BITMASK[w]
    return m


# NOTE: This cache is process-wide (module-level lru_cache), so it persists
# for the entire lifetime of a self-play worker across all iterations. Each
# entry is a Dict[Pos, int] of ~99 cells for our 11x9 board and costs ~1.66 KB
# (measured). At the old maxsize=200_000 that's ~6.5 GB per worker; with 200+
# workers running a long loop, workers steadily filled the cache and OOM'd the
# host despite MALLOC_TRIM (memory was genuinely referenced by lru_cache, not
# arena fragmentation). At 5,000 entries the cache is capped at ~8 MB per
# worker (~2 GB across 251 workers) while still catching most within-game
# transpositions — MCTS bursts >5k unique masks per game are rare and LRU
# eviction covers them.
@lru_cache(maxsize=5_000)
def _dist_table_from(goal: Pos, blocked_mask: int) -> Dict[Pos, int]:
    dist: Dict[Pos, int] = {goal: 0}
    frontier = [goal]
    while frontier:
        nxt = []
        for cell in frontier:
            d = dist[cell] + 1
            for nb in _ADJ[cell]:
                if nb in dist:
                    continue
                bit = _EDGE_BIT.get(_edge_key(cell, nb))
                if bit is not None and (blocked_mask >> bit) & 1:
                    continue
                dist[nb] = d
                nxt.append(nb)
        frontier = nxt
    return dist


# goal -> per-cell neighbor tuples (nb_idx, nb_bit, edge_mask), ordered
# farthest-from-goal first so a LIFO stack pops the most goal-ward neighbor
# first. Built lazily per goal cell (at most 99 goals, tiny tables).
_GOAL_ORDERED_NEIGHBORS: Dict[Pos, Tuple[Tuple[Tuple[int, int, int], ...], ...]] = {}


def _neighbors_ordered_for(goal: Pos) -> Tuple[Tuple[Tuple[int, int, int], ...], ...]:
    dist = _dist_table_from(goal, 0)
    far = max(dist.values()) + 1
    table = []
    for r in range(NROWS):
        for c in range(NCOLS):
            cell = (r, c)
            entries = []
            for nb in _ADJ[cell]:
                nb_idx = nb[0] * NCOLS + nb[1]
                entries.append((dist.get(nb, far), nb_idx, 1 << nb_idx,
                                _EDGE_MASK[(cell, nb)]))
            entries.sort(key=lambda e: -e[0])
            table.append(tuple(e[1:] for e in entries))
    result = tuple(table)
    _GOAL_ORDERED_NEIGHBORS[goal] = result
    return result


def has_path(start: Pos, goal: Pos, blocked_mask: int) -> bool:
    """Return whether start can reach goal.

    Uses a goal-directed depth-first search: neighbors are pre-ordered by
    unwalled distance to the goal, so when a path exists (the common case for
    wall-legality checks) it is found in roughly path-length steps instead of
    a full breadth-first flood. Exact — falls back to exhausting the whole
    reachable component before answering False.
    """
    if _ENGINE is not None:
        return _ENGINE.has_path(start[0] * NCOLS + start[1],
                                goal[0] * NCOLS + goal[1], blocked_mask)
    return _py_has_path(start, goal, blocked_mask)


def _py_has_path(start: Pos, goal: Pos, blocked_mask: int) -> bool:
    if start == goal:
        return True
    neighbors = (_GOAL_ORDERED_NEIGHBORS.get(goal)
                 or _neighbors_ordered_for(goal))
    goal_idx = goal[0] * NCOLS + goal[1]
    seen = 1 << (start[0] * NCOLS + start[1])
    stack = [start[0] * NCOLS + start[1]]
    while stack:
        for nb_idx, nb_bit, edge_mask in neighbors[stack.pop()]:
            if nb_bit & seen or edge_mask & blocked_mask:
                continue
            if nb_idx == goal_idx:
                return True
            seen |= nb_bit
            stack.append(nb_idx)
    return False


def shortest_dist(start: Pos, goal: Pos, blocked_mask: int) -> Optional[int]:
    if _ENGINE is not None:
        return _ENGINE.shortest_dist(start[0] * NCOLS + start[1],
                                     goal[0] * NCOLS + goal[1], blocked_mask)
    return _dist_table_from(goal, blocked_mask).get(start)


@dataclass(frozen=True)
class Board:
    """Immutable per-game configuration: the two chosen goal cells."""
    p1_goal: Pos  # P1 pawn must reach this cell (in row 0, P2's start col)
    p2_goal: Pos  # P2 pawn must reach this cell (in row 10, P1's start col)

    def __post_init__(self) -> None:
        if not is_endzone(*self.p1_goal) or self.p1_goal[0] != P2_END_ROW:
            raise ValueError("P1's goal must be a valid cell in P2's end zone")
        if not is_endzone(*self.p2_goal) or self.p2_goal[0] != P1_END_ROW:
            raise ValueError("P2's goal must be a valid cell in P1's end zone")


@dataclass(frozen=True)
class State:
    p1: Pos
    p2: Pos
    p1_walls_left: int
    p2_walls_left: int
    walls: FrozenSet[Wall]
    turn: int  # 1 or 2

    @staticmethod
    def start(p1_col: int, p2_col: int, walls: int = WALLS_PER_PLAYER) -> "Tuple[Board, State]":
        if not 0 <= p1_col < NCOLS or not 0 <= p2_col < NCOLS:
            raise ValueError(f"starting columns must be between 0 and {NCOLS - 1}")
        if walls < 0:
            raise ValueError("starting wall count cannot be negative")
        board = Board(p1_goal=(P2_END_ROW, p2_col), p2_goal=(P1_END_ROW, p1_col))
        state = State(
            p1=(P1_END_ROW, p1_col),
            p2=(P2_END_ROW, p2_col),
            p1_walls_left=walls,
            p2_walls_left=walls,
            walls=frozenset(),
            turn=1,
        )
        return board, state

    def winner(self, board: Board) -> Optional[int]:
        if self.p1 == board.p1_goal:
            return 1
        if self.p2 == board.p2_goal:
            return 2
        return None


def is_threefold_repetition(states: Sequence[State]) -> bool:
    """Whether the latest exact position has now occurred three times.

    ``State`` includes both pawns, side to move, placed walls, and walls left.
    Goals live on ``Board`` and remain fixed for the duration of a game.
    """
    if not states:
        return False
    latest = states[-1]
    return sum(state == latest for state in states) >= REPETITIONS_FOR_DRAW


def _pawn_targets(me: Pos, opp: Pos, blocked_mask: int) -> Iterator[Pos]:
    r, c = me
    edge_mask = _EDGE_MASK.get
    for dr, dc in _DIRS:
        step = (r + dr, c + dc)
        mask = edge_mask((me, step))
        # mask is None when (me, step) is not a board edge; a set bit under
        # blocked_mask means a wall closed it.
        if mask is None or mask & blocked_mask:
            continue
        if step != opp:
            yield step
            continue
        # The opponent may be crossed only with a straight jump. If the cell
        # behind them is blocked or outside the board, this direction has no
        # legal pawn move; this variant does not allow side-jumps.
        straight = (step[0] + dr, step[1] + dc)
        mask = edge_mask((step, straight))
        if mask is not None and not (mask & blocked_mask):
            yield straight


def _mover(state: State) -> Tuple[Pos, Pos, int]:
    if state.turn == 1:
        return state.p1, state.p2, state.p1_walls_left
    return state.p2, state.p1, state.p2_walls_left


def _pawn_moves_unchecked(state: State, board: Board,
                          blocked_mask: Optional[int] = None) -> List[Pos]:
    """Physical pawn moves, before checking whether they immobilize the opponent."""
    me, opp, _ = _mover(state)
    m = blocked_mask_for(state.walls) if blocked_mask is None else blocked_mask
    # A player's own end-zone row is a one-way starting area: the initial pawn
    # move enters the board, and the pawn may never return to that row. End-zone
    # cells are already disconnected laterally by the adjacency graph.
    own_start_row = P1_END_ROW if state.turn == 1 else P2_END_ROW
    opponent_end_row = P2_END_ROW if state.turn == 1 else P1_END_ROW
    goal = board.p1_goal if state.turn == 1 else board.p2_goal
    out = []
    for t in _pawn_targets(me, opp, m):
        if t[0] == own_start_row:
            continue
        if t[0] == opponent_end_row and t != goal:
            continue
        out.append(t)
    return out


def _side_has_pawn_move(me: Pos, opp: Pos, own_start_row: int,
                        opp_end_row: int, goal: Pos, blocked_mask: int) -> bool:
    """Whether the pawn at `me` has any physical move; early-exits on the first.

    Same rules as _pawn_moves_unchecked but takes positions directly, so hot
    callers (wall legality) never construct a State just to ask this.
    """
    r, c = me
    edge_mask = _EDGE_MASK.get
    for dr, dc in _DIRS:
        step = (r + dr, c + dc)
        mask = edge_mask((me, step))
        if mask is None or mask & blocked_mask:
            continue
        if step == opp:
            straight = (step[0] + dr, step[1] + dc)
            mask = edge_mask((step, straight))
            if mask is None or mask & blocked_mask:
                continue
            step = straight
        if step[0] == own_start_row:
            continue
        if step[0] == opp_end_row and step != goal:
            continue
        return True
    return False


def _mover_rows_goal(turn: int, board: Board) -> Tuple[int, int, Pos]:
    """(own_start_row, opponent_end_row, goal) for the given side."""
    if turn == 1:
        return P1_END_ROW, P2_END_ROW, board.p1_goal
    return P2_END_ROW, P1_END_ROW, board.p2_goal


def _has_pawn_move(state: State, board: Board, blocked_mask: int) -> bool:
    me, opp, _ = _mover(state)
    own_row, opp_row, goal = _mover_rows_goal(state.turn, board)
    return _side_has_pawn_move(me, opp, own_row, opp_row, goal, blocked_mask)


def _pawn_mobility_edge_mask(me: Pos, opp: Pos) -> int:
    """Blockable edges whose closure could change this side's pawn moves."""
    mask = _CELL_EDGE_MASK[me]
    if (me, opp) in _EDGE_MASK:  # adjacent: jumps over opp are also in play
        mask |= _CELL_EDGE_MASK[opp]
    return mask


def legal_pawn_moves(state: State, board: Board) -> List[Pos]:
    if state.winner(board) is not None:
        return []
    blocked = blocked_mask_for(state.walls)
    if _ENGINE is not None:
        me, opp, _ = _mover(state)
        return [_INDEX_POS[i] for i in _ENGINE.legal_pawn_moves(
            me[0] * NCOLS + me[1], opp[0] * NCOLS + opp[1], state.turn,
            board.p1_goal[0] * NCOLS + board.p1_goal[1],
            board.p2_goal[0] * NCOLS + board.p2_goal[1], blocked)]
    return _py_legal_pawn_moves(state, board, blocked)


def _py_legal_pawn_moves(state: State, board: Board, blocked: int) -> List[Pos]:
    if state.turn == 1:
        opp, my_goal = state.p2, board.p1_goal
        opp_own_row, opp_end_row, opp_goal = P2_END_ROW, P1_END_ROW, board.p2_goal
    else:
        opp, my_goal = state.p1, board.p2_goal
        opp_own_row, opp_end_row, opp_goal = P1_END_ROW, P2_END_ROW, board.p1_goal
    out = []
    for target in _pawn_moves_unchecked(state, board, blocked):
        # Winning ends the game. Otherwise a pawn may not occupy the opponent's
        # last usable exit and leave them without any physical pawn move.
        if target == my_goal or _side_has_pawn_move(
                opp, target, opp_own_row, opp_end_row, opp_goal, blocked):
            out.append(target)
    return out


def _shortest_path_mask(pos: Pos, goal: Pos, blocked_mask: int) -> int:
    """Blockable-edge bitmask of ONE shortest path from pos to goal.

    Greedy walk down the BFS dist table, taking the first neighbor in _ADJ
    order that is one step closer; the edges come back as a bitmask so
    wall-touch tests reduce to a single integer AND against _WALL_BITMASK.
    End-zone entry edges have no bit and contribute nothing — walls can never
    touch them anyway.
    """
    dist = _dist_table_from(goal, blocked_mask)
    if pos not in dist:
        return 0
    mask = 0
    cur = pos
    while cur != goal:
        d = dist[cur]
        for nb in _ADJ[cur]:
            if nb not in dist:
                continue
            em = _EDGE_MASK[(cur, nb)]
            if em & blocked_mask:
                continue
            if dist[nb] == d - 1:
                mask |= em
                cur = nb
                break
        else:
            break
    return mask


def legal_wall_moves(state: State, board: Board) -> List[Wall]:
    if state.winner(board) is not None:
        return []
    _, _, walls_left = _mover(state)
    if walls_left <= 0:
        return []
    if _ENGINE is not None:
        placed_bits = 0
        for w in state.walls:
            placed_bits |= 1 << _WALL_INDEX[w]
        return [ALL_WALLS[i] for i in _ENGINE.legal_wall_moves(
            state.p1[0] * NCOLS + state.p1[1],
            state.p2[0] * NCOLS + state.p2[1],
            board.p1_goal[0] * NCOLS + board.p1_goal[1],
            board.p2_goal[0] * NCOLS + board.p2_goal[1],
            state.turn, placed_bits, blocked_mask_for(state.walls))]
    return _py_legal_wall_moves(state, board)


def _py_legal_wall_moves(state: State, board: Board) -> List[Wall]:
    placed = state.walls
    m = blocked_mask_for(placed)
    conflicting = set()
    for w in placed:
        conflicting.update(_WALL_CONFLICTS[w])
    # A wall cannot invalidate a path whose edges it does not touch. Keep one
    # existing path for each player (as an edge bitmask) and only run
    # reachability when the candidate intersects that path. This is an exact
    # shortcut, not a legality heuristic.
    p1, p2 = state.p1, state.p2
    p1_goal, p2_goal = board.p1_goal, board.p2_goal
    if state.turn == 1:
        me, my_goal, opp, opp_goal = p1, p1_goal, p2, p2_goal
    else:
        me, my_goal, opp, opp_goal = p2, p2_goal, p1, p1_goal
    # Adding a wall can only remove moves. A legacy/malformed position that is
    # already immobilized cannot be repaired by placing another wall.
    if not _side_has_pawn_move(p1, p2, P1_END_ROW, P2_END_ROW, p1_goal, m):
        return []
    if not _side_has_pawn_move(p2, p1, P2_END_ROW, P1_END_ROW, p2_goal, m):
        return []
    my_path_mask = _shortest_path_mask(me, my_goal, m)
    opp_path_mask = _shortest_path_mask(opp, opp_goal, m)
    p1_mobility_edges = _pawn_mobility_edge_mask(p1, p2)
    p2_mobility_edges = _pawn_mobility_edge_mask(p2, p1)
    # Second path per player, avoiding the primary path's edges (computed
    # lazily; 0 when no such detour exists). A candidate that leaves either
    # path of the pair untouched provably keeps that player connected, so the
    # reachability search only runs for candidates crossing both paths. Exact.
    my_alt_mask: Optional[int] = None
    opp_alt_mask: Optional[int] = None
    out: List[Wall] = []
    for w, wall_mask in _WALL_MASK_ITEMS:
        if w in placed or w in conflicting:
            continue
        # Walls may not immobilize either pawn. Checking both sides also stops
        # a player from walling in their own pawn and creating a forced no-move
        # state on the following turn.
        if (wall_mask & p1_mobility_edges
                and not _side_has_pawn_move(p1, p2, P1_END_ROW, P2_END_ROW,
                                            p1_goal, m | wall_mask)):
            continue
        if (wall_mask & p2_mobility_edges
                and not _side_has_pawn_move(p2, p1, P2_END_ROW, P1_END_ROW,
                                            p2_goal, m | wall_mask)):
            continue
        if wall_mask & my_path_mask:
            if my_alt_mask is None:
                my_alt_mask = _shortest_path_mask(me, my_goal, m | my_path_mask)
            if ((not my_alt_mask or wall_mask & my_alt_mask)
                    and not _py_has_path(me, my_goal, m | wall_mask)):
                continue
        if wall_mask & opp_path_mask:
            if opp_alt_mask is None:
                opp_alt_mask = _shortest_path_mask(opp, opp_goal,
                                                   m | opp_path_mask)
            if ((not opp_alt_mask or wall_mask & opp_alt_mask)
                    and not _py_has_path(opp, opp_goal, m | wall_mask)):
                continue
        out.append(w)
    return out


# Interned Move tuples so legal_moves returns shared objects: hot MCTS paths
# hold thousands of these lists at once, and shared tuples keep caches small.
_PAWN_MOVE: Dict[Pos, Move] = {
    (r, c): ("m", (r, c)) for r in range(NROWS) for c in range(NCOLS)
}
_WALL_MOVE: Dict[Wall, Move] = {w: ("w", w) for w in ALL_WALLS}


def legal_moves(state: State, board: Board) -> List[Move]:
    if state.winner(board) is not None:
        return []
    moves: List[Move] = list(map(_PAWN_MOVE.__getitem__,
                                 legal_pawn_moves(state, board)))
    moves.extend(map(_WALL_MOVE.__getitem__, legal_wall_moves(state, board)))
    return moves


# ---------------------------------------------------------------------------
# Solver support: distance/branching queries that dispatch to the compiled
# engine when available. Pure fallbacks reuse the cached dict tables.
# ---------------------------------------------------------------------------

def dist_and_alt(pos: Pos, goal: Pos, blocked_mask: int) -> Tuple[int, int]:
    """(shortest distance, count of neighbors one step closer to goal).

    Distance is 10_000 when the goal is unreachable.
    """
    if _ENGINE is not None:
        return _ENGINE.dist_and_alt(pos[0] * NCOLS + pos[1],
                                    goal[0] * NCOLS + goal[1], blocked_mask)
    dist = _dist_table_from(goal, blocked_mask)
    d = dist.get(pos)
    if d is None:
        return 10_000, 0
    cnt = 0
    for nb in _ADJ[pos]:
        nd = dist.get(nb)
        if nd is None:
            continue
        if _EDGE_MASK[(pos, nb)] & blocked_mask:
            continue
        if nd == d - 1:
            cnt += 1
    return d, cnt


def dist_reader(goal: Pos, blocked_mask: int):
    """pos -> distance callable for one (goal, walls) pair; 10_000 = unreachable.

    Amortizes the table build across many lookups (move ordering probes every
    pawn target against the same table).
    """
    if _ENGINE is not None:
        table = _ENGINE.dist_table_bytes(goal[0] * NCOLS + goal[1],
                                         blocked_mask)

        def _rd(pos: Pos, _t: bytes = table) -> int:
            d = _t[pos[0] * NCOLS + pos[1]]
            return 10_000 if d == 255 else d
        return _rd
    table = _dist_table_from(goal, blocked_mask)
    return lambda pos: table.get(pos, 10_000)


def shortest_path_mask(pos: Pos, goal: Pos, blocked_mask: int) -> int:
    """Blockable-edge bitmask of ONE shortest path from pos to goal."""
    if _ENGINE is not None:
        return _ENGINE.shortest_path_mask(pos[0] * NCOLS + pos[1],
                                          goal[0] * NCOLS + goal[1],
                                          blocked_mask)
    return _shortest_path_mask(pos, goal, blocked_mask)


def apply_move(state: State, move: Move) -> State:
    # Constructs State directly: dataclasses.replace() costs roughly twice as
    # much and this is one of the hottest calls in tree search.
    kind, arg = move
    if kind == "m":
        if state.turn == 1:
            return State(arg, state.p2, state.p1_walls_left,
                         state.p2_walls_left, state.walls, 2)
        return State(state.p1, arg, state.p1_walls_left,
                     state.p2_walls_left, state.walls, 1)
    if kind == "w":
        new_walls = state.walls | {arg}
        if state.turn == 1:
            return State(state.p1, state.p2, state.p1_walls_left - 1,
                         state.p2_walls_left, new_walls, 2)
        return State(state.p1, state.p2, state.p1_walls_left,
                     state.p2_walls_left - 1, new_walls, 1)
    raise ValueError(f"bad move: {move}")


# ---------------------------------------------------------------------------
# Compiled engine bootstrap. The optional _engine extension (Cython) holds C
# reimplementations of the hot functions above; when it is importable, the
# dispatchers hand off to it, otherwise the pure-Python bodies run. Parity
# tests exercise both via the _py_* names.
# ---------------------------------------------------------------------------

_INDEX_POS: Tuple[Pos, ...] = tuple(
    (i // NCOLS, i % NCOLS) for i in range(NROWS * NCOLS))
_WALL_INDEX: Dict[Wall, int] = {w: i for i, w in enumerate(ALL_WALLS)}


def _engine_tables():
    """Board topology in the flat index/bitmask form _engine.init expects."""
    adj_dir = []
    adj_full = []
    cell_masks = []
    for cell in _INDEX_POS:
        by_dir = []
        for dr, dc in _DIRS:
            nb = (cell[0] + dr, cell[1] + dc)
            m = _EDGE_MASK.get((cell, nb))
            # m == 0 is a real (unblockable end-zone) edge; None means no edge.
            by_dir.append((-1, 0) if m is None else (nb[0] * NCOLS + nb[1], m))
        adj_dir.append(by_dir)
        adj_full.append([(nb[0] * NCOLS + nb[1], _EDGE_MASK[(cell, nb)])
                         for nb in _ADJ[cell]])
        cell_masks.append(_CELL_EDGE_MASK[cell])
    wall_data = []
    for w in ALL_WALLS:
        conf = 0
        for other in _WALL_CONFLICTS[w]:
            conf |= 1 << _WALL_INDEX[other]
        wall_data.append((_WALL_BITMASK[w], conf))
    return adj_dir, adj_full, cell_masks, wall_data


try:
    from . import _engine as _ENGINE
except ImportError:
    _ENGINE = None
else:
    _ENGINE.init(*_engine_tables())
