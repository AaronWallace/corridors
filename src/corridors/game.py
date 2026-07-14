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

from dataclasses import dataclass, replace
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
_CELL_NEIGHBORS: Tuple[Tuple[Tuple[int, int], ...], ...] = ()


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


def _init_cell_neighbors() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """Precompute compact adjacency for bitset reachability searches.

    Each entry is ``(neighbor_cell_bit, blocking_edge_bit)``. End-zone entry
    edges cannot be blocked and therefore use a zero blocking mask.
    """
    out = []
    for r in range(NROWS):
        for c in range(NCOLS):
            cell = (r, c)
            neighbors = []
            for nb in _ADJ[cell]:
                edge_idx = _EDGE_BIT.get(_edge_key(cell, nb))
                edge_mask = 0 if edge_idx is None else 1 << edge_idx
                neighbors.append((1 << (nb[0] * NCOLS + nb[1]), edge_mask))
            out.append(tuple(neighbors))
    return tuple(out)


_CELL_NEIGHBORS = _init_cell_neighbors()


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


def blocked_mask_for(walls) -> int:
    m = 0
    for w in walls:
        m |= _WALL_BITMASK[w]
    return m


@lru_cache(maxsize=200_000)
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


def has_path(start: Pos, goal: Pos, blocked_mask: int) -> bool:
    """Return whether start can reach goal using an allocation-light bitset BFS."""
    if start == goal:
        return True
    start_bit = 1 << (start[0] * NCOLS + start[1])
    goal_bit = 1 << (goal[0] * NCOLS + goal[1])
    seen = start_bit
    frontier = start_bit
    while frontier:
        nxt = 0
        cells = frontier
        while cells:
            cell_bit = cells & -cells
            cells ^= cell_bit
            cell_idx = cell_bit.bit_length() - 1
            for neighbor_bit, edge_mask in _CELL_NEIGHBORS[cell_idx]:
                if seen & neighbor_bit or edge_mask & blocked_mask:
                    continue
                if neighbor_bit == goal_bit:
                    return True
                seen |= neighbor_bit
                nxt |= neighbor_bit
        frontier = nxt
    return False


def shortest_dist(start: Pos, goal: Pos, blocked_mask: int) -> Optional[int]:
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


def _edge_open(a: Pos, b: Pos, blocked_mask: int) -> bool:
    """True iff (a,b) is an adjacency edge on the board and no wall blocks it."""
    if b not in _ADJ[a]:
        return False
    bit = _EDGE_BIT.get(_edge_key(a, b))
    if bit is None:
        return True  # end-zone entry edges are never blocked
    return not ((blocked_mask >> bit) & 1)


def _pawn_targets(me: Pos, opp: Pos, blocked_mask: int) -> Iterator[Pos]:
    r, c = me
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        step = (r + dr, c + dc)
        if not valid(*step) or not _edge_open(me, step, blocked_mask):
            continue
        if step != opp:
            yield step
            continue
        # The opponent may be crossed only with a straight jump. If the cell
        # behind them is blocked or outside the board, this direction has no
        # legal pawn move; this variant does not allow side-jumps.
        straight = (step[0] + dr, step[1] + dc)
        if valid(*straight) and _edge_open(step, straight, blocked_mask):
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
    seen = set()
    out = []
    for t in _pawn_targets(me, opp, m):
        if t == me or t in seen or t[0] == own_start_row:
            continue
        if t[0] == opponent_end_row and t != goal:
            continue
        seen.add(t)
        out.append(t)
    return out


def _has_pawn_move(state: State, board: Board, blocked_mask: int) -> bool:
    return bool(_pawn_moves_unchecked(state, board, blocked_mask))


def _pawn_mobility_edge_mask(state: State) -> int:
    """Blockable edges whose closure could change this turn's pawn moves."""
    me, opp, _ = _mover(state)
    cells = (me, opp) if opp in _ADJ[me] else (me,)
    mask = 0
    for cell in cells:
        for neighbor in _ADJ[cell]:
            edge = _EDGE_BIT.get(_edge_key(cell, neighbor))
            if edge is not None:
                mask |= 1 << edge
    return mask


def legal_pawn_moves(state: State, board: Board) -> List[Pos]:
    if state.winner(board) is not None:
        return []
    blocked = blocked_mask_for(state.walls)
    out = []
    for target in _pawn_moves_unchecked(state, board, blocked):
        child = apply_move(state, ("m", target))
        # Winning ends the game. Otherwise a pawn may not occupy the opponent's
        # last usable exit and leave them without any physical pawn move.
        if child.winner(board) is not None or _has_pawn_move(child, board, blocked):
            out.append(target)
    return out


def _wall_touches_shortest_path(w: Wall, path_edges: FrozenSet[Tuple[Pos, Pos]]) -> bool:
    e1, e2 = _wall_edges(w)
    return e1 in path_edges or e2 in path_edges


def _shortest_path_edges(pos: Pos, goal: Pos, blocked_mask: int) -> FrozenSet[Tuple[Pos, Pos]]:
    """Set of edges on ONE shortest path from pos to goal (empty if unreachable)."""
    dist = _dist_table_from(goal, blocked_mask)
    if pos not in dist:
        return frozenset()
    edges = set()
    cur = pos
    while cur != goal:
        d = dist[cur]
        for nb in _ADJ[cur]:
            if nb not in dist:
                continue
            bit = _EDGE_BIT.get(_edge_key(cur, nb))
            if bit is not None and (blocked_mask >> bit) & 1:
                continue
            if dist[nb] == d - 1:
                edges.add(_edge_key(cur, nb))
                cur = nb
                break
        else:
            break
    return frozenset(edges)


def legal_wall_moves(state: State, board: Board) -> List[Wall]:
    if state.winner(board) is not None:
        return []
    _, _, walls_left = _mover(state)
    if walls_left <= 0:
        return []
    m = blocked_mask_for(state.walls)
    conflicting = set()
    for w in state.walls:
        conflicting.update(_WALL_CONFLICTS[w])
    # A wall cannot invalidate a path whose edges it does not touch. Keep one
    # existing path for each player and only run reachability when the candidate
    # intersects that path. This is an exact shortcut, not a legality heuristic.
    me = state.p1 if state.turn == 1 else state.p2
    my_goal = board.p1_goal if state.turn == 1 else board.p2_goal
    opp = state.p2 if state.turn == 1 else state.p1
    opp_goal = board.p2_goal if state.turn == 1 else board.p1_goal
    my_path_edges = _shortest_path_edges(me, my_goal, m)
    opp_path_edges = _shortest_path_edges(opp, opp_goal, m)
    p1_view = replace(state, turn=1)
    p2_view = replace(state, turn=2)
    # Adding a wall can only remove moves. A legacy/malformed position that is
    # already immobilized cannot be repaired by placing another wall.
    if not _has_pawn_move(p1_view, board, m) or not _has_pawn_move(p2_view, board, m):
        return []
    p1_mobility_edges = _pawn_mobility_edge_mask(p1_view)
    p2_mobility_edges = _pawn_mobility_edge_mask(p2_view)
    out: List[Wall] = []
    for w in ALL_WALLS:
        if w in state.walls or w in conflicting:
            continue
        m2 = m | _WALL_BITMASK[w]
        # Walls may not immobilize either pawn. Checking both sides also stops
        # a player from walling in their own pawn and creating a forced no-move
        # state on the following turn.
        wall_mask = _WALL_BITMASK[w]
        if (wall_mask & p1_mobility_edges
                and not _has_pawn_move(p1_view, board, m2)):
            continue
        if (wall_mask & p2_mobility_edges
                and not _has_pawn_move(p2_view, board, m2)):
            continue
        if _wall_touches_shortest_path(w, my_path_edges):
            if not has_path(me, my_goal, m2):
                continue
        if _wall_touches_shortest_path(w, opp_path_edges):
            if not has_path(opp, opp_goal, m2):
                continue
        out.append(w)
    return out


def legal_moves(state: State, board: Board) -> List[Move]:
    if state.winner(board) is not None:
        return []
    moves: List[Move] = [("m", p) for p in legal_pawn_moves(state, board)]
    moves.extend(("w", w) for w in legal_wall_moves(state, board))
    return moves


def apply_move(state: State, move: Move) -> State:
    kind, arg = move
    if kind == "m":
        if state.turn == 1:
            return replace(state, p1=arg, turn=2)
        return replace(state, p2=arg, turn=1)
    if kind == "w":
        new_walls = state.walls | {arg}
        if state.turn == 1:
            return replace(state, walls=new_walls, p1_walls_left=state.p1_walls_left - 1, turn=2)
        return replace(state, walls=new_walls, p2_walls_left=state.p2_walls_left - 1, turn=1)
    raise ValueError(f"bad move: {move}")
