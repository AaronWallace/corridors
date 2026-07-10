"""Bidirectional mapping between game Moves and fixed action indices.

Action space (227 total):
    [0..98]    pawn moves: index = row * 9 + col  (11 rows × 9 cols = 99)
    [99..226]  wall moves: index = 99 + (r - 1) * 16 + c * 2 + orient
               r ∈ [1..8], c ∈ [0..7], orient: 0=H, 1=V  (8×8×2 = 128)

Not every index is reachable in every position — the policy head outputs logits
for all 227 and illegal moves are masked to -inf before softmax.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..game import (
    ALL_WALLS, Board, Move, NCOLS, NROWS, PLAY_MIN, State,
    legal_moves,
)

NUM_ACTIONS = 99 + 128  # 227

# --- index ↔ move -----------------------------------------------------------

def move_to_index(move: Move) -> int:
    kind, arg = move
    if kind == "m":
        r, c = arg
        return r * NCOLS + c
    # wall: (r, c, orient)
    r, c, o = arg
    return 99 + (r - PLAY_MIN) * 16 + c * 2 + (0 if o == "H" else 1)


def index_to_move(idx: int) -> Move:
    if idx < 99:
        return ("m", (idx // NCOLS, idx % NCOLS))
    wi = idx - 99
    r = wi // 16 + PLAY_MIN
    c = (wi % 16) // 2
    o = "H" if wi % 2 == 0 else "V"
    return ("w", (r, c, o))


# --- legal move mask --------------------------------------------------------

def legal_move_mask(state: State, board: Board) -> Tuple[List[Move], List[int]]:
    """Returns (moves, indices) for all legal moves in this position."""
    moves = legal_moves(state, board)
    indices = [move_to_index(m) for m in moves]
    return moves, indices
