"""State → tensor encoding for the value network.

Planes are (9, 11, 9) float32, board-shaped [row, col]:

    0: P1 pawn one-hot
    1: P2 pawn one-hot
    2: H-walls (1.0 at the wall slot's anchor cell (r, c))
    3: V-walls (same)
    4: P1 walls left / 9, broadcast
    5: P2 walls left / 9, broadcast
    6: side-to-move (1.0 everywhere if P1 to move, else 0.0)
    7: P1 goal one-hot   (this variant has per-game goal cells, so the net
    8: P2 goal one-hot    must see them — unlike fixed-goal Quoridor)

Value targets are always from the side-to-move's perspective: +1 means the
player about to move eventually won.

This module is torch-free so data generation workers don't import CUDA.
"""

from __future__ import annotations

import numpy as np

from ..game import Board, NCOLS, NROWS, State, WALLS_PER_PLAYER

NUM_PLANES = 9
SHAPE = (NUM_PLANES, NROWS, NCOLS)

# Score normalization for tt_score targets: tanh-squash classical centipawn-ish
# scores into [-1, 1]. 300 ≈ a comfortable 3-step distance advantage.
SCORE_SCALE = 300.0


def encode_state(state: State, board: Board) -> np.ndarray:
    t = np.zeros(SHAPE, dtype=np.float32)
    t[0, state.p1[0], state.p1[1]] = 1.0
    t[1, state.p2[0], state.p2[1]] = 1.0
    for (r, c, o) in state.walls:
        t[2 if o == "H" else 3, r, c] = 1.0
    t[4, :, :] = state.p1_walls_left / WALLS_PER_PLAYER
    t[5, :, :] = state.p2_walls_left / WALLS_PER_PLAYER
    t[6, :, :] = 1.0 if state.turn == 1 else 0.0
    t[7, board.p1_goal[0], board.p1_goal[1]] = 1.0
    t[8, board.p2_goal[0], board.p2_goal[1]] = 1.0
    return t


def reflect_encoded(encoded: np.ndarray) -> np.ndarray:
    """Reflect an encoded state, or batch of states, across the center file."""
    out = np.flip(encoded, axis=-1).copy()
    # Wall planes store anchors in columns 0..7, so their reflection is around
    # that eight-slot grid rather than the nine-cell board grid.
    out[..., 2:4, :, :] = 0
    out[..., 2:4, :, :NCOLS - 1] = np.flip(
        encoded[..., 2:4, :, :NCOLS - 1], axis=-1)
    return out


def normalize_score(score: int) -> float:
    """Squash a classical engine score (side-to-move perspective) into [-1, 1]."""
    return float(np.tanh(score / SCORE_SCALE))


def outcome_for_mover(winner: int | None, turn: int) -> int:
    """Game outcome from the perspective of the side to move at this position."""
    if winner is None:
        return 0
    return 1 if winner == turn else -1
