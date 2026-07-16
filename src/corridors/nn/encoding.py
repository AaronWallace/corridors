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

from ..game import ALL_WALLS, Board, NCOLS, NROWS, State, WALLS_PER_PLAYER

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


# --- Compact wire format for inference requests -----------------------------
#
# The GPU inference server's throughput is bounded by per-request IPC cost, so
# self-play workers ship a 23-byte packed state instead of the 3.6 KB encoded
# tensor; the server decodes whole batches vectorized (unpack_states_batch).
#
# Layout (all uint8):
#   [0] p1 cell index (row * NCOLS + col)
#   [1] p2 cell index
#   [2] p1_walls_left
#   [3] p2_walls_left
#   [4] turn (1 or 2)
#   [5] p1_goal col (p1_goal row is always 0)
#   [6] p2_goal col (p2_goal row is always NROWS - 1)
#   [7:23] wall bitmask, bit i (little-endian within each byte) = ALL_WALLS[i]

PACKED_SIZE = 7 + (len(ALL_WALLS) + 7) // 8  # 23

_WALL_BIT: dict = {w: i for i, w in enumerate(ALL_WALLS)}
# Per wall-bit scatter targets: plane 2 for H, 3 for V, at the anchor (r, c).
_WALL_PLANE = np.array([2 if o == "H" else 3 for _, _, o in ALL_WALLS], dtype=np.intp)
_WALL_ROW = np.array([r for r, _, _ in ALL_WALLS], dtype=np.intp)
_WALL_COL = np.array([c for _, c, _ in ALL_WALLS], dtype=np.intp)


def pack_state(state: State, board: Board) -> bytes:
    """Pack a position into the fixed-size wire format (see layout above)."""
    buf = bytearray(PACKED_SIZE)
    buf[0] = state.p1[0] * NCOLS + state.p1[1]
    buf[1] = state.p2[0] * NCOLS + state.p2[1]
    buf[2] = state.p1_walls_left
    buf[3] = state.p2_walls_left
    buf[4] = state.turn
    buf[5] = board.p1_goal[1]
    buf[6] = board.p2_goal[1]
    for w in state.walls:
        i = _WALL_BIT[w]
        buf[7 + (i >> 3)] |= 1 << (i & 7)
    return bytes(buf)


def unpack_states_batch(packed_list) -> np.ndarray:
    """Decode packed states into a (B, 9, 11, 9) float32 batch, bit-identical
    to stacking encode_state over the same positions."""
    n = len(packed_list)
    raw = np.frombuffer(b"".join(packed_list), dtype=np.uint8).reshape(n, PACKED_SIZE)
    out = np.zeros((n, *SHAPE), dtype=np.float32)
    b = np.arange(n)
    out[b, 0, raw[:, 0] // NCOLS, raw[:, 0] % NCOLS] = 1.0
    out[b, 1, raw[:, 1] // NCOLS, raw[:, 1] % NCOLS] = 1.0
    # bitorder="little" matches pack_state's `1 << (i & 7)` within each byte.
    bits = np.unpackbits(raw[:, 7:], axis=1, bitorder="little")[:, :len(ALL_WALLS)]
    bi, wi = np.nonzero(bits)
    out[bi, _WALL_PLANE[wi], _WALL_ROW[wi], _WALL_COL[wi]] = 1.0
    # float64 division then float32 cast — the exact arithmetic encode_state's
    # scalar `walls_left / WALLS_PER_PLAYER` assignment performs.
    out[:, 4] = (raw[:, 2] / WALLS_PER_PLAYER).astype(np.float32)[:, None, None]
    out[:, 5] = (raw[:, 3] / WALLS_PER_PLAYER).astype(np.float32)[:, None, None]
    out[:, 6] = (raw[:, 4] == 1).astype(np.float32)[:, None, None]
    out[b, 7, 0, raw[:, 5]] = 1.0
    out[b, 8, NROWS - 1, raw[:, 6]] = 1.0
    return out


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
