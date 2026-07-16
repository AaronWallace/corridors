"""Round-trip tests for the compact inference wire format.

unpack_states_batch(pack_state(...)) must be bitwise-identical to
encode_state — the packed protocol replaces the encoded tensor on the
GPU inference request path, so any drift silently corrupts self-play.
"""

import random

import numpy as np
import pytest

from corridors.game import (
    ALL_WALLS, NCOLS, WALLS_PER_PLAYER, State, apply_move, legal_moves,
)
from corridors.nn.encoding import (
    PACKED_SIZE, encode_state, pack_state, unpack_states_batch,
)


def _random_game_positions(rng, max_plies=60):
    """(state, board) pairs sampled from one random-playout game."""
    board, state = State.start(
        p1_col=rng.randrange(NCOLS), p2_col=rng.randrange(NCOLS),
        walls=WALLS_PER_PLAYER)
    positions = [(state, board)]
    for _ in range(max_plies):
        if state.winner(board) is not None:
            break
        moves = legal_moves(state, board)
        if not moves:
            break
        state = apply_move(state, rng.choice(moves))
        positions.append((state, board))
    return positions


def _assert_roundtrip(state, board):
    packed = pack_state(state, board)
    assert len(packed) == PACKED_SIZE
    decoded = unpack_states_batch([packed])[0]
    assert np.array_equal(decoded, encode_state(state, board))


def test_packed_size():
    assert PACKED_SIZE == 23


def test_roundtrip_start_states():
    for p1_col in range(NCOLS):
        for p2_col in range(NCOLS):
            board, state = State.start(p1_col=p1_col, p2_col=p2_col)
            _assert_roundtrip(state, board)


def test_roundtrip_random_games():
    rng = random.Random(12345)
    total = 0
    for _ in range(40):
        for state, board in _random_game_positions(rng):
            _assert_roundtrip(state, board)
            total += 1
    assert total > 500  # sanity: the sweep actually covered many positions


def test_roundtrip_turn_two_and_no_walls_left():
    board, state = State.start(p1_col=4, p2_col=4, walls=0)
    _assert_roundtrip(state, board)  # 0 walls left from the start
    state = apply_move(state, legal_moves(state, board)[0])
    assert state.turn == 2
    _assert_roundtrip(state, board)


def test_roundtrip_every_wall_bit():
    # Each ALL_WALLS entry individually, so every bit of the 16-byte mask
    # (both bytes-boundary and bit-order behavior) is exercised.
    board, base = State.start(p1_col=0, p2_col=8)
    for w in ALL_WALLS:
        state = State(base.p1, base.p2, 3, 0, frozenset({w}), 2)
        _assert_roundtrip(state, board)


def test_roundtrip_synthetic_wall_heavy_state():
    # 18 non-conflicting walls (all placed), pawns mid-board, p2 to move.
    walls = frozenset(
        w for w in ALL_WALLS
        if w[2] == "H" and w[0] in (2, 5, 8) and w[1] in (0, 2, 4, 6))
    assert len(walls) == 12
    extra = frozenset({(3, 1, "V"), (4, 3, "V"), (6, 5, "V"),
                       (7, 7, "V"), (1, 0, "V"), (1, 6, "V")})
    state = State((5, 4), (6, 4), 0, 0, walls | extra, 2)
    board, _ = State.start(p1_col=2, p2_col=7)
    _assert_roundtrip(state, board)


def test_batch_decode_matches_per_state():
    rng = random.Random(777)
    pairs = []
    while len(pairs) < 64:
        pairs.extend(_random_game_positions(rng, max_plies=25))
    pairs = pairs[:64]
    batch = unpack_states_batch([pack_state(s, b) for s, b in pairs])
    assert batch.shape == (64, 9, 11, 9)
    assert batch.dtype == np.float32
    for i, (s, b) in enumerate(pairs):
        assert np.array_equal(batch[i], encode_state(s, b))


def test_walls_left_planes_bitwise_float32():
    # walls_left / 9 is not exactly representable; verify the decoded planes
    # carry the identical float32 rounding for every possible count.
    board, base = State.start(p1_col=3, p2_col=5)
    for n in range(WALLS_PER_PLAYER + 1):
        state = State(base.p1, base.p2, n, WALLS_PER_PLAYER - n, frozenset(), 1)
        packed = unpack_states_batch([pack_state(state, board)])[0]
        ref = encode_state(state, board)
        assert packed[4].tobytes() == ref[4].tobytes()
        assert packed[5].tobytes() == ref[5].tobytes()
