"""Parity between the pure-Python engine and the compiled Cython engine.

Every dispatched function must return exactly the same values (including list
ordering — move ordering feeds the solver's tie-breaking) whether the
compiled engine is active or not. The pure path is exercised by temporarily
nulling game._ENGINE, which every dispatcher reads at call time.

Skipped entirely when the extension is not built.
"""

import random

import pytest

from corridors import game
from corridors.game import State, blocked_mask_for

pytestmark = pytest.mark.skipif(
    game._ENGINE is None, reason="corridors._engine extension not built")


def _playout_positions(seed: int, max_moves: int = 60):
    """Board/state pairs along a wall-heavy random playout."""
    rng = random.Random(seed)
    board, state = State.start(rng.randint(0, 8), rng.randint(0, 8))
    yield board, state
    for _ in range(max_moves):
        if state.winner(board) is not None:
            break
        moves = game.legal_moves(state, board)
        if not moves:
            break
        # Bias toward walls so parity is checked on cluttered boards, where
        # the wall-legality shortcuts actually trigger.
        walls = [m for m in moves if m[0] == "w"]
        mv = rng.choice(walls) if walls and rng.random() < 0.5 else rng.choice(moves)
        state = game.apply_move(state, mv)
        yield board, state


@pytest.mark.parametrize("seed", range(8))
def test_playout_parity(seed, monkeypatch):
    positions = list(_playout_positions(seed))
    for board, state in positions:
        mask = blocked_mask_for(state.walls)
        got = {
            "pawn": game.legal_pawn_moves(state, board),
            "wall": game.legal_wall_moves(state, board),
            "moves": game.legal_moves(state, board),
            "da1": game.dist_and_alt(state.p1, board.p1_goal, mask),
            "da2": game.dist_and_alt(state.p2, board.p2_goal, mask),
            "sd1": game.shortest_dist(state.p1, board.p1_goal, mask),
            "spm1": game.shortest_path_mask(state.p1, board.p1_goal, mask),
            "spm2": game.shortest_path_mask(state.p2, board.p2_goal, mask),
        }
        with monkeypatch.context() as mp:
            mp.setattr(game, "_ENGINE", None)
            assert game.legal_pawn_moves(state, board) == got["pawn"]
            assert game.legal_wall_moves(state, board) == got["wall"]
            assert game.legal_moves(state, board) == got["moves"]
            assert game.dist_and_alt(state.p1, board.p1_goal, mask) == got["da1"]
            assert game.dist_and_alt(state.p2, board.p2_goal, mask) == got["da2"]
            assert game.shortest_dist(state.p1, board.p1_goal, mask) == got["sd1"]
            assert game.shortest_path_mask(
                state.p1, board.p1_goal, mask) == got["spm1"]
            assert game.shortest_path_mask(
                state.p2, board.p2_goal, mask) == got["spm2"]


@pytest.mark.parametrize("seed", range(4))
def test_has_path_parity_under_heavy_walls(seed, monkeypatch):
    """has_path agreement on masks dense enough to actually cut the board."""
    rng = random.Random(1000 + seed)
    board, state = State.start(rng.randint(0, 8), rng.randint(0, 8))
    walls = []
    for _ in range(60):
        candidates = [w for w in game.ALL_WALLS
                      if w not in walls
                      and not (set(walls) & game._WALL_CONFLICTS[w])]
        if not candidates:
            break
        walls.append(rng.choice(candidates))
        mask = blocked_mask_for(walls)
        got = game.has_path(state.p1, board.p1_goal, mask)
        with monkeypatch.context() as mp:
            mp.setattr(game, "_ENGINE", None)
            assert game.has_path(state.p1, board.p1_goal, mask) == got


def test_solver_evaluate_parity(monkeypatch):
    from corridors import solver
    for board, state in _playout_positions(42, max_moves=40):
        got = solver.evaluate(state, board)
        with monkeypatch.context() as mp:
            mp.setattr(game, "_ENGINE", None)
            assert solver.evaluate(state, board) == got
