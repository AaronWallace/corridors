"""Anti-loop wiring: the solver must steer away from repeated positions.

The game loops (parallel.py workers, play.py autoplay) pass positions seen
twice as avoid_child_hashes so the root never completes a threefold
repetition when any alternative exists.
"""

from corridors import solver
from corridors.game import State, apply_move, legal_moves


def _best(state, board, avoid=None):
    mv, _score, _stats, _pv = solver.best_move(
        state, board, max_depth=3, avoid_child_hashes=avoid,
        verbose=False, flush_on_exit=False)
    return mv


def test_avoided_child_changes_choice():
    board, state = State.start(4, 4)
    preferred = _best(state, board)
    child_hash = solver.zobrist(apply_move(state, preferred), board)
    rerouted = _best(state, board, avoid={child_hash})
    assert rerouted is not None
    assert rerouted != preferred


def test_all_children_avoided_still_returns_a_move():
    board, state = State.start(4, 4)
    avoid = {solver.zobrist(apply_move(state, mv), board)
             for mv in legal_moves(state, board)}
    mv = _best(state, board, avoid=avoid)
    assert mv in legal_moves(state, board)
