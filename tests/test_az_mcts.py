"""Repetition and draw-horizon behavior in AlphaZero MCTS."""

import numpy as np

from corridors.game import State, apply_move, legal_moves
from corridors.nn.actions import NUM_ACTIONS, move_to_index
from corridors.nn.mcts import Node, run_mcts


def _favor(move, calls):
    def evaluate(_state, _board):
        calls.append(1)
        logits = np.full(NUM_ACTIONS, -1_000_000.0, dtype=np.float32)
        logits[move_to_index(move)] = 0.0
        return logits, 0.75
    return evaluate


def test_equal_logits_give_equal_aggregate_pawn_and_wall_prior():
    board, state = State.start(4, 6)
    node = Node(state, board)
    node.expand(np.zeros(NUM_ACTIONS, dtype=np.float32))

    pawn = np.array([move[0] == "m" for move in node.child_moves])
    assert np.isclose(node.P[pawn].sum(), 0.5)
    assert np.isclose(node.P[~pawn].sum(), 0.5)


def test_dirichlet_noise_is_balanced_between_action_types():
    board, state = State.start(4, 6)
    node = Node(state, board)
    node.expand(np.zeros(NUM_ACTIONS, dtype=np.float32))
    np.random.seed(731)
    node.add_dirichlet_noise(alpha=0.3, frac=1.0)

    pawn = np.array([move[0] == "m" for move in node.child_moves])
    assert np.isclose(node.P[pawn].sum(), 0.5)
    assert np.isclose(node.P[~pawn].sum(), 0.5)


def test_policy_target_marks_every_legal_action_for_balanced_training():
    board, state = State.start(4, 6)
    move = ("m", (9, 4))
    calls = []
    pi, _value, _chosen, _reuse = run_mcts(
        state, board, _favor(move, calls), num_simulations=1,
        temperature=0.0, add_noise=False,
    )

    legal_indices = {move_to_index(candidate) for candidate in legal_moves(state, board)}
    assert set(np.flatnonzero(pi)) == legal_indices
    assert np.isclose(pi.sum(), 1.0)


def test_mcts_scores_third_position_occurrence_as_draw_without_evaluating_it():
    board, root = State.start(4, 6)
    move = ("m", (9, 4))
    repeated = apply_move(root, move)
    calls = []

    _pi, root_value, chosen, _reuse = run_mcts(
        root, board, _favor(move, calls), num_simulations=1,
        temperature=0.0, add_noise=False,
        state_history=[repeated, repeated, root],
    )

    assert chosen == move
    assert root_value == 0.0
    assert len(calls) == 1  # root only; the repeated child is adjudicated


def test_mcts_scores_maximum_ply_horizon_as_draw_without_evaluating_it():
    board, root = State.start(4, 6)
    move = ("m", (9, 4))
    calls = []

    _pi, root_value, chosen, _reuse = run_mcts(
        root, board, _favor(move, calls), num_simulations=1,
        temperature=0.0, add_noise=False,
        state_history=[root], remaining_plies=1,
    )

    assert chosen == move
    assert root_value == 0.0
    assert len(calls) == 1
