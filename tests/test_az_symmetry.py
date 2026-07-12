"""AlphaZero horizontal-symmetry augmentation tests."""

import numpy as np

from corridors.game import State, apply_move
from corridors.nn.actions import (
    NUM_ACTIONS,
    index_to_move,
    move_to_index,
    reflect_move,
    reflect_policy,
)
from corridors.nn.encoding import encode_state, reflect_encoded
from corridors.nn.mcts import Node


def test_reflecting_every_action_twice_is_identity():
    for index in range(NUM_ACTIONS):
        move = index_to_move(index)
        assert reflect_move(reflect_move(move)) == move
        assert move_to_index(reflect_move(move)) < NUM_ACTIONS


def test_policy_reflection_is_an_involution():
    policy = np.arange(NUM_ACTIONS, dtype=np.float32)
    assert np.array_equal(reflect_policy(reflect_policy(policy)), policy)


def test_encoded_reflection_matches_reflected_position():
    board, state = State.start(2, 7)
    state = apply_move(state, ("w", (4, 1, "H")))
    state = apply_move(state, ("w", (6, 5, "V")))

    reflected_board, reflected_state = State.start(6, 1)
    reflected_state = apply_move(reflected_state, reflect_move(("w", (4, 1, "H"))))
    reflected_state = apply_move(reflected_state, reflect_move(("w", (6, 5, "V"))))

    assert np.array_equal(
        reflect_encoded(encode_state(state, board)),
        encode_state(reflected_state, reflected_board),
    )


def test_puct_constant_changes_exploration_pressure():
    board, state = State.start(4, 4)
    node = Node(state, board)
    logits = np.zeros(NUM_ACTIONS, dtype=np.float32)
    node.expand(logits)
    node.N[:] = 1
    node.W[:] = 0
    node.P[:] = 0
    node.P[-1] = 1
    assert node.select_child(c_puct=2.0) == len(node.P) - 1
