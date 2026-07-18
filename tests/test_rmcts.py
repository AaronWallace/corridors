"""RMCTS allocation, posterior, and Corridors integration tests."""

import numpy as np

from corridors.game import State, apply_move, legal_moves
from corridors.nn.actions import NUM_ACTIONS, move_to_index
from corridors.nn.rmcts import (
    assign_simulations,
    optimized_posterior,
    run_rmcts_batch,
)


def test_assign_simulations_has_exact_budget_and_expected_support():
    counts = assign_simulations(
        17, np.array([0.1, 0.2, 0.7]), np.random.default_rng(3))
    assert counts.sum() == 17
    assert np.all(counts >= 0)
    assert counts[2] > counts[1] > counts[0]


def test_optimized_posterior_favors_value_but_retains_prior_support():
    posterior = optimized_posterior(
        np.array([-0.5, 0.75], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32), 100, 1.5)
    assert np.isclose(posterior.sum(), 1.0)
    assert 0 < posterior[0] < posterior[1] < 1


def test_rmcts_uses_exact_node_budget_and_returns_legal_policy():
    board, state = State.start(4, 6)

    def evaluate(states, _boards):
        return (np.zeros((len(states), NUM_ACTIONS), dtype=np.float32),
                np.zeros(len(states), dtype=np.float32))

    result = run_rmcts_batch(
        [state], [board], evaluate, num_simulations=25,
        add_noise=False, temperature=0.0, rng=np.random.default_rng(7))[0]
    legal_indices = {move_to_index(move) for move in legal_moves(state, board)}
    assert result.nodes == 25
    assert result.evaluations == 25
    assert set(np.flatnonzero(result.policy)) == legal_indices
    assert result.move in legal_moves(state, board)
    assert np.isclose(result.policy.sum(), 1.0)


def test_rmcts_adjudicates_repetition_without_evaluating_child():
    board, state = State.start(4, 6)
    move = ("m", (9, 4))
    repeated = apply_move(state, move)

    def evaluate(states, _boards):
        logits = np.full((len(states), NUM_ACTIONS), -1e6, dtype=np.float32)
        logits[:, move_to_index(move)] = 0.0
        return logits, np.full(len(states), 0.5, dtype=np.float32)

    result = run_rmcts_batch(
        [state], [board], evaluate, num_simulations=2,
        add_noise=False, temperature=0.0,
        state_histories=[[repeated, repeated, state]],
        rng=np.random.default_rng(1))[0]
    assert result.move == move
    assert result.evaluations == 1
    assert result.value == 0.25  # one root-network vote + one draw vote

