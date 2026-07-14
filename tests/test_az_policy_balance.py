"""Action-type-balanced AlphaZero policy training tests."""

import math

import torch

from corridors.nn.actions import NUM_ACTIONS, PAWN_ACTIONS, WALL_ACTIONS
from corridors.nn.az_train import balanced_policy_log_probs, balanced_policy_loss


def _opening_target():
    target = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    target[0, 9 * 9 + 4] = 0.5
    target[0, PAWN_ACTIONS:] = 0.5 / WALL_ACTIONS
    return target


def test_balanced_training_probabilities_do_not_favor_larger_action_type():
    logits = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    probs = balanced_policy_log_probs(logits, _opening_target()).exp()

    assert torch.isclose(probs[0, :PAWN_ACTIONS].sum(), torch.tensor(0.5))
    assert torch.isclose(probs[0, PAWN_ACTIONS:].sum(), torch.tensor(0.5))


def test_network_can_still_learn_to_prefer_walls_when_evidence_supports_it():
    logits = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32)
    logits[:, PAWN_ACTIONS:] = math.log(4.0)
    probs = balanced_policy_log_probs(logits, _opening_target()).exp()

    assert torch.isclose(probs[0, PAWN_ACTIONS:].sum(), torch.tensor(0.8))


def test_balanced_policy_loss_is_finite_and_differentiable():
    logits = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, requires_grad=True)
    loss = balanced_policy_loss(logits, _opening_target())
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
