"""Draw games are labeled with the configured draw_value for both sides."""

import random

import numpy as np

from corridors.nn.actions import NUM_ACTIONS
from corridors.nn.az_selfplay import SelfPlayConfig, _play_one_game


class _Collect:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _stub_eval(_state, _board):
    return np.zeros(NUM_ACTIONS, dtype=np.float32), 0.0


def _play_drawn_game(draw_value):
    config = SelfPlayConfig(num_games=1, simulations=4, max_plies=4,
                            temperature_moves=0, draw_value=draw_value)
    queue = _Collect()
    _play_one_game(0, 0, config, _stub_eval, random.Random(7), queue)
    results = [i for i in queue.items if i[0] == 0]  # skip heartbeats
    assert results, "game produced no record"
    _wid, _game, _states, _policies, outcomes, winner, _ply = results[0]
    assert winner is None  # 4-ply cap cannot produce a win from the start
    return outcomes


def test_draw_outcomes_use_configured_draw_value():
    outcomes = _play_drawn_game(-0.2)
    assert np.allclose(outcomes, -0.2)


def test_neutral_draw_value_matches_legacy_zero():
    outcomes = _play_drawn_game(0.0)
    assert np.allclose(outcomes, 0.0)
