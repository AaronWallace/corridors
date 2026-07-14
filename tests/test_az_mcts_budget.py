import random

from corridors.nn.az_selfplay import (
    SelfPlayConfig,
    expected_mcts_budget,
    mcts_budget_bounds,
    sample_mcts_budget,
)


def test_legacy_simulation_setting_remains_fixed():
    config = SelfPlayConfig(simulations=200)

    assert mcts_budget_bounds(config) == (200, 200)
    assert expected_mcts_budget(config) == 200
    assert {sample_mcts_budget(config, random.Random(seed)) for seed in range(20)} == {200}


def test_weighted_budget_covers_range_and_favors_stronger_searches():
    config = SelfPlayConfig(simulations=250, min_mcts=100, max_mcts=250)
    rng = random.Random(12345)
    samples = [sample_mcts_budget(config, rng) for _ in range(20_000)]

    assert min(samples) >= 100
    assert max(samples) <= 250
    assert abs(sum(samples) / len(samples) - 212.5) < 1.0
    assert sum(sample >= 200 for sample in samples) / len(samples) > 0.68


def test_reversed_bounds_are_safely_normalized():
    config = SelfPlayConfig(simulations=100, min_mcts=250, max_mcts=100)

    assert mcts_budget_bounds(config) == (100, 250)
