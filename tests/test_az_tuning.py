"""Tests for persisted, hardware-specific AlphaZero benchmark defaults."""

from types import SimpleNamespace

import numpy as np

from corridors import settings
from corridors.nn import az_net, az_selfplay
from corridors.nn.actions import NUM_ACTIONS
from corridors.nn.az_net import AZNet
from corridors.nn.az_menu import _EpochFitTrend
from corridors.nn.az_train import (
    AZTrainConfig, _ValidationEarlyStopper, _resolved_early_stop_min_epochs,
    _resolved_max_epochs, train_az,
)
from corridors.nn.encoding import NUM_PLANES, NCOLS, NROWS


def test_tuning_profiles_are_isolated_by_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_PATH", tmp_path / "corridors.json")
    cpu_key = az_selfplay.hardware_tuning_key("cpu", 64)
    gpu_key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1)

    az_selfplay.save_tuning_profile(cpu_key, {"workers": 63})
    az_selfplay.save_tuning_profile(
        gpu_key, {"workers": 14, "inference_batch": 64, "concurrency": 10})

    profiles = settings.load()["az_tuning_profiles"]
    assert profiles[cpu_key] == {"workers": 63}
    assert profiles[gpu_key] == {
        "workers": 14, "inference_batch": 64, "concurrency": 10,
    }


def test_profile_overrides_only_selfplay_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_PATH", tmp_path / "corridors.json")
    key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1)
    az_selfplay.save_tuning_profile(
        key, {"workers": 12, "inference_batch": 96, "concurrency": 8})
    defaults = {
        "hardware_key": key,
        "workers": 14,
        "inference_batch": 64,
        "concurrency": 10,
        "games_per_iter": 140,
        "train_batch": 128,
        "benchmark_tuned": False,
    }

    tuned = az_selfplay._apply_tuning_profile(defaults)
    assert tuned["workers"] == 12
    assert tuned["inference_batch"] == 96
    assert tuned["concurrency"] == 8
    assert tuned["train_batch"] == 128
    assert tuned["benchmark_tuned"] is True


def _epoch(train, val):
    return SimpleNamespace(train_loss=train, val_loss=val)


def test_epoch_fit_trend_labels_productive_generalization():
    trend = _EpochFitTrend()
    assert "baseline" in trend.update(_epoch(3.0, 3.1))
    status = trend.update(_epoch(2.8, 2.9))
    assert "productive" in status
    assert "fit █████ 100%" in status


def test_epoch_fit_trend_distinguishes_plateau_from_overfitting():
    plateau = _EpochFitTrend()
    plateau.update(_epoch(3.0, 3.0))
    assert "plateau" in plateau.update(_epoch(2.8, 3.0001))

    diverging = _EpochFitTrend()
    diverging.update(_epoch(3.0, 3.0))
    assert "watch divergence" in diverging.update(_epoch(2.8, 3.02))
    assert "overfitting" in diverging.update(_epoch(2.6, 3.04))


def test_validation_early_stopper_ignores_tiny_improvements():
    stopper = _ValidationEarlyStopper(patience=3, min_epochs=5, min_delta=0.001)
    values = [3.0, 2.8, 2.7, 2.699, 2.6985, 2.698]
    decisions = [stopper.update(epoch, value) for epoch, value in enumerate(values, 1)]

    assert all(not stop for stop, _reason in decisions[:-1])
    assert decisions[-1][0] is True
    assert "0.10% validation improvement" in decisions[-1][1]


def test_validation_early_stopper_resets_after_meaningful_gain():
    stopper = _ValidationEarlyStopper(patience=2, min_epochs=3, min_delta=0.001)
    stopper.update(1, 3.0)
    stopper.update(2, 2.999)
    stop, _reason = stopper.update(3, 2.99)

    assert stop is False
    assert stopper.stale_epochs == 0


def test_automatic_minimum_epochs_scales_with_iteration_budget():
    assert _resolved_early_stop_min_epochs(AZTrainConfig(epochs=20)) == 7
    assert _resolved_early_stop_min_epochs(AZTrainConfig(epochs=10)) == 5
    assert _resolved_early_stop_min_epochs(AZTrainConfig(epochs=4)) == 4
    assert _resolved_max_epochs(AZTrainConfig(epochs=20)) == 30
    assert _resolved_max_epochs(AZTrainConfig(epochs=10)) == 15
    assert _resolved_max_epochs(AZTrainConfig(epochs=10, max_epochs=12)) == 12


def test_early_stopper_would_end_the_observed_plateau_at_epoch_18():
    values = [
        2.6970, 2.6644, 2.6350, 2.6258, 2.6056, 2.5976, 2.5959,
        2.5819, 2.5862, 2.5652, 2.5598, 2.5563, 2.5509, 2.5460,
        2.5407, 2.5406, 2.5395, 2.5407, 2.5408, 2.5396,
    ]
    stopper = _ValidationEarlyStopper(patience=3, min_epochs=7, min_delta=0.001)
    stopped_at = next(
        (epoch for epoch, value in enumerate(values, 1)
         if stopper.update(epoch, value)[0]),
        None,
    )

    assert stopped_at == 18


def test_train_az_actually_ends_iteration_and_keeps_stop_reason(monkeypatch):
    monkeypatch.setattr(az_net, "AZNet", lambda: AZNet(channels=4, blocks=1))
    monkeypatch.setattr(az_net, "save_checkpoint", lambda *_args, **_kwargs: None)
    states = np.zeros((20, NUM_PLANES, NROWS, NCOLS), dtype=np.float32)
    policies = np.zeros((20, NUM_ACTIONS), dtype=np.float32)
    policies[:, 0] = 1.0
    outcomes = np.zeros((20,), dtype=np.float32)
    seen = []

    result = train_az(
        states, policies, outcomes,
        AZTrainConfig(
            epochs=10, batch_size=4, lr=0, lr_min=0,
            reflection_prob=0, device="cpu",
            early_stop_patience=2, early_stop_min_epochs=2,
        ),
        on_epoch=seen.append,
    )

    assert result["stopped_early"] is True
    assert result["epochs_completed"] == 3
    assert result["best_epoch"] == 1
    assert result["stop_reason"]
    assert seen[-1].will_stop is True


def test_train_az_extends_past_soft_target_when_stopping_is_disabled(monkeypatch):
    monkeypatch.setattr(az_net, "AZNet", lambda: AZNet(channels=4, blocks=1))
    monkeypatch.setattr(az_net, "save_checkpoint", lambda *_args, **_kwargs: None)
    states = np.zeros((20, NUM_PLANES, NROWS, NCOLS), dtype=np.float32)
    policies = np.zeros((20, NUM_ACTIONS), dtype=np.float32)
    policies[:, 0] = 1.0
    outcomes = np.zeros((20,), dtype=np.float32)
    seen = []

    result = train_az(
        states, policies, outcomes,
        AZTrainConfig(
            epochs=2, batch_size=4, lr=0, lr_min=0,
            reflection_prob=0, device="cpu", early_stopping=False,
        ),
        on_epoch=seen.append,
    )

    assert result["target_epochs"] == 2
    assert result["max_epochs"] == 3
    assert result["epochs_completed"] == 3
    assert result["extended"] is True
    assert seen[1].extension_started is True
