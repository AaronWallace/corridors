"""Tests for persisted, hardware-specific AlphaZero benchmark defaults."""

from corridors import settings
from corridors.nn import az_selfplay


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
