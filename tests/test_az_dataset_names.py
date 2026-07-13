"""Self-play dataset naming tests."""

import json

from corridors.nn.az_menu import _auto_dataset_name
from corridors.nn import az_selfplay


def test_cpu_dataset_name_is_short_and_descriptive():
    name = _auto_dataset_name(
        prefix="az", games=62, simulations=50, max_plies=30,
        device="cpu", workers=31, batch_size=64, concurrency=1,
        search_params={}, timestamp="20260712-120000",
    )
    assert name == "az_20260712-120000_g62_s50"


def test_gpu_dataset_name_keeps_details_in_metadata_not_path():
    name = _auto_dataset_name(
        prefix="azloop", games=140, simulations=200, max_plies=150,
        device="cuda", workers=14, batch_size=64, concurrency=10,
        search_params={"c_puct": 2.0, "dirichlet_alpha": 0.05},
        timestamp="20260712-120000",
    )
    assert name == "azloop_20260712-120000_g140_s200"


def test_full_generation_settings_are_saved_beside_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(az_selfplay, "AZ_DATA_ROOT", tmp_path)
    config = az_selfplay.SelfPlayConfig(
        num_games=140, simulations=500, workers=14, batch_size=64,
        concurrent_games=0, device="cuda", c_puct=1.5,
    )
    path = az_selfplay.save_run_config(
        "azloop_short", config, mode="loop", promotion_score=0.51)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["selfplay"]["simulations"] == 500
    assert saved["selfplay"]["batch_size"] == 64
    assert saved["promotion_score"] == 0.51
