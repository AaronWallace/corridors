"""Dataset listing/deletion: nested AlphaZero runs must be manageable
individually, and the ``alphazero`` container must never be deletable as a
single "dataset" (that once wiped every run at once)."""

import json

import numpy as np
import pytest

from corridors.nn import datasets as ds


def _write_shard(d, idx=0):
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"shard_{idx:04d}.npz",
        states=np.zeros((1, 1), dtype=np.float32),
        policies=np.zeros((1, 1), dtype=np.float32),
        outcomes=np.zeros((1,), dtype=np.int8),
    )


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "nn_data"
    root.mkdir()
    monkeypatch.setattr(ds, "DATA_ROOT", root)
    return root


def test_container_is_expanded_into_individual_runs(data_root):
    _write_shard(data_root / "classic")                     # leaf dataset
    _write_shard(data_root / "alphazero" / "run_a")         # nested run
    _write_shard(data_root / "alphazero" / "run_b")         # nested run

    names = {d["name"] for d in ds.list_datasets()}
    assert names == {"classic", "alphazero/run_a", "alphazero/run_b"}
    # The container itself is never presented as a dataset.
    assert "alphazero" not in names


def test_legacy_alphazero_run_reports_useful_metadata(data_root):
    run = data_root / "alphazero" / "run_a"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "mode": "loop",
        "iterations": 3,
        "selfplay": {
            "num_games": 50,
            "simulations": 200,
            "max_plies": 120,
            "device": "cuda",
            "workers": 64,
        },
    }), encoding="utf-8")
    for index, positions in enumerate((3, 5)):
        np.savez_compressed(
            run / f"shard_{index:04d}.npz",
            states=np.zeros((positions, 1), dtype=np.float32),
            policies=np.zeros((positions, 1), dtype=np.float32),
            outcomes=np.zeros((positions,), dtype=np.int8),
        )

    [item] = ds.list_datasets()
    assert item["games"] == 50
    assert item["positions"] == 8
    assert item["kind"] == "alphazero"
    assert item["config"]["simulations"] == 200
    assert item["config"]["max_plies"] == 120


def test_pre_metadata_alphazero_name_recovers_useful_fields(data_root):
    run = data_root / "alphazero" / "azloop_20260712_g100_s200_p150_cuda_w14"
    for index in range(4):
        _write_shard(run, index)

    [item] = ds.list_datasets()
    assert item["games"] == 100
    assert item["positions"] == 4
    assert item["kind"] == "alphazero"
    assert item["config"] == {"simulations": 200, "max_plies": 150}


def test_delete_container_is_refused(data_root):
    _write_shard(data_root / "alphazero" / "run_a")
    _write_shard(data_root / "alphazero" / "run_b")

    assert ds.delete_dataset("alphazero") is False
    assert (data_root / "alphazero" / "run_a").exists()
    assert (data_root / "alphazero" / "run_b").exists()


def test_delete_single_run_leaves_siblings(data_root):
    _write_shard(data_root / "alphazero" / "run_a")
    _write_shard(data_root / "alphazero" / "run_b")

    assert ds.delete_dataset("alphazero/run_a") is True
    assert not (data_root / "alphazero" / "run_a").exists()
    assert (data_root / "alphazero" / "run_b").exists()


def test_delete_refuses_paths_outside_data_root(data_root, tmp_path):
    outside = tmp_path / "outside"
    _write_shard(outside)
    assert ds.delete_dataset("../outside") is False
    assert outside.exists()


def test_delete_missing_dataset_returns_false(data_root):
    assert ds.delete_dataset("nope") is False
