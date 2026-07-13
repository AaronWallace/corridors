"""Dataset listing/deletion: nested AlphaZero runs must be manageable
individually, and the ``alphazero`` container must never be deletable as a
single "dataset" (that once wiped every run at once)."""

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
