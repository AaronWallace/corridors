"""load_dataset progress reporting and memory-budget truncation."""

import numpy as np
import pytest

from corridors.nn import datasets as ds


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "nn_data"
    root.mkdir()
    monkeypatch.setattr(ds, "DATA_ROOT", root)
    return root


def _make_shard(name: str, index: int, positions: int, outcome: int) -> int:
    """Write one shard whose outcomes all equal `outcome`; returns nbytes."""
    tensors = np.zeros((positions, 9, 11, 9), dtype=np.float32)
    outcomes = np.full(positions, outcome, dtype=np.int8)
    tt_scores = np.zeros(positions, dtype=np.float32)
    ds.write_shard(name, index, tensors, outcomes, tt_scores)
    return tensors.nbytes + outcomes.nbytes + tt_scores.nbytes


def test_available_memory_is_detectable():
    avail = ds.available_memory_bytes()
    assert avail is not None and avail > 0


def test_uncompressed_size_estimate_matches_headers(data_root):
    nbytes = _make_shard("est", 0, positions=50, outcome=1)
    path = ds.dataset_dir("est") / "shard_000.npz"
    assert ds._npz_uncompressed_nbytes(path) == nbytes


def test_load_reports_progress_per_shard(data_root):
    for i in range(3):
        _make_shard("prog", i, positions=10, outcome=1)
    events = []
    tensors, outcomes, tt = ds.load_dataset(
        "prog", on_progress=lambda done, total, msg: events.append((done, msg)))
    assert len(tensors) == len(outcomes) == len(tt) == 30
    loaded = [msg for _done, msg in events if msg.startswith("loaded ")]
    assert len(loaded) == 3
    assert "shard_000.npz" in loaded[0] and "shard_002.npz" in loaded[2]
    assert not any(msg.startswith("WARNING") for _done, msg in events)


def test_oversized_dataset_truncates_to_newest_shards(data_root):
    per_shard = _make_shard("big", 0, positions=100, outcome=0)
    _make_shard("big", 1, positions=100, outcome=1)
    _make_shard("big", 2, positions=100, outcome=2)
    events = []
    tensors, outcomes, _tt = ds.load_dataset(
        "big", memory_budget=int(per_shard * 1.5),
        on_progress=lambda done, total, msg: events.append(msg))
    warnings = [msg for msg in events if msg.startswith("WARNING")]
    assert len(warnings) == 1
    assert "1/3 shards" in warnings[0]
    # Only the newest shard survives a 1.5-shard budget.
    assert len(tensors) == 100
    assert set(outcomes.tolist()) == {2}


def test_budget_keeps_at_least_one_shard(data_root):
    _make_shard("tiny", 0, positions=20, outcome=0)
    _make_shard("tiny", 1, positions=20, outcome=1)
    tensors, outcomes, _tt = ds.load_dataset("tiny", memory_budget=1)
    assert len(tensors) == 20
    assert set(outcomes.tolist()) == {1}


def test_budget_none_disables_truncation(data_root):
    _make_shard("all", 0, positions=25, outcome=0)
    _make_shard("all", 1, positions=25, outcome=1)
    tensors, _outcomes, _tt = ds.load_dataset("all", memory_budget=None)
    assert len(tensors) == 50
