"""Dataset listing/deletion: nested AlphaZero runs must be manageable
individually, and the ``alphazero`` container must never be deletable as a
single "dataset" (that once wiped every run at once)."""

import json
import os

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


def test_dataset_modified_time_uses_newest_shard_or_manifest(data_root):
    run = data_root / "alphazero" / "dated"
    _write_shard(run)
    manifest = run / "run.json"
    manifest.write_text(json.dumps({"selfplay": {}}), encoding="utf-8")
    os.utime(run / "shard_0000.npz", (1_000, 1_000))
    os.utime(manifest, (2_000, 2_000))

    [item] = ds.list_datasets()

    assert item["modified"] == 2_000


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


def test_listing_indexes_only_new_or_changed_shards(data_root, monkeypatch):
    run = data_root / "alphazero" / "cached"
    _write_shard(run, 0)
    calls = []
    original = ds._npz_array_rows

    def counted(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(ds, "_npz_array_rows", counted)
    ds.list_datasets()
    ds.list_datasets()
    _write_shard(run, 1)
    ds.list_datasets()

    assert calls == ["shard_0000.npz", "shard_0001.npz"]
    index = json.loads((run / ds.INDEX_FILENAME).read_text(encoding="utf-8"))
    assert set(index["shards"]) == {"shard_0000.npz", "shard_0001.npz"}
    assert index["summary"]["positions"] == 2


def test_index_prunes_removed_shards_and_updates_positions(data_root):
    run = data_root / "alphazero" / "trimmed"
    _write_shard(run, 0)
    _write_shard(run, 1)
    [before] = ds.list_datasets()
    assert before["positions"] == 2

    (run / "shard_0000.npz").unlink()
    [after] = ds.list_datasets()

    assert after["positions"] == 1
    index = json.loads((run / ds.INDEX_FILENAME).read_text(encoding="utf-8"))
    assert set(index["shards"]) == {"shard_0001.npz"}
    assert index["summary"]["positions"] == 1
    assert index["summary"]["shards"] == 1


def test_archive_hides_dataset_and_restore_returns_it(data_root):
    _write_shard(data_root / "alphazero" / "run_a")

    assert ds.archive_dataset("alphazero/run_a") is True
    assert ds.list_datasets() == []
    archived = ds.list_archived_datasets()
    assert [item["name"] for item in archived] == ["alphazero/run_a"]
    assert archived[0]["archived"] is True

    assert ds.restore_dataset("alphazero/run_a") is True
    assert [item["name"] for item in ds.list_datasets()] == ["alphazero/run_a"]
    assert ds.list_archived_datasets() == []


def test_archiving_refuses_to_overwrite_existing_archive(data_root):
    _write_shard(data_root / "same")
    _write_shard(data_root / ds.ARCHIVE_DIRNAME / "same")

    assert ds.archive_dataset("same") is False
    assert (data_root / "same").exists()


def test_active_and_archived_lists_are_newest_first(data_root):
    old = data_root / "old"
    new = data_root / "new"
    _write_shard(old)
    _write_shard(new)
    os.utime(old / "shard_0000.npz", (1_000, 1_000))
    os.utime(new / "shard_0000.npz", (2_000, 2_000))

    assert [item["name"] for item in ds.list_datasets()] == ["new", "old"]
    assert ds.archive_dataset("new") is True
    assert ds.archive_dataset("old") is True
    assert [item["name"] for item in ds.list_archived_datasets()] == ["new", "old"]


def test_share_moves_nested_dataset_and_keeps_it_trainable(data_root):
    original = data_root / "alphazero" / "valuable"
    _write_shard(original)
    (original / "run.json").write_text(
        json.dumps({"games": 1, "positions": 1, "selfplay": {"simulations": 200}}),
        encoding="utf-8",
    )
    ds.list_datasets()  # create the machine-local metadata index first

    shared_name = ds.share_dataset("alphazero/valuable")

    assert shared_name == "shared/alphazero/valuable"
    assert not original.exists()
    shared = data_root / "shared" / "alphazero" / "valuable"
    assert (shared / "shard_0000.npz").exists()
    assert (shared / ds.INDEX_FILENAME).exists()
    [item] = ds.list_datasets()
    assert item["name"] == shared_name
    assert item["kind"] == "alphazero"


def test_share_refuses_shared_archive_and_duplicate_destinations(data_root):
    _write_shard(data_root / "active")
    _write_shard(data_root / "shared" / "active")
    _write_shard(data_root / "shared" / "already")

    assert ds.share_dataset("active") is None
    assert ds.share_dataset("shared/already") is None
    assert ds.share_dataset("archive/nope") is None
    assert (data_root / "active").exists()
