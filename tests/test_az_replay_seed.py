"""One-time seed datasets share the loop's replay window correctly."""

import numpy as np

from corridors.nn import az_train
from corridors.nn.actions import NUM_ACTIONS
from corridors.nn.encoding import NCOLS, NROWS, NUM_PLANES


def _shard(root, run, index, positions, marker):
    directory = root / run
    directory.mkdir(parents=True, exist_ok=True)
    states = np.full(
        (positions, NUM_PLANES, NROWS, NCOLS), marker, dtype=np.float32)
    policies = np.zeros((positions, NUM_ACTIONS), dtype=np.float32)
    policies[:, 0] = 1
    outcomes = np.zeros((positions,), dtype=np.float32)
    np.savez_compressed(
        directory / f"shard_{index:04d}.npz",
        states=states, policies=policies, outcomes=outcomes,
    )


def test_combined_replay_cap_prioritizes_current_run(tmp_path, monkeypatch):
    monkeypatch.setattr(az_train, "AZ_DATA_ROOT", tmp_path)
    _shard(tmp_path, "seed", 0, 2, 10)
    _shard(tmp_path, "seed", 1, 2, 11)
    _shard(tmp_path, "current", 0, 2, 20)
    _shard(tmp_path, "current", 1, 2, 21)

    states, policies, outcomes = az_train.load_training_datasets(
        ["seed", "current"], max_positions=5)

    # The cap is shard-granular: newest current data is retained first, then
    # the freshest seed shard fills the remaining replay capacity.
    assert len(states) == 6
    assert list(states[:, 0, 0, 0]) == [11, 11, 20, 20, 21, 21]
    assert len(policies) == len(outcomes) == 6


def test_combined_replay_provenance_matches_selected_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(az_train, "AZ_DATA_ROOT", tmp_path)
    _shard(tmp_path, "seed", 0, 3, 10)
    _shard(tmp_path, "current", 0, 3, 20)
    _shard(tmp_path, "current", 1, 3, 21)

    provenance = az_train.dataset_provenance(
        ["seed", "current"], max_positions=5)

    assert provenance["data_runs"] == ["current"]
    assert provenance["data_shards"] == 2
    assert provenance["data_shard_names"] == [
        "current/shard_0000.npz", "current/shard_0001.npz",
    ]


def test_shared_alphazero_dataset_loads_from_shared_tree(tmp_path, monkeypatch):
    az_root = tmp_path / "alphazero"
    monkeypatch.setattr(az_train, "AZ_DATA_ROOT", az_root)
    _shard(tmp_path, "shared/alphazero/curated", 0, 3, 42)

    states, policies, outcomes = az_train.load_training_data(
        "shared/alphazero/curated")

    assert len(states) == len(policies) == len(outcomes) == 3
    assert list(states[:, 0, 0, 0]) == [42, 42, 42]
