import json

from corridors.nn.checkpoints import (
    ranked_checkpoint_paths, resolve_checkpoint_path, stale_elo_checkpoints,
)


def _checkpoint(root, name, elo=None):
    path = root / f"{name}.safetensors"
    path.write_bytes(b"weights")
    if elo is not None:
        path.with_suffix(".meta.json").write_text(
            json.dumps({"elo": elo}), encoding="utf-8"
        )


def test_checkpoints_rank_by_descending_elo_then_unrated_name(tmp_path):
    _checkpoint(tmp_path, "unrated_b")
    _checkpoint(tmp_path, "low", -20)
    _checkpoint(tmp_path, "high", 250)
    _checkpoint(tmp_path, "unrated_a")

    assert [path.stem for path in ranked_checkpoint_paths(tmp_path)] == [
        "high", "low", "unrated_a", "unrated_b"
    ]


def test_tournament_elo_takes_priority_over_checkpoint_metadata(tmp_path):
    _checkpoint(tmp_path, "a", 500)
    _checkpoint(tmp_path, "b", 100)
    (tmp_path / "elo.json").write_text(
        json.dumps({"ratings": {"a": 10, "b": 20}}), encoding="utf-8"
    )

    assert [path.stem for path in ranked_checkpoint_paths(tmp_path)] == ["b", "a"]


def test_curated_checkpoints_are_discovered_and_local_copy_takes_priority(tmp_path):
    best = tmp_path / "best"
    best.mkdir()
    _checkpoint(best, "curated", 50)

    assert resolve_checkpoint_path(tmp_path, "curated") == best / "curated.safetensors"
    assert [path.stem for path in ranked_checkpoint_paths(tmp_path)] == ["curated"]

    _checkpoint(tmp_path, "curated", 100)
    assert resolve_checkpoint_path(tmp_path, "curated") == tmp_path / "curated.safetensors"
    assert ranked_checkpoint_paths(tmp_path) == [tmp_path / "curated.safetensors"]


def test_stale_elo_flags_rated_checkpoints_absent_from_latest_round_robin(tmp_path):
    (tmp_path / "elo.json").write_text(json.dumps({
        "anchor": "classical",
        "ratings": {"classical": 0.0, "a": 120.0, "b": 80.0, "c": -30.0},
        "last_run": {"checkpoints": ["a", "c"]},
    }), encoding="utf-8")

    assert stale_elo_checkpoints(tmp_path) == {"b"}


def test_stale_elo_is_empty_without_a_recorded_round_robin(tmp_path):
    assert stale_elo_checkpoints(tmp_path) == set()  # no elo.json at all

    (tmp_path / "elo.json").write_text(json.dumps({
        "ratings": {"classical": 0.0, "a": 120.0},
    }), encoding="utf-8")
    assert stale_elo_checkpoints(tmp_path) == set()  # no last_run participant list
