import json

import pytest

from corridors.nn import model, tournament


def test_rename_checkpoint_moves_weights_metadata_and_self_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "CHECKPOINT_ROOT", tmp_path)
    (tmp_path / "old.safetensors").write_bytes(b"weights")
    (tmp_path / "old.meta.json").write_text(
        json.dumps({"checkpoint": "old", "resumed_from": "old", "epoch": 4}),
        encoding="utf-8",
    )

    assert model.rename_checkpoint("old", "new") is True

    assert not (tmp_path / "old.safetensors").exists()
    assert not (tmp_path / "old.meta.json").exists()
    assert (tmp_path / "new.safetensors").read_bytes() == b"weights"
    assert model.read_meta("new") == {
        "checkpoint": "new",
        "resumed_from": "new",
        "epoch": 4,
    }


def test_rename_checkpoint_will_not_overwrite_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "CHECKPOINT_ROOT", tmp_path)
    (tmp_path / "old.safetensors").write_bytes(b"old")
    (tmp_path / "new.safetensors").write_bytes(b"new")

    with pytest.raises(FileExistsError):
        model.rename_checkpoint("old", "new")

    assert (tmp_path / "old.safetensors").read_bytes() == b"old"
    assert (tmp_path / "new.safetensors").read_bytes() == b"new"


def test_rename_elo_checkpoint_updates_saved_history(tmp_path, monkeypatch):
    elo_path = tmp_path / "elo.json"
    monkeypatch.setattr(tournament, "ELO_PATH", elo_path)
    tournament.save_elo({
        "ratings": {"old": 123.0, "classical": 0.0},
        "games": [["old", "classical", 1.0]],
        "last_run": {
            "checkpoints": ["old"],
            "results": [["classical", "old", 0.0]],
        },
    })

    assert tournament.rename_elo_checkpoint("old", "new") is True

    data = tournament.load_elo()
    assert data["ratings"] == {"new": 123.0, "classical": 0.0}
    assert data["games"] == [["new", "classical", 1.0]]
    assert data["last_run"]["checkpoints"] == ["new"]
    assert data["last_run"]["results"] == [["classical", "new", 0.0]]
