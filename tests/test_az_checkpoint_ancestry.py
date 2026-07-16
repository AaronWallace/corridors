"""AlphaZero checkpoint ancestry tree tests."""

import json
from io import StringIO

from rich.console import Console

from corridors.nn import az_menu


def _checkpoint(root, name: str, meta: dict) -> None:
    (root / f"{name}.safetensors").write_bytes(b"weights")
    (root / f"{name}.meta.json").write_text(
        json.dumps({"arch": "az", **meta}), encoding="utf-8"
    )


def _render(tree) -> str:
    output = StringIO()
    Console(file=output, force_terminal=False, width=180).print(tree)
    return output.getvalue()


def test_ancestry_tree_shows_available_and_deleted_checkpoint_details(
        tmp_path, monkeypatch):
    monkeypatch.setattr(az_menu, "CHECKPOINT_ROOT", tmp_path)
    _checkpoint(tmp_path, "az_root", {"epoch": 4, "elo": 100})
    _checkpoint(tmp_path, "az_child", {
        "epoch": 8,
        "data_run": "run_8",
        "promoted_from": "az_20260714-112052_candidate",
        "seeded_from": "az_root",
    })
    (tmp_path / "elo.json").write_text(json.dumps({
        "ratings": {
            "az_child": 250,
            "az_root": 100,
            "az_20260714-112052_candidate": 175,
        }
    }), encoding="utf-8")

    rendered = _render(az_menu._checkpoint_ancestry_tree())

    assert "az_child" in rendered
    assert "Elo +250" in rendered
    assert "epoch 8" in rendered
    assert "data run_8" in rendered
    assert "promoted from" in rendered
    assert "az_20260714-112052_candidate  [deleted]" in rendered
    assert "Elo +175" in rendered
    assert "date ~2026-07-14 11:20" in rendered
    assert "seeded from" in rendered
    assert "az_root" in rendered


def test_ancestry_tree_ignores_self_resume_and_marks_real_cycles(
        tmp_path, monkeypatch):
    monkeypatch.setattr(az_menu, "CHECKPOINT_ROOT", tmp_path)
    _checkpoint(tmp_path, "az_a", {"resumed_from": "az_a", "seeded_from": "az_b"})
    _checkpoint(tmp_path, "az_b", {"resumed_from": "az_a"})

    rendered = _render(az_menu._checkpoint_ancestry_tree())

    assert "cycle in recorded ancestry" in rendered
    assert "resumed from  az_a" in rendered
