"""Local web-game API tests."""

import re
from types import SimpleNamespace

from corridors import web
from corridors.nn.az_net import AZNet


def test_rapid_autoplay_delays_allow_zero_seconds():
    html = (web.STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<input id="turnDelay"[^>]+min="0"', html)
    assert re.search(r'<input id="endgameDelay"[^>]+min="0"', html)


def test_new_human_game_exposes_legal_moves():
    game_id, game = web._new_game({
        "mode": "human-ai", "humanSide": "1",
        "ai": {"kind": "classical", "depth": 2, "timeLimit": 0.1},
        "p1Col": 4, "p2Col": 5,
    })
    payload = web._game_json(game_id, game)
    try:
        assert payload["players"]["1"]["kind"] == "human"
        assert payload["players"]["2"]["kind"] == "classical"
        assert payload["plies"] == 0
        assert {tuple(m["at"]) for m in payload["legal"] if m["kind"] == "m"} == {(9, 4)}
        assert len([m for m in payload["legal"] if m["kind"] == "w"]) == 128
    finally:
        web.GAMES.pop(game_id, None)


def test_web_game_stops_on_threefold_repetition():
    game_id, game = web._new_game({
        "mode": "human-ai", "humanSide": "1",
        "ai": {"kind": "classical", "depth": 1, "timeLimit": 0.1},
        "p1Col": 4, "p2Col": 5,
    })
    try:
        game.state_history = [game.state, game.state, game.state]
        web._adjudicate_draw(game)
        payload = web._game_json(game_id, game)
        assert payload["gameOver"] is True
        assert payload["drawReason"] == "threefold repetition"
        assert payload["winner"] is None
        assert payload["legal"] == []
    finally:
        web.GAMES.pop(game_id, None)


def test_web_game_retains_maximum_ply_fallback():
    game_id, game = web._new_game({
        "mode": "human-ai", "humanSide": "1",
        "ai": {"kind": "classical", "depth": 1, "timeLimit": 0.1},
        "p1Col": 4, "p2Col": 5, "maxPlies": 2,
    })
    try:
        game.history = [{}, {}]
        web._adjudicate_draw(game)
        assert game.draw_reason == "maximum plies"
    finally:
        web.GAMES.pop(game_id, None)


def test_agent_spec_rejects_missing_checkpoint():
    try:
        web._agent_spec({"kind": "model", "checkpoint": "does_not_exist"})
    except ValueError as exc:
        assert "checkpoint not found" in str(exc)
    else:
        raise AssertionError("missing checkpoint accepted")


def test_agent_spec_accepts_curated_checkpoint(tmp_path, monkeypatch):
    best = tmp_path / "best"
    best.mkdir()
    (best / "shared.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(web, "CHECKPOINT_ROOT", tmp_path)

    assert web._agent_spec({"kind": "model", "checkpoint": "shared"}) == {
        "kind": "model",
        "checkpoint": "shared",
    }


def test_web_rejects_an_illegal_ai_move(monkeypatch):
    game_id, game = web._new_game({
        "mode": "ai-ai",
        "p1": {"kind": "classical", "depth": 1, "timeLimit": 0},
        "p2": {"kind": "classical", "depth": 1, "timeLimit": 0},
        "p1Col": 4,
        "p2Col": 5,
    })
    try:
        monkeypatch.setattr(
            web.REGISTRY,
            "move",
            lambda *_args: (("m", (10, 4)), {"solver": "broken", "elapsed": 0}),
        )
        try:
            web._validated_ai_move(game, game.players["1"])
        except RuntimeError as exc:
            assert "illegal move" in str(exc)
        else:
            raise AssertionError("web backend accepted an illegal AI move")
    finally:
        web.GAMES.pop(game_id, None)


def test_cnn_explorer_analyzes_a_live_web_game():
    checkpoint = "test_explorer_model"
    web.REGISTRY.models[checkpoint] = SimpleNamespace(
        model=AZNet(channels=4, blocks=1).eval(), arch="az",
    )
    game_id, _game = web._new_game({
        "mode": "human-ai", "humanSide": "1",
        "ai": {"kind": "classical", "depth": 1, "timeLimit": 0},
        "p1Col": 2, "p2Col": 7,
    })
    try:
        result = web._explore_game({
            "checkpoint": checkpoint, "gameId": game_id,
            "layer": "block_1", "channel": 2,
        })
        assert result["checkpoint"] == checkpoint
        assert result["game"]["id"] == game_id
        assert result["selectedActivation"]["layer"] == "block_1"
        assert result["policy"]
    finally:
        web.GAMES.pop(game_id, None)
        web.REGISTRY.models.pop(checkpoint, None)
