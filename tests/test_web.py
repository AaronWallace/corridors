"""Local web-game API tests."""

from corridors import web


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
        assert {tuple(m["at"]) for m in payload["legal"] if m["kind"] == "m"} == {(9, 4)}
        assert len([m for m in payload["legal"] if m["kind"] == "w"]) == 128
    finally:
        web.GAMES.pop(game_id, None)


def test_agent_spec_rejects_missing_checkpoint():
    try:
        web._agent_spec({"kind": "model", "checkpoint": "does_not_exist"})
    except ValueError as exc:
        assert "checkpoint not found" in str(exc)
    else:
        raise AssertionError("missing checkpoint accepted")
