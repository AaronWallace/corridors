"""Candidate arena scoring tests."""

from corridors.nn import az_arena


def test_arena_scores_candidate_across_color_swaps(monkeypatch):
    results = iter([1.0, 1.0, 0.5, 0.0])
    monkeypatch.setattr(az_arena, "play_pair_game", lambda *args, **kwargs: next(results))
    arena = az_arena.run_arena("incumbent", "candidate", games=4)
    # Odd games return the incumbent's score and are inverted for the candidate.
    assert arena["wins"] == 2
    assert arena["draws"] == 1
    assert arena["losses"] == 1
    assert arena["score"] == 0.625


def test_arena_reports_plies_sides_time_and_terminations(monkeypatch):
    raw_games = iter([
        {"score": 1.0, "plies": 20, "elapsed": 1.0, "termination": "goal"},
        {"score": 0.5, "plies": 50, "elapsed": 2.0, "termination": "threefold"},
        {"score": 0.0, "plies": 30, "elapsed": 1.5, "termination": "goal"},
        {"score": 1.0, "plies": 40, "elapsed": 2.5, "termination": "goal"},
    ])
    seen = []
    monkeypatch.setattr(
        az_arena, "play_pair_game", lambda *args, **kwargs: next(raw_games))

    arena = az_arena.run_arena(
        "incumbent", "candidate", games=4,
        on_game=lambda done, total, result, info: seen.append(info),
    )

    assert (arena["wins"], arena["draws"], arena["losses"]) == (1, 1, 2)
    assert arena["avg_plies"] == 35
    assert arena["min_plies"] == 20
    assert arena["max_plies_played"] == 50
    assert arena["by_side"]["P1"] == {"wins": 1, "draws": 0, "losses": 1}
    assert arena["by_side"]["P2"] == {"wins": 0, "draws": 1, "losses": 1}
    assert arena["terminations"] == {"goal": 3, "threefold": 1}
    assert [info["candidate_side"] for info in seen] == ["P1", "P2", "P1", "P2"]
