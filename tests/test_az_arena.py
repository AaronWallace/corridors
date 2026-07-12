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
