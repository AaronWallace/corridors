"""Round-robin game adjudication tests."""

from corridors.game import State
from corridors.nn import tournament


def test_tournament_defensively_scores_legacy_no_move_state_as_loss(monkeypatch):
    board, _ = State.start(4, 4)
    state = State(
        p1=(5, 4), p2=(4, 4),
        p1_walls_left=0, p2_walls_left=0,
        walls=frozenset({
            (5, 3, "H"), (4, 3, "V"), (5, 4, "V"), (3, 3, "H"),
        }),
        turn=1,
    )

    class FixedStart:
        @staticmethod
        def start(**_kwargs):
            return board, state

    def no_move_agent(*_args, **_kwargs):
        def pick(_state, _board):
            raise RuntimeError("no legal moves")
        return pick

    monkeypatch.setattr(tournament, "State", FixedStart)
    monkeypatch.setattr(tournament, "_get_mover", no_move_agent)
    a = tournament.AgentSpec("classical", "a")
    b = tournament.AgentSpec("classical", "b")

    assert tournament.play_pair_game(a, b, 0) == 0.0
    details = tournament.play_pair_game(a, b, 0, return_details=True)
    assert details["score"] == 0.0
    assert details["plies"] == 0
    assert details["termination"] == "no legal moves"
    assert details["p1"] == "a"
    assert details["p2"] == "b"
