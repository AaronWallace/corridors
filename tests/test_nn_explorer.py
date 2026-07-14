"""CNN explorer returns real, legal, JSON-ready model internals."""

from corridors.game import State, legal_moves
from corridors.nn.az_net import AZNet
from corridors.nn.explorer import analyze_model
from corridors.nn.model import ValueNet


def test_alphazero_explorer_reports_activations_policy_and_weights():
    board, state = State.start(3, 6)
    model = AZNet(channels=8, blocks=2).eval()

    result = analyze_model(
        model, "az", state, board, selected_layer="block_1",
        channel=3, weight_out=2, weight_in=1,
    )

    assert result["architecture"] == "az"
    assert result["parameters"] == model.num_params()
    assert len(result["inputPlanes"]) == 9
    assert result["selectedActivation"]["layer"] == "block_1"
    assert result["selectedActivation"]["channel"] == 3
    assert result["selectedActivation"]["channels"] == 8
    assert len(result["selectedActivation"]["map"]) == 11
    assert result["weights"]["kernel"]["out"] == 2
    assert sum(bin_["count"] for bin_ in result["weights"]["histogram"]) == result["weights"]["count"]
    legal = set(legal_moves(state, board))
    assert result["policy"]
    assert all((item["move"]["kind"], tuple(item["move"]["at"])) in legal
               for item in result["policy"])


def test_value_network_explorer_has_value_but_no_policy_head():
    board, state = State.start(4, 4)
    model = ValueNet(channels=8, blocks=2).eval()

    result = analyze_model(model, "value", state, board)

    assert result["architecture"] == "value"
    assert result["policy"] == []
    assert -1 <= result["value"] <= 1
    assert any(layer["name"] == "global_pool" for layer in result["layers"])
