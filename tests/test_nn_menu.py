from corridors.nn.menu import _numbered_selections, _ranked_ratings


def test_numbered_selections_accepts_one_or_many_numbers():
    items = [{"name": "one"}, {"name": "two"}, {"name": "three"}]

    assert _numbered_selections(items, "2") == [{"name": "two"}]
    assert _numbered_selections(items, "3, 1") == [
        {"name": "three"},
        {"name": "one"},
    ]


def test_numbered_selections_ignores_invalid_and_duplicate_numbers():
    items = [{"name": "one"}, {"name": "two"}]

    assert _numbered_selections(items, "2, nope, 0, 3, 2") == [{"name": "two"}]


def test_ranked_ratings_returns_only_top_ten_with_stable_ties():
    ratings = {f"model_{i:02d}": float(i) for i in range(12)}
    ratings["aaa_tie"] = 10.0

    ranked = _ranked_ratings(ratings)

    assert len(ranked) == 10
    assert ranked[:3] == [
        ("model_11", 11.0),
        ("aaa_tie", 10.0),
        ("model_10", 10.0),
    ]
