from corridors.nn.menu import _numbered_selections


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
