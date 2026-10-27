import pytest

from metis.worlds.discoverphysics.query_unit import QueryUnit


@pytest.mark.parametrize(
    "duration, expected",
    [
        (2.0, 5.0),
        (5.0, 5.0),
        (8.5, 8.5),
    ],
)
def test_effective_duration(duration, expected):
    query = QueryUnit(1, duration, 1)

    assert query.effective_duration() == expected


def test_count_query_returns_one():
    query = QueryUnit(1, 0.5, 1)

    assert query.count_query() == 1
