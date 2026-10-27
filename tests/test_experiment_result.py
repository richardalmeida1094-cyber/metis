import pytest

from metis.experiment_result import ExperimentResult


def test_completed_result():
    result = ExperimentResult("exp", 2, ["one", "two"], True, None)
    assert result.experiment_id == "exp"
    assert result.requested_repetitions == 2
    assert result.results == ["one", "two"]
    assert result.completed is True
    assert result.failure_reason is None


def test_partial_result():
    result = ExperimentResult("exp", 3, ["one", "two"], False, "failure")
    assert result.requested_repetitions == 3
    assert result.results == ["one", "two"]
    assert result.completed is False
    assert result.failure_reason == "failure"


@pytest.mark.parametrize(
    "requested,results,completed,reason",
    [
        (0, [], False, "failure"),
        (-1, [], False, "failure"),
        (1, ["one", "two"], True, None),
        (2, ["one"], True, None),
        (1, ["one"], True, "failure"),
        (2, ["one"], False, None),
        (2, ["one"], False, ""),
        (2, ["one", "two"], False, "failure"),
    ],
)
def test_invalid_result(requested, results, completed, reason):
    with pytest.raises(ValueError):
        ExperimentResult("exp", requested, results, completed, reason)
