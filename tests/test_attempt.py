import pytest

from metis.attempt import Attempt


@pytest.mark.parametrize(
    "attempt_id, run_id, cost, status",
    [
        ("ATT-001", "RUN-001", 1, "RESERVED"),
        ("ATT-002", "RUN-001", 10, "DISPATCHED"),
        ("ATT-003", "RUN-002", 2, "COMPLETED"),
        ("ATT-004", "RUN-002", 3, "UNKNOWN"),
    ],
)
def test_valid_attempt(attempt_id, run_id, cost, status):
    attempt = Attempt(attempt_id, run_id, cost, status)

    assert attempt.attempt_id == attempt_id
    assert attempt.run_id == run_id
    assert attempt.cost == cost
    assert attempt.status == status


@pytest.mark.parametrize(
    "attempt_id, run_id, cost, status",
    [
        ("", "RUN-001", 1, "RESERVED"),
        ("   ", "RUN-001", 1, "RESERVED"),
        ("ATT-001", "", 1, "RESERVED"),
        ("ATT-001", "   ", 1, "RESERVED"),
        ("ATT-001", "RUN-001", 0, "RESERVED"),
        ("ATT-001", "RUN-001", -1, "RESERVED"),
        ("ATT-001", "RUN-001", True, "RESERVED"),
        ("ATT-001", "RUN-001", False, "RESERVED"),
        ("ATT-001", "RUN-001", 1.5, "RESERVED"),
        ("ATT-001", "RUN-001", "1", "RESERVED"),
        ("ATT-001", "RUN-001", 1, "RUNNING"),
        ("ATT-001", "RUN-001", 1, ""),
    ],
)
def test_invalid_attempt(attempt_id, run_id, cost, status):
    with pytest.raises(ValueError):
        Attempt(attempt_id, run_id, cost, status)
