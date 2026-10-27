import pytest

from metis.worlds.family_a import A0Environment


def test_reset_returns_start():
    environment = A0Environment()

    assert environment.reset() == "start"


def test_right_returns_success():
    environment = A0Environment()

    assert environment.step("right") == "success"


def test_left_returns_failure():
    environment = A0Environment()

    assert environment.step("left") == "failure"


def test_invalid_action_raises_value_error():
    environment = A0Environment()

    with pytest.raises(ValueError):
        environment.step("up")
