import pytest

from metis.basic_environment_adapter import BasicEnvironmentAdapter
from metis.experiment import Experiment


class FakeEnvironment:
    def __init__(self, fail_on=None, reset_fail_on=None, error_type=ValueError):
        self.actions = []
        self.events = []
        self.resets = 0
        self.fail_on = fail_on
        self.reset_fail_on = reset_fail_on
        self.error_type = error_type

    def reset(self) -> str:
        self.events.append("reset")
        self.resets += 1
        if self.resets == self.reset_fail_on:
            raise self.error_type("reset failure")
        return "initial observation"

    def step(self, action: str) -> str:
        self.events.append("step")
        self.actions.append(action)
        if len(self.actions) == self.fail_on:
            raise self.error_type("external failure")
        return f"observation {len(self.actions)}"


def test_reset_updates_observe():
    adapter = BasicEnvironmentAdapter(FakeEnvironment())
    assert adapter.reset() == "initial observation"
    assert adapter.observe() == "initial observation"


def test_step_updates_observe():
    environment = FakeEnvironment()
    adapter = BasicEnvironmentAdapter(environment)
    assert adapter.step("move") == "observation 1"
    assert adapter.observe() == "observation 1"
    assert environment.actions == ["move"]


def test_capabilities():
    capabilities = BasicEnvironmentAdapter(FakeEnvironment()).capabilities()
    assert capabilities.supports_actions is True
    assert capabilities.supports_experiments is True
    assert capabilities.supports_snapshot is False
    assert capabilities.supports_restore is False


def test_experiment_executes_one_action_per_repetition():
    environment = FakeEnvironment()
    adapter = BasicEnvironmentAdapter(environment)
    result = adapter.execute_experiment(Experiment("exp", "objective", "move", 3))
    assert environment.actions == ["move"] * 3
    assert environment.resets == 3
    assert environment.events == ["reset", "step"] * 3
    assert result.experiment_id == "exp"
    assert result.completed is True
    assert result.requested_repetitions == 3
    assert result.results == ["observation 1", "observation 2", "observation 3"]
    assert result.failure_reason is None
    assert adapter.observe() == "observation 3"


def test_experiment_preserves_partial_results_without_extra_steps():
    environment = FakeEnvironment(fail_on=3)
    adapter = BasicEnvironmentAdapter(environment)
    result = adapter.execute_experiment(Experiment("exp", "objective", "move", 5))
    assert environment.actions == ["move"] * 3
    assert result.experiment_id == "exp"
    assert result.requested_repetitions == 5
    assert result.completed is False
    assert result.results == ["observation 1", "observation 2"]
    assert result.failure_reason == "ValueError: external failure"
    assert environment.events == ["reset", "step"] * 3
    assert adapter.observe() == "initial observation"


def test_reset_value_error_preserves_completed_repetitions():
    environment = FakeEnvironment(reset_fail_on=3)
    adapter = BasicEnvironmentAdapter(environment)
    result = adapter.execute_experiment(Experiment("exp", "objective", "move", 5))
    assert result.completed is False
    assert result.results == ["observation 1", "observation 2"]
    assert result.failure_reason == "ValueError: reset failure"
    assert result.requested_repetitions == 5
    assert environment.actions == ["move"] * 2
    assert environment.events == ["reset", "step", "reset", "step", "reset"]


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, AttributeError, AssertionError])
@pytest.mark.parametrize("source", ["reset", "step"])
def test_programming_exceptions_propagate(error_type, source):
    environment = FakeEnvironment(
        fail_on=3 if source == "step" else None,
        reset_fail_on=3 if source == "reset" else None,
        error_type=error_type,
    )
    adapter = BasicEnvironmentAdapter(environment)
    with pytest.raises(error_type):
        adapter.execute_experiment(Experiment("exp", "objective", "move", 5))
    assert environment.resets == 3
    assert len(environment.actions) == (3 if source == "step" else 2)
