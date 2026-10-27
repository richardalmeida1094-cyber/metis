from typing import Protocol, runtime_checkable

from metis.environment_capabilities import EnvironmentCapabilities
from metis.experiment import Experiment
from metis.experiment_result import ExperimentResult
from metis.interfaces import Environment


@runtime_checkable
class SnapshotEnvironment(Protocol):
    def snapshot(self) -> object: ...

    def restore(self, state: object) -> None: ...


class BasicEnvironmentAdapter:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.last_observation = ""

    def reset(self) -> str:
        self.last_observation = self.environment.reset()
        return self.last_observation

    def step(self, action: str) -> str:
        self.last_observation = self.environment.step(action)
        return self.last_observation

    def observe(self) -> str:
        return self.last_observation

    def capabilities(self) -> EnvironmentCapabilities:
        supports_state = isinstance(
            self.environment,
            SnapshotEnvironment,
        )

        return EnvironmentCapabilities(
            supports_actions=True,
            supports_experiments=True,
            supports_snapshot=supports_state,
            supports_restore=supports_state,
        )

    def snapshot(self) -> object:
        if not isinstance(self.environment, SnapshotEnvironment):
            raise RuntimeError("Environment does not support snapshots")

        return {
            "environment_state": self.environment.snapshot(),
            "last_observation": self.last_observation,
        }

    def restore(self, state: object) -> None:
        if not isinstance(self.environment, SnapshotEnvironment):
            raise RuntimeError("Environment does not support restoration")

        if not isinstance(state, dict):
            raise ValueError("Invalid environment snapshot")

        if set(state) != {
            "environment_state",
            "last_observation",
        }:
            raise ValueError("Invalid environment snapshot")

        observation = state["last_observation"]

        if not isinstance(observation, str):
            raise ValueError("Invalid snapshot observation")

        self.environment.restore(state["environment_state"])
        self.last_observation = observation

    def execute_experiment(self, experiment: Experiment) -> ExperimentResult:
        results: list[str] = []

        for _ in range(experiment.repetitions):
            try:
                self.reset()
                results.append(self.step(experiment.action))
            except ValueError as exc:
                return ExperimentResult(
                    experiment_id=experiment.experiment_id,
                    requested_repetitions=experiment.repetitions,
                    results=results,
                    completed=False,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            requested_repetitions=experiment.repetitions,
            results=results,
            completed=True,
            failure_reason=None,
        )
