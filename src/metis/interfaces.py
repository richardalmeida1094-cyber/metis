from typing import Protocol

from metis.environment_capabilities import EnvironmentCapabilities
from metis.experiment import Experiment
from metis.experiment_result import ExperimentResult


class EnvironmentAdapter(Protocol):
    def capabilities(self) -> EnvironmentCapabilities: ...

    def reset(self) -> str: ...

    def observe(self) -> str: ...

    def step(self, action: str) -> str: ...

    def execute_experiment(self, experiment: Experiment) -> ExperimentResult: ...


class Environment(Protocol):
    def reset(self) -> str: ...

    def step(self, action: str) -> str: ...


class Agent(Protocol):
    def act(self, observation: str) -> str: ...


class Evaluator(Protocol):
    def evaluate(self, result: str) -> float: ...
