"""Shared pieces for the DiscoverPhysics integration tests.

No physics: the fake executor returns a deterministic trajectory that depends on the law
family, so a regime change is visible inside the environment and can be searched for in
what the agent observes. The path exercised is the same as a real run:
run_loop -> BasicEnvironmentAdapter -> DiscoverPhysicsEnvironment -> executor ->
allowlisted payload -> evaluator.
"""

from pathlib import Path

import pytest

from metis.agents.random_agent import RandomAgent
from metis.basic_environment_adapter import BasicEnvironmentAdapter
from metis.budget_ledger import BudgetLedger
from metis.evidence import load_evidence
from metis.recovery import RecoveryState, load_recovery
from metis.run_loop import run_loop
from metis.seed_manager import get_rng_state, set_seed
from metis.worlds.discoverphysics.environment import DiscoverPhysicsEnvironment, FinalDistanceEvaluator

DEV_SEED = 2026101101
OTHER_DEV_SEED = 2026101102
GRAVITY = [(0, "gravity")]
RUN_ID = "integration-test"


class SimulatedCrash(BaseException):
    """Process death. A BaseException so no `except Exception` in the core catches it."""


class FakeExecutor:
    def __init__(self, case, extra_fields=None):
        self.case = case
        self.extra_fields = extra_fields or {}

    def run(self, experiments):
        experiment = experiments[0]
        pull = 1.0 if self.case.family == "gravity" else 0.25
        x0, y0 = experiment["pos2"]
        times = experiment["measurement_times"]
        pos2 = [[float(x0 - pull * t * t), float(y0 + experiment["velocity2"][1] * t)] for t in times]
        zero = [[0.0, 0.0] for _ in times]
        observation = {
            "measurement_times": list(times),
            "pos1": zero,
            "pos2": pos2,
            "velocity1": zero,
            "velocity2": [[0.0, 0.1] for _ in times],
        }
        return [{**observation, **self.extra_fields}]


class RecordingFactory:
    """Factory that records every physics query."""

    def __init__(self, extra_fields=None, crash_on_query=None):
        self.cases = []
        self.extra_fields = extra_fields
        self.crash_on_query = crash_on_query

    def __call__(self, case):
        self.cases.append(case)
        if self.crash_on_query == len(self.cases):
            raise SimulatedCrash(f"process dies during query {len(self.cases)}")
        return FakeExecutor(case, self.extra_fields)

    @property
    def families(self):
        return [case.family for case in self.cases]


class RecordingAgent:
    """Real random agent that records everything it receives."""

    def __init__(self, crash_on_act=None):
        self.inner = RandomAgent()
        self.observations = []
        self.crash_on_act = crash_on_act

    def act(self, observation):
        if self.crash_on_act == len(self.observations) + 1:
            raise SimulatedCrash(f"process dies before decision {self.crash_on_act}")
        self.observations.append(observation)
        return self.inner.act(observation)


def make_environment(factory, regimes=GRAVITY, seed=DEV_SEED):
    return BasicEnvironmentAdapter(DiscoverPhysicsEnvironment(factory, seed, regimes))


def fresh_state(budget, seed=DEV_SEED, run_id=RUN_ID):
    set_seed(seed)
    return RecoveryState(run_id, 0, budget, 0, "start", get_rng_state())


def crash_after_completed(ledger, step, run_id=RUN_ID):
    """Kill the process right after the ledger marks attempt `step` as COMPLETED."""
    original = ledger.set_attempt_status

    def set_status(attempt_id, status):
        original(attempt_id, status)
        if status == "COMPLETED" and attempt_id == f"{run_id}:{step}":
            raise SimulatedCrash(f"process dies after step {step}")

    ledger.set_attempt_status = set_status


class Workdir:
    """The three files of a run in one directory; each process reopens the ledger."""

    def __init__(self, path: Path):
        self.path = path
        self.recovery = path / "recovery.json"
        self.evidence = path / "evidence.jsonl"
        self.db = path / "budget.sqlite"

    def ledger(self, budget, run_id=RUN_ID):
        return BudgetLedger(self.db, budget, run_id)

    def process(
        self, state, agent, factory, budget, regimes=GRAVITY, seed=DEV_SEED, crash_after=None, evaluator=None
    ):
        """One process: open the ledger, run run_loop until done or crash, close the ledger."""
        ledger = self.ledger(budget)
        if crash_after is not None:
            crash_after_completed(ledger, crash_after)
        try:
            run_loop(
                agent,
                make_environment(factory, regimes, seed),
                evaluator or FinalDistanceEvaluator(),
                state,
                self.recovery,
                evidence_path=self.evidence,
                ledger=ledger,
            )
        finally:
            ledger.close()

    def resume_state(self):
        return load_recovery(self.recovery)

    def evidence_text(self):
        return self.evidence.read_text(encoding="utf-8")

    def evidence_steps(self):
        return [record.step for record in load_evidence(self.evidence)]

    def statuses(self, budget):
        ledger = self.ledger(budget)
        try:
            return [
                ledger.get_attempt_status(f"{RUN_ID}:{n}") for n in range(1, budget + 1)
            ], ledger.get_used()
        finally:
            ledger.close()


@pytest.fixture
def workdir(tmp_path):
    return Workdir(tmp_path)


@pytest.fixture
def make_workdir(tmp_path):
    def make(name):
        path = tmp_path / name
        path.mkdir()
        return Workdir(path)

    return make
