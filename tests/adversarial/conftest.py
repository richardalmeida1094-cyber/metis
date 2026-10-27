"""Fixtures and fault injectors for the adversarial core test suite.

Rules for this suite:
- It never changes production code.
- xfail(strict=True) is used only for reproduced defects (D1..D6). When a defect is
  fixed the test turns into XPASS and fails, so the marker must be removed by hand.
- XFAIL is not evidence of correctness.

Adaptation points (explicit dependency on the current API):
- start_run(seed): seeds the global RNG through set_seed().
- resume_run(...): load_recovery + set_rng_state + run_loop with a NEW agent and
  environment (process-restart semantics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from metis.agents.random_agent import RandomAgent
from metis.environment_capabilities import EnvironmentCapabilities
from metis.recovery import RecoveryState, load_recovery
from metis.run_loop import run_loop
from metis.seed_manager import get_rng_state, set_rng_state, set_seed


class SimulatedCrash(BaseException):
    """Simulates process death. A BaseException so no `except Exception` catches it."""


@dataclass
class Ledger:
    """Accounting outside the core that survives the crash (the observer)."""

    queries_executed: int = 0  # step() calls that returned (physics query done)
    queries_attempted: int = 0  # step() calls started
    resets: int = 0  # reset() calls (not scientific queries)


class HistoryEnv:
    """Environment whose observation depends on history: t{n}:{action}.
    A spurious reset() shows up as t1 again."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.t = 0
        self.last = "start"

    def reset(self) -> str:
        self.ledger.resets += 1
        self.t = 0
        self.last = "start"
        return "start"

    def step(self, action: str) -> str:
        self.ledger.queries_attempted += 1
        self.t += 1
        obs = f"t{self.t}:{action}"
        self.ledger.queries_executed += 1
        self.last = obs
        return obs

    # EnvironmentAdapter surface required by run_loop, which refuses to resume
    # without a snapshot. snapshot/restore count neither as a query nor as a reset.
    def observe(self) -> str:
        return self.last

    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            supports_actions=True,
            supports_experiments=False,
            supports_snapshot=True,
            supports_restore=True,
        )

    def snapshot(self) -> dict:
        return {"t": self.t, "last": self.last}

    def restore(self, state: dict) -> None:
        self.t = state["t"]
        self.last = state["last"]


class NoisyHistoryEnv(HistoryEnv):
    """Like HistoryEnv, but consumes the global RNG during the run,
    as a third-party library would."""

    def step(self, action: str) -> str:
        import random

        random.random()
        return super().step(action)


class ZeroEvaluator:
    def evaluate(self, result: str) -> float:
        return 0.0


class RecordingAgent:
    """Wraps an agent and records the observations it receives and the actions it takes."""

    def __init__(self, inner, crash_on_act: int | None = None):
        self.inner = inner
        self.crash_on_act = crash_on_act
        self.observations: list[str] = []
        self.actions: list[str] = []

    def act(self, observation: str) -> str:
        if self.crash_on_act is not None and len(self.actions) + 1 == self.crash_on_act:
            raise SimulatedCrash(f"crash before decision {self.crash_on_act}")
        self.observations.append(observation)
        action = self.inner.act(observation)
        self.actions.append(action)
        return action


class CyclingAgent:
    """Deterministic agent without RNG: separates environment continuity from the RNG."""

    def __init__(self):
        self.i = 0

    def act(self, observation: str) -> str:
        action = ("left", "right", "right")[self.i % 3]
        self.i += 1
        return action


# ---------------- adaptation points ----------------


def start_run(seed: int, run_id: str, budget: int) -> tuple[RandomAgent, RecoveryState]:
    """Adaptation point: the seed enters through the global RNG."""
    set_seed(seed)
    return RandomAgent(), RecoveryState(run_id, 0, budget, 0, "start", get_rng_state())


def resume_run(agent, env, recovery_path: Path, evidence_path: Path | None = None) -> RecoveryState:
    """Adaptation point: resume with a new agent and environment (restarted process)."""
    state = load_recovery(recovery_path)
    set_rng_state(state.rng_state)
    run_loop(agent, env, ZeroEvaluator(), state, recovery_path, evidence_path)
    return state


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "recovery.json", tmp_path / "evidence.jsonl"
