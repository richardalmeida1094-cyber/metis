"""Per-run RNG isolation (D1, D2).

Agents are compared under the same budget and the same seed. If the result depends
on who else touched the global RNG, the comparison is unfair and not reproducible.
"""

import pytest

from metis.recovery import load_recovery
from metis.run_loop import run_loop

from .conftest import HistoryEnv, Ledger, NoisyHistoryEnv, RecordingAgent, ZeroEvaluator, start_run

SEED = 42
BUDGET = 20


@pytest.mark.xfail(
    strict=True, reason="D1: the agent uses the global RNG; interference during the run changes its actions"
)
def test_interference_during_run_does_not_change_agent_actions(paths):
    """Same seed, same budget. In the second run the environment consumes the global
    RNG on every step(), during the loop. Expected: identical action sequence."""
    rec, _ = paths
    agent, state = start_run(SEED, "clean", BUDGET)
    clean = RecordingAgent(agent)
    run_loop(clean, HistoryEnv(Ledger()), ZeroEvaluator(), state, rec)

    agent, state = start_run(SEED, "noisy", BUDGET)
    noisy = RecordingAgent(agent)
    run_loop(noisy, NoisyHistoryEnv(Ledger()), ZeroEvaluator(), state, rec)

    assert noisy.actions == clean.actions


@pytest.mark.xfail(strict=True, reason="D1: two runs share the global RNG and contaminate each other")
def test_interleaved_runs_match_isolated_runs():
    """Two runs (seeds 1 and 2) deciding in interleaved order A,B,A,B, as when running
    baselines side by side in one process. Expected: each equals its isolated run."""
    n = 20
    a, _ = start_run(1, "a", n)
    iso_a = [a.act("o") for _ in range(n)]
    b, _ = start_run(2, "b", n)
    iso_b = [b.act("o") for _ in range(n)]

    a, _ = start_run(1, "a", n)
    b, _ = start_run(2, "b", n)
    mix_a, mix_b = [], []
    for _ in range(n):
        mix_a.append(a.act("o"))
        mix_b.append(b.act("o"))

    assert (mix_a, mix_b) == (iso_a, iso_b)


def test_saved_rng_state_advances_after_draws(paths):
    """Random agent for 5 steps. Expected: the persisted rng_state differs from the
    initial state, i.e. the saved state follows the draws."""
    rec, _ = paths
    agent, state = start_run(SEED, "r", 5)
    initial = state.rng_state
    run_loop(agent, HistoryEnv(Ledger()), ZeroEvaluator(), state, rec)
    assert load_recovery(rec).rng_state != initial
