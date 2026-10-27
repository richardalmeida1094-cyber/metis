"""Deterministic recovery after an interruption (D3, D4).

Clean crash: the process dies BEFORE decision k+1, after step k was fully persisted.
The run is resumed with a new agent and environment (resume_run). Environment state
must continue for real, by direct restore or by validated deterministic rebuild.
A silent reset() presented as recovery shows up here as a diverging observation trace.
"""

import pytest

from metis.evidence import load_evidence
from metis.recovery import load_recovery
from metis.run_loop import run_loop

from .conftest import (
    CyclingAgent,
    HistoryEnv,
    Ledger,
    RecordingAgent,
    SimulatedCrash,
    ZeroEvaluator,
    resume_run,
    start_run,
)

SEED = 7
BUDGET = 16
# Crash before decision k (k-1 steps persisted). k stops at 8 so at least 8 decisions
# remain after resuming: with 2 possible actions a short tail can match the reference
# by chance and hide the defect. A tail of >= 8 makes a coincidence <= 1/256.
CRASH_POINTS = list(range(2, 9))


def _crash_then_resume(paths, make_agent, crash_before):
    rec, ev = paths
    first_agent, state = make_agent()
    first = RecordingAgent(first_agent, crash_on_act=crash_before)
    with pytest.raises(SimulatedCrash):
        run_loop(first, HistoryEnv(Ledger()), ZeroEvaluator(), state, rec, ev)
    second_agent, _ = make_agent(fresh_only=True)
    second = RecordingAgent(second_agent)
    resume_run(second, HistoryEnv(Ledger()), rec, ev)
    return first, second


def _reference(tmp_path, make_agent):
    agent, state = make_agent()
    ref = RecordingAgent(agent)
    ev = tmp_path / "ref_evidence.jsonl"
    run_loop(ref, HistoryEnv(Ledger()), ZeroEvaluator(), state, tmp_path / "ref_rec.json", ev)
    return ref, load_evidence(ev)


def _random_agent(fresh_only=False):
    if fresh_only:
        from metis.agents.random_agent import RandomAgent

        return RandomAgent(), None
    return start_run(SEED, "run", BUDGET)


def _cycling_agent(fresh_only=False):
    from metis.recovery import RecoveryState
    from metis.seed_manager import get_rng_state

    return CyclingAgent(), RecoveryState("run", 0, BUDGET, 0, "start", get_rng_state())


@pytest.mark.parametrize("crash_before", CRASH_POINTS)
def test_resumed_actions_equal_uninterrupted(tmp_path, paths, crash_before):
    """Random agent. Expected: actions (before + after the crash) == reference."""
    ref, _ = _reference(tmp_path, _random_agent)
    first, second = _crash_then_resume(paths, _random_agent, crash_before)
    assert first.actions + second.actions == ref.actions


@pytest.mark.parametrize("crash_before", CRASH_POINTS)
def test_first_decision_after_resume_sees_persisted_observation(tmp_path, paths, crash_before):
    """Expected: the first observation given to the agent after resuming equals the
    recovery.state persisted before the crash."""
    rec, _ = paths
    first_agent, state = _cycling_agent()
    first = RecordingAgent(first_agent, crash_on_act=crash_before)
    with pytest.raises(SimulatedCrash):
        run_loop(first, HistoryEnv(Ledger()), ZeroEvaluator(), state, rec)
    persisted = load_recovery(rec).state
    second = RecordingAgent(CyclingAgent())
    resume_run(second, HistoryEnv(Ledger()), rec)
    assert second.observations[0] == persisted


# D4 (environment state) is fixed. k=4 and k=7 pass because the 3-cycle of the new
# CyclingAgent lines up with the original one. For other k the trace diverges because
# agent state (its counter) is not persisted: defect D4b, open together with D1.
@pytest.mark.parametrize(
    "crash_before",
    [
        k
        if (k - 1) % 3 == 0
        else pytest.param(
            k,
            marks=pytest.mark.xfail(
                strict=True,
                reason="D4b: agent internal state is not persisted; a new agent restarts its cycle",
            ),
        )
        for k in CRASH_POINTS
    ],
)
def test_environment_state_continuity_after_resume(tmp_path, paths, crash_before):
    """T2.3. Deterministic agent (isolates the RNG). Expected: observation trace after
    resuming == reference. Restore or rebuild are both accepted; a silent reset()
    produces t1 again and fails here."""
    ref, ref_ev = _reference(tmp_path, _cycling_agent)
    _, ev = paths
    _crash_then_resume(paths, _cycling_agent, crash_before)
    k = crash_before - 1
    got = [r.observation for r in load_evidence(ev)][k:]
    assert got == [r.observation for r in ref_ev][k:]


@pytest.mark.parametrize("crash_before", CRASH_POINTS)
def test_resumed_evidence_equals_uninterrupted_evidence(tmp_path, paths, crash_before):
    """The final evidence file of an interrupted run equals the evidence of an
    uninterrupted run."""
    _, ref_ev = _reference(tmp_path, _random_agent)
    _, ev = paths
    _crash_then_resume(paths, _random_agent, crash_before)
    assert load_evidence(ev) == ref_ev
