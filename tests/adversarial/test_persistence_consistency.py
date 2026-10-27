"""Consistency between evidence and recovery under injected failures (D5, D6).

A failure is injected at step CRASH_STEP, at each intermediate point of the cycle:
  after_query      step() already ran; crash before anything is written
  in_evidence      crash inside append_evidence, before writing
  after_evidence   evidence written; crash before save_recovery writes
  before_replace   .tmp written; crash before os.replace
The run is resumed with a new agent and environment (resume_run). The Ledger lives
outside the core and survives the crash: it counts the physics queries actually run.

No write order is assumed: the invariants hold for any order. Reversing the current
order trades duplicates for gaps (T3.2 covers that).
"""

import os
import types

import pytest

import metis.recovery
import metis.run_loop
from metis.evidence import load_evidence
from metis.recovery import RecoveryState, load_recovery, save_recovery
from metis.run_loop import run_loop
from metis.seed_manager import get_rng_state

from .conftest import CyclingAgent, HistoryEnv, Ledger, SimulatedCrash, ZeroEvaluator, resume_run

BUDGET = 5
CRASH_STEP = 3
POINTS = ["after_query", "in_evidence", "after_evidence", "before_replace"]


class CrashingEvaluator:
    def __init__(self, on_call):
        self.on_call, self.calls = on_call, 0

    def evaluate(self, result):
        self.calls += 1
        if self.calls == self.on_call:
            raise SimulatedCrash("crash after the query")
        return 0.0


def _nth(fn, n, msg):
    calls = {"n": 0}

    def wrapper(*a, **k):
        calls["n"] += 1
        if calls["n"] == n:
            raise SimulatedCrash(msg)
        return fn(*a, **k)

    return wrapper


def _run_with_crash(monkeypatch, point, rec, ev, ledger):
    evaluator = ZeroEvaluator()
    if point == "after_query":
        evaluator = CrashingEvaluator(CRASH_STEP)
    elif point == "in_evidence":
        monkeypatch.setattr(
            metis.run_loop,
            "append_evidence",
            _nth(metis.run_loop.append_evidence, CRASH_STEP, "crash in append_evidence"),
        )
    elif point == "after_evidence":
        monkeypatch.setattr(
            metis.run_loop,
            "save_recovery",
            _nth(metis.run_loop.save_recovery, CRASH_STEP, "crash before save_recovery"),
        )
    elif point == "before_replace":
        fake_os = types.SimpleNamespace(**{k: getattr(os, k) for k in ("fsync",)})
        fake_os.replace = _nth(os.replace, CRASH_STEP, "crash before os.replace")
        monkeypatch.setattr(metis.recovery, "os", fake_os)

    state = RecoveryState("run", 0, BUDGET, 0, "start", get_rng_state())
    with pytest.raises(SimulatedCrash):
        run_loop(CyclingAgent(), HistoryEnv(ledger), evaluator, state, rec, ev)
    monkeypatch.undo()


def _crash_and_resume(monkeypatch, point, paths):
    rec, ev = paths
    ledger = Ledger()
    _run_with_crash(monkeypatch, point, rec, ev, ledger)
    resume_run(CyclingAgent(), HistoryEnv(ledger), rec, ev)
    return ledger, [r.step for r in load_evidence(ev)], load_recovery(rec)


def _xfail_on(points, reason):
    return [
        pytest.param(p, marks=pytest.mark.xfail(strict=True, reason=reason)) if p in points else p
        for p in POINTS
    ]


@pytest.mark.parametrize(
    "point",
    _xfail_on(
        {"after_evidence", "before_replace"}, "D5: evidence written without recovery; resume repeats the step"
    ),
)
def test_no_duplicate_evidence_steps(monkeypatch, paths, point):
    """T3.1. Expected: each (run_id, step) appears exactly once."""
    _, steps, _ = _crash_and_resume(monkeypatch, point, paths)
    assert len(steps) == len(set(steps)), steps


@pytest.mark.parametrize("point", POINTS)
def test_no_missing_evidence_steps(monkeypatch, paths, point):
    """T3.2. Expected: every step 1..BUDGET has evidence. Passes today; guards
    against 'fixing' D5 by reversing the write order and creating gaps."""
    _, steps, final = _crash_and_resume(monkeypatch, point, paths)
    assert set(steps) == set(range(1, BUDGET + 1)), steps
    assert final.step == final.budget_used == BUDGET


@pytest.mark.parametrize(
    "point",
    _xfail_on(set(POINTS), "D6: a query that ran but was not persisted is repeated without being charged"),
)
def test_executed_queries_never_exceed_budget(monkeypatch, paths, point):
    """T3.4. Separates executed scientific queries (external Ledger, cannot be
    undone by the crash), persisted operations (evidence/budget_used) and
    reconstruction (reset, not a query).
    Expected: physics queries executed across all attempts <= budget_total.
    A query made before the crash has spent budget even if its result is lost."""
    ledger, steps, final = _crash_and_resume(monkeypatch, point, paths)
    assert ledger.queries_executed <= BUDGET, (
        f"executed={ledger.queries_executed} persisted={len(steps)} "
        f"budget_used={final.budget_used} resets={ledger.resets}"
    )


def test_atomic_recovery_survives_crash_before_replace(monkeypatch, tmp_path):
    """T3.5 (regression). Crash before os.replace: the previous recovery.json
    stays intact and readable."""
    rec = tmp_path / "recovery.json"
    state = RecoveryState("run", 1, BUDGET, 1, "t1:left", get_rng_state())
    save_recovery(state, rec)
    fake_os = types.SimpleNamespace(fsync=os.fsync, replace=_nth(os.replace, 1, "crash"))
    monkeypatch.setattr(metis.recovery, "os", fake_os)
    newer = RecoveryState("run", 2, BUDGET, 2, "t2:right", get_rng_state())
    with pytest.raises(SimulatedCrash):
        save_recovery(newer, rec)
    monkeypatch.undo()
    assert load_recovery(rec) == state
