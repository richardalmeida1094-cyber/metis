"""Crash: at any point inside a step, nothing is duplicated or silently redone.

With a BudgetLedger, an attempt that never reached COMPLETED blocks the resume
("Reconciliation required"): no refund and no retry. The crash is injected at each point
of step 3 and the resume uses a new agent, environment, factory and ledger.

Expected at every point inside the step: resume blocked, zero new queries, no repeated
evidence step, charge kept. At the step boundary (after COMPLETED) the resume finishes
the run. Automatic resume from inside a step is not implemented.
"""

import os
import subprocess
import sys
import textwrap
import types

import pytest

import metis.recovery
import metis.run_loop

from .conftest import RUN_ID, RecordingAgent, RecordingFactory, SimulatedCrash, fresh_state, make_environment

BUDGET = 5
STEP = 3


def _nth(fn, n):
    calls = {"n": 0}

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            raise SimulatedCrash(f"crash on call {n} of {fn.__name__}")
        return fn(*args, **kwargs)

    return wrapper


class CrashingEvaluator:
    def __init__(self, on_call):
        self.on_call, self.calls = on_call, 0

    def evaluate(self, result):
        self.calls += 1
        if self.calls == self.on_call:
            raise SimulatedCrash("crash after the query, before scoring")
        return 0.0


def _crash_inside_step(monkeypatch, work, point):
    """Run process 1 until it dies at `point` of step STEP. Return the number of queries made."""
    agent = RecordingAgent(crash_on_act=STEP if point == "before_act" else None)
    factory = RecordingFactory(crash_on_query=STEP if point == "in_dispatch" else None)
    evaluator = CrashingEvaluator(STEP) if point == "after_query" else None
    if point == "in_evidence":
        monkeypatch.setattr(metis.run_loop, "append_evidence", _nth(metis.run_loop.append_evidence, STEP))
    elif point == "after_evidence":
        monkeypatch.setattr(metis.run_loop, "save_recovery", _nth(metis.run_loop.save_recovery, STEP))
    elif point == "before_replace":
        fake_os = types.SimpleNamespace(fsync=os.fsync, replace=_nth(os.replace, STEP))
        monkeypatch.setattr(metis.recovery, "os", fake_os)

    ledger = work.ledger(BUDGET)
    if point == "after_recovery":
        original = ledger.set_attempt_status

        def die_before_completed(attempt_id, status):
            if status == "COMPLETED" and attempt_id == f"{RUN_ID}:{STEP}":
                raise SimulatedCrash("crash after recovery, before COMPLETED")
            original(attempt_id, status)

        ledger.set_attempt_status = die_before_completed
    try:
        with pytest.raises(SimulatedCrash):
            metis.run_loop.run_loop(
                agent,
                make_environment(factory),
                evaluator or _ZeroEvaluator(),
                fresh_state(BUDGET),
                work.recovery,
                evidence_path=work.evidence,
                ledger=ledger,
            )
    finally:
        ledger.close()
        monkeypatch.undo()
    return len(factory.cases)


class _ZeroEvaluator:
    def evaluate(self, result):
        return 0.0


POINTS = [
    "before_act",
    "in_dispatch",
    "after_query",
    "in_evidence",
    "after_evidence",
    "before_replace",
    "after_recovery",
]


@pytest.mark.parametrize("point", POINTS)
def test_crash_inside_step_blocks_resume_without_requery_or_duplicate(monkeypatch, workdir, point):
    queries_before = _crash_inside_step(monkeypatch, workdir, point)
    statuses_before, used_before = workdir.statuses(BUDGET)
    steps_before = workdir.evidence_steps()

    resume_factory = RecordingFactory()
    state = workdir.resume_state()
    ledger = workdir.ledger(BUDGET)
    try:
        with pytest.raises(RuntimeError, match="(?i)reconciliation required"):
            metis.run_loop.run_loop(
                RecordingAgent(),
                make_environment(resume_factory),
                _ZeroEvaluator(),
                state,
                workdir.recovery,
                evidence_path=workdir.evidence,
                ledger=ledger,
            )
    finally:
        ledger.close()

    assert resume_factory.cases == []
    assert workdir.evidence_steps() == steps_before
    assert len(steps_before) == len(set(steps_before))
    assert workdir.statuses(BUDGET) == (statuses_before, used_before)
    assert used_before == STEP
    assert queries_before <= STEP


def test_crash_at_step_boundary_resumes_to_the_end(workdir):
    first_factory = RecordingFactory()
    with pytest.raises(SimulatedCrash):
        workdir.process(
            fresh_state(BUDGET),
            RecordingAgent(),
            first_factory,
            BUDGET,
            crash_after=STEP,
            evaluator=_ZeroEvaluator(),
        )

    second_factory = RecordingFactory()
    workdir.process(
        workdir.resume_state(), RecordingAgent(), second_factory, BUDGET, evaluator=_ZeroEvaluator()
    )

    assert len(first_factory.cases) + len(second_factory.cases) == BUDGET
    assert workdir.evidence_steps() == list(range(1, BUDGET + 1))
    assert workdir.statuses(BUDGET) == (["COMPLETED"] * BUDGET, BUDGET)


CHILD = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[2])
    from tests.integration.conftest import (RecordingAgent, RecordingFactory, Workdir, fresh_state)

    work = Workdir(Path(sys.argv[1]))
    ledger = work.ledger(5)
    original = ledger.set_attempt_status

    def die(attempt_id, status):
        original(attempt_id, status)
        if status == "COMPLETED" and attempt_id.endswith(":3"):
            os._exit(37)

    ledger.set_attempt_status = die
    from metis.run_loop import run_loop
    from tests.integration.conftest import make_environment
    from metis.worlds.discoverphysics.environment import FinalDistanceEvaluator
    run_loop(RecordingAgent(), make_environment(RecordingFactory()), FinalDistanceEvaluator(),
             fresh_state(5), work.recovery, evidence_path=work.evidence, ledger=ledger)
    """
)


def test_real_process_death_at_boundary_then_resume_in_new_process_matches_uninterrupted(make_workdir):
    """Real death (os._exit) in a child process; resume in the test process."""
    repo = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    crashed = make_workdir("crashed")
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(crashed.path), repo],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 37, result.stdout + result.stderr
    assert crashed.evidence_steps() == [1, 2, 3]

    resume_factory = RecordingFactory()
    crashed.process(crashed.resume_state(), RecordingAgent(), resume_factory, BUDGET)

    reference = make_workdir("reference")
    reference.process(fresh_state(BUDGET), RecordingAgent(), RecordingFactory(), BUDGET)

    assert len(resume_factory.cases) == BUDGET - 3
    assert crashed.evidence_text() == reference.evidence_text()
    assert crashed.statuses(BUDGET) == (["COMPLETED"] * BUDGET, BUDGET)
