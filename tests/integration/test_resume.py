"""Resume: continues from the saved state, never resets, and rejects incompatible state."""

import random

import pytest

from metis.worlds.discoverphysics.environment import DiscoverPhysicsEnvironment

from .conftest import GRAVITY, RecordingAgent, RecordingFactory, SimulatedCrash, fresh_state

BUDGET = 6
CUT = 3


class CountingResetEnvironment(DiscoverPhysicsEnvironment):
    resets = 0

    def reset(self):
        CountingResetEnvironment.resets += 1
        return super().reset()


def _crash_at_cut(work, regimes=GRAVITY):
    with pytest.raises(SimulatedCrash):
        work.process(
            fresh_state(BUDGET),
            RecordingAgent(),
            RecordingFactory(),
            BUDGET,
            regimes=regimes,
            crash_after=CUT,
        )


def test_first_observation_after_resume_is_the_persisted_one(workdir):
    _crash_at_cut(workdir)
    persisted = workdir.resume_state().state
    agent = RecordingAgent()
    workdir.process(workdir.resume_state(), agent, RecordingFactory(), BUDGET)
    assert agent.observations[0] == persisted
    assert persisted != "start"


def test_resume_never_calls_reset(workdir, monkeypatch):
    from . import conftest

    _crash_at_cut(workdir)
    CountingResetEnvironment.resets = 0
    monkeypatch.setattr(conftest, "DiscoverPhysicsEnvironment", CountingResetEnvironment)
    workdir.process(workdir.resume_state(), RecordingAgent(), RecordingFactory(), BUDGET)
    assert CountingResetEnvironment.resets == 0
    assert workdir.evidence_steps() == list(range(1, BUDGET + 1))


def test_resume_restores_rng_even_if_global_rng_was_disturbed(make_workdir):
    reference = make_workdir("reference")
    reference.process(fresh_state(BUDGET), RecordingAgent(), RecordingFactory(), BUDGET)

    work = make_workdir("work")
    _crash_at_cut(work)
    random.seed(123)
    [random.random() for _ in range(10)]
    work.process(work.resume_state(), RecordingAgent(), RecordingFactory(), BUDGET)
    assert work.evidence_text() == reference.evidence_text()


def test_resume_refuses_snapshot_from_another_regime_schedule(workdir):
    """Moving the hidden change after a restart would silently swap the world."""
    _crash_at_cut(workdir, regimes=[(0, "gravity"), (4, "yukawa")])
    factory = RecordingFactory()
    with pytest.raises(ValueError, match="different environment configuration"):
        workdir.process(
            workdir.resume_state(), RecordingAgent(), factory, BUDGET, regimes=[(0, "gravity"), (5, "yukawa")]
        )
    assert factory.cases == []
    assert workdir.evidence_steps() == [1, 2, 3]


def test_resume_refuses_snapshot_from_another_seed(workdir):
    _crash_at_cut(workdir)
    factory = RecordingFactory()
    with pytest.raises(ValueError, match="different environment configuration"):
        workdir.process(workdir.resume_state(), RecordingAgent(), factory, BUDGET, seed=2026101103)
    assert factory.cases == []


def test_completed_run_cannot_be_resumed_into_extra_queries(workdir):
    workdir.process(fresh_state(BUDGET), RecordingAgent(), RecordingFactory(), BUDGET)
    factory = RecordingFactory()
    workdir.process(workdir.resume_state(), RecordingAgent(), factory, BUDGET)
    assert factory.cases == []
    assert workdir.evidence_steps() == list(range(1, BUDGET + 1))
