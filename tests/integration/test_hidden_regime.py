"""Hidden regime: the change reaches the environment, never leaks to the agent and survives restarts.

Plumbing only: no change detection or new intelligence. The regime is the law family the
executor uses from query n on (counting from 0). Leakage is covered in test_leakage.py.
"""

import pytest

from .conftest import GRAVITY, RecordingAgent, RecordingFactory, SimulatedCrash, fresh_state

BUDGET = 8
CHANGE_AT = 4
REGIMES = [(0, "gravity"), (CHANGE_AT, "yukawa")]
EXPECTED = ["gravity"] * CHANGE_AT + ["yukawa"] * (BUDGET - CHANGE_AT)


def test_hidden_change_reaches_the_executor_at_the_scheduled_query(workdir):
    factory = RecordingFactory()
    workdir.process(fresh_state(BUDGET), RecordingAgent(), factory, BUDGET, regimes=REGIMES)
    assert factory.families == EXPECTED


def test_hidden_change_changes_observations_only_after_the_change(make_workdir):
    plain = make_workdir("plain")
    plain_agent = RecordingAgent()
    plain.process(fresh_state(BUDGET), plain_agent, RecordingFactory(), BUDGET, regimes=GRAVITY)
    changed = make_workdir("changed")
    changed_agent = RecordingAgent()
    changed.process(fresh_state(BUDGET), changed_agent, RecordingFactory(), BUDGET, regimes=REGIMES)

    # observations[i] is what the agent saw before decision i+1; the result of query q appears at q+1.
    assert changed_agent.observations[: CHANGE_AT + 1] == plain_agent.observations[: CHANGE_AT + 1]
    assert all(
        a != b
        for a, b in zip(
            changed_agent.observations[CHANGE_AT + 1 :],
            plain_agent.observations[CHANGE_AT + 1 :],
            strict=True,
        )
    )


@pytest.mark.parametrize("cut", [CHANGE_AT - 1, CHANGE_AT, CHANGE_AT + 1], ids=["before", "at", "after"])
def test_hidden_change_survives_restart(make_workdir, cut):
    reference = make_workdir("reference")
    reference.process(fresh_state(BUDGET), RecordingAgent(), RecordingFactory(), BUDGET, regimes=REGIMES)

    work = make_workdir("work")
    first = RecordingFactory()
    with pytest.raises(SimulatedCrash):
        work.process(fresh_state(BUDGET), RecordingAgent(), first, BUDGET, regimes=REGIMES, crash_after=cut)
    second = RecordingFactory()
    work.process(work.resume_state(), RecordingAgent(), second, BUDGET, regimes=REGIMES)

    assert first.families + second.families == EXPECTED
    assert work.evidence_text() == reference.evidence_text()


def test_regime_schedule_must_use_active_families():
    from metis.worlds.discoverphysics.environment import DiscoverPhysicsEnvironment

    with pytest.raises(ValueError, match="not ACTIVE"):
        DiscoverPhysicsEnvironment(RecordingFactory(), 2026101101, [(0, "gravity"), (3, "fractional")])
    with pytest.raises(ValueError, match="start at query 0"):
        DiscoverPhysicsEnvironment(RecordingFactory(), 2026101101, [(2, "gravity")])
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscoverPhysicsEnvironment(RecordingFactory(), 2026101101, [(0, "gravity"), (0, "yukawa")])
