"""Segmented determinism: running in pieces, with restarts, gives the same run as running at once.

Each segment ends with process death right after a COMPLETED step and restarts with a new
agent, environment, factory and ledger, from disk only. Before each resume the global RNG
is scrambled on purpose: run_loop itself must restore it.
"""

import random

import pytest

from metis.replay import replay_hash

from .conftest import GRAVITY, RecordingAgent, RecordingFactory, SimulatedCrash, fresh_state

BUDGET = 12
REGIME_CHANGE = [(0, "gravity"), (5, "yukawa")]


def _uninterrupted(make_workdir, regimes):
    work = make_workdir("uninterrupted")
    factory = RecordingFactory()
    work.process(fresh_state(BUDGET), RecordingAgent(), factory, BUDGET, regimes=regimes)
    return work, factory


def _segmented(make_workdir, regimes, cuts):
    work = make_workdir("segmented")
    factories = []
    state = fresh_state(BUDGET)
    for cut in cuts:
        factory = RecordingFactory()
        factories.append(factory)
        with pytest.raises(SimulatedCrash):
            work.process(state, RecordingAgent(), factory, BUDGET, regimes=regimes, crash_after=cut)
        random.seed(999_999)  # another process: the global RNG starts anywhere
        state = work.resume_state()
    factory = RecordingFactory()
    factories.append(factory)
    work.process(state, RecordingAgent(), factory, BUDGET, regimes=regimes)
    return work, factories


@pytest.mark.parametrize("regimes", [GRAVITY, REGIME_CHANGE], ids=["gravity", "gravity_to_yukawa_at_5"])
@pytest.mark.parametrize(
    "cuts",
    [[1], [4], [2, 4], [3, 6, 9], list(range(1, BUDGET))],
    ids=["1", "4", "2-4", "3-6-9", "every_step"],
)
def test_segmented_run_equals_uninterrupted_run(make_workdir, regimes, cuts):
    reference, reference_factory = _uninterrupted(make_workdir, regimes)
    segmented, factories = _segmented(make_workdir, regimes, cuts)

    assert replay_hash(segmented.evidence_text()) == replay_hash(reference.evidence_text())
    assert segmented.evidence_text() == reference.evidence_text()
    assert sum(len(f.cases) for f in factories) == BUDGET
    assert [fam for f in factories for fam in f.families] == reference_factory.families
    assert segmented.statuses(BUDGET) == (["COMPLETED"] * BUDGET, BUDGET)
    final = segmented.resume_state()
    assert final.rng_state == reference.resume_state().rng_state
    assert final.step == final.budget_used == BUDGET
