"""Seeds: the same seed reproduces a run and a different seed changes it.

Known limitation (D1): the agent uses the global RNG, so these guarantees hold for one
run per process with nothing else drawing from `random`.
"""

from metis.replay import replay_hash

from .conftest import DEV_SEED, OTHER_DEV_SEED, RecordingAgent, RecordingFactory, fresh_state

BUDGET = 16


def _run(make_workdir, name, seed):
    work = make_workdir(name)
    agent = RecordingAgent()
    work.process(fresh_state(BUDGET, seed), agent, RecordingFactory(), BUDGET, seed=seed)
    return work


def test_same_dev_seed_gives_same_replay_hash(make_workdir):
    first = _run(make_workdir, "a", DEV_SEED)
    second = _run(make_workdir, "b", DEV_SEED)
    assert replay_hash(first.evidence_text()) == replay_hash(second.evidence_text())


def test_different_dev_seed_changes_agent_actions(make_workdir):
    """16 binary decisions: a chance match would have probability 1/65536."""
    first = _run(make_workdir, "a", DEV_SEED)
    second = _run(make_workdir, "b", OTHER_DEV_SEED)
    assert replay_hash(first.evidence_text()) != replay_hash(second.evidence_text())
