"""Leakage: the agent only receives the allowlisted numeric payload.

Regime, law family, seed and score never reach it. Threat model: trusted host. This checks
the fields and text given to the agent, not side channels such as timing or size.
"""

import json

import pytest

from .conftest import DEV_SEED, RecordingAgent, RecordingFactory, fresh_state

BUDGET = 8
REGIMES = [(0, "gravity"), (4, "yukawa")]
ALLOWED = {"measurement_times", "pos1", "pos2", "velocity1", "velocity2"}
FORBIDDEN_TOKENS = (
    "gravity",
    "yukawa",
    "family",
    "regime",
    "seed",
    "score",
    "laplacian",
    "screening",
    str(DEV_SEED),
)


class MarkerEvaluator:
    MARKER = 987654.125

    def evaluate(self, result):
        return self.MARKER


def _leaves(value):
    if isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def test_agent_sees_only_allowlisted_numeric_payload(workdir):
    agent = RecordingAgent()
    workdir.process(
        fresh_state(BUDGET), agent, RecordingFactory(), BUDGET, regimes=REGIMES, evaluator=MarkerEvaluator()
    )

    assert agent.observations[0] == "start"
    for text in agent.observations[1:]:
        payload = json.loads(text)
        assert set(payload) == ALLOWED
        assert all(type(leaf) is float for key in ALLOWED for leaf in _leaves(payload[key]))
        lowered = text.lower()
        assert not [token for token in FORBIDDEN_TOKENS if token in lowered]
        assert str(MarkerEvaluator.MARKER) not in text
    assert set(vars(agent)) == {"inner", "observations", "crash_on_act"}


@pytest.mark.parametrize(
    "extra", [{"family": "gravity"}, {"true_law": "1/r"}, {"regime_id": 1.0}, {"hidden_change_at": 4.0}]
)
def test_extra_field_from_simulator_is_rejected_and_never_reaches_agent(workdir, extra):
    agent = RecordingAgent()
    with pytest.raises(ValueError, match="fields not allowed"):
        workdir.process(fresh_state(BUDGET), agent, RecordingFactory(extra_fields=extra), BUDGET)
    assert agent.observations == ["start"]
    statuses, used = workdir.statuses(BUDGET)
    assert statuses[0] == "UNKNOWN" and used == 1
    assert not workdir.evidence.exists()


def test_regime_change_is_invisible_in_payload_structure(workdir):
    """Before and after the change the agent receives exactly the same keys and shapes."""
    agent = RecordingAgent()
    workdir.process(fresh_state(BUDGET), agent, RecordingFactory(), BUDGET, regimes=REGIMES)

    def structure(text):
        payload = json.loads(text)
        return {
            key: (len(payload[key]), len(payload[key][0]) if isinstance(payload[key][0], list) else 0)
            for key in sorted(payload)
        }

    before = {json.dumps(structure(t)) for t in agent.observations[1:5]}
    after = {json.dumps(structure(t)) for t in agent.observations[5:]}
    assert before == after and len(before) == 1
