"""Allowlist for observations sent to the agent.

Only numeric trajectory fields pass. Unknown, missing or non-numeric fields are rejected,
so simulator metadata (law family, regime, seed) can never leak to the agent.
"""

import copy

AGENT_PAYLOAD_KEYS = ("measurement_times", "pos1", "pos2", "velocity1", "velocity2")


class PayloadRejected(ValueError):
    """Observation that cannot be delivered to the agent."""


def _numeric(value):
    if isinstance(value, list):
        return bool(value) and all(_numeric(item) for item in value)
    return type(value) is float


def agent_payload(observation):
    """Return only the allowlisted numeric fields, or raise PayloadRejected."""
    if not isinstance(observation, dict):
        raise PayloadRejected("observation is not a dict")
    unknown = sorted(set(observation) - set(AGENT_PAYLOAD_KEYS))
    if unknown:
        raise PayloadRejected(f"fields not allowed: {unknown}")
    missing = [key for key in AGENT_PAYLOAD_KEYS if key not in observation]
    if missing:
        raise PayloadRejected(f"missing fields: {missing}")
    for key in AGENT_PAYLOAD_KEYS:
        if not _numeric(observation[key]):
            raise PayloadRejected(f"{key}: non-numeric content")
    return {key: copy.deepcopy(observation[key]) for key in AGENT_PAYLOAD_KEYS}
