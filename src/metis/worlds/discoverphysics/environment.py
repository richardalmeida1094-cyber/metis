"""DiscoverPhysics world exposed through the core `Environment` protocol.

Each `step(action)` is one physics query: the action selects an experiment, a fresh
executor is built by the factory (rebuilding is how the world is reset), and only the
allowlisted numeric payload is returned to the agent, as canonical JSON.

Hidden regime: `regimes` is a list of `(first_query, family)` pairs. The active family
is passed to the executor and never appears in the observation. Snapshots store the
next query index and a fingerprint of the configuration, so a snapshot from a different
configuration is rejected and the hidden change cannot move after a restart.

The core never imports this module (see tests/test_architecture_boundaries.py).
"""

import hashlib
import json

from metis.worlds.discoverphysics.executor import TrajectoryCase
from metis.worlds.discoverphysics.families import audit_family
from metis.worlds.discoverphysics.payload import agent_payload

MEASUREMENT_TIMES = [0.0, 0.25, 0.5]
DURATION = 0.5  # 50 steps of dt = 0.01

ACTIONS = {
    "left": {"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.1]},
    "right": {"p1": 1.0, "p2": 1.0, "pos2": [5.0, 0.0], "velocity2": [0.0, 0.1]},
}

INITIAL_OBSERVATION = "start"


def experiment_for(action):
    if action not in ACTIONS:
        raise ValueError(f"Invalid action: {action}")
    return {**ACTIONS[action], "measurement_times": list(MEASUREMENT_TIMES), "duration": DURATION}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class DiscoverPhysicsEnvironment:
    def __init__(self, factory, seed, regimes):
        if not regimes or regimes[0][0] != 0:
            raise ValueError("regimes must start at query 0")
        starts = [start for start, _ in regimes]
        if starts != sorted(set(starts)):
            raise ValueError("regime starts must be strictly increasing")
        for _, family in regimes:
            if not audit_family(family).counts_as_evidence:
                raise ValueError(f"family is not ACTIVE: {family}")

        self.factory = factory
        self.seed = seed
        self.regimes = [(int(start), family) for start, family in regimes]
        self.next_query = 0
        self.config_sha256 = hashlib.sha256(
            canonical_json(
                {
                    "seed": seed,
                    "regimes": self.regimes,
                    "actions": ACTIONS,
                    "times": MEASUREMENT_TIMES,
                    "duration": DURATION,
                }
            ).encode()
        ).hexdigest()

    def family_for(self, query_index):
        family = self.regimes[0][1]
        for start, name in self.regimes:
            if query_index >= start:
                family = name
        return family

    def reset(self) -> str:
        self.next_query = 0
        return INITIAL_OBSERVATION

    def step(self, action: str) -> str:
        experiment = experiment_for(action)
        case = TrajectoryCase(family=self.family_for(self.next_query), seed=self.seed, experiment=experiment)
        executor = self.factory(case)
        observations = executor.run([experiment])
        self.next_query += 1
        return canonical_json(agent_payload(observations[0]))

    def snapshot(self) -> dict:
        return {"next_query": self.next_query, "config_sha256": self.config_sha256}

    def restore(self, state: dict) -> None:
        if state.get("config_sha256") != self.config_sha256:
            raise ValueError("snapshot belongs to a different environment configuration")
        next_query = state.get("next_query")
        if type(next_query) is not int or next_query < 0:
            raise ValueError("invalid next_query in snapshot")
        self.next_query = next_query


class FinalDistanceEvaluator:
    """Placeholder score: final distance of particle 2 from the origin.

    Not a scientific metric. It shows that the evaluator receives the trajectory and
    returns a deterministic number that is never sent back to the agent.
    """

    def evaluate(self, result: str) -> float:
        x, y = json.loads(result)["pos2"][-1]
        return float((x * x + y * y) ** 0.5)
