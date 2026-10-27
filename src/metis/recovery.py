import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RecoveryState:
    run_id: str
    step: int
    budget_total: int
    budget_used: int
    state: str
    rng_state: tuple
    environment_snapshot: Any = None

    def __post_init__(self):
        if self.step < 0:
            raise ValueError("step cannot be negative")

        if self.budget_total <= 0:
            raise ValueError("budget_total must be greater than 0")

        if self.budget_used < 0:
            raise ValueError("budget_used cannot be negative")

        if self.budget_used > self.budget_total:
            raise ValueError("budget_used cannot be greater than budget_total")


def save_recovery(state, path):
    data = asdict(state)
    temp_path = path.parent / (path.name + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, path)


def load_recovery(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    data["rng_state"] = deserialize_rng_state(data["rng_state"])

    return RecoveryState(**data)


def deserialize_rng_state(rng_state):
    return (
        rng_state[0],
        tuple(rng_state[1]),
        rng_state[2],
    )
