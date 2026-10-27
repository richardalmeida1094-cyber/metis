from dataclasses import dataclass


@dataclass
class Attempt:
    attempt_id: str
    run_id: str
    cost: int
    status: str

    def __post_init__(self):
        if type(self.attempt_id) is not str or not self.attempt_id.strip():
            raise ValueError("Attempt ID must be a non-empty string.")

        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("Run ID must be a non-empty string.")

        if type(self.cost) is not int or self.cost <= 0:
            raise ValueError("Cost must be a positive integer.")

        valid_statuses = {
            "RESERVED",
            "DISPATCHED",
            "COMPLETED",
            "UNKNOWN",
        }

        if type(self.status) is not str or self.status not in valid_statuses:
            raise ValueError("Invalid attempt status.")
