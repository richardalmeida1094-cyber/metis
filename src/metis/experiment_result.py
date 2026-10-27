from dataclasses import dataclass


@dataclass
class ExperimentResult:
    experiment_id: str
    requested_repetitions: int
    results: list[str]
    completed: bool
    failure_reason: str | None

    def __post_init__(self):
        if self.requested_repetitions <= 0:
            raise ValueError

        if len(self.results) > self.requested_repetitions:
            raise ValueError

        if self.completed and len(self.results) != self.requested_repetitions:
            raise ValueError

        if self.completed and self.failure_reason is not None:
            raise ValueError

        if not self.completed and not self.failure_reason:
            raise ValueError

        if not self.completed and len(self.results) == self.requested_repetitions:
            raise ValueError
