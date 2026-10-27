from dataclasses import dataclass


@dataclass
class Experiment:
    experiment_id: str
    objective: str
    action: str
    repetitions: int

    def __post_init__(self):
        if self.repetitions <= 0:
            raise ValueError
