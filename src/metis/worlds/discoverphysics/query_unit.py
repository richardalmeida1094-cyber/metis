from dataclasses import dataclass


@dataclass
class QueryUnit:
    measurements: int
    duration: float
    particles: int

    def effective_duration(self) -> float:
        if self.duration < 5.0:
            return 5.0

        return self.duration

    def count_query(self) -> int:
        return 1

    def __post_init__(self):
        if self.measurements <= 0:
            raise ValueError("measurements must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.particles <= 0:
            raise ValueError("particles must be positive")
