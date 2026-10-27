"""Factory for the real DiscoverPhysics field executor (ScienceAgent / PhysicsSchool).

Every call builds a new executor, which is how a world is reset. Heavy imports
(jax, scienceagent) stay inside the factory so only processes that run physics pay for them.
"""

from dataclasses import dataclass

FIELD_GRID = 32
FIELD_DOMAIN = 40.0
FIELD_DT = 0.01

OPERATORS = {
    "gravity": [{"type": "laplacian", "params": {"strength": 1.0}}],
    "yukawa": [{"type": "screening", "params": {"strength": 1.0, "screening_length": 2.0}}],
}


@dataclass(frozen=True)
class TrajectoryCase:
    family: str
    seed: int
    experiment: dict


def make_field_factory(noise_std=0.0):
    """Return a factory that maps a TrajectoryCase to a new SimulationExecutor."""

    def factory(case):
        if case.family not in OPERATORS:
            raise ValueError(f"no physics executor for family: {case.family}")
        import scienceagent.executor as api

        return api.SimulationExecutor(
            operators=OPERATORS[case.family],
            temporal_order=0,
            grid_size=(FIELD_GRID, FIELD_GRID),
            domain_size=FIELD_DOMAIN,
            dt=FIELD_DT,
            noise_std=noise_std,
            noise_seed=case.seed,
        )

    return factory
