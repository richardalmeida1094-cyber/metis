import random


def set_seed(seed: int) -> None:
    random.seed(seed)


def get_rng_state():
    return random.getstate()


def set_rng_state(state):
    random.setstate(state)
