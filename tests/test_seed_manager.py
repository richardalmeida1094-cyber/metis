import random

from metis.seed_manager import get_rng_state, set_rng_state, set_seed


def test_same_seed_repeats_sequence():
    set_seed(42)
    sequence_a = [
        random.randint(1, 100),
        random.randint(1, 100),
    ]

    set_seed(42)
    sequence_b = [
        random.randint(1, 100),
        random.randint(1, 100),
    ]

    assert sequence_a == sequence_b


def test_rng_state_can_be_restored():
    set_seed(42)

    saved_state = get_rng_state()

    number_a = random.randint(1, 100)

    set_rng_state(saved_state)

    number_b = random.randint(1, 100)

    assert number_a == number_b
