import random

import pytest

from metis.recovery import RecoveryState, load_recovery, save_recovery
from metis.seed_manager import get_rng_state, set_rng_state, set_seed


def test_recovery_rejects_negative_step():
    with pytest.raises(ValueError):
        RecoveryState(
            run_id="20", step=-1, budget_total=100, budget_used=60, state="running", rng_state=get_rng_state()
        )


def test_recovery_rejects_zero_budget_total():
    with pytest.raises(ValueError):
        RecoveryState(
            run_id="20", step=11, budget_total=0, budget_used=0, state="running", rng_state=get_rng_state()
        )


def test_recovery_rejects_negative_budget_used():
    with pytest.raises(ValueError):
        RecoveryState(
            run_id="20", step=11, budget_total=50, budget_used=-1, state="running", rng_state=get_rng_state()
        )


def test_recovery_rejects_budget_used_greater_than_total():
    with pytest.raises(ValueError):
        RecoveryState(
            run_id="20", step=12, budget_total=20, budget_used=26, state="running", rng_state=get_rng_state()
        )


def test_recovery_accepts_valid_boundary_values():
    state_full = RecoveryState(
        run_id="20", step=0, budget_total=20, budget_used=20, state="running", rng_state=get_rng_state()
    )

    state_empty = RecoveryState(
        run_id="20", step=0, budget_total=20, budget_used=0, state="running", rng_state=get_rng_state()
    )

    assert state_full.budget_used == 20
    assert state_empty.budget_used == 0


def test_recovery_round_trip(tmp_path):
    original_state = RecoveryState(
        run_id="1", step=7, budget_total=10, budget_used=7, state="running", rng_state=get_rng_state()
    )

    path = tmp_path / "recovery.json"

    save_recovery(original_state, path)

    restored_state = load_recovery(path)

    assert restored_state == original_state


def test_recovery_restores_rng_sequence():
    set_seed(42)
    saved_rng_state = get_rng_state()
    expected_number = random.randint(1, 100)
    set_rng_state(saved_rng_state)
    restored_number = random.randint(1, 100)
    assert restored_number == expected_number
