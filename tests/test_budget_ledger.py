import sqlite3
import threading
from pathlib import Path

import pytest

from metis.budget_ledger import BudgetLedger


def count_attempts(ledger: BudgetLedger, run_id: str) -> int:
    return ledger.connection.execute(
        "SELECT COUNT(*) FROM attempts WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]


# ============================================================
# PART 1: RUN REGISTRATION AND PERSISTENCE
# ============================================================


def test_creates_new_run(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        result = ledger.connection.execute("SELECT run_id, budget_total FROM runs").fetchone()

        assert result == ("run-001", 100)
    finally:
        ledger.close()


def test_reopens_existing_run(tmp_path: Path):
    db_path = tmp_path / "ledger.db"

    BudgetLedger(db_path, 100, "run-001").close()
    second = BudgetLedger(db_path, 100, "run-001")

    try:
        result = second.connection.execute("SELECT run_id, budget_total FROM runs").fetchone()

        assert result == ("run-001", 100)
    finally:
        second.close()


def test_rejects_conflicting_budget(tmp_path: Path):
    db_path = tmp_path / "ledger.db"

    BudgetLedger(db_path, 100, "run-001").close()

    with pytest.raises(ValueError, match="Run budget does not match"):
        BudgetLedger(db_path, 200, "run-001")


def test_does_not_duplicate_run(tmp_path: Path):
    db_path = tmp_path / "ledger.db"

    BudgetLedger(db_path, 100, "run-001").close()
    second = BudgetLedger(db_path, 100, "run-001")

    try:
        count = second.connection.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?",
            ("run-001",),
        ).fetchone()[0]

        assert count == 1
    finally:
        second.close()


def test_persists_after_connection_closes(tmp_path: Path):
    db_path = tmp_path / "ledger.db"

    BudgetLedger(db_path, 100, "run-001").close()

    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            "SELECT budget_total FROM runs WHERE run_id = ?",
            ("run-001",),
        ).fetchone()

    assert result == (100,)


def test_separates_different_runs(tmp_path: Path):
    db_path = tmp_path / "ledger.db"

    BudgetLedger(db_path, 100, "run-001").close()
    BudgetLedger(db_path, 200, "run-002").close()

    with sqlite3.connect(db_path) as connection:
        result = connection.execute("SELECT run_id, budget_total FROM runs ORDER BY run_id").fetchall()

    assert result == [("run-001", 100), ("run-002", 200)]


# ============================================================
# PART 2: RESERVE CONTRACT
# ============================================================


def test_reserve_within_budget_writes_one_row(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        assert count_attempts(ledger, "run-001") == 0
        assert ledger.reserve("A001", 40) is True

        rows = ledger.connection.execute("SELECT run_id, attempt_id, cost FROM attempts").fetchall()

        assert rows == [("run-001", "A001", 40)]
    finally:
        ledger.close()


def test_reserve_exact_total_then_excess_writes_no_row(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        assert ledger.reserve("A001", 30) is True
        assert ledger.reserve("A002", 30) is True
        assert ledger.reserve("A003", 40) is True
        assert ledger.get_used() == 100

        before = count_attempts(ledger, "run-001")

        with pytest.raises(ValueError, match="Insufficient budget"):
            ledger.reserve("A004", 1)

        assert count_attempts(ledger, "run-001") == before == 3
        assert ledger.get_used() == 100
    finally:
        ledger.close()


def test_reserve_same_id_and_cost_does_not_debit_twice(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        assert ledger.reserve("A001", 25) is True
        assert ledger.reserve("A001", 25) is False

        assert ledger.get_used() == 25
        assert count_attempts(ledger, "run-001") == 1
    finally:
        ledger.close()


def test_reserve_same_id_with_different_cost_keeps_original(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        assert ledger.reserve("A001", 25) is True

        with pytest.raises(ValueError, match="different cost"):
            ledger.reserve("A001", 30)

        rows = ledger.connection.execute(
            "SELECT attempt_id, cost FROM attempts WHERE run_id = ?",
            ("run-001",),
        ).fetchall()

        assert rows == [("A001", 25)]
        assert ledger.get_used() == 25
        assert ledger.reserve("A001", 25) is False
    finally:
        ledger.close()


@pytest.mark.parametrize(
    ("attempt_id", "cost", "message"),
    [
        ("", 7, "attempt_id cannot be empty"),
        ("A001", 0, "positive integer"),
        ("A001", -1, "positive integer"),
        ("A001", True, "positive integer"),
        ("A001", 1.5, "positive integer"),
        ("A001", "1", "positive integer"),
        ("A001", None, "positive integer"),
    ],
)
def test_reserve_rejects_invalid_arguments(tmp_path: Path, attempt_id, cost, message):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        with pytest.raises(ValueError, match=message):
            ledger.reserve(attempt_id, cost)

        assert count_attempts(ledger, "run-001") == 0
    finally:
        ledger.close()


def test_exhausted_run_does_not_block_other_run(tmp_path: Path):
    db_path = tmp_path / "ledger.db"

    first = BudgetLedger(db_path, 50, "run-001")
    second = BudgetLedger(db_path, 50, "run-002")

    try:
        assert first.reserve("A001", 50) is True

        with pytest.raises(ValueError, match="Insufficient budget"):
            first.reserve("A002", 1)

        assert second.reserve("A001", 50) is True
        assert first.get_used() == 50
        assert second.get_used() == 50
    finally:
        first.close()
        second.close()


def test_used_budget_still_counts_after_reopening(tmp_path: Path):
    db_path = tmp_path / "ledger.db"

    first = BudgetLedger(db_path, 100, "run-001")

    try:
        assert first.reserve("A001", 60) is True
    finally:
        first.close()

    second = BudgetLedger(db_path, 100, "run-001")

    try:
        assert second.get_used() == 60
        assert second.reserve("A001", 60) is False

        with pytest.raises(ValueError, match="Insufficient budget"):
            second.reserve("A002", 41)

        assert second.reserve("A002", 40) is True
        assert second.get_remaining() == 0
    finally:
        second.close()


def test_reservation_visible_to_separate_sqlite_connection(tmp_path: Path):
    db_path = tmp_path / "ledger.db"
    ledger = BudgetLedger(db_path, 100, "run-001")

    try:
        assert ledger.reserve("A001", 25) is True
    finally:
        ledger.close()

    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            "SELECT attempt_id, cost FROM attempts WHERE run_id = ?",
            ("run-001",),
        ).fetchall()

    assert result == [("A001", 25)]


def test_connection_is_usable_after_value_error(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        assert ledger.reserve("A001", 90) is True

        with pytest.raises(ValueError, match="Insufficient budget"):
            ledger.reserve("A002", 20)

        assert ledger.connection.in_transaction is False

        with pytest.raises(ValueError, match="different cost"):
            ledger.reserve("A001", 5)

        assert ledger.connection.in_transaction is False
        assert ledger.reserve("A003", 10) is True
        assert count_attempts(ledger, "run-001") == 2
    finally:
        ledger.close()


# ============================================================
# PART 3: BUDGET QUERIES AND CONNECTION MANAGEMENT
# ============================================================


def test_get_used_and_remaining(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    try:
        assert ledger.get_used() == 0
        assert ledger.get_remaining() == 100

        ledger.reserve("A001", 20)
        ledger.reserve("A002", 35)

        assert ledger.get_used() == 55
        assert ledger.get_remaining() == 45
    finally:
        ledger.close()


def test_close_closes_connection(tmp_path: Path):
    ledger = BudgetLedger(tmp_path / "ledger.db", 100, "run-001")

    ledger.close()

    with pytest.raises(sqlite3.ProgrammingError):
        ledger.connection.execute("SELECT 1")


# ============================================================
# PART 4: CONCURRENT RESERVATIONS
# ============================================================


def test_concurrent_reservations_do_not_exceed_budget(tmp_path: Path):
    db_path = tmp_path / "ledger.db"
    budget_total = 10
    run_id = "run-001"

    # Initialize the database before starting the threads.
    BudgetLedger(db_path, budget_total, run_id).close()

    barrier = threading.Barrier(2)
    results = {}
    errors = {}

    def worker(attempt_id: str) -> None:
        ledger = None

        try:
            # Each thread must create its own SQLite connection.
            ledger = BudgetLedger(db_path, budget_total, run_id)
            barrier.wait(timeout=10)
            results[attempt_id] = ledger.reserve(attempt_id, 10)
        except Exception as exc:
            errors[attempt_id] = exc
        finally:
            if ledger is not None:
                ledger.close()

    thread_a = threading.Thread(target=worker, args=("A001",))
    thread_b = threading.Thread(target=worker, args=("A002",))

    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=15)
    thread_b.join(timeout=15)

    assert not thread_a.is_alive(), "Thread A did not finish"
    assert not thread_b.is_alive(), "Thread B did not finish"

    # Exactly one reservation succeeds; the other fails for lack of budget.
    assert sum(value is True for value in results.values()) == 1
    assert len(errors) == 1

    error = next(iter(errors.values()))
    assert isinstance(error, ValueError)
    assert "Insufficient budget" in str(error)

    verification = BudgetLedger(db_path, budget_total, run_id)

    try:
        assert verification.get_used() == 10
        assert verification.get_remaining() == 0
        assert count_attempts(verification, run_id) == 1
    finally:
        verification.close()


# ============================================================
# PART 5: TRANSACTION FAILURE AND DATABASE INTEGRITY
# ============================================================


def test_reserve_rolls_back_when_insert_fails(tmp_path: Path):
    db_path = tmp_path / "ledger.db"
    ledger = BudgetLedger(db_path, 100, "run-001")

    try:
        # Simulate a database failure after an INSERT is attempted.
        ledger.connection.execute(
            """
            CREATE TRIGGER fail_attempt_insert
            AFTER INSERT ON attempts
            BEGIN
                SELECT RAISE(ABORT, 'Injected database failure');
            END;
            """
        )
        ledger.connection.commit()

        with pytest.raises(
            sqlite3.DatabaseError,
            match="Injected database failure",
        ):
            ledger.reserve("A001", 30)

        # The failed transaction must not consume any budget.
        assert ledger.get_used() == 0
        assert ledger.get_remaining() == 100
        assert count_attempts(ledger, "run-001") == 0
        assert ledger.connection.in_transaction is False

        # Remove the injected failure and verify recovery.
        ledger.connection.execute("DROP TRIGGER fail_attempt_insert")
        ledger.connection.commit()

        assert ledger.reserve("A002", 30) is True
        assert ledger.get_used() == 30

    finally:
        ledger.close()


def test_rejects_invalid_initial_budget_without_persisting_run(
    tmp_path: Path,
):
    db_path = tmp_path / "ledger.db"

    with pytest.raises(ValueError, match="Run record was not found"):
        BudgetLedger(db_path, -100, "run-001")

    # Verify that the invalid run was not persisted.
    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?",
            ("run-001",),
        ).fetchone()[0]

    assert count == 0
