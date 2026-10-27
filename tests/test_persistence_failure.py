import pytest

from metis import run_loop as run_loop_module
from metis.basic_environment_adapter import BasicEnvironmentAdapter
from metis.budget_ledger import BudgetLedger
from metis.evidence import load_evidence
from metis.recovery import RecoveryState
from metis.seed_manager import get_rng_state


class CountingEnvironment:
    def __init__(self):
        self.step_calls = 0

    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> str:
        self.step_calls += 1
        return "completed"


class FixedAgent:
    def act(self, observation: str) -> str:
        return "move"


class FixedEvaluator:
    def evaluate(self, result: str) -> float:
        return 1.0


def test_failure_after_evidence_before_recovery(
    tmp_path,
    monkeypatch,
):
    recovery_path = tmp_path / "recovery.json"
    evidence_path = tmp_path / "evidence.jsonl"
    db_path = tmp_path / "budget.sqlite"

    state = RecoveryState(
        run_id="persistence-failure",
        step=0,
        budget_total=2,
        budget_used=0,
        state="start",
        rng_state=get_rng_state(),
    )

    environment = CountingEnvironment()

    ledger = BudgetLedger(
        db_path,
        2,
        state.run_id,
    )

    original_save_recovery = run_loop_module.save_recovery

    def failing_save_recovery(state, path):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr(
        run_loop_module,
        "save_recovery",
        failing_save_recovery,
    )

    try:
        with pytest.raises(
            OSError,
            match="Simulated disk failure",
        ):
            run_loop_module.run_loop(
                FixedAgent(),
                BasicEnvironmentAdapter(environment),
                FixedEvaluator(),
                state,
                recovery_path,
                evidence_path=evidence_path,
                ledger=ledger,
            )

        assert environment.step_calls == 1
        assert ledger.get_used() == 1
        assert ledger.get_attempt_status("persistence-failure:1") == "UNKNOWN"

        assert state.step == 0
        assert not recovery_path.exists()

        records = load_evidence(evidence_path)
        assert len(records) == 1
        assert records[0].step == 1

    finally:
        ledger.close()

    monkeypatch.setattr(
        run_loop_module,
        "save_recovery",
        original_save_recovery,
    )

    reopened = BudgetLedger(
        db_path,
        2,
        "persistence-failure",
    )

    retry_environment = CountingEnvironment()

    try:
        with pytest.raises(
            RuntimeError,
            match="reconciliation required",
        ):
            run_loop_module.run_loop(
                FixedAgent(),
                BasicEnvironmentAdapter(retry_environment),
                FixedEvaluator(),
                state,
                recovery_path,
                evidence_path=evidence_path,
                ledger=reopened,
            )

        assert retry_environment.step_calls == 0
        assert reopened.get_used() == 1

    finally:
        reopened.close()
