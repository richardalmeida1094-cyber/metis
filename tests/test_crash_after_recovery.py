import subprocess
import sys
import textwrap

import pytest

from metis.basic_environment_adapter import BasicEnvironmentAdapter
from metis.budget_ledger import BudgetLedger
from metis.recovery import load_recovery
from metis.run_loop import run_loop


class FixedAgent:
    def act(self, observation: str) -> str:
        return "move"


class CountingEnvironment:
    def __init__(self):
        self.step_calls = 0

    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> str:
        self.step_calls += 1
        return "completed"


class FixedEvaluator:
    def evaluate(self, result: str) -> float:
        return 1.0


def test_process_death_after_recovery_before_completed(tmp_path):
    db_path = tmp_path / "budget.sqlite"
    recovery_path = tmp_path / "recovery.json"
    evidence_path = tmp_path / "evidence.jsonl"

    child_code = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import metis.run_loop as run_loop_module

        from metis.basic_environment_adapter import BasicEnvironmentAdapter
        from metis.budget_ledger import BudgetLedger
        from metis.recovery import RecoveryState
        from metis.seed_manager import get_rng_state

        class Agent:
            def act(self, observation):
                return "move"

        class Environment:
            def reset(self):
                return "start"

            def step(self, action):
                return "completed"

        class Evaluator:
            def evaluate(self, result):
                return 1.0

        db_path = Path(sys.argv[1])
        recovery_path = Path(sys.argv[2])
        evidence_path = Path(sys.argv[3])

        ledger = BudgetLedger(
            db_path,
            2,
            "crash-after-recovery",
        )

        state = RecoveryState(
            run_id="crash-after-recovery",
            step=0,
            budget_total=2,
            budget_used=0,
            state="start",
            rng_state=get_rng_state(),
        )

        original_set_status = ledger.set_attempt_status

        def crash_before_completed(attempt_id, status):
            if status == "COMPLETED":
                os._exit(37)

            return original_set_status(attempt_id, status)

        ledger.set_attempt_status = crash_before_completed

        run_loop_module.run_loop(
            Agent(),
            BasicEnvironmentAdapter(Environment()),
            Evaluator(),
            state,
            recovery_path,
            evidence_path=evidence_path,
            ledger=ledger,
        )
        """
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(db_path),
            str(recovery_path),
            str(evidence_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 37, result.stdout + result.stderr

    assert recovery_path.exists()
    assert evidence_path.exists()

    recovered = load_recovery(recovery_path)

    assert recovered.step == 1
    assert recovered.budget_used == 1
    assert recovered.state == "completed"

    reopened = BudgetLedger(
        db_path,
        2,
        "crash-after-recovery",
    )

    try:
        assert reopened.get_used() == 1
        assert reopened.get_attempt_status("crash-after-recovery:1") == "DISPATCHED"

        retry_environment = CountingEnvironment()

        with pytest.raises(
            RuntimeError,
            match="Reconciliation required",
        ):
            run_loop(
                FixedAgent(),
                BasicEnvironmentAdapter(retry_environment),
                FixedEvaluator(),
                recovered,
                recovery_path,
                evidence_path=evidence_path,
                ledger=reopened,
            )

        assert retry_environment.step_calls == 0
        assert reopened.get_used() == 1

    finally:
        reopened.close()
