import subprocess
import sys
import textwrap

import pytest

from metis.basic_environment_adapter import BasicEnvironmentAdapter
from metis.budget_ledger import BudgetLedger
from metis.evidence import EvidenceRecord, load_evidence
from metis.recovery import RecoveryState, load_recovery, save_recovery
from metis.run_loop import run_loop
from metis.seed_manager import get_rng_state


class FakeAgent:
    def act(self, observation: str) -> str:
        return "move"


class FakeExternalEnvironment:
    def __init__(self):
        self.actions = []

    def reset(self) -> str:
        return "Stat"

    def step(self, action: str) -> str:
        self.actions.append(action)
        return "Next"


class CrashingEnvironment:
    def __init__(self):
        self.step_calls = 0

    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> str:
        self.step_calls += 1
        raise RuntimeError("Simulator crashed")


class FakeEvaluator:
    def evaluate(self, result: str) -> float:
        return 1.0


class TrackingAgent:
    def __init__(self):
        self.observations = []

    def act(self, observation: str) -> str:
        assert isinstance(observation, str)
        self.observations.append(observation)
        return "move"


class IdentifiableEvaluator:
    def __init__(self):
        self.observations = []

    def evaluate(self, result: str) -> float:
        self.observations.append(result)
        return 987654.0


def test_run_loop_stops_when_budget_is_exhausted(tmp_path):
    agent = FakeAgent()
    environment = BasicEnvironmentAdapter(FakeExternalEnvironment())
    evaluator = FakeEvaluator()

    recovery_state = RecoveryState(
        run_id="1",
        step=0,
        budget_total=3,
        budget_used=0,
        state="start",
        rng_state=get_rng_state(),
    )

    recovery_path = tmp_path / "recovery.json"

    run_loop(
        agent,
        environment,
        evaluator,
        recovery_state,
        recovery_path,
    )

    assert recovery_state.step == 3
    assert recovery_state.budget_used == 3
    assert recovery_path.exists()

    restored = load_recovery(recovery_path)

    assert restored.state == "Next"
    assert restored.step == 3
    assert restored.budget_used == 3


def test_run_loop_resumes_from_partial_budget(tmp_path):
    class SnapshotEnvironment:
        def __init__(self):
            self.position = 0
            self.reset_calls = 0
            self.actions = []

        def reset(self):
            self.reset_calls += 1
            self.position = 0
            return "position:0"

        def step(self, action):
            self.actions.append(action)
            self.position += 1
            return f"position:{self.position}"

        def snapshot(self):
            return {"position": self.position}

        def restore(self, state):
            self.position = state["position"]

    original_environment = SnapshotEnvironment()
    original_adapter = BasicEnvironmentAdapter(original_environment)
    original_adapter.reset()
    original_adapter.step("move")
    original_adapter.step("move")

    recovery_state = RecoveryState(
        run_id="1",
        step=2,
        budget_total=3,
        budget_used=2,
        state="position:2",
        rng_state=get_rng_state(),
        environment_snapshot=original_adapter.snapshot(),
    )

    recovery_path = tmp_path / "recovery.json"
    save_recovery(recovery_state, recovery_path)
    restored = load_recovery(recovery_path)

    new_environment = SnapshotEnvironment()
    new_adapter = BasicEnvironmentAdapter(new_environment)

    run_loop(
        FakeAgent(),
        new_adapter,
        FakeEvaluator(),
        restored,
        recovery_path,
    )

    assert new_environment.reset_calls == 0
    assert new_environment.position == 3
    assert new_environment.actions == ["move"]
    assert restored.step == 3
    assert restored.budget_used == 3

    final_state = load_recovery(recovery_path)
    assert final_state.step == 3
    assert final_state.budget_used == 3
    assert final_state.state == "position:3"
    assert final_state.environment_snapshot is not None


def test_run_loop_blocks_resume_without_snapshot(tmp_path):
    environment = BasicEnvironmentAdapter(FakeExternalEnvironment())
    recovery_state = RecoveryState(
        run_id="missing-snapshot",
        step=2,
        budget_total=3,
        budget_used=2,
        state="start",
        rng_state=get_rng_state(),
    )

    with pytest.raises(
        RuntimeError,
        match="environment snapshot is missing",
    ):
        run_loop(
            FakeAgent(),
            environment,
            FakeEvaluator(),
            recovery_state,
            tmp_path / "recovery.json",
        )


def test_external_environment_harness_evidence_and_evaluator_isolation(
    tmp_path,
):
    external = FakeExternalEnvironment()
    adapter = BasicEnvironmentAdapter(external)
    agent = TrackingAgent()
    evaluator = IdentifiableEvaluator()

    state = RecoveryState(
        "integration",
        0,
        2,
        0,
        "start",
        get_rng_state(),
    )

    recovery_path = tmp_path / "recovery.json"
    evidence_path = tmp_path / "evidence.jsonl"

    run_loop(
        agent,
        adapter,
        evaluator,
        state,
        recovery_path,
        evidence_path,
    )

    assert agent.observations == ["Stat", "Next"]
    assert all("987654" not in observation for observation in agent.observations)
    assert 987654.0 not in agent.observations
    assert all(observation is not evaluator for observation in agent.observations)
    assert vars(agent) == {"observations": ["Stat", "Next"]}
    assert vars(adapter) == {
        "environment": external,
        "last_observation": "Next",
    }
    assert evaluator.observations == ["Next", "Next"]
    assert external.actions == ["move", "move"]
    assert state.step == state.budget_used == state.budget_total == 2
    assert load_recovery(recovery_path) == state
    assert evidence_path.exists()

    assert load_evidence(evidence_path) == [
        EvidenceRecord("integration", 1, "move", "Next", 987654.0),
        EvidenceRecord("integration", 2, "move", "Next", 987654.0),
    ]


# BudgetLedger integration


def test_run_loop_reserves_budget_before_simulator_execution(tmp_path):
    external = FakeExternalEnvironment()
    adapter = BasicEnvironmentAdapter(external)

    state = RecoveryState(
        "ledger-integration",
        0,
        3,
        0,
        "start",
        get_rng_state(),
    )

    ledger = BudgetLedger(
        tmp_path / "budget.sqlite",
        3,
        "ledger-integration",
    )

    try:
        run_loop(
            FakeAgent(),
            adapter,
            FakeEvaluator(),
            state,
            tmp_path / "recovery.json",
            ledger=ledger,
        )

        assert external.actions == ["move", "move", "move"]
        assert ledger.get_used() == 3
        assert ledger.get_remaining() == 0
        assert state.budget_used == 3
        assert state.step == 3

    finally:
        ledger.close()


def test_run_loop_rejects_mismatched_ledger_run_id(tmp_path):
    external = FakeExternalEnvironment()
    adapter = BasicEnvironmentAdapter(external)

    state = RecoveryState(
        "correct-run",
        0,
        2,
        0,
        "start",
        get_rng_state(),
    )

    ledger = BudgetLedger(
        tmp_path / "budget.sqlite",
        2,
        "wrong-run",
    )

    try:
        with pytest.raises(
            ValueError,
            match="Ledger run_id does not match recovery run_id",
        ):
            run_loop(
                FakeAgent(),
                adapter,
                FakeEvaluator(),
                state,
                tmp_path / "recovery.json",
                ledger=ledger,
            )

        assert external.actions == []
        assert ledger.get_used() == 0

    finally:
        ledger.close()


def test_run_loop_does_not_repeat_reserved_attempt(tmp_path):
    external = FakeExternalEnvironment()
    adapter = BasicEnvironmentAdapter(external)

    state = RecoveryState(
        "reserved-run",
        0,
        2,
        0,
        "start",
        get_rng_state(),
    )

    ledger = BudgetLedger(
        tmp_path / "budget.sqlite",
        2,
        "reserved-run",
    )

    try:
        assert ledger.reserve("reserved-run:1", 1) is True

        with pytest.raises(
            RuntimeError,
            match="already reserved",
        ):
            run_loop(
                FakeAgent(),
                adapter,
                FakeEvaluator(),
                state,
                tmp_path / "recovery.json",
                ledger=ledger,
            )

        assert external.actions == []
        assert ledger.get_used() == 1
        assert ledger.get_remaining() == 1
        assert state.step == 0
        assert state.budget_used == 0

    finally:
        ledger.close()


def test_run_loop_keeps_reservation_after_simulator_crash(tmp_path):
    external = CrashingEnvironment()
    adapter = BasicEnvironmentAdapter(external)

    state = RecoveryState(
        "crash-run",
        0,
        3,
        0,
        "start",
        get_rng_state(),
    )

    ledger = BudgetLedger(
        tmp_path / "budget.sqlite",
        3,
        "crash-run",
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="Simulator crashed",
        ):
            run_loop(
                FakeAgent(),
                adapter,
                FakeEvaluator(),
                state,
                tmp_path / "recovery.json",
                ledger=ledger,
            )

        assert external.step_calls == 1
        assert ledger.get_used() == 1
        assert ledger.get_remaining() == 2
        assert state.step == 0
        assert state.budget_used == 0

    finally:
        ledger.close()


def test_completed_attempt_status_survives_reopening(tmp_path):
    db_path = tmp_path / "budget.sqlite"

    state = RecoveryState(
        "completed-run",
        0,
        1,
        0,
        "start",
        get_rng_state(),
    )

    ledger = BudgetLedger(
        db_path,
        1,
        "completed-run",
    )

    try:
        run_loop(
            FakeAgent(),
            BasicEnvironmentAdapter(FakeExternalEnvironment()),
            FakeEvaluator(),
            state,
            tmp_path / "recovery.json",
            ledger=ledger,
        )

        assert ledger.get_attempt_status("completed-run:1") == "COMPLETED"
        assert ledger.get_used() == 1
        assert state.step == 1

    finally:
        ledger.close()

    reopened = BudgetLedger(
        db_path,
        1,
        "completed-run",
    )

    try:
        assert reopened.get_attempt_status("completed-run:1") == "COMPLETED"
        assert reopened.get_used() == 1

    finally:
        reopened.close()


def test_unknown_attempt_blocks_retry_after_crash(tmp_path):
    db_path = tmp_path / "budget.sqlite"
    recovery_path = tmp_path / "recovery.json"

    state = RecoveryState(
        "unknown-run",
        0,
        3,
        0,
        "start",
        get_rng_state(),
    )

    external = CrashingEnvironment()
    ledger = BudgetLedger(
        db_path,
        3,
        "unknown-run",
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="Simulator crashed",
        ):
            run_loop(
                FakeAgent(),
                BasicEnvironmentAdapter(external),
                FakeEvaluator(),
                state,
                recovery_path,
                ledger=ledger,
            )

        assert external.step_calls == 1
        assert ledger.get_attempt_status("unknown-run:1") == "UNKNOWN"
        assert ledger.get_used() == 1
        assert state.step == 0

    finally:
        ledger.close()

    reopened = BudgetLedger(
        db_path,
        3,
        "unknown-run",
    )

    retry_environment = CrashingEnvironment()

    try:
        with pytest.raises(
            RuntimeError,
            match="reconciliation required",
        ):
            run_loop(
                FakeAgent(),
                BasicEnvironmentAdapter(retry_environment),
                FakeEvaluator(),
                state,
                recovery_path,
                ledger=reopened,
            )

        assert retry_environment.step_calls == 0
        assert reopened.get_attempt_status("unknown-run:1") == "UNKNOWN"
        assert reopened.get_used() == 1

    finally:
        reopened.close()


def test_process_death_preserves_dispatch_and_blocks_retry(tmp_path):
    db_path = tmp_path / "budget.sqlite"
    recovery_path = tmp_path / "recovery.json"

    child_code = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        from metis.run_loop import run_loop
        from metis.budget_ledger import BudgetLedger
        from metis.recovery import RecoveryState
        from metis.seed_manager import get_rng_state
        from metis.basic_environment_adapter import BasicEnvironmentAdapter

        class Agent:
            def act(self, observation):
                return "move"

        class Environment:
            def reset(self):
                return "start"

            def step(self, action):
                os._exit(37)

        class Evaluator:
            def evaluate(self, result):
                return 1.0

        db_path = Path(sys.argv[1])
        recovery_path = Path(sys.argv[2])

        ledger = BudgetLedger(db_path, 3, "power-test")

        state = RecoveryState(
            "power-test",
            0,
            3,
            0,
            "start",
            get_rng_state(),
        )

        run_loop(
            Agent(),
            BasicEnvironmentAdapter(Environment()),
            Evaluator(),
            state,
            recovery_path,
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
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 37, result.stdout + result.stderr

    ledger = BudgetLedger(
        db_path,
        3,
        "power-test",
    )

    try:
        assert ledger.get_used() == 1
        assert ledger.get_remaining() == 2
        assert ledger.get_attempt_status("power-test:1") == "DISPATCHED"
        assert not recovery_path.exists()

        external = FakeExternalEnvironment()

        state = RecoveryState(
            "power-test",
            0,
            3,
            0,
            "start",
            get_rng_state(),
        )

        with pytest.raises(
            RuntimeError,
            match="reconciliation required",
        ):
            run_loop(
                FakeAgent(),
                BasicEnvironmentAdapter(external),
                FakeEvaluator(),
                state,
                recovery_path,
                ledger=ledger,
            )

        assert external.actions == []
        assert ledger.get_used() == 1

    finally:
        ledger.close()


def test_reconciliation_blocks_missing_evidence(tmp_path):
    state = RecoveryState(
        "missing-evidence",
        0,
        2,
        0,
        "start",
        get_rng_state(),
    )

    ledger = BudgetLedger(
        tmp_path / "budget.sqlite",
        2,
        "missing-evidence",
    )

    recovery_path = tmp_path / "recovery.json"
    evidence_path = tmp_path / "evidence.jsonl"

    try:
        run_loop(
            FakeAgent(),
            BasicEnvironmentAdapter(FakeExternalEnvironment()),
            FakeEvaluator(),
            state,
            recovery_path,
            evidence_path=evidence_path,
            ledger=ledger,
        )

        # Simulate loss of the evidence file.
        evidence_path.unlink()

        restored = load_recovery(recovery_path)

        with pytest.raises(
            RuntimeError,
            match="Reconciliation required: evidence",
        ):
            run_loop(
                FakeAgent(),
                BasicEnvironmentAdapter(FakeExternalEnvironment()),
                FakeEvaluator(),
                restored,
                recovery_path,
                evidence_path=evidence_path,
                ledger=ledger,
            )

    finally:
        ledger.close()


def test_reconciliation_blocks_changed_observation(tmp_path):
    import json

    run_id = "changed-observation"
    recovery_path = tmp_path / "recovery.json"
    evidence_path = tmp_path / "evidence.jsonl"

    state = RecoveryState(
        run_id,
        0,
        2,
        0,
        "start",
        get_rng_state(),
    )

    ledger = BudgetLedger(
        tmp_path / "budget.sqlite",
        2,
        run_id,
    )

    try:
        run_loop(
            FakeAgent(),
            BasicEnvironmentAdapter(FakeExternalEnvironment()),
            FakeEvaluator(),
            state,
            recovery_path,
            evidence_path=evidence_path,
            ledger=ledger,
        )

        records = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]

        records[-1]["observation"] = "Alterado"

        evidence_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        restored = load_recovery(recovery_path)

        with pytest.raises(
            RuntimeError,
            match="evidence observation mismatch",
        ):
            run_loop(
                FakeAgent(),
                BasicEnvironmentAdapter(FakeExternalEnvironment()),
                FakeEvaluator(),
                restored,
                recovery_path,
                evidence_path=evidence_path,
                ledger=ledger,
            )

    finally:
        ledger.close()
