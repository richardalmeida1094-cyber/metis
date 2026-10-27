from pathlib import Path

from .budget_ledger import BudgetLedger
from .evidence import EvidenceRecord, append_evidence, load_evidence
from .interfaces import Agent, EnvironmentAdapter, Evaluator
from .recovery import RecoveryState, save_recovery
from .seed_manager import get_rng_state, set_rng_state


def reconcile_run(
    recovery_state: RecoveryState,
    ledger: BudgetLedger,
    evidence_path: Path | None = None,
) -> None:
    """
    Check consistency between ledger, recovery and evidence.

    Unresolved attempts block execution.
    No automatic refunds or retries.
    """

    if ledger.run_id != recovery_state.run_id:
        raise ValueError("Ledger run_id does not match recovery run_id")

    if ledger.budget_total != recovery_state.budget_total:
        raise ValueError("Ledger budget does not match recovery budget")

    used = ledger.get_used()
    completed = recovery_state.budget_used

    if used < completed:
        raise RuntimeError("Reconciliation required: recovery exceeds ledger")

    if used > completed:
        raise RuntimeError("Attempt already reserved; reconciliation required: unresolved attempts exist")

    for step in range(1, recovery_state.step + 1):
        attempt_id = f"{recovery_state.run_id}:{step}"
        status = ledger.get_attempt_status(attempt_id)

        if status != "COMPLETED":
            raise RuntimeError(f"Reconciliation required: attempt {attempt_id} has status {status}")

    if evidence_path is None:
        return

    if not evidence_path.exists():
        if recovery_state.step > 0:
            raise RuntimeError("Reconciliation required: evidence file is missing")
        return

    try:
        records = load_evidence(evidence_path)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("Reconciliation required: evidence is unreadable") from error

    if any(record.run_id != recovery_state.run_id for record in records):
        raise RuntimeError("Reconciliation required: unexpected evidence run_id")

    expected_steps = list(range(1, recovery_state.step + 1))
    actual_steps = [record.step for record in records]

    if actual_steps != expected_steps:
        raise RuntimeError("Reconciliation required: evidence steps mismatch")

    if records:
        if records[-1].observation != recovery_state.state:
            raise RuntimeError("Reconciliation required: evidence observation mismatch")


def _initial_observation(
    environment: EnvironmentAdapter,
    recovery_state: RecoveryState,
) -> str:
    """
    Start a new run or restore a saved environment.

    Never reset a partially completed run.
    """

    if recovery_state.step == 0:
        return environment.reset()

    snapshot = recovery_state.environment_snapshot

    if snapshot is None:
        raise RuntimeError("Cannot resume: environment snapshot is missing")

    capabilities = environment.capabilities()

    if not capabilities.supports_restore:
        raise RuntimeError("Cannot resume: environment does not support restore")

    restore = getattr(environment, "restore", None)

    if not callable(restore):
        raise RuntimeError("Cannot resume: restore method is unavailable")

    restore(snapshot)

    observation = environment.observe()

    if observation != recovery_state.state:
        raise RuntimeError("Cannot resume: restored observation mismatch")

    # Restore Python's RNG only after validating the environment.
    set_rng_state(recovery_state.rng_state)

    return observation


def run_loop(
    agent: Agent,
    environment: EnvironmentAdapter,
    evaluator: Evaluator,
    recovery_state: RecoveryState,
    recovery_path: Path,
    evidence_path: Path | None = None,
    ledger: BudgetLedger | None = None,
) -> None:
    """
    Coordinate execution, accounting, evidence and recovery.

    Reserve budget before simulator dispatch.
    Persist attempt status.
    Save environment snapshots and Python RNG state.
    Block unresolved attempts.
    """

    if ledger is not None:
        reconcile_run(
            recovery_state,
            ledger,
            evidence_path,
        )

    observation = _initial_observation(
        environment,
        recovery_state,
    )

    while recovery_state.budget_used < recovery_state.budget_total:
        attempt_id = f"{recovery_state.run_id}:{recovery_state.step + 1}"

        if ledger is not None:
            reserved = ledger.reserve(attempt_id, 1)

            if not reserved:
                raise RuntimeError(f"Attempt {attempt_id} already reserved; reconciliation required")

        try:
            action = agent.act(observation)

            if ledger is not None:
                ledger.set_attempt_status(
                    attempt_id,
                    "DISPATCHED",
                )

            observation = environment.step(action)
            score = evaluator.evaluate(observation)

            next_step = recovery_state.step + 1
            next_budget_used = recovery_state.budget_used + 1

            environment_snapshot = None
            capabilities = environment.capabilities()

            if capabilities.supports_snapshot and capabilities.supports_restore:
                snapshot_method = getattr(
                    environment,
                    "snapshot",
                    None,
                )

                if not callable(snapshot_method):
                    raise RuntimeError("Environment advertises snapshot but does not implement it")

                environment_snapshot = snapshot_method()

            # Capture the current RNG after processing the experiment.
            next_rng_state = get_rng_state()

            if evidence_path is not None:
                append_evidence(
                    EvidenceRecord(
                        run_id=recovery_state.run_id,
                        step=next_step,
                        action=action,
                        observation=observation,
                        score=score,
                    ),
                    evidence_path,
                )

            next_state = RecoveryState(
                run_id=recovery_state.run_id,
                step=next_step,
                budget_total=recovery_state.budget_total,
                budget_used=next_budget_used,
                state=observation,
                rng_state=next_rng_state,
                environment_snapshot=environment_snapshot,
            )

            save_recovery(next_state, recovery_path)

            recovery_state.step = next_state.step
            recovery_state.budget_used = next_state.budget_used
            recovery_state.state = next_state.state
            recovery_state.rng_state = next_state.rng_state
            recovery_state.environment_snapshot = next_state.environment_snapshot

            if ledger is not None:
                ledger.set_attempt_status(
                    attempt_id,
                    "COMPLETED",
                )

        except Exception:
            if ledger is not None:
                current_status = ledger.get_attempt_status(attempt_id)

                if current_status in {
                    "RESERVED",
                    "DISPATCHED",
                }:
                    ledger.set_attempt_status(
                        attempt_id,
                        "UNKNOWN",
                    )

            raise
