"""Run the random baseline agent on DiscoverPhysics, with crash-safe resume.

Example:
    python scripts/run_discoverphysics.py --out runs/demo --regimes 0:gravity,3:yukawa
    python scripts/run_discoverphysics.py --out runs/demo2 --regimes 0:gravity,3:yukawa --stop-after 2
    python scripts/run_discoverphysics.py --out runs/demo2 --regimes 0:gravity,3:yukawa --resume

`--stop-after K` kills the process (os._exit, no cleanup) right after step K is committed.
`--resume` continues from the files on disk. The final evidence is identical to an
uninterrupted run with the same arguments.
"""

import argparse
import os
import sys
from pathlib import Path

from metis.agents.random_agent import RandomAgent
from metis.basic_environment_adapter import BasicEnvironmentAdapter
from metis.budget_ledger import BudgetLedger
from metis.recovery import RecoveryState, load_recovery
from metis.replay import replay_hash
from metis.run_loop import run_loop
from metis.seed_manager import get_rng_state, set_seed
from metis.worlds.discoverphysics.environment import DiscoverPhysicsEnvironment, FinalDistanceEvaluator
from metis.worlds.discoverphysics.executor import make_field_factory

RUN_ID = "discoverphysics-demo"


def parse_regimes(text):
    regimes = []
    for item in text.split(","):
        start, _, family = item.partition(":")
        regimes.append((int(start), family.strip()))
    return regimes


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="run directory")
    parser.add_argument("--seed", type=int, default=2026101101)
    parser.add_argument("--budget", type=int, default=6, help="number of physics queries")
    parser.add_argument("--regimes", default="0:gravity", help="hidden law schedule, e.g. 0:gravity,3:yukawa")
    parser.add_argument("--stop-after", type=int, help="kill the process right after this step")
    parser.add_argument("--resume", action="store_true", help="continue a run from disk")
    return parser.parse_args(argv)


def stop_after(ledger, step):
    original = ledger.set_attempt_status

    def set_status(attempt_id, status):
        original(attempt_id, status)
        if status == "COMPLETED" and attempt_id == f"{RUN_ID}:{step}":
            print(f"stopping the process after step {step}")
            os._exit(1)

    ledger.set_attempt_status = set_status


def main(argv=None, factory=None):
    args = parse_args(argv)
    recovery_path = args.out / "recovery.json"
    evidence_path = args.out / "evidence.jsonl"

    if args.resume:
        state = load_recovery(recovery_path)
    else:
        if recovery_path.exists():
            sys.exit(f"{args.out} already holds a run; use --resume or another --out")
        args.out.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed)
        state = RecoveryState(RUN_ID, 0, args.budget, 0, "start", get_rng_state())

    environment = BasicEnvironmentAdapter(
        DiscoverPhysicsEnvironment(factory or make_field_factory(), args.seed, parse_regimes(args.regimes))
    )
    ledger = BudgetLedger(args.out / "budget.sqlite", args.budget, RUN_ID)
    if args.stop_after:
        stop_after(ledger, args.stop_after)
    try:
        run_loop(
            RandomAgent(),
            environment,
            FinalDistanceEvaluator(),
            state,
            recovery_path,
            evidence_path=evidence_path,
            ledger=ledger,
        )
    finally:
        ledger.close()

    evidence = evidence_path.read_text(encoding="utf-8")
    print(f"steps: {state.step}/{state.budget_total}")
    print(f"replay hash: {replay_hash(evidence)}")
    return replay_hash(evidence)


if __name__ == "__main__":
    main()
