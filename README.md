# METIS

A crash-safe, reproducible experiment loop for agents that discover physical laws.

METIS runs an agent against [DiscoverPhysics](https://github.com/SampsonML/DiscoverPhysics)
([arXiv:2605.26087](https://arxiv.org/abs/2605.26087)), a simulator of worlds with hidden
governing laws. The agent designs experiments, sees only particle trajectories, and must
work out the physics. Every experiment costs budget, the law can change silently mid-run,
and the run has to survive a crash without repeating, losing or double-charging a query.

> **Status:** the experiment infrastructure is complete and tested. The only agent today is a
> random baseline. Learning agents are the next milestone (see [Roadmap](#roadmap)).

## What it guarantees

| Guarantee | How | Tests |
| --- | --- | --- |
| A crashed run resumes to the **same evidence** as an uninterrupted one | atomic recovery file (`fsync` + `os.replace`), environment snapshot/restore, RNG state persisted every step | `tests/integration/test_segmented_determinism.py`, `tests/adversarial/test_resume_determinism.py` |
| No experiment is run twice or charged twice | SQLite budget ledger reserves each attempt *before* dispatch; status `RESERVED → DISPATCHED → COMPLETED`; an unfinished attempt blocks resume until reconciled | `tests/integration/test_crash.py`, `tests/test_budget_ledger.py` |
| The agent never sees the hidden law | observations pass an allowlist of numeric fields only; any extra simulator field is rejected | `tests/integration/test_leakage.py` |
| A hidden regime change stays at the same query across restarts | snapshots carry a fingerprint of the world configuration; a mismatched snapshot is refused | `tests/integration/test_hidden_regime.py`, `tests/integration/test_resume.py` |
| The core never depends on a specific simulator | an AST test fails if anything outside `metis/worlds/` imports a world | `tests/test_architecture_boundaries.py` |

Crashes are tested at every point inside a step (before the agent acts, during the physics
call, before and after writing evidence, before the atomic rename), including real process
death with `os._exit` in a child process.

## Architecture

```
            ┌──────────────────────── run_loop ────────────────────────┐
            │                                                          │
 Agent ──action──▶ EnvironmentAdapter ──▶ World (DiscoverPhysics) ──▶ allowlisted payload
   ▲                                                                     │
   └───────────────────────────── observation ◀──────────────────────────┤
                                                                         ▼
                                                              Evaluator (score never
                                                              returned to the agent)
            │
            ├── BudgetLedger   (SQLite, one row per attempt)
            ├── evidence.jsonl (append-only, fsync per record)
            └── recovery.json  (atomic: step, budget, RNG state, env snapshot)
```

- `src/metis/` is the simulator-agnostic core: typed `Protocol` interfaces for agent,
  environment and evaluator, the run loop, recovery, evidence and the budget ledger.
- `src/metis/worlds/discoverphysics/` adapts DiscoverPhysics to those interfaces. Each step
  builds a fresh executor (that is how a world is reset) and runs one experiment.

## Known limitations

Open defects are kept as strict `xfail` tests, so fixing one makes the suite fail until the
marker is removed:

- **D1:** agents draw from Python's global RNG, so two runs in one process can interfere.
- **D4b:** internal agent state is not persisted; a stateful agent restarts its own state on resume.
- **D5/D6:** without a ledger, a crash between writing evidence and saving recovery can
  repeat a step. With the ledger enabled the resume is blocked instead of repeating it.

## Quickstart

Requires Python 3.12+. CI runs the suite on Linux, Windows and macOS.

```bash
git clone https://github.com/richardalmeida1094-cyber/metis.git
cd metis
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The test suite needs no simulator: it uses a deterministic fake executor.

### Running with real physics

Install DiscoverPhysics next to the repo (tested at commit `33b7fa9`):

```bash
git clone https://github.com/SampsonML/DiscoverPhysics.git
pip install -e DiscoverPhysics/PhysicsSchool -e DiscoverPhysics/ScienceAgent
```

Run 6 experiments where the law switches from gravity to a Yukawa potential at query 3:

```bash
python scripts/run_discoverphysics.py --out runs/reference --regimes 0:gravity,3:yukawa
```

Now kill a run halfway and resume it:

```bash
python scripts/run_discoverphysics.py --out runs/crashed --regimes 0:gravity,3:yukawa --stop-after 2
python scripts/run_discoverphysics.py --out runs/crashed --regimes 0:gravity,3:yukawa --resume
```

Both runs print the same replay hash.

## Roadmap

1. Learning agent that chooses experiments to reduce uncertainty about the hidden law.
2. Detection of and adaptation to hidden regime changes.
3. Bounded self-improvement: the agent tunes its own experiment-selection strategy, judged by an evaluator it cannot modify.

## Project layout

```
src/metis/
  run_loop.py                   orchestration: budget, dispatch, evidence, recovery
  recovery.py                   atomic save/load of run state
  budget_ledger.py              SQLite attempt ledger
  evidence.py                   append-only evidence log
  interfaces.py                 Agent / Environment / Evaluator protocols
  basic_environment_adapter.py  adapter with snapshot/restore and experiment repetitions
  agents/random_agent.py        random baseline
  worlds/discoverphysics/       DiscoverPhysics world, executor factory, payload allowlist
scripts/run_discoverphysics.py  end-to-end run with real physics
tests/                          unit, adversarial and integration tests
```
