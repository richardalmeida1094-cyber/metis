import sqlite3
from pathlib import Path


class BudgetLedger:
    def __init__(
        self,
        db_path: Path,
        budget_total: int,
        run_id: str,
    ):
        self.db_path = db_path
        self.budget_total = budget_total
        self.run_id = run_id

        self.connection = sqlite3.connect(self.db_path)

        # Create the runs table if it does not exist.
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                budget_total INTEGER NOT NULL CHECK (budget_total > 0)
            );
            """
        )

        # Register the run without replacing an existing record.
        self.connection.execute(
            """
            INSERT OR IGNORE INTO runs (run_id, budget_total)
            VALUES (?, ?);
            """,
            (self.run_id, self.budget_total),
        )

        # Retrieve the stored budget for this run.
        cursor = self.connection.execute(
            """
            SELECT budget_total
            FROM runs
            WHERE run_id = ?;
            """,
            (self.run_id,),
        )

        row = cursor.fetchone()

        if row is None:
            raise ValueError("Run record was not found.")

        if row[0] != self.budget_total:
            raise ValueError("Run budget does not match the stored budget.")

        # Create the attempts table with status.
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                cost INTEGER NOT NULL CHECK (cost > 0),
                status TEXT NOT NULL DEFAULT 'RESERVED'
                    CHECK (
                        status IN (
                            'RESERVED',
                            'DISPATCHED',
                            'COMPLETED',
                            'UNKNOWN'
                        )
                    ),
                PRIMARY KEY (run_id, attempt_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            """
        )

        # Upgrade databases created before status existed.
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(attempts)")}

        if "status" not in columns:
            self.connection.execute(
                """
                ALTER TABLE attempts
                ADD COLUMN status TEXT NOT NULL DEFAULT 'UNKNOWN'
                    CHECK (
                        status IN (
                            'RESERVED',
                            'DISPATCHED',
                            'COMPLETED',
                            'UNKNOWN'
                        )
                    );
                """
            )

        # Persist the registration and schema.
        self.connection.commit()

    def reserve(self, attempt_id: str, cost: int) -> bool:
        if not attempt_id:
            raise ValueError("attempt_id cannot be empty")

        if type(cost) is not int or cost <= 0:
            raise ValueError("cost must be a positive integer")

        self.connection.execute("BEGIN IMMEDIATE")

        try:
            existing = self.connection.execute(
                """
                SELECT cost
                FROM attempts
                WHERE run_id = ? AND attempt_id = ?;
                """,
                (self.run_id, attempt_id),
            ).fetchone()

            if existing is not None:
                if existing[0] != cost:
                    raise ValueError("Attempt already exists with a different cost")

                self.connection.commit()
                return False

            used = self.connection.execute(
                """
                SELECT COALESCE(SUM(cost), 0)
                FROM attempts
                WHERE run_id = ?;
                """,
                (self.run_id,),
            ).fetchone()[0]

            if used + cost > self.budget_total:
                raise ValueError("Insufficient budget")

            self.connection.execute(
                """
                INSERT INTO attempts (run_id, attempt_id, cost)
                VALUES (?, ?, ?);
                """,
                (self.run_id, attempt_id, cost),
            )

            self.connection.commit()
            return True

        except Exception:
            self.connection.rollback()
            raise

    def get_used(self) -> int:
        result = self.connection.execute(
            """
            SELECT COALESCE(SUM(cost), 0)
            FROM attempts
            WHERE run_id = ?;
            """,
            (self.run_id,),
        ).fetchone()

        return result[0]

    def get_remaining(self) -> int:
        return self.budget_total - self.get_used()

    def get_attempt_status(self, attempt_id: str) -> str | None:
        result = self.connection.execute(
            """
            SELECT status
            FROM attempts
            WHERE run_id = ? AND attempt_id = ?;
            """,
            (self.run_id, attempt_id),
        ).fetchone()

        if result is None:
            return None

        return result[0]

    def set_attempt_status(
        self,
        attempt_id: str,
        new_status: str,
    ) -> None:
        allowed_transitions = {
            "RESERVED": {"DISPATCHED", "UNKNOWN"},
            "DISPATCHED": {"COMPLETED", "UNKNOWN"},
            "COMPLETED": set(),
            "UNKNOWN": set(),
        }

        if new_status not in allowed_transitions:
            raise ValueError("Invalid attempt status")

        self.connection.execute("BEGIN IMMEDIATE")

        try:
            result = self.connection.execute(
                """
                SELECT status
                FROM attempts
                WHERE run_id = ? AND attempt_id = ?;
                """,
                (self.run_id, attempt_id),
            ).fetchone()

            if result is None:
                raise ValueError("Attempt does not exist")

            current_status = result[0]

            if new_status not in allowed_transitions[current_status]:
                raise ValueError(f"Invalid attempt transition: {current_status} -> {new_status}")

            self.connection.execute(
                """
                UPDATE attempts
                SET status = ?
                WHERE run_id = ? AND attempt_id = ?;
                """,
                (new_status, self.run_id, attempt_id),
            )

            self.connection.commit()

        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        self.connection.close()
