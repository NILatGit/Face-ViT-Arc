"""
This module owns the entire database. No SQL appears anywhere else in the
codebase. All queries are parameterised (using ? placeholders), preventing
SQL injection regardless of input values.

Two tables:

    identities
        faiss_idx     INTEGER PRIMARY KEY  - must match the ID used in the
                                             FAISS index (set during register)
        name          TEXT NOT NULL        - human-readable label
        registered_at TEXT NOT NULL        - ISO 8601 UTC timestamp

    logs
        id            INTEGER PRIMARY KEY AUTOINCREMENT
        operation     TEXT NOT NULL        - "verify", "identify", "register"
        result        TEXT NOT NULL        - outcome string (name, match, etc.)
        confidence    REAL NOT NULL        - similarity score from FAISS/dot product
        feedback      TEXT                 - NULL until user submits via /api/feedback
        created_at    TEXT NOT NULL        - ISO 8601 UTC timestamp
"""

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.config import DB_PATH
from app.models import LogEntry


class Database:
    def __init__(self) -> None:
        """
        Open the SQLite connection and initialise the schema.

        connect() is called with check_same_thread=False because the
        connection is shared across multiple request handler coroutines
        within a single container. SQLite's write serialisation (via the
        GIL and WAL mode) makes this safe for the read-heavy, low-write
        workload of this application.

        row_factory = sqlite3.Row makes fetchone()/fetchall() return
        dict-like Row objects, allowing column access by name (r["id"])
        instead of by position (r[0]), which is more robust to schema changes.
        """
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    def _bootstrap(self) -> None:
        """
        Create tables and indices if they do not already exist.

        executescript runs all statements as a single transaction. The
        IF NOT EXISTS guards make this idempotent - safe to call on every
        container startup without risk of losing data.

        Indices on logs(operation) and logs(created_at) speed up filtered
        history queries. If you add a WHERE clause on a new column to
        get_recent_logs(), add a corresponding index here.
        """
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS identities (
                    faiss_idx    INTEGER PRIMARY KEY,
                    name         TEXT    NOT NULL,
                    registered_at TEXT   NOT NULL
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation   TEXT    NOT NULL,
                    result      TEXT    NOT NULL,
                    confidence  REAL    NOT NULL,
                    feedback    TEXT,
                    created_at  TEXT    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_logs_operation  ON logs(operation);
                CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at);
            """)

    def _now(self) -> str:
        """
        Return the current UTC time as an ISO 8601 string.

        All timestamps in the database use this format:
            2026-02-22T10:45:00.123456+00:00

        Using UTC consistently avoids timezone confusion when the Modal
        container, the client, and the developer are in different timezones.
        """
        return datetime.now(timezone.utc).isoformat()

    def insert_identity(self, faiss_idx: int, name: str) -> None:
        """
        Insert a new identity record after a successful face registration.

        Called by server.py immediately after engine.register() returns True.
        The faiss_idx must be the same value passed to engine.register().
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO identities (faiss_idx, name, registered_at) VALUES (?, ?, ?)",
                (faiss_idx, name, self._now()),
            )

    def get_identity(self, faiss_idx: int) -> Optional[str]:
        """
        Look up the name for a given FAISS index ID.

        Called by server.py after engine.identify() returns a faiss_idx,
        to resolve the integer ID to a display name..
        """
        row = self._conn.execute(
            "SELECT name FROM identities WHERE faiss_idx = ?",
            (faiss_idx,),
        ).fetchone()
        return row["name"] if row else None

    def count_identities(self) -> int:
        """
        Return the total number of registered identities.

        Used by server.py to assign the next faiss_idx on registration
        (next_idx = count_identities()). This works as a monotonically
        increasing ID generator as long as identities are never deleted.
        If you add deletion support, switch to MAX(faiss_idx) + 1 instead
        to avoid reusing IDs of deleted identities.
        """
        row = self._conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()
        return row["n"]

    def delete_identity(self, faiss_idx: int) -> bool:
        """
        Delete an identity record by its FAISS index ID.

        This method only removes the database record. The caller must also
        call engine.remove(faiss_idx) to remove the corresponding embedding
        from the FAISS index. If only one of the two is removed, the index
        and database will be out of sync.
        """
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM identities WHERE faiss_idx = ?",
                (faiss_idx,),
            )
        return cur.rowcount > 0

    def insert_log(self, operation: str, result: str, confidence: float) -> int:
        """
        Append an audit log entry for a completed operation.

        Called by server.py at the end of every successful verify, identify,
        and register request. feedback is always NULL on insertion and can
        be updated later via update_feedback().
        """
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO logs (operation, result, confidence, feedback, created_at) "
                "VALUES (?, ?, ?, NULL, ?)",
                (operation, result, confidence, self._now()),
            )
        return cur.lastrowid

    def update_feedback(self, log_id: int, feedback: str) -> bool:
        """
        Attach a feedback label to an existing log entry.

        Called by server.py when a client POSTs to /api/feedback. The
        feedback value is a free-form string; the API does not enforce
        a fixed vocabulary. Typical values: "correct", "incorrect".
        """
        with self._conn:
            cur = self._conn.execute(
                "UPDATE logs SET feedback = ? WHERE id = ?",
                (feedback, log_id),
            )
        return cur.rowcount > 0

    def get_recent_logs(self, limit: int = 50) -> list[LogEntry]:
        """
        Return the most recent log entries in descending order.

        Called by server.py for the GET /api/history endpoint. Returns
        LogEntry Pydantic objects which FastAPI serialises to JSON.

        To filter by operation type, add a WHERE clause:
            WHERE operation = 'verify'
        To filter by date range, add:
            WHERE created_at >= '2026-01-01T00:00:00+00:00'
        """
        rows = self._conn.execute(
            "SELECT id, operation, result, confidence, feedback, created_at "
            "FROM logs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            LogEntry(
                id=r["id"],
                operation=r["operation"],
                result=r["result"],
                confidence=r["confidence"],
                feedback=r["feedback"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_log_by_id(self, log_id: int) -> Optional[LogEntry]:
        """
        Fetch a single log entry by its primary key.

        Not currently used by any endpoint but available for use cases such
        as confirming a log was written after an operation, or building a
        GET /api/history/<id> detail endpoint.
        """
        row = self._conn.execute(
            "SELECT id, operation, result, confidence, feedback, created_at "
            "FROM logs WHERE id = ?",
            (log_id,),
        ).fetchone()
        if not row:
            return None
        return LogEntry(
            id=row["id"],
            operation=row["operation"],
            result=row["result"],
            confidence=row["confidence"],
            feedback=row["feedback"],
            created_at=row["created_at"],
        )
