import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.config import DB_PATH
from app.models import LogEntry


class Database:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    def _bootstrap(self) -> None:
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
        return datetime.now(timezone.utc).isoformat()

    # Identity methods

    def insert_identity(self, faiss_idx: int, name: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO identities (faiss_idx, name, registered_at) VALUES (?, ?, ?)",
                (faiss_idx, name, self._now()),
            )

    def get_identity(self, faiss_idx: int) -> Optional[str]:
        row = self._conn.execute(
            "SELECT name FROM identities WHERE faiss_idx = ?",
            (faiss_idx,),
        ).fetchone()
        return row["name"] if row else None

    def count_identities(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()
        return row["n"]

    def delete_identity(self, faiss_idx: int) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM identities WHERE faiss_idx = ?",
                (faiss_idx,),
            )
        return cur.rowcount > 0

    # Log methods

    def insert_log(self, operation: str, result: str, confidence: float) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO logs (operation, result, confidence, feedback, created_at) "
                "VALUES (?, ?, ?, NULL, ?)",
                (operation, result, confidence, self._now()),
            )
        return cur.lastrowid

    def update_feedback(self, log_id: int, feedback: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE logs SET feedback = ? WHERE id = ?",
                (feedback, log_id),
            )
        return cur.rowcount > 0

    def get_recent_logs(self, limit: int = 50) -> list[LogEntry]:
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
