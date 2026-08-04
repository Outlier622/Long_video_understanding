"""
Shared clip-job state interfaces for standalone VideoThinker workers.

The SQLite backend is a local concurrency-safe development implementation.
It models the same operations later required from DynamoDB:

- query successful clips
- atomically claim a pending/retryable/stale clip
- renew a running lease
- mark success
- mark retryable or final failure

SQLite is used only for local validation. The cloud implementation will use
DynamoDB conditional updates behind the same interface.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Set


PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
RETRYABLE_FAILED = "RETRYABLE_FAILED"
FINAL_FAILED = "FINAL_FAILED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class ClaimResult:
    claimed: bool
    reason: str
    attempt: int
    status: str


class JobStateStore(ABC):
    @abstractmethod
    def initialize_clips(self, run_id: str, clip_ids: Set[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_succeeded_clip_ids(self, run_id: str) -> Set[int]:
        raise NotImplementedError

    @abstractmethod
    def import_succeeded_clip_ids(
        self,
        *,
        run_id: str,
        clip_ids: Set[int],
        output_path: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def claim_clip(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> ClaimResult:
        raise NotImplementedError

    @abstractmethod
    def renew_lease(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def mark_succeeded(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        output_path: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_failed(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        error_type: str,
        error_message: str,
        retryable: bool,
        max_attempts: int,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_status_counts(self, run_id: str) -> Dict[str, int]:
        raise NotImplementedError


class SQLiteJobStateStore(JobStateStore):
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clip_jobs (
                    run_id TEXT NOT NULL,
                    clip_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    output_path TEXT,
                    PRIMARY KEY (run_id, clip_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_clip_jobs_run_status
                ON clip_jobs (run_id, status)
                """
            )

    def initialize_clips(self, run_id: str, clip_ids: Set[int]) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO clip_jobs (
                        run_id, clip_id, status, attempt, updated_at
                    ) VALUES (?, ?, ?, 0, ?)
                    """,
                    [
                        (run_id, int(clip_id), PENDING, now)
                        for clip_id in sorted(clip_ids)
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get_succeeded_clip_ids(self, run_id: str) -> Set[int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT clip_id
                FROM clip_jobs
                WHERE run_id = ? AND status = ?
                """,
                (run_id, SUCCEEDED),
            ).fetchall()
        return {int(row["clip_id"]) for row in rows}

    def import_succeeded_clip_ids(
        self,
        *,
        run_id: str,
        clip_ids: Set[int],
        output_path: str,
    ) -> None:
        if not clip_ids:
            return

        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for clip_id in sorted(clip_ids):
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO clip_jobs (
                            run_id, clip_id, status, attempt, updated_at
                        ) VALUES (?, ?, ?, 0, ?)
                        """,
                        (run_id, int(clip_id), PENDING, now),
                    )
                    connection.execute(
                        """
                        UPDATE clip_jobs
                        SET status = ?,
                            completed_at = COALESCE(completed_at, ?),
                            updated_at = ?,
                            lease_expires_at = NULL,
                            output_path = ?,
                            error_type = NULL,
                            error_message = NULL
                        WHERE run_id = ? AND clip_id = ?
                        """,
                        (
                            SUCCEEDED,
                            now,
                            now,
                            output_path,
                            run_id,
                            int(clip_id),
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def claim_clip(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> ClaimResult:
        now = utc_now()
        now_iso = now.isoformat()
        lease_expires = (now + timedelta(seconds=lease_seconds)).isoformat()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT status, worker_id, attempt, lease_expires_at
                    FROM clip_jobs
                    WHERE run_id = ? AND clip_id = ?
                    """,
                    (run_id, int(clip_id)),
                ).fetchone()

                if row is None:
                    connection.execute(
                        """
                        INSERT INTO clip_jobs (
                            run_id, clip_id, status, attempt, updated_at
                        ) VALUES (?, ?, ?, 0, ?)
                        """,
                        (run_id, int(clip_id), PENDING, now_iso),
                    )
                    status = PENDING
                    attempt = 0
                    current_worker = None
                    current_lease = None
                else:
                    status = str(row["status"])
                    attempt = int(row["attempt"])
                    current_worker = row["worker_id"]
                    current_lease = parse_iso(row["lease_expires_at"])

                if status == SUCCEEDED:
                    connection.execute("COMMIT")
                    return ClaimResult(False, "already_succeeded", attempt, status)

                if status == FINAL_FAILED:
                    connection.execute("COMMIT")
                    return ClaimResult(False, "final_failed", attempt, status)

                if attempt >= max_attempts and status != RUNNING:
                    connection.execute(
                        """
                        UPDATE clip_jobs
                        SET status = ?, updated_at = ?
                        WHERE run_id = ? AND clip_id = ?
                        """,
                        (FINAL_FAILED, now_iso, run_id, int(clip_id)),
                    )
                    connection.execute("COMMIT")
                    return ClaimResult(
                        False,
                        "max_attempts_reached",
                        attempt,
                        FINAL_FAILED,
                    )

                if (
                    status == RUNNING
                    and current_lease is not None
                    and current_lease > now
                    and current_worker != worker_id
                ):
                    connection.execute("COMMIT")
                    return ClaimResult(
                        False,
                        "leased_by_another_worker",
                        attempt,
                        status,
                    )

                next_attempt = attempt + 1
                connection.execute(
                    """
                    UPDATE clip_jobs
                    SET status = ?,
                        worker_id = ?,
                        attempt = ?,
                        lease_expires_at = ?,
                        started_at = COALESCE(started_at, ?),
                        completed_at = NULL,
                        error_type = NULL,
                        error_message = NULL,
                        updated_at = ?
                    WHERE run_id = ? AND clip_id = ?
                    """,
                    (
                        RUNNING,
                        worker_id,
                        next_attempt,
                        lease_expires,
                        now_iso,
                        now_iso,
                        run_id,
                        int(clip_id),
                    ),
                )
                connection.execute("COMMIT")
                return ClaimResult(True, "claimed", next_attempt, RUNNING)
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def renew_lease(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        now = utc_now()
        lease_expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE clip_jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE run_id = ?
                  AND clip_id = ?
                  AND status = ?
                  AND worker_id = ?
                """,
                (
                    lease_expires,
                    now.isoformat(),
                    run_id,
                    int(clip_id),
                    RUNNING,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def mark_succeeded(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        output_path: str,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE clip_jobs
                SET status = ?,
                    completed_at = ?,
                    updated_at = ?,
                    lease_expires_at = NULL,
                    output_path = ?,
                    error_type = NULL,
                    error_message = NULL
                WHERE run_id = ?
                  AND clip_id = ?
                  AND status = ?
                  AND worker_id = ?
                """,
                (
                    SUCCEEDED,
                    now,
                    now,
                    output_path,
                    run_id,
                    int(clip_id),
                    RUNNING,
                    worker_id,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Could not mark clip {clip_id} succeeded for worker {worker_id}."
            )

    def mark_failed(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        error_type: str,
        error_message: str,
        retryable: bool,
        max_attempts: int,
    ) -> str:
        now = utc_now_iso()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attempt
                FROM clip_jobs
                WHERE run_id = ? AND clip_id = ?
                """,
                (run_id, int(clip_id)),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"State record missing for run={run_id}, clip={clip_id}."
                )

            attempt = int(row["attempt"])
            next_status = (
                RETRYABLE_FAILED
                if retryable and attempt < max_attempts
                else FINAL_FAILED
            )

            cursor = connection.execute(
                """
                UPDATE clip_jobs
                SET status = ?,
                    completed_at = ?,
                    updated_at = ?,
                    lease_expires_at = NULL,
                    error_type = ?,
                    error_message = ?
                WHERE run_id = ?
                  AND clip_id = ?
                  AND status = ?
                  AND worker_id = ?
                """,
                (
                    next_status,
                    now,
                    now,
                    error_type,
                    error_message,
                    run_id,
                    int(clip_id),
                    RUNNING,
                    worker_id,
                ),
            )

        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Could not mark clip {clip_id} failed for worker {worker_id}."
            )
        return next_status

    def get_status_counts(self, run_id: str) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM clip_jobs
                WHERE run_id = ?
                GROUP BY status
                """,
                (run_id,),
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}