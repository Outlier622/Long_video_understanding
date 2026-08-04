"""
Fast local tests for SQLiteJobStateStore.

No model or GPU is loaded.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from job_state_store import (
    FINAL_FAILED,
    RETRYABLE_FAILED,
    SUCCEEDED,
    SQLiteJobStateStore,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-path", default=None)
    args = parser.parse_args()

    if args.database_path:
        database_path = Path(args.database_path)
        database_path.unlink(missing_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="videothinker-state-test-"))
        database_path = temp_dir / "job_state.db"

    store = SQLiteJobStateStore(database_path)
    run_id = "state-test"
    store.initialize_clips(run_id, {1, 2})

    first = store.claim_clip(
        run_id=run_id,
        clip_id=1,
        worker_id="worker-a",
        lease_seconds=300,
        max_attempts=3,
    )
    assert first.claimed and first.attempt == 1

    blocked = store.claim_clip(
        run_id=run_id,
        clip_id=1,
        worker_id="worker-b",
        lease_seconds=300,
        max_attempts=3,
    )
    assert not blocked.claimed
    assert blocked.reason == "leased_by_another_worker"

    status = store.mark_failed(
        run_id=run_id,
        clip_id=1,
        worker_id="worker-a",
        error_type="TEST_ERROR",
        error_message="intentional retryable failure",
        retryable=True,
        max_attempts=3,
    )
    assert status == RETRYABLE_FAILED

    second = store.claim_clip(
        run_id=run_id,
        clip_id=1,
        worker_id="worker-b",
        lease_seconds=300,
        max_attempts=3,
    )
    assert second.claimed and second.attempt == 2

    store.mark_succeeded(
        run_id=run_id,
        clip_id=1,
        worker_id="worker-b",
        output_path="worker_001.jsonl",
    )
    assert store.get_succeeded_clip_ids(run_id) == {1}

    after_success = store.claim_clip(
        run_id=run_id,
        clip_id=1,
        worker_id="worker-a",
        lease_seconds=300,
        max_attempts=3,
    )
    assert not after_success.claimed
    assert after_success.status == SUCCEEDED

    for attempt_worker in ("worker-a", "worker-b"):
        claim = store.claim_clip(
            run_id=run_id,
            clip_id=2,
            worker_id=attempt_worker,
            lease_seconds=300,
            max_attempts=2,
        )
        assert claim.claimed
        final_status = store.mark_failed(
            run_id=run_id,
            clip_id=2,
            worker_id=attempt_worker,
            error_type="TEST_ERROR",
            error_message="intentional failure",
            retryable=True,
            max_attempts=2,
        )

    assert final_status == FINAL_FAILED
    counts = store.get_status_counts(run_id)
    assert counts == {FINAL_FAILED: 1, SUCCEEDED: 1}

    print("SQLite job-state tests passed.")
    print("Database:", database_path)
    print("Status counts:", counts)


if __name__ == "__main__":
    main()