"""
No-GPU live integration test for DynamoDBJobStateStore.

The test uses a unique run_id and validates:
- initialization
- exclusive claim
- retryable failure
- reclaim by another worker
- success terminal state
- max-attempt terminal failure

Example:
    python .\src\tests\test_dynamodb_job_state_store.py `
      --table-name videothinker-clip-jobs `
      --region us-east-2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.state.dynamodb_job_state_store import DynamoDBJobStateStore
from src.state.job_state_store import (
    FINAL_FAILED,
    RETRYABLE_FAILED,
    SUCCEEDED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table-name",
        default="videothinker-clip-jobs",
    )
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--keep-test-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = (
        "dynamodb-state-test-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )

    store = DynamoDBJobStateStore(
        table_name=args.table_name,
        region_name=args.region,
    )

    try:
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

        failed_status = store.mark_failed(
            run_id=run_id,
            clip_id=1,
            worker_id="worker-a",
            error_type="TEST_ERROR",
            error_message="intentional retryable failure",
            retryable=True,
            max_attempts=3,
        )
        assert failed_status == RETRYABLE_FAILED

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
            output_path="s3://example/test-output.jsonl",
        )
        assert store.get_succeeded_clip_ids(run_id) == {1}

        after_success = store.claim_clip(
            run_id=run_id,
            clip_id=1,
            worker_id="worker-c",
            lease_seconds=300,
            max_attempts=3,
        )
        assert not after_success.claimed
        assert after_success.status == SUCCEEDED

        for worker_id in ("worker-a", "worker-b"):
            claim = store.claim_clip(
                run_id=run_id,
                clip_id=2,
                worker_id=worker_id,
                lease_seconds=300,
                max_attempts=2,
            )
            assert claim.claimed
            final_status = store.mark_failed(
                run_id=run_id,
                clip_id=2,
                worker_id=worker_id,
                error_type="TEST_ERROR",
                error_message="intentional repeated failure",
                retryable=True,
                max_attempts=2,
            )

        assert final_status == FINAL_FAILED
        counts = store.get_status_counts(run_id)
        assert counts == {FINAL_FAILED: 1, SUCCEEDED: 1}

        print("DynamoDB job-state integration test passed.")
        print("Table:", args.table_name)
        print("Region:", args.region)
        print("Run ID:", run_id)
        print("Status counts:", counts)

    finally:
        if args.keep_test_data:
            print("Keeping test records for inspection:", run_id)
        else:
            deleted = store.delete_run(run_id)
            print("Deleted test records:", deleted)


if __name__ == "__main__":
    main()