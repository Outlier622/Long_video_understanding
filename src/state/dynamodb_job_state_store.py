"""
DynamoDB implementation of the shared VideoThinker clip-job state interface.

Table schema:
    Partition key: run_id (String)
    Sort key:      clip_id (Number)

Each clip is one independently claimable job. Conditional UpdateItem calls
ensure that only one worker can hold a valid lease for a clip at a time.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Dict, Iterable, Optional, Set

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from .job_state_store import (
    FINAL_FAILED,
    PENDING,
    RETRYABLE_FAILED,
    RUNNING,
    SUCCEEDED,
    ClaimResult,
    JobStateStore,
    utc_now,
    utc_now_iso,
)


class DynamoDBJobStateStore(JobStateStore):
    def __init__(
        self,
        *,
        table_name: str,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
        consistent_reads: bool = True,
    ) -> None:
        if not table_name.strip():
            raise ValueError("table_name must not be empty.")

        self.table_name = table_name
        self.consistent_reads = consistent_reads

        # Local development can use an AWS CLI profile. In AWS Batch,
        # profile_name should be omitted so boto3 uses the task/job role.
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        self.resource = session.resource(
            "dynamodb",
            endpoint_url=endpoint_url,
        )
        self.table = self.resource.Table(table_name)

    @staticmethod
    def _is_conditional_failure(error: ClientError) -> bool:
        return (
            error.response.get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        )

    def initialize_clips(self, run_id: str, clip_ids: Set[int]) -> None:
        now_iso = utc_now_iso()

        for clip_id in sorted(clip_ids):
            try:
                self.table.put_item(
                    Item={
                        "run_id": run_id,
                        "clip_id": int(clip_id),
                        "status": PENDING,
                        "attempt": 0,
                        "updated_at": now_iso,
                        "lease_expires_at_epoch": 0,
                    },
                    ConditionExpression=(
                        "attribute_not_exists(#run_id) "
                        "AND attribute_not_exists(#clip_id)"
                    ),
                    ExpressionAttributeNames={
                        "#run_id": "run_id",
                        "#clip_id": "clip_id",
                    },
                )
            except ClientError as error:
                if not self._is_conditional_failure(error):
                    raise

    def _query_items(
        self,
        *,
        run_id: str,
        projection_expression: Optional[str] = None,
        expression_attribute_names: Optional[Dict[str, str]] = None,
        filter_expression=None,
    ) -> Iterable[dict]:
        kwargs = {
            "KeyConditionExpression": Key("run_id").eq(run_id),
            "ConsistentRead": self.consistent_reads,
        }

        if projection_expression:
            kwargs["ProjectionExpression"] = projection_expression
        if expression_attribute_names:
            kwargs["ExpressionAttributeNames"] = expression_attribute_names
        if filter_expression is not None:
            kwargs["FilterExpression"] = filter_expression

        while True:
            response = self.table.query(**kwargs)
            yield from response.get("Items", [])

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key

    def _get_item(self, run_id: str, clip_id: int) -> Optional[dict]:
        response = self.table.get_item(
            Key={"run_id": run_id, "clip_id": int(clip_id)},
            ConsistentRead=self.consistent_reads,
        )
        return response.get("Item")

    def get_succeeded_clip_ids(self, run_id: str) -> Set[int]:
        items = self._query_items(
            run_id=run_id,
            projection_expression="clip_id, #status",
            expression_attribute_names={"#status": "status"},
            filter_expression=Attr("status").eq(SUCCEEDED),
        )
        return {int(item["clip_id"]) for item in items}

    def import_succeeded_clip_ids(
        self,
        *,
        run_id: str,
        clip_ids: Set[int],
        output_path: str,
    ) -> None:
        if not clip_ids:
            return

        self.initialize_clips(run_id, clip_ids)
        now_iso = utc_now_iso()

        for clip_id in sorted(clip_ids):
            self.table.update_item(
                Key={"run_id": run_id, "clip_id": int(clip_id)},
                UpdateExpression=(
                    "SET #status = :succeeded, "
                    "#completed_at = if_not_exists(#completed_at, :now_iso), "
                    "#updated_at = :now_iso, "
                    "#output_path = :output_path "
                    "REMOVE #worker_id, #lease_epoch, #lease_iso, "
                    "#error_type, #error_message"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                    "#completed_at": "completed_at",
                    "#updated_at": "updated_at",
                    "#output_path": "output_path",
                    "#worker_id": "worker_id",
                    "#lease_epoch": "lease_expires_at_epoch",
                    "#lease_iso": "lease_expires_at",
                    "#error_type": "error_type",
                    "#error_message": "error_message",
                },
                ExpressionAttributeValues={
                    ":succeeded": SUCCEEDED,
                    ":now_iso": now_iso,
                    ":output_path": output_path,
                },
            )

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
        now_epoch = int(now.timestamp())
        now_iso = now.isoformat()
        lease_time = now + timedelta(seconds=lease_seconds)
        lease_epoch = int(lease_time.timestamp())
        lease_iso = lease_time.isoformat()

        try:
            response = self.table.update_item(
                Key={"run_id": run_id, "clip_id": int(clip_id)},
                UpdateExpression=(
                    "SET #status = :running, "
                    "#worker_id = :worker_id, "
                    "#attempt = if_not_exists(#attempt, :zero) + :one, "
                    "#lease_epoch = :lease_epoch, "
                    "#lease_iso = :lease_iso, "
                    "#started_at = if_not_exists(#started_at, :now_iso), "
                    "#updated_at = :now_iso "
                    "REMOVE #completed_at, #error_type, #error_message"
                ),
                ConditionExpression=(
                    "(#status IN (:pending, :retryable_failed) "
                    "OR (#status = :running AND #lease_epoch < :now_epoch)) "
                    "AND (attribute_not_exists(#attempt) "
                    "OR #attempt < :max_attempts)"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                    "#worker_id": "worker_id",
                    "#attempt": "attempt",
                    "#lease_epoch": "lease_expires_at_epoch",
                    "#lease_iso": "lease_expires_at",
                    "#started_at": "started_at",
                    "#updated_at": "updated_at",
                    "#completed_at": "completed_at",
                    "#error_type": "error_type",
                    "#error_message": "error_message",
                },
                ExpressionAttributeValues={
                    ":pending": PENDING,
                    ":retryable_failed": RETRYABLE_FAILED,
                    ":running": RUNNING,
                    ":worker_id": worker_id,
                    ":zero": 0,
                    ":one": 1,
                    ":lease_epoch": lease_epoch,
                    ":lease_iso": lease_iso,
                    ":now_epoch": now_epoch,
                    ":now_iso": now_iso,
                    ":max_attempts": int(max_attempts),
                },
                ReturnValues="ALL_NEW",
            )
            attributes = response["Attributes"]
            return ClaimResult(
                claimed=True,
                reason="claimed",
                attempt=int(attributes["attempt"]),
                status=str(attributes["status"]),
            )

        except ClientError as error:
            if not self._is_conditional_failure(error):
                raise

        item = self._get_item(run_id, clip_id)
        if item is None:
            return ClaimResult(False, "missing_state", 0, "MISSING")

        status = str(item.get("status", PENDING))
        attempt = int(item.get("attempt", 0))
        current_lease_epoch = int(item.get("lease_expires_at_epoch", 0) or 0)

        if status == SUCCEEDED:
            return ClaimResult(False, "already_succeeded", attempt, status)

        if status == FINAL_FAILED:
            return ClaimResult(False, "final_failed", attempt, status)

        if status == RUNNING and current_lease_epoch >= now_epoch:
            return ClaimResult(
                False,
                "leased_by_another_worker",
                attempt,
                status,
            )

        if attempt >= max_attempts:
            try:
                self.table.update_item(
                    Key={"run_id": run_id, "clip_id": int(clip_id)},
                    UpdateExpression=(
                        "SET #status = :final_failed, "
                        "#updated_at = :now_iso, "
                        "#completed_at = :now_iso, "
                        "#error_type = :error_type, "
                        "#error_message = :error_message "
                        "REMOVE #worker_id, #lease_epoch, #lease_iso"
                    ),
                    ConditionExpression=(
                        "#attempt >= :max_attempts "
                        "AND #status <> :succeeded "
                        "AND #status <> :final_failed "
                        "AND (#status <> :running "
                        "OR #lease_epoch < :now_epoch)"
                    ),
                    ExpressionAttributeNames={
                        "#status": "status",
                        "#attempt": "attempt",
                        "#updated_at": "updated_at",
                        "#completed_at": "completed_at",
                        "#error_type": "error_type",
                        "#error_message": "error_message",
                        "#worker_id": "worker_id",
                        "#lease_epoch": "lease_expires_at_epoch",
                        "#lease_iso": "lease_expires_at",
                    },
                    ExpressionAttributeValues={
                        ":final_failed": FINAL_FAILED,
                        ":succeeded": SUCCEEDED,
                        ":running": RUNNING,
                        ":max_attempts": int(max_attempts),
                        ":now_epoch": now_epoch,
                        ":now_iso": now_iso,
                        ":error_type": "MAX_ATTEMPTS_REACHED",
                        ":error_message": (
                            "The clip reached the maximum number of attempts."
                        ),
                    },
                )
                status = FINAL_FAILED
            except ClientError as error:
                if not self._is_conditional_failure(error):
                    raise

            return ClaimResult(
                False,
                "max_attempts_reached",
                attempt,
                status,
            )

        return ClaimResult(False, "claim_condition_failed", attempt, status)

    def renew_lease(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        now = utc_now()
        lease_time = now + timedelta(seconds=lease_seconds)

        try:
            self.table.update_item(
                Key={"run_id": run_id, "clip_id": int(clip_id)},
                UpdateExpression=(
                    "SET #lease_epoch = :lease_epoch, "
                    "#lease_iso = :lease_iso, "
                    "#updated_at = :now_iso"
                ),
                ConditionExpression=(
                    "#status = :running AND #worker_id = :worker_id"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                    "#worker_id": "worker_id",
                    "#lease_epoch": "lease_expires_at_epoch",
                    "#lease_iso": "lease_expires_at",
                    "#updated_at": "updated_at",
                },
                ExpressionAttributeValues={
                    ":running": RUNNING,
                    ":worker_id": worker_id,
                    ":lease_epoch": int(lease_time.timestamp()),
                    ":lease_iso": lease_time.isoformat(),
                    ":now_iso": now.isoformat(),
                },
            )
            return True
        except ClientError as error:
            if self._is_conditional_failure(error):
                return False
            raise

    def mark_succeeded(
        self,
        *,
        run_id: str,
        clip_id: int,
        worker_id: str,
        output_path: str,
    ) -> None:
        now_iso = utc_now_iso()

        try:
            self.table.update_item(
                Key={"run_id": run_id, "clip_id": int(clip_id)},
                UpdateExpression=(
                    "SET #status = :succeeded, "
                    "#completed_at = :now_iso, "
                    "#updated_at = :now_iso, "
                    "#output_path = :output_path "
                    "REMOVE #lease_epoch, #lease_iso, "
                    "#error_type, #error_message"
                ),
                ConditionExpression=(
                    "#status = :running AND #worker_id = :worker_id"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                    "#worker_id": "worker_id",
                    "#completed_at": "completed_at",
                    "#updated_at": "updated_at",
                    "#output_path": "output_path",
                    "#lease_epoch": "lease_expires_at_epoch",
                    "#lease_iso": "lease_expires_at",
                    "#error_type": "error_type",
                    "#error_message": "error_message",
                },
                ExpressionAttributeValues={
                    ":running": RUNNING,
                    ":succeeded": SUCCEEDED,
                    ":worker_id": worker_id,
                    ":now_iso": now_iso,
                    ":output_path": output_path,
                },
            )
        except ClientError as error:
            if self._is_conditional_failure(error):
                raise RuntimeError(
                    f"Could not mark clip {clip_id} succeeded for "
                    f"worker {worker_id}; the worker no longer owns the claim."
                ) from error
            raise

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
        item = self._get_item(run_id, clip_id)
        if item is None:
            raise RuntimeError(
                f"State record missing for run={run_id}, clip={clip_id}."
            )

        attempt = int(item.get("attempt", 0))
        next_status = (
            RETRYABLE_FAILED
            if retryable and attempt < max_attempts
            else FINAL_FAILED
        )
        now_iso = utc_now_iso()

        try:
            self.table.update_item(
                Key={"run_id": run_id, "clip_id": int(clip_id)},
                UpdateExpression=(
                    "SET #status = :next_status, "
                    "#completed_at = :now_iso, "
                    "#updated_at = :now_iso, "
                    "#error_type = :error_type, "
                    "#error_message = :error_message "
                    "REMOVE #lease_epoch, #lease_iso"
                ),
                ConditionExpression=(
                    "#status = :running AND #worker_id = :worker_id"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                    "#worker_id": "worker_id",
                    "#completed_at": "completed_at",
                    "#updated_at": "updated_at",
                    "#error_type": "error_type",
                    "#error_message": "error_message",
                    "#lease_epoch": "lease_expires_at_epoch",
                    "#lease_iso": "lease_expires_at",
                },
                ExpressionAttributeValues={
                    ":running": RUNNING,
                    ":worker_id": worker_id,
                    ":next_status": next_status,
                    ":now_iso": now_iso,
                    ":error_type": error_type,
                    ":error_message": error_message[:4000],
                },
            )
        except ClientError as error:
            if self._is_conditional_failure(error):
                raise RuntimeError(
                    f"Could not mark clip {clip_id} failed for "
                    f"worker {worker_id}; the worker no longer owns the claim."
                ) from error
            raise

        return next_status

    def get_status_counts(self, run_id: str) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        items = self._query_items(
            run_id=run_id,
            projection_expression="#status",
            expression_attribute_names={"#status": "status"},
        )
        for item in items:
            counter[str(item["status"])] += 1
        return dict(sorted(counter.items()))

    def delete_run(self, run_id: str) -> int:
        keys = [
            {
                "run_id": item["run_id"],
                "clip_id": int(item["clip_id"]),
            }
            for item in self._query_items(
                run_id=run_id,
                projection_expression="run_id, clip_id",
            )
        ]

        if not keys:
            return 0

        with self.table.batch_writer() as batch:
            for key in keys:
                batch.delete_item(Key=key)

        return len(keys)