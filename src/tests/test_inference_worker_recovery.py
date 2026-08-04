"""
Integration test for inference_worker_v2 retry and recovery behavior.

This test does not load the VideoThinker model and does not run GPU inference.
It monkeypatches the model loader and inference function, then executes the
real worker orchestration twice:

1. worker-a claims one clip and receives an injected RuntimeError.
   Expected state: RETRYABLE_FAILED, attempt 1.
2. worker-b claims the same clip and returns valid synthetic JSON.
   Expected state: SUCCEEDED, attempt 2.

The worker output should contain two records for the clip: one failed attempt
and one successful retry. The normal aggregator will later select the success.

Example:
    python .\src\tests\test_inference_worker_recovery.py `
      --manifest-path "D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_flow.jsonl" `
      --clip-id 1 `
      --test-dir "D:\projects\longvideo\episodes\ep02\outputs\worker_runs\state_recovery_test"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.state.job_state_store import (  # noqa: E402
    RETRYABLE_FAILED,
    SUCCEEDED,
    SQLiteJobStateStore,
)
from src.workers import inference_worker_v2 as worker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a no-GPU retry/recovery integration test."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--clip-id", type=int, default=1)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument(
        "--run-id",
        default="ep02-state-recovery-integration-test",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
    return records


def load_target_record(
    manifest_path: Path,
    clip_id: int,
) -> Dict[str, Any]:
    for record in worker.load_manifest(manifest_path):
        if int(record["clip_id"]) == clip_id:
            return record
    raise ValueError(
        f"Clip {clip_id} was not found in manifest: {manifest_path}"
    )


def build_config(
    *,
    manifest_path: Path,
    output_dir: Path,
    state_db_path: Path,
    run_id: str,
    clip_id: int,
    worker_id: str,
) -> worker.WorkerConfig:
    return worker.WorkerConfig(
        model_path=PROJECT_ROOT,
        manifest_path=manifest_path,
        output_dir=output_dir,
        shard_index=0,
        num_shards=1,
        clip_ids={clip_id},
        version="worker_state_recovery_integration_test",
        dtype="bf16",
        max_new_tokens=32,
        min_pixels=4 * 28 * 28,
        max_pixels=128 * 28 * 28,
        max_keyframes_per_clip=12,
        use_top_score_when_too_many_frames=True,
        force=False,
        dry_run=False,
        worker_id=worker_id,
        state_backend="sqlite",
        state_db_path=state_db_path,
        run_id=run_id,
        lease_seconds=300,
        max_attempts=3,
    )


def fake_model_loader(
    config: worker.WorkerConfig,
) -> Tuple[object, object, str]:
    print("Injected model loader: model loading skipped.")
    return object(), object(), "integration-test"


def injected_failure(
    processor: object,
    model: object,
    record: Dict[str, Any],
    config: worker.WorkerConfig,
):
    raise RuntimeError("Injected retryable failure for integration test")


def build_success_json(record: Dict[str, Any]) -> str:
    payload = {
        "clip_id": int(record["clip_id"]),
        "clip_file": record.get("clip_file"),
        "actual_start_time": record.get("actual_start_time"),
        "actual_end_time": record.get("actual_end_time"),
        "summary": "Synthetic successful retry used for state recovery validation.",
        "visual_information_level": "medium",
        "setting": "integration test",
        "main_subjects": ["synthetic test subject"],
        "events": [
            {
                "action": "successful retry completed",
                "objects": ["worker state"],
                "scene": "integration test",
                "visual_evidence": "Synthetic output generated without model inference.",
                "frame_references": ["Frame 00"],
                "confidence": "high",
            }
        ],
        "uncertain_parts": [],
        "possible_hallucination_risks": [],
    }
    return json.dumps(payload, ensure_ascii=False)


def injected_success(
    processor: object,
    model: object,
    record: Dict[str, Any],
    config: worker.WorkerConfig,
):
    frames, missing_paths = worker.choose_keyframes(record, config)
    if not frames:
        raise AssertionError("Target manifest record contains no usable frames.")
    return build_success_json(record), frames, missing_paths


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    test_dir = Path(args.test_dir)
    output_dir = test_dir / "outputs"
    state_db_path = test_dir / "job_state.db"
    output_path = output_dir / "worker_000_of_001.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Confirm the target is valid before deleting the prior test directory.
    target_record = load_target_record(manifest_path, args.clip_id)
    selected_frames = target_record.get("selected_frames") or []
    if not selected_frames:
        raise ValueError(
            f"Clip {args.clip_id} has no selected frames in the manifest."
        )

    if test_dir.exists():
        shutil.rmtree(test_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_loader = worker.load_model_bundle
    original_infer = worker.infer_record

    try:
        worker.load_model_bundle = fake_model_loader

        print("\n=== Attempt 1: injected retryable failure ===")
        worker.infer_record = injected_failure
        worker.run_worker(
            build_config(
                manifest_path=manifest_path,
                output_dir=output_dir,
                state_db_path=state_db_path,
                run_id=args.run_id,
                clip_id=args.clip_id,
                worker_id="worker-a",
            )
        )

        store = SQLiteJobStateStore(state_db_path)
        counts_after_failure = store.get_status_counts(args.run_id)
        assert counts_after_failure == {RETRYABLE_FAILED: 1}, (
            "Expected one RETRYABLE_FAILED clip after attempt 1, got "
            f"{counts_after_failure}"
        )

        print("\n=== Attempt 2: injected successful retry ===")
        worker.infer_record = injected_success
        worker.run_worker(
            build_config(
                manifest_path=manifest_path,
                output_dir=output_dir,
                state_db_path=state_db_path,
                run_id=args.run_id,
                clip_id=args.clip_id,
                worker_id="worker-b",
            )
        )

    finally:
        worker.load_model_bundle = original_loader
        worker.infer_record = original_infer

    store = SQLiteJobStateStore(state_db_path)
    final_counts = store.get_status_counts(args.run_id)
    assert final_counts == {SUCCEEDED: 1}, (
        f"Expected one SUCCEEDED clip, got {final_counts}"
    )

    records = read_jsonl(output_path)
    assert len(records) == 2, (
        f"Expected two attempt records, found {len(records)}"
    )

    failed_record, successful_record = records
    assert int(failed_record["clip_id"]) == args.clip_id
    assert failed_record.get("error_type") == "RuntimeError"
    assert failed_record.get("json_parse_ok") is False

    assert int(successful_record["clip_id"]) == args.clip_id
    assert successful_record.get("error") is None
    assert successful_record.get("json_parse_ok") is True
    assert successful_record.get("worker_id") == "worker-b"

    print("\nWorker recovery integration test passed.")
    print("Clip ID:", args.clip_id)
    print("Attempt records:", len(records))
    print("Attempt 1 worker:", failed_record.get("worker_id"))
    print("Attempt 1 error type:", failed_record.get("error_type"))
    print("Attempt 2 worker:", successful_record.get("worker_id"))
    print("Attempt 2 JSON parse:", successful_record.get("json_parse_ok"))
    print("Final state counts:", final_counts)
    print("State database:", state_db_path)
    print("Worker output:", output_path)


if __name__ == "__main__":
    main()