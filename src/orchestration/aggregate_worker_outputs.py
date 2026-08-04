r"""
aggregate_worker_outputs.py

Merge standalone worker JSONL outputs into one deterministic clip-level result.

Behavior:
- Reads worker_*.jsonl files from one run directory.
- Parses valid JSON records.
- Groups records by clip_id.
- Prefers a successful record over failed or invalid records.
- If several equally successful records exist, keeps the latest finished record.
- Writes one record per clip, sorted by clip_id.
- Optionally validates coverage against a manifest.
- Writes a machine-readable aggregation summary.

Example:
    python .\src\orchestration\aggregate_worker_outputs.py `
      --input-dir "D:\projects\longvideo\episodes\ep02\outputs\worker_runs\bf16_4w" `
      --output-path "D:\projects\longvideo\episodes\ep02\outputs\ep02_bf16_4w_merged.jsonl" `
      --manifest-path "D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_flow.jsonl" `
      --require-complete
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and validate sharded VideoThinker worker outputs."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing worker_*.jsonl files.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Merged JSONL output path.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest used to validate expected clip IDs.",
    )
    parser.add_argument(
        "--clip-ids",
        default=None,
        help=(
            "Optional comma-separated expected clip IDs. Applied after manifest "
            "loading when both are supplied."
        ),
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional aggregation summary JSON path.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit with an error when expected clips are missing or unsuccessful.",
    )
    return parser.parse_args()


def parse_clip_ids(text: Optional[str]) -> Optional[List[int]]:
    if text is None or not text.strip():
        return None

    values = sorted(
        {
            int(part.strip())
            for part in text.split(",")
            if part.strip()
        }
    )
    if not values:
        raise ValueError("--clip-ids did not contain any valid IDs.")
    return values


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    invalid_line_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_line_count += 1
                print(
                    f"Warning: invalid JSON in {path.name} line "
                    f"{line_number}: {exc}"
                )
                continue

            if record.get("clip_id") is None:
                invalid_line_count += 1
                print(
                    f"Warning: record without clip_id in {path.name} "
                    f"line {line_number}; skipped."
                )
                continue

            record["_aggregation_source_file"] = path.name
            record["_aggregation_source_line"] = line_number
            records.append(record)

    return records, invalid_line_count


def load_manifest_clip_ids(path: Path) -> List[int]:
    records, invalid_count = read_jsonl(path)
    if invalid_count:
        raise ValueError(
            f"Manifest contains {invalid_count} invalid record(s): {path}"
        )

    clip_ids = [int(record["clip_id"]) for record in records]
    duplicates = sorted(
        clip_id
        for clip_id in set(clip_ids)
        if clip_ids.count(clip_id) > 1
    )
    if duplicates:
        raise ValueError(
            f"Manifest contains duplicate clip IDs: {duplicates[:10]}"
        )

    if not clip_ids:
        raise ValueError(f"Manifest is empty: {path}")

    return sorted(clip_ids)


def is_success(record: Dict[str, Any]) -> bool:
    return (
        record.get("error") is None
        and record.get("json_parse_ok") is True
    )


def parse_finished_time(record: Dict[str, Any]) -> datetime:
    value = (
        record.get("finished_at_utc")
        or record.get("completed_at_utc")
        or record.get("finished_at")
        or ""
    )
    if not value:
        return datetime.min

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
            tzinfo=None
        )
    except ValueError:
        return datetime.min


def record_rank(record: Dict[str, Any]) -> Tuple[int, datetime, int]:
    """
    Higher is better:
    1. Successful record beats failed/invalid record.
    2. Latest finished timestamp wins.
    3. Later source line wins as a deterministic fallback.
    """
    return (
        1 if is_success(record) else 0,
        parse_finished_time(record),
        int(record.get("_aggregation_source_line", 0) or 0),
    )


def choose_best_record(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return max(records, key=record_rank)


def strip_internal_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_aggregation_")
    }


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_path.with_suffix(".summary.json")
    )
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else None
    )
    explicit_clip_ids = parse_clip_ids(args.clip_ids)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    worker_files = sorted(input_dir.glob("worker_*.jsonl"))
    if not worker_files:
        raise FileNotFoundError(
            f"No worker_*.jsonl files found in: {input_dir}"
        )

    print("Worker files:")
    for path in worker_files:
        print(" ", path.name)

    records_by_clip: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    total_records = 0
    invalid_line_count = 0

    for worker_file in worker_files:
        records, invalid_count = read_jsonl(worker_file)
        invalid_line_count += invalid_count
        total_records += len(records)

        for record in records:
            records_by_clip[int(record["clip_id"])].append(record)

    selected_records: List[Dict[str, Any]] = []
    duplicate_clip_ids: List[int] = []

    for clip_id in sorted(records_by_clip):
        candidates = records_by_clip[clip_id]
        if len(candidates) > 1:
            duplicate_clip_ids.append(clip_id)

        selected = choose_best_record(candidates)
        selected_records.append(strip_internal_fields(selected))

    actual_ids = {int(record["clip_id"]) for record in selected_records}

    if manifest_path is not None:
        expected_ids = set(load_manifest_clip_ids(manifest_path))
    else:
        expected_ids = set(actual_ids)

    if explicit_clip_ids is not None:
        expected_ids &= set(explicit_clip_ids)

    selected_records = [
        record
        for record in selected_records
        if int(record["clip_id"]) in expected_ids
    ]
    actual_ids = {int(record["clip_id"]) for record in selected_records}

    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)
    unsuccessful_ids = sorted(
        int(record["clip_id"])
        for record in selected_records
        if not is_success(record)
    )
    successful_count = sum(
        1 for record in selected_records if is_success(record)
    )

    write_jsonl(selected_records, output_path)

    summary = {
        "input_dir": str(input_dir),
        "worker_files": [str(path) for path in worker_files],
        "worker_file_count": len(worker_files),
        "raw_valid_record_count": total_records,
        "invalid_line_count": invalid_line_count,
        "unique_clip_count_before_filter": len(records_by_clip),
        "selected_clip_count": len(selected_records),
        "successful_clip_count": successful_count,
        "unsuccessful_clip_count": len(unsuccessful_ids),
        "duplicate_clip_count": len(duplicate_clip_ids),
        "duplicate_clip_ids": duplicate_clip_ids,
        "expected_clip_count": len(expected_ids),
        "missing_clip_ids": missing_ids,
        "unexpected_clip_ids": unexpected_ids,
        "unsuccessful_clip_ids": unsuccessful_ids,
        "complete": (
            not missing_ids
            and not unsuccessful_ids
            and len(selected_records) == len(expected_ids)
        ),
        "output_path": str(output_path),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nAggregation summary")
    print("Raw valid records:", total_records)
    print("Unique clips:", len(records_by_clip))
    print("Selected clips:", len(selected_records))
    print("Successful clips:", successful_count)
    print("Duplicate clip IDs:", duplicate_clip_ids)
    print("Missing clip IDs:", missing_ids)
    print("Unsuccessful clip IDs:", unsuccessful_ids)
    print("Complete:", summary["complete"])
    print("Merged output:", output_path)
    print("Summary:", summary_path)

    if args.require_complete and not summary["complete"]:
        raise RuntimeError(
            "Aggregation is incomplete. See the summary JSON for details."
        )


if __name__ == "__main__":
    main()