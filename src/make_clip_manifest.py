"""
make_clip_manifest.py

Convert content-aware sampler output into a clip-level manifest.

Input:
    D:\projects\longvideo\keyframes\sampling_summary.jsonl

Outputs:
    D:\projects\longvideo\keyframes\clip_manifest.jsonl
    D:\projects\longvideo\keyframes\clip_manifest.csv

The important fix:
    Do not assume every clip is exactly 20 seconds.
    This script accumulates each clip's real duration_sec from sampling_summary.jsonl.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SUMMARY_PATH = r"D:\projects\longvideo\keyframes\sampling_summary.jsonl"
DEFAULT_OUTPUT_JSONL = r"D:\projects\longvideo\keyframes\clip_manifest.jsonl"
DEFAULT_OUTPUT_CSV = r"D:\projects\longvideo\keyframes\clip_manifest.csv"


def format_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def extract_clip_id(record: Dict[str, Any], fallback_id: int) -> int:
    clip_file = record.get("clip_file") or ""
    match = re.search(r"clip[_-](\d+)", clip_file)
    if match:
        return int(match.group(1))
    return fallback_id


def load_summary(summary_path: Path) -> List[Dict[str, Any]]:
    records = []
    with summary_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            record["_line_num"] = line_num
            records.append(record)

    if not records:
        raise ValueError(f"No records found in summary file: {summary_path}")

    return records


def build_manifest(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []

    for fallback_id, record in enumerate(records):
        clip_id = extract_clip_id(record, fallback_id)
        video_metadata = record.get("video_metadata") or {}
        duration_sec = float(video_metadata.get("duration_sec", 0.0) or 0.0)

        enriched.append({
            "clip_id": clip_id,
            "record": record,
            "duration_sec": duration_sec,
        })

    enriched.sort(key=lambda x: x["clip_id"])

    manifest = []
    current_start = 0.0

    for item in enriched:
        record = item["record"]
        clip_id = item["clip_id"]
        duration_sec = item["duration_sec"]
        current_end = current_start + duration_sec

        selected_frame_records = []
        for frame_order, frame in enumerate(record.get("selected_frames") or []):
            local_timestamp_sec = float(frame.get("timestamp_sec", 0.0) or 0.0)
            global_timestamp_sec = current_start + local_timestamp_sec

            selected_frame_records.append({
                "frame_order": frame_order,
                "frame_index": frame.get("frame_index"),
                "local_timestamp_sec": local_timestamp_sec,
                "local_timestamp": frame.get("timestamp") or format_time(local_timestamp_sec),
                "global_timestamp_sec": global_timestamp_sec,
                "global_timestamp": format_time(global_timestamp_sec),
                "image_file": frame.get("image_file"),
                "image_path": frame.get("image_path"),
                "selected_reason": frame.get("selected_reason"),
                "brightness": frame.get("brightness"),
                "sharpness": frame.get("sharpness"),
                "motion_score": frame.get("motion_score"),
                "hist_diff_score": frame.get("hist_diff_score"),
                "combined_score": frame.get("combined_score"),
            })

        video_metadata = record.get("video_metadata") or {}

        manifest.append({
            "manifest_version": "clip_manifest_v1",
            "clip_id": clip_id,
            "clip_file": record.get("clip_file"),
            "clip_stem": record.get("clip_stem"),
            "video_path": video_metadata.get("video_path"),
            "duration_sec": duration_sec,
            "actual_start_sec": current_start,
            "actual_end_sec": current_end,
            "actual_start_time": format_time(current_start),
            "actual_end_time": format_time(current_end),
            "selected_frame_count": len(selected_frame_records),
            "selected_frames": selected_frame_records,
            "keyframe_output_dir": record.get("output_dir"),
            "contact_sheet_path": record.get("contact_sheet_path"),
            "sampler_metadata_path": record.get("metadata_path"),
            "sampler_version": record.get("sampler_version"),
            "sampler_error": record.get("error"),
        })

        current_start = current_end

    return manifest


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "clip_id",
        "clip_file",
        "duration_sec",
        "actual_start_time",
        "actual_end_time",
        "selected_frame_count",
        "video_path",
        "keyframe_output_dir",
        "contact_sheet_path",
        "sampler_metadata_path",
        "sampler_error",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def print_summary(records: List[Dict[str, Any]]) -> None:
    durations = [float(r.get("duration_sec", 0.0) or 0.0) for r in records]
    selected_counts = [int(r.get("selected_frame_count", 0) or 0) for r in records]
    errors = [r for r in records if r.get("sampler_error")]
    total_duration = sum(durations)

    print("\nManifest summary")
    print("Clip count:", len(records))
    print("Total duration:", format_time(total_duration), f"({total_duration:.2f}s)")
    print("Min clip duration:", f"{min(durations):.2f}s")
    print("Max clip duration:", f"{max(durations):.2f}s")
    print("Min selected frames:", min(selected_counts))
    print("Max selected frames:", max(selected_counts))
    print("Sampler errors:", len(errors))

    abnormal = [
        r for r in records
        if float(r.get("duration_sec", 0.0) or 0.0) < 18.0
        or float(r.get("duration_sec", 0.0) or 0.0) > 22.0
    ]

    if abnormal:
        print("\nClips with duration outside 18-22 seconds:")
        for r in abnormal:
            print(
                f"  clip {r['clip_id']:04d}: "
                f"{r['clip_file']} | "
                f"{r['duration_sec']:.2f}s | "
                f"{r['actual_start_time']} - {r['actual_end_time']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clip_manifest.jsonl from content-aware sampler summary."
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to sampling_summary.jsonl generated by content_aware_sampler.py.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=DEFAULT_OUTPUT_JSONL,
        help="Output path for clip_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Output path for clip_manifest.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    print("Reading sampler summary:", summary_path)
    records = load_summary(summary_path)
    manifest = build_manifest(records)

    write_jsonl(manifest, output_jsonl)
    write_csv(manifest, output_csv)

    print("Saved JSONL manifest:", output_jsonl)
    print("Saved CSV manifest:", output_csv)
    print_summary(manifest)


if __name__ == "__main__":
    main()
