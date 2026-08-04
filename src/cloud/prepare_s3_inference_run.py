"""
Upload a local keyframe manifest and its selected frame images to S3.

The generated cloud manifest preserves the original record structure, but
replaces every selected_frames[*].image_path with an s3:// URI.

Example:
    python .\src\cloud\prepare_s3_inference_run.py `
      --manifest-path "D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_flow.jsonl" `
      --bucket videothinker-longvideo-535534157295-us-east-2 `
      --run-id ep02-bf16-cloud-v1 `
      --region us-east-2 `
      --profile videothinker-dev
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.s3_artifact_store import S3ArtifactStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--prefix",
        default="runs",
        help="Top-level S3 key prefix.",
    )
    parser.add_argument(
        "--local-output-path",
        default=None,
        help="Optional local path for the generated cloud manifest.",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Upload files even when the target S3 object already exists.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on manifest line {line_number}: {exc}"
                ) from exc
            if record.get("clip_id") is None:
                raise ValueError(
                    f"Manifest line {line_number} has no clip_id."
                )
            records.append(record)

    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    output_path = (
        Path(args.local_output_path)
        if args.local_output_path
        else manifest_path.with_name(
            f"{manifest_path.stem}_{args.run_id}_cloud.jsonl"
        )
    )

    store = S3ArtifactStore(
        region_name=args.region,
        profile_name=args.profile,
    )
    records = load_jsonl(manifest_path)

    uploaded_uris: Set[str] = set()
    skipped_existing = 0
    total_frame_references = 0
    cloud_records: List[Dict[str, Any]] = []

    base_key = f"{args.prefix.strip('/')}/{args.run_id}/inputs"

    for record in records:
        cloud_record = dict(record)
        cloud_frames: List[Dict[str, Any]] = []
        clip_id = int(record["clip_id"])

        for frame_index, frame in enumerate(
            record.get("selected_frames") or []
        ):
            total_frame_references += 1
            cloud_frame = dict(frame)
            local_image = Path(str(frame.get("image_path") or ""))

            if not local_image.exists():
                raise FileNotFoundError(
                    f"Missing frame for clip {clip_id}: {local_image}"
                )

            object_name = (
                f"{frame_index:02d}_{local_image.name}"
            )
            image_uri = (
                f"s3://{args.bucket}/{base_key}/frames/"
                f"clip_{clip_id:04d}/{object_name}"
            )

            if image_uri not in uploaded_uris:
                if args.force_upload or not store.exists(image_uri):
                    content_type = (
                        mimetypes.guess_type(local_image.name)[0]
                        or "application/octet-stream"
                    )
                    store.upload_file(
                        local_image,
                        image_uri,
                        content_type=content_type,
                    )
                    print("Uploaded:", image_uri)
                else:
                    skipped_existing += 1
                uploaded_uris.add(image_uri)

            cloud_frame["source_image_path"] = str(local_image)
            cloud_frame["image_path"] = image_uri
            cloud_frames.append(cloud_frame)

        cloud_record["selected_frames"] = cloud_frames
        cloud_records.append(cloud_record)

    write_jsonl(cloud_records, output_path)

    manifest_uri = (
        f"s3://{args.bucket}/{base_key}/manifest/"
        f"{manifest_path.stem}_cloud.jsonl"
    )
    store.upload_file(
        output_path,
        manifest_uri,
        content_type="application/x-ndjson",
    )

    summary = {
        "run_id": args.run_id,
        "source_manifest": str(manifest_path),
        "cloud_manifest_local": str(output_path),
        "cloud_manifest_uri": manifest_uri,
        "record_count": len(cloud_records),
        "frame_reference_count": total_frame_references,
        "unique_uploaded_frame_count": len(uploaded_uris),
        "skipped_existing_object_count": skipped_existing,
        "output_prefix": f"s3://{args.bucket}/{args.prefix.strip('/')}/{args.run_id}/outputs",
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nS3 preparation complete.")
    print("Records:", len(cloud_records))
    print("Frame references:", total_frame_references)
    print("Unique frame objects:", len(uploaded_uris))
    print("Existing objects skipped:", skipped_existing)
    print("Cloud manifest:", manifest_uri)
    print("Output prefix:", summary["output_prefix"])
    print("Local summary:", summary_path)


if __name__ == "__main__":
    main()