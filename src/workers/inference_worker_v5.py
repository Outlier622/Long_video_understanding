r"""
Standalone sharded inference worker with local/S3 artifacts and shared state.

The worker:
- Reads one clip manifest.
- Selects an optional explicit clip subset.
- Selects one deterministic shard using clip order modulo num_shards.
- Loads the model once.
- Processes all clips assigned to that shard.
- Writes only to its own JSONL output file.
- Resumes by skipping successful clip records already present in that file.

Example dry run:
    python .\src\workers\inference_worker_v5.py `
      --model-path "D:\projects\VideoThinker-R1-3B" `
      --manifest-path "D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_flow.jsonl" `
      --output-dir "D:\projects\longvideo\episodes\ep02\outputs\worker_runs\bf16_4w" `
      --shard-index 0 `
      --num-shards 4 `
      --dry-run

Example inference run:
    python .\src\workers\inference_worker_v5.py `
      --model-path "D:\projects\VideoThinker-R1-3B" `
      --manifest-path "D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_flow.jsonl" `
      --output-dir "D:\projects\longvideo\episodes\ep02\outputs\worker_runs\bf16_1w" `
      --shard-index 0 `
      --num-shards 1 `
      --clip-ids "1,3" `
      --dtype bf16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.state.job_state_store import SQLiteJobStateStore
from src.state.dynamodb_job_state_store import DynamoDBJobStateStore
from src.storage.s3_artifact_store import (
    S3ArtifactStore,
    is_s3_uri,
    join_s3_uri,
    parse_s3_uri,
)


DEFAULT_VERSION = "keyframe_v7_s3_model_worker"

KEYFRAME_PROMPT_BODY = """
You are analyzing a sequence of selected keyframes from one short video clip.

The images are ordered chronologically. Each image represents a selected moment from
the same clip. Your task is to describe the visible events based only on these images.

Important rules:
1. Only describe what is directly visible in the provided keyframes. If motion is not directly visible across multiple keyframes, describe it as a visible pose or state change instead of a continuous action.
2. Do not infer events from other clips, previous scenes, future scenes, or general story knowledge.
3. Do not invent characters, vehicles, locations, monsters, screens, holograms, explosions, beams, or control rooms unless they are clearly visible.
4. If a subject is unclear, write "unclear object", "unclear figure", or "unclear structure" instead of naming it.
5. If the keyframes mostly show darkness, credits, subtitles, static text, or low-information content, say that clearly.
6. Prioritize major visible state changes and scene changes over small background details.
7. Because the input is a sequence of still keyframes, do not overstate continuous motion.
8. Describe visible state changes across frames, not unseen actions between frames.
9. If there is a major visible event, such as a creature appearing, a statue or structure falling, a vehicle moving, a fight, smoke, fire, explosion, or a screen display, include it.
10. Every event must include visual_evidence and confidence.
11. Use frame references such as "Frame 03" or "Frame 07" when describing evidence.
12. If the evidence is weak, set confidence to "low" and list the uncertainty.

Output length rules:
- summary must be no more than 35 words.
- include at most 3 events.
- each visual_evidence must be no more than 20 words.
- each event can reference at most 3 frames.
- possible_hallucination_risks must be non-empty if any object or action is uncertain.

Return valid JSON only. Do not use markdown. Return all fields in English only.

Use this exact JSON schema:
{
  "clip_id": number,
  "clip_file": string,
  "actual_start_time": string,
  "actual_end_time": string,
  "summary": string,
  "visual_information_level": "high | medium | low",
  "setting": string,
  "main_subjects": [string],
  "events": [
    {
      "action": string,
      "objects": [string],
      "scene": string,
      "visual_evidence": string,
      "frame_references": [string],
      "confidence": "high | medium | low"
    }
  ],
  "uncertain_parts": [string],
  "possible_hallucination_risks": [string]
}
""".strip()


@dataclass(frozen=True)
class WorkerConfig:
    model_path: Path
    model_s3_prefix: Optional[str]
    verify_model_sha256: bool
    manifest_path: Path
    manifest_s3_uri: Optional[str]
    output_dir: Path
    output_s3_prefix: Optional[str]
    cache_dir: Path
    shard_index: int
    num_shards: int
    clip_ids: Optional[Set[int]]
    version: str
    dtype: str
    max_new_tokens: int
    min_pixels: int
    max_pixels: int
    max_keyframes_per_clip: int
    use_top_score_when_too_many_frames: bool
    force: bool
    dry_run: bool
    worker_id: str
    state_backend: str
    state_db_path: Optional[Path]
    dynamodb_table: Optional[str]
    aws_region: Optional[str]
    aws_profile: Optional[str]
    run_id: str
    lease_seconds: int
    max_attempts: int

    @property
    def output_path(self) -> Path:
        return self.output_dir / (
            f"worker_{self.shard_index:03d}_of_{self.num_shards:03d}.jsonl"
        )

    @property
    def output_s3_uri(self) -> Optional[str]:
        if not self.output_s3_prefix:
            return None
        return join_s3_uri(
            self.output_s3_prefix,
            self.output_path.name,
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_clip_ids(text: Optional[str]) -> Optional[Set[int]]:
    if text is None or not text.strip():
        return None

    values: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if part:
            values.add(int(part))

    if not values:
        raise ValueError("--clip-ids did not contain any valid clip IDs.")

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic shard of keyframe-based VideoThinker inference."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help=(
            "Local model directory or local cache destination when "
            "--model-s3-prefix is supplied."
        ),
    )
    parser.add_argument(
        "--model-s3-prefix",
        default=None,
        help=(
            "Optional S3 model bundle prefix, for example "
            "s3://bucket/models/videothinker-r1-3b."
        ),
    )
    parser.add_argument(
        "--verify-model-sha256",
        action="store_true",
        help="Verify SHA-256 after downloading every model file.",
    )
    parser.add_argument("--manifest-path", required=True, help="Input clip manifest JSONL.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for worker-specific JSONL outputs.",
    )
    parser.add_argument(
        "--output-s3-prefix",
        default=None,
        help=(
            "Optional S3 prefix for worker JSONL outputs, for example "
            "s3://bucket/runs/run-id/outputs."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Local cache for downloaded S3 manifest and frames. Defaults to "
            "<output-dir>/cache."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--clip-ids",
        default=None,
        help="Optional comma-separated clip IDs applied before sharding.",
    )
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp32"],
        default="auto",
        help="Model loading dtype. auto uses BF16 on CUDA and FP32 on CPU.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-keyframes-per-clip", type=int, default=12)
    parser.add_argument(
        "--use-first-frames",
        action="store_true",
        help="When too many frames exist, use the first N instead of top-score N.",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Optional worker identifier. Defaults to AWS Batch job ID or host/process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete this worker's prior output and rerun its assigned clips.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assignment and validate paths without loading the model.",
    )
    parser.add_argument(
        "--state-backend",
        choices=["output", "sqlite", "dynamodb"],
        default="output",
        help=(
            "Completion/claim backend. output uses only this worker's JSONL; "
            "sqlite provides local shared state; dynamodb provides cloud shared state."
        ),
    )
    parser.add_argument(
        "--state-db-path",
        default=None,
        help="SQLite state database path when --state-backend sqlite is used.",
    )
    parser.add_argument(
        "--dynamodb-table",
        default=None,
        help="DynamoDB table name when --state-backend dynamodb is used.",
    )
    parser.add_argument(
        "--aws-region",
        default=None,
        help="AWS region for DynamoDB, for example us-east-2.",
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
        help=(
            "Optional AWS CLI profile for local execution. Omit this in AWS Batch "
            "so boto3 uses the job role."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Logical job run ID. Required for shared state; otherwise derived.",
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=7200,
        help="Running-claim lease duration.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum claims for one clip before FINAL_FAILED.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> WorkerConfig:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards - 1}], "
            f"but received {args.shard_index}."
        )
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.max_keyframes_per_clip < 1:
        raise ValueError("--max-keyframes-per-clip must be positive.")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Pixel limits are invalid.")
    if args.lease_seconds < 1:
        raise ValueError("--lease-seconds must be positive.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive.")
    if args.state_backend == "sqlite" and not args.state_db_path:
        raise ValueError("--state-db-path is required for sqlite state.")
    if args.state_backend == "dynamodb" and not args.dynamodb_table:
        raise ValueError("--dynamodb-table is required for dynamodb state.")
    if args.state_backend != "output" and args.force:
        raise ValueError(
            "--force is not supported with shared state. Use a new --run-id "
            "for a clean rerun."
        )

    output_dir = Path(args.output_dir)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_dir / "cache"
    )

    manifest_source = str(args.manifest_path)
    if is_s3_uri(manifest_source):
        manifest_location = parse_s3_uri(manifest_source)
        manifest_path = (
            cache_dir
            / "manifest"
            / Path(manifest_location.key).name
        )
        manifest_s3_uri = manifest_source
        manifest_stem = Path(manifest_location.key).stem
    else:
        manifest_path = Path(manifest_source)
        manifest_s3_uri = None
        manifest_stem = manifest_path.stem

    worker_id = (
        args.worker_id
        or os.getenv("AWS_BATCH_JOB_ID")
        or f"{socket.gethostname()}-pid-{os.getpid()}"
    )

    return WorkerConfig(
        model_path=Path(args.model_path),
        model_s3_prefix=args.model_s3_prefix,
        verify_model_sha256=args.verify_model_sha256,
        manifest_path=manifest_path,
        manifest_s3_uri=manifest_s3_uri,
        output_dir=output_dir,
        output_s3_prefix=args.output_s3_prefix,
        cache_dir=cache_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        clip_ids=parse_clip_ids(args.clip_ids),
        version=args.version,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_keyframes_per_clip=args.max_keyframes_per_clip,
        use_top_score_when_too_many_frames=not args.use_first_frames,
        force=args.force,
        dry_run=args.dry_run,
        worker_id=worker_id,
        state_backend=args.state_backend,
        state_db_path=Path(args.state_db_path) if args.state_db_path else None,
        dynamodb_table=args.dynamodb_table,
        aws_region=args.aws_region,
        aws_profile=args.aws_profile,
        run_id=(
            args.run_id
            or f"{manifest_stem}-{args.dtype}-{args.num_shards}shards"
        ),
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
    )



def build_s3_store(config: WorkerConfig) -> Optional[S3ArtifactStore]:
    needs_s3 = bool(
        config.model_s3_prefix
        or config.manifest_s3_uri
        or config.output_s3_prefix
    )
    if not needs_s3:
        return None

    return S3ArtifactStore(
        region_name=config.aws_region,
        profile_name=config.aws_profile,
    )



def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def model_bundle_manifest_uri(config: WorkerConfig) -> Optional[str]:
    if not config.model_s3_prefix:
        return None
    return join_s3_uri(config.model_s3_prefix, "bundle_manifest.json")


def validate_model_source(
    config: WorkerConfig,
    s3_store: Optional[S3ArtifactStore],
) -> None:
    if config.model_s3_prefix:
        if s3_store is None:
            raise RuntimeError("S3 store is required for an S3 model bundle.")
        manifest_uri = model_bundle_manifest_uri(config)
        assert manifest_uri is not None
        if not s3_store.exists(manifest_uri):
            raise FileNotFoundError(
                f"Model bundle manifest not found: {manifest_uri}"
            )
        print("Model bundle manifest:", manifest_uri)
        return

    if not config.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {config.model_path}")


def ensure_model_local(
    config: WorkerConfig,
    s3_store: Optional[S3ArtifactStore],
) -> None:
    if not config.model_s3_prefix:
        if not config.model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {config.model_path}"
            )
        return

    if s3_store is None:
        raise RuntimeError("S3 store is required for an S3 model bundle.")

    manifest_uri = model_bundle_manifest_uri(config)
    assert manifest_uri is not None
    local_manifest = (
        config.cache_dir
        / "model_bundle"
        / "bundle_manifest.json"
    )

    print("Downloading model bundle manifest:", manifest_uri)
    s3_store.download_file(manifest_uri, local_manifest)
    manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
    files = manifest.get("files") or []
    if not files:
        raise ValueError(
            f"Model bundle manifest contains no files: {manifest_uri}"
        )

    print("Model bundle files:", len(files))
    print("Model bundle size GiB:", manifest.get("total_gib"))
    config.model_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    reused = 0
    for index, item in enumerate(files, start=1):
        relative_path = Path(str(item["relative_path"]))
        expected_size = int(item["size_bytes"])
        expected_sha256 = str(item.get("sha256") or "")
        source_uri = str(item["s3_uri"])
        local_path = config.model_path / relative_path

        valid_existing = (
            local_path.exists()
            and local_path.is_file()
            and local_path.stat().st_size == expected_size
        )

        if (
            valid_existing
            and config.verify_model_sha256
            and expected_sha256
        ):
            valid_existing = (
                sha256_file(local_path) == expected_sha256
            )

        if valid_existing:
            reused += 1
            continue

        print(
            f"Downloading model file {index}/{len(files)}:",
            source_uri,
        )
        s3_store.download_file(source_uri, local_path)

        if local_path.stat().st_size != expected_size:
            raise IOError(
                f"Downloaded model file has wrong size: {local_path}"
            )
        if (
            config.verify_model_sha256
            and expected_sha256
            and sha256_file(local_path) != expected_sha256
        ):
            raise IOError(
                f"Downloaded model file has wrong SHA-256: {local_path}"
            )

        downloaded += 1

    required_config = config.model_path / "config.json"
    if not required_config.exists():
        raise FileNotFoundError(
            f"Downloaded model bundle has no config.json: "
            f"{config.model_path}"
        )

    print("Model bundle ready:", config.model_path)
    print("Model files downloaded:", downloaded)
    print("Model files reused:", reused)

def ensure_manifest_local(
    config: WorkerConfig,
    s3_store: Optional[S3ArtifactStore],
) -> None:
    if config.manifest_s3_uri is None:
        return
    if s3_store is None:
        raise RuntimeError("S3 store is required for an S3 manifest.")

    print("Downloading manifest:", config.manifest_s3_uri)
    s3_store.download_file(
        config.manifest_s3_uri,
        config.manifest_path,
    )
    print("Local manifest:", config.manifest_path)


def frame_cache_path(
    *,
    config: WorkerConfig,
    clip_id: int,
    frame_index: int,
    source_uri: str,
) -> Path:
    location = parse_s3_uri(source_uri)
    source_name = Path(location.key).name
    return (
        config.cache_dir
        / "frames"
        / f"clip_{clip_id:04d}"
        / f"{frame_index:02d}_{source_name}"
    )


def validate_assigned_frame_sources(
    records: Sequence[Dict[str, Any]],
    s3_store: Optional[S3ArtifactStore],
) -> Tuple[int, int]:
    total_paths = 0
    missing_paths = 0

    for record in records:
        for frame in record.get("selected_frames") or []:
            total_paths += 1
            image_path = str(frame.get("image_path") or "")
            if is_s3_uri(image_path):
                if s3_store is None or not s3_store.exists(image_path):
                    missing_paths += 1
            elif not image_path or not Path(image_path).exists():
                missing_paths += 1

    return total_paths, missing_paths


def materialize_assigned_frames(
    records: Sequence[Dict[str, Any]],
    config: WorkerConfig,
    s3_store: Optional[S3ArtifactStore],
) -> None:
    for record in records:
        clip_id = int(record["clip_id"])
        for frame_index, frame in enumerate(
            record.get("selected_frames") or []
        ):
            source = str(frame.get("image_path") or "")
            if not is_s3_uri(source):
                continue
            if s3_store is None:
                raise RuntimeError(
                    f"S3 store is required to download frame: {source}"
                )

            local_path = frame_cache_path(
                config=config,
                clip_id=clip_id,
                frame_index=frame_index,
                source_uri=source,
            )
            if not local_path.exists():
                print("Downloading frame:", source)
                s3_store.download_file(source, local_path)

            frame["source_image_path"] = source
            frame["image_path"] = str(local_path)


def restore_remote_worker_output(
    config: WorkerConfig,
    s3_store: Optional[S3ArtifactStore],
) -> None:
    if not config.output_s3_uri or s3_store is None:
        return
    if config.output_path.exists():
        return

    downloaded = s3_store.download_if_exists(
        config.output_s3_uri,
        config.output_path,
    )
    if downloaded:
        print(
            "Restored prior worker output from S3:",
            config.output_s3_uri,
        )


def sync_worker_output(
    config: WorkerConfig,
    s3_store: Optional[S3ArtifactStore],
) -> None:
    if not config.output_s3_uri:
        return
    if s3_store is None:
        raise RuntimeError("S3 store is required for S3 output sync.")

    s3_store.upload_file(
        config.output_path,
        config.output_s3_uri,
        content_type="application/x-ndjson",
    )
    print("Synced worker output:", config.output_s3_uri)


def state_output_location(config: WorkerConfig) -> str:
    return config.output_s3_uri or str(config.output_path)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
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
                    f"Invalid JSON in manifest line {line_number}: {exc}"
                ) from exc

            if record.get("clip_id") is None:
                raise ValueError(
                    f"Manifest line {line_number} does not contain clip_id."
                )
            records.append(record)

    if not records:
        raise ValueError(f"No records found in manifest: {path}")

    clip_ids = [int(record["clip_id"]) for record in records]
    duplicate_ids = sorted(
        clip_id for clip_id in set(clip_ids) if clip_ids.count(clip_id) > 1
    )
    if duplicate_ids:
        raise ValueError(f"Manifest contains duplicate clip IDs: {duplicate_ids[:10]}")

    return sorted(records, key=lambda record: int(record["clip_id"]))


def select_assigned_records(
    records: Sequence[Dict[str, Any]],
    config: WorkerConfig,
) -> List[Dict[str, Any]]:
    selected = list(records)

    if config.clip_ids is not None:
        selected = [
            record
            for record in selected
            if int(record["clip_id"]) in config.clip_ids
        ]

        found_ids = {int(record["clip_id"]) for record in selected}
        missing_ids = sorted(config.clip_ids - found_ids)
        if missing_ids:
            raise ValueError(
                f"Requested clip IDs are missing from the manifest: {missing_ids}"
            )

    # Shard by position in the sorted selected list. This keeps the distribution
    # deterministic even when clip IDs are not contiguous.
    return [
        record
        for position, record in enumerate(selected)
        if position % config.num_shards == config.shard_index
    ]


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return text.strip()


def try_extract_json(text: str) -> str:
    text = clean_json_text(text)
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text


def normalize_video_kwargs(video_kwargs: Optional[dict]) -> dict:
    if not video_kwargs:
        return {}

    cleaned: Dict[str, Any] = {}
    for key, value in video_kwargs.items():
        if isinstance(value, list):
            if not value:
                continue
            cleaned[key] = value[0] if len(value) == 1 else value
        else:
            cleaned[key] = value

    return cleaned


def choose_keyframes(
    record: Dict[str, Any],
    config: WorkerConfig,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    frames = list(record.get("selected_frames") or [])
    existing: List[Dict[str, Any]] = []
    missing_paths: List[str] = []

    for frame in frames:
        image_path = frame.get("image_path")
        if image_path and Path(image_path).exists():
            existing.append(frame)
        else:
            missing_paths.append(str(image_path))

    if len(existing) <= config.max_keyframes_per_clip:
        selected = existing
    elif config.use_top_score_when_too_many_frames:
        selected = sorted(
            existing,
            key=lambda frame: float(frame.get("combined_score", 0.0) or 0.0),
            reverse=True,
        )[: config.max_keyframes_per_clip]
    else:
        selected = existing[: config.max_keyframes_per_clip]

    selected = sorted(
        selected,
        key=lambda frame: float(frame.get("local_timestamp_sec", 0.0) or 0.0),
    )
    return selected, missing_paths


def build_frame_metadata_text(frames: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for index, frame in enumerate(frames):
        frame_label = f"Frame {index:02d}"
        score = frame.get("combined_score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "unknown"

        lines.append(
            f"{frame_label}: "
            f"local_time={frame.get('local_timestamp') or ''}, "
            f"global_time={frame.get('global_timestamp') or ''}, "
            f"selection_reason={frame.get('selected_reason') or ''}, "
            f"content_score={score_text}"
        )

    return "\n".join(lines)


def build_prompt(
    record: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
) -> str:
    metadata = f"""
Clip metadata:
clip_id: {record.get("clip_id")}
clip_file: {record.get("clip_file")}
actual_start_time: {record.get("actual_start_time")}
actual_end_time: {record.get("actual_end_time")}
duration_sec: {record.get("duration_sec")}
selected_keyframe_count: {len(frames)}

Keyframe order and timestamps:
{build_frame_metadata_text(frames)}
""".strip()

    return metadata + "\n\n" + KEYFRAME_PROMPT_BODY


def build_messages(
    record: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    config: WorkerConfig,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []

    for frame in frames:
        content.append(
            {
                "type": "image",
                "image": frame["image_path"],
                "min_pixels": config.min_pixels,
                "max_pixels": config.max_pixels,
            }
        )

    content.append({"type": "text", "text": build_prompt(record, frames)})
    return [{"role": "user", "content": content}]


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        if not torch.cuda.is_available():
            raise RuntimeError("--dtype bf16 requires CUDA.")
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model_bundle(config: WorkerConfig):
    print("Loading processor:", config.model_path)
    processor = AutoProcessor.from_pretrained(str(config.model_path))

    torch_dtype = resolve_torch_dtype(config.dtype)
    print("Loading model with dtype:", torch_dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(config.model_path),
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    return processor, model, torch_dtype


def infer_record(
    processor,
    model,
    record: Dict[str, Any],
    config: WorkerConfig,
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    frames, missing_paths = choose_keyframes(record, config)

    if not frames:
        raise RuntimeError(
            f"No existing keyframe images for clip {record.get('clip_id')}"
        )

    messages = build_messages(record, frames, config)

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )
    video_kwargs = normalize_video_kwargs(video_kwargs)

    processor_kwargs: Dict[str, Any] = {
        "text": [text],
        "images": image_inputs,
        "padding": True,
        "return_tensors": "pt",
    }

    if video_inputs:
        processor_kwargs["videos"] = video_inputs
        processor_kwargs.update(video_kwargs)

    inputs = processor(**processor_kwargs)
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output, frames, missing_paths


def build_output_record(
    *,
    record: Dict[str, Any],
    config: WorkerConfig,
    raw_output: str,
    used_frames: Sequence[Dict[str, Any]],
    missing_frame_paths: Sequence[str],
    started_at: str,
    elapsed_seconds: float,
    error: Optional[str],
    error_type: Optional[str],
) -> Dict[str, Any]:
    output_record: Dict[str, Any] = {
        "version": config.version,
        "input_mode": "content_aware_keyframes_as_images",
        "worker_id": config.worker_id,
        "shard_index": config.shard_index,
        "num_shards": config.num_shards,
        "clip_id": record.get("clip_id"),
        "clip_file": record.get("clip_file"),
        "actual_start_time": record.get("actual_start_time"),
        "actual_end_time": record.get("actual_end_time"),
        "duration_sec": record.get("duration_sec"),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "selected_frame_count": len(used_frames),
        "missing_frame_paths": list(missing_frame_paths),
        "selected_frames_used": [
            {
                "frame_order": index,
                "image_path": frame.get("image_path"),
                "local_timestamp": frame.get("local_timestamp"),
                "global_timestamp": frame.get("global_timestamp"),
                "selected_reason": frame.get("selected_reason"),
                "combined_score": frame.get("combined_score"),
            }
            for index, frame in enumerate(used_frames)
        ],
        "max_keyframes_per_clip": config.max_keyframes_per_clip,
        "max_new_tokens": config.max_new_tokens,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "dtype_requested": config.dtype,
        "raw_output": raw_output,
        "cleaned_output": None,
        "parsed_json": None,
        "json_parse_ok": False,
        "error_type": error_type,
        "error": error,
    }

    if error is not None:
        return output_record

    cleaned_output = try_extract_json(raw_output)
    output_record["cleaned_output"] = cleaned_output

    try:
        output_record["parsed_json"] = json.loads(cleaned_output)
        output_record["json_parse_ok"] = True
    except json.JSONDecodeError:
        output_record["parsed_json"] = None
        output_record["json_parse_ok"] = False

    return output_record


def load_completed_clip_ids(output_path: Path) -> Set[int]:
    completed: Set[int] = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: invalid JSON in {output_path} line {line_number}; "
                    "the line will not count as completed."
                )
                continue

            clip_id = item.get("clip_id")
            if (
                clip_id is not None
                and item.get("json_parse_ok") is True
                and item.get("error") is None
            ):
                completed.add(int(clip_id))

    return completed


def print_assignment(
    records: Sequence[Dict[str, Any]],
    config: WorkerConfig,
) -> None:
    clip_ids = [int(record["clip_id"]) for record in records]
    print("\nWorker assignment")
    print("Worker ID:", config.worker_id)
    print("Shard:", f"{config.shard_index}/{config.num_shards}")
    print("Assigned clip count:", len(clip_ids))
    print("Assigned clip IDs:", ",".join(str(value) for value in clip_ids))
    print("Local output path:", config.output_path)
    if config.output_s3_uri:
        print("S3 output path:", config.output_s3_uri)




def run_worker(config: WorkerConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    s3_store = build_s3_store(config)
    validate_model_source(config, s3_store)
    ensure_manifest_local(config, s3_store)

    if not config.manifest_path.exists():
        raise FileNotFoundError(f"Manifest path not found: {config.manifest_path}")

    records = load_manifest(config.manifest_path)
    assigned_records = select_assigned_records(records, config)
    print_assignment(assigned_records, config)

    total_frame_paths, missing_frame_paths = validate_assigned_frame_sources(
        assigned_records,
        s3_store,
    )
    print("Assigned frame paths:", total_frame_paths)
    print("Missing frame sources:", missing_frame_paths)

    if not assigned_records:
        print("This shard has no assigned clips.")
        return

    if missing_frame_paths:
        raise FileNotFoundError(
            f"{missing_frame_paths} assigned frame source(s) are missing."
        )

    if config.dry_run:
        print("Dry run complete. The model was not loaded.")
        return

    restore_remote_worker_output(config, s3_store)

    if config.force and config.output_path.exists():
        print("Removing prior worker output:", config.output_path)
        config.output_path.unlink()

    state_store = None
    if config.state_backend == "sqlite":
        assert config.state_db_path is not None
        state_store = SQLiteJobStateStore(config.state_db_path)
        print("State backend: sqlite")
        print("State database:", config.state_db_path)
    elif config.state_backend == "dynamodb":
        assert config.dynamodb_table is not None
        state_store = DynamoDBJobStateStore(
            table_name=config.dynamodb_table,
            region_name=config.aws_region,
            profile_name=config.aws_profile,
        )
        print("State backend: dynamodb")
        print("DynamoDB table:", config.dynamodb_table)
        print("AWS region:", config.aws_region or "boto3 default")
        print("AWS profile:", config.aws_profile or "default credential chain")
    else:
        completed_clip_ids = load_completed_clip_ids(config.output_path)
        print("State backend: output")

    if state_store is not None:
        state_store.initialize_clips(
            config.run_id,
            {int(record["clip_id"]) for record in assigned_records},
        )
        completed_in_output = load_completed_clip_ids(config.output_path)
        if completed_in_output:
            state_store.import_succeeded_clip_ids(
                run_id=config.run_id,
                clip_ids=completed_in_output,
                output_path=state_output_location(config),
            )
            print(
                "Imported successful clips from existing worker output:",
                len(completed_in_output),
            )
        completed_clip_ids = state_store.get_succeeded_clip_ids(config.run_id)
        print("Run ID:", config.run_id)
        print("State counts:", state_store.get_status_counts(config.run_id))

    remaining_records = [
        record
        for record in assigned_records
        if int(record["clip_id"]) not in completed_clip_ids
    ]

    print("Already completed:", len(completed_clip_ids))
    print("Remaining assigned clips:", len(remaining_records))

    if not remaining_records:
        print("No remaining clips for this worker.")
        return

    ensure_model_local(config, s3_store)

    materialize_assigned_frames(
        remaining_records,
        config,
        s3_store,
    )

    processor, model, actual_dtype = load_model_bundle(config)
    print("Model loaded.")
    print("CUDA available:", torch.cuda.is_available())
    print("Actual model dtype:", actual_dtype)

    with config.output_path.open("a", encoding="utf-8") as output_file:
        for position, record in enumerate(remaining_records, start=1):
            clip_id = int(record["clip_id"])

            if state_store is not None:
                claim = state_store.claim_clip(
                    run_id=config.run_id,
                    clip_id=clip_id,
                    worker_id=config.worker_id,
                    lease_seconds=config.lease_seconds,
                    max_attempts=config.max_attempts,
                )
                if not claim.claimed:
                    print(
                        f"Worker {config.shard_index}: skip clip {clip_id}; "
                        f"claim reason={claim.reason}, status={claim.status}, "
                        f"attempt={claim.attempt}"
                    )
                    continue
                print(
                    f"Claimed clip {clip_id}; attempt={claim.attempt}, "
                    f"lease_seconds={config.lease_seconds}"
                )

            print(
                f"\nWorker {config.shard_index}: "
                f"processing {position}/{len(remaining_records)}, clip {clip_id}"
            )

            started_at = utc_now_iso()
            started_perf = time.perf_counter()
            used_frames: List[Dict[str, Any]] = []
            missing_paths: List[str] = []

            try:
                raw_output, used_frames, missing_paths = infer_record(
                    processor=processor,
                    model=model,
                    record=record,
                    config=config,
                )
                elapsed = time.perf_counter() - started_perf
                output_record = build_output_record(
                    record=record,
                    config=config,
                    raw_output=raw_output,
                    used_frames=used_frames,
                    missing_frame_paths=missing_paths,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    error=None,
                    error_type=None,
                )
                print("JSON parse ok:", output_record["json_parse_ok"])
                print("Frames used:", len(used_frames))
                print("Elapsed seconds:", f"{elapsed:.3f}")

            except torch.OutOfMemoryError as exc:
                elapsed = time.perf_counter() - started_perf
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                output_record = build_output_record(
                    record=record,
                    config=config,
                    raw_output="",
                    used_frames=used_frames,
                    missing_frame_paths=missing_paths,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    error=str(exc) or "CUDA out of memory",
                    error_type="CUDA_OUT_OF_MEMORY",
                )
                print("CUDA out of memory on clip:", clip_id)

            except Exception as exc:
                elapsed = time.perf_counter() - started_perf
                output_record = build_output_record(
                    record=record,
                    config=config,
                    raw_output="",
                    used_frames=used_frames,
                    missing_frame_paths=missing_paths,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    error=repr(exc),
                    error_type=type(exc).__name__,
                )
                print("Error on clip:", clip_id)
                print(repr(exc))

            output_file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            try:
                sync_worker_output(config, s3_store)
            except Exception as sync_error:
                print("S3 output sync failed for clip:", clip_id)
                print(repr(sync_error))
                if state_store is not None:
                    final_status = state_store.mark_failed(
                        run_id=config.run_id,
                        clip_id=clip_id,
                        worker_id=config.worker_id,
                        error_type="S3_OUTPUT_SYNC_FAILED",
                        error_message=repr(sync_error),
                        retryable=True,
                        max_attempts=config.max_attempts,
                    )
                    print("State:", clip_id, final_status)
                continue

            if state_store is not None:
                if (
                    output_record.get("error") is None
                    and output_record.get("json_parse_ok") is True
                ):
                    state_store.mark_succeeded(
                        run_id=config.run_id,
                        clip_id=clip_id,
                        worker_id=config.worker_id,
                        output_path=state_output_location(config),
                    )
                    print("State:", clip_id, "SUCCEEDED")
                else:
                    error_type = (
                        output_record.get("error_type")
                        or "JSON_PARSE_FAILED"
                    )
                    error_message = (
                        output_record.get("error")
                        or "Model output was not valid JSON."
                    )
                    retryable = error_type not in {
                        "FileNotFoundError",
                        "NO_KEYFRAMES",
                    }
                    final_status = state_store.mark_failed(
                        run_id=config.run_id,
                        clip_id=clip_id,
                        worker_id=config.worker_id,
                        error_type=str(error_type),
                        error_message=str(error_message),
                        retryable=retryable,
                        max_attempts=config.max_attempts,
                    )
                    print("State:", clip_id, final_status)

    if state_store is not None:
        print("Final state counts:", state_store.get_status_counts(config.run_id))

    print("\nWorker finished.")
    print("Local output:", config.output_path)
    if config.output_s3_uri:
        print("S3 output:", config.output_s3_uri)


def main() -> None:
    config = build_config(parse_args())
    run_worker(config)


if __name__ == "__main__":
    main()