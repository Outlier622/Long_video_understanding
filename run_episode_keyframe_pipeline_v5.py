r"""
run_episode_keyframe_pipeline_v5.py

Version 3 keeps the existing clip/keyframe/manifest preparation workflow, but
replaces dynamic source-code patching with a direct call to the standalone
inference worker.

Two supported modes:

1. Full local preparation + inference:
   python .\run_episode_keyframe_pipeline_v5.py `
     --video "D:\projects\longvideo\raw\ep02.mp4" `
     --episode ep02 `
     --run-mode all

2. Reuse an existing manifest, including the current optical-flow manifest:
   python .\run_episode_keyframe_pipeline_v5.py `
     --episode ep02 `
     --existing-manifest "D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_flow.jsonl" `
     --run-mode test `
     --test-clip-ids "1,3" `
     --dtype bf16

The worker writes one output file per shard:
    worker_000_of_001.jsonl
    worker_000_of_004.jsonl
    worker_001_of_004.jsonl
    ...

This pipeline launches one shard per invocation. An external orchestrator, such
as AWS Batch array jobs, can launch multiple shard indexes concurrently later.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple


DEFAULT_WORK_ROOT = r"D:\projects\longvideo\episodes"
DEFAULT_MODEL_DIR = r"D:\projects\VideoThinker-R1-3B"
DEFAULT_WORKER_RELATIVE_PATH = Path("src") / "workers" / "inference_worker_v3.py"


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("\nRunning command:")
    print(" ".join(f'"{item}"' if " " in str(item) else str(item) for item in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def check_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"{name} was not found in PATH. Install FFmpeg and make sure "
            f"{name}.exe is available in PowerShell."
        )


def get_video_duration_sec(video_path: Path) -> float:
    check_executable("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_clip_files(clips_dir: Path) -> List[Path]:
    if not clips_dir.exists():
        return []
    return sorted(clips_dir.glob("*.mp4"))


def split_video_exact(
    video_path: Path,
    clips_dir: Path,
    episode: str,
    segment_seconds: float,
    force_clips: bool,
    max_clips: Optional[int],
    reencode: bool,
) -> int:
    check_executable("ffmpeg")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ensure_clean_dir(clips_dir, overwrite=force_clips)

    duration = get_video_duration_sec(video_path)
    total_clips = int(math.ceil(duration / segment_seconds))
    if max_clips is not None:
        total_clips = min(total_clips, max_clips)

    existing_clips = list_clip_files(clips_dir)
    print("\nVideo duration:", f"{duration:.3f}s")
    print("Segment seconds:", segment_seconds)
    print("Expected clips:", total_clips)
    print("Existing clips:", len(existing_clips))
    print("Output clips dir:", clips_dir)

    for clip_id in range(total_clips):
        start = clip_id * segment_seconds
        clip_duration = min(segment_seconds, max(0.0, duration - start))
        if clip_duration <= 0:
            break

        out_path = clips_dir / f"{episode}_clip_{clip_id:04d}.mp4"
        if out_path.exists() and not force_clips:
            print(f"Skip existing clip: {out_path.name}")
            continue

        if reencode:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{clip_duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-reset_timestamps",
                "1",
                str(out_path),
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{clip_duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c",
                "copy",
                "-reset_timestamps",
                "1",
                str(out_path),
            ]

        print(
            f"\nCreating clip {clip_id:04d}: "
            f"start={start:.3f}s duration={clip_duration:.3f}s"
        )
        run_cmd(cmd)

    return total_clips


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"Warning: invalid JSON in {path} line {line_number}; "
                    "ignoring that line."
                )
    return records


def clip_file_set_from_clips(clips_dir: Path) -> Set[str]:
    return {path.name for path in list_clip_files(clips_dir)}


def summary_is_complete(summary_path: Path, clips_dir: Path) -> Tuple[bool, str]:
    expected = clip_file_set_from_clips(clips_dir)
    if not expected:
        return False, "No clip mp4 files found."
    if not summary_path.exists():
        return False, "sampling_summary.jsonl does not exist."

    records = read_jsonl(summary_path)
    if not records:
        return False, "sampling_summary.jsonl is empty or invalid."

    completed: Set[str] = set()
    errors = []

    for record in records:
        clip_file = record.get("clip_file")
        error = record.get("error")
        selected_count = int(record.get("selected_count", 0) or 0)
        if error:
            errors.append((clip_file, error))
            continue
        if clip_file and selected_count > 0:
            completed.add(clip_file)

    missing = sorted(expected - completed)
    if missing:
        return (
            False,
            f"Keyframe sampling incomplete. Missing {len(missing)} clips, "
            f"e.g. {missing[:5]}.",
        )
    if errors:
        return False, f"Sampler errors exist, e.g. {errors[:3]}."

    return True, f"sampling_summary.jsonl covers all {len(expected)} clips."


def manifest_is_complete(manifest_path: Path, clips_dir: Path) -> Tuple[bool, str]:
    expected = clip_file_set_from_clips(clips_dir)
    if not expected:
        return False, "No clip mp4 files found."
    if not manifest_path.exists():
        return False, "clip_manifest.jsonl does not exist."

    records = read_jsonl(manifest_path)
    if not records:
        return False, "clip_manifest.jsonl is empty or invalid."

    completed = {
        record.get("clip_file")
        for record in records
        if record.get("clip_file")
        and int(record.get("selected_frame_count", 0) or 0) > 0
    }

    missing = sorted(expected - completed)
    if missing:
        return (
            False,
            f"Manifest incomplete. Missing {len(missing)} clips, e.g. {missing[:5]}.",
        )

    return True, f"clip_manifest.jsonl covers all {len(expected)} clips."


def run_sampler_if_needed(
    python_exe: str,
    model_dir: Path,
    clips_dir: Path,
    keyframes_dir: Path,
    num_frames: int,
    candidate_fps: float,
    black_threshold: float,
    min_gap_sec: float,
    force_keyframes: bool,
) -> None:
    summary_path = keyframes_dir / "sampling_summary.jsonl"
    complete, reason = summary_is_complete(summary_path, clips_dir)
    print("\nKeyframe sampling status:", reason)

    if complete and not force_keyframes:
        print("Skip keyframe sampling. Existing selected keyframes will be reused.")
        return

    if force_keyframes and keyframes_dir.exists():
        print("Force keyframe regeneration. Removing old keyframes:", keyframes_dir)
        shutil.rmtree(keyframes_dir)

    sampler_script = model_dir / "content_aware_sampler.py"
    if not sampler_script.exists():
        raise FileNotFoundError(f"Missing script: {sampler_script}")

    keyframes_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(sampler_script),
        "--input",
        str(clips_dir),
        "--output",
        str(keyframes_dir),
        "--num-frames",
        str(num_frames),
        "--candidate-fps",
        str(candidate_fps),
        "--black-threshold",
        str(black_threshold),
        "--min-gap-sec",
        str(min_gap_sec),
    ]
    run_cmd(cmd, cwd=model_dir)

    complete_after, reason_after = summary_is_complete(summary_path, clips_dir)
    print("Keyframe sampling status after run:", reason_after)
    if not complete_after:
        raise RuntimeError("Keyframe sampling did not complete successfully.")


def run_manifest_if_needed(
    python_exe: str,
    model_dir: Path,
    clips_dir: Path,
    keyframes_dir: Path,
    force_manifest: bool,
) -> Path:
    manifest_script = model_dir / "make_clip_manifest.py"
    if not manifest_script.exists():
        raise FileNotFoundError(f"Missing script: {manifest_script}")

    summary_path = keyframes_dir / "sampling_summary.jsonl"
    output_jsonl = keyframes_dir / "clip_manifest.jsonl"
    output_csv = keyframes_dir / "clip_manifest.csv"

    complete, reason = manifest_is_complete(output_jsonl, clips_dir)
    print("\nManifest status:", reason)

    if complete and not force_manifest:
        print("Skip manifest generation. Existing clip_manifest.jsonl will be reused.")
        return output_jsonl

    if not summary_path.exists():
        raise FileNotFoundError(f"Sampler summary not found: {summary_path}")

    cmd = [
        python_exe,
        str(manifest_script),
        "--summary",
        str(summary_path),
        "--output-jsonl",
        str(output_jsonl),
        "--output-csv",
        str(output_csv),
    ]
    run_cmd(cmd, cwd=model_dir)

    complete_after, reason_after = manifest_is_complete(output_jsonl, clips_dir)
    print("Manifest status after run:", reason_after)
    if not complete_after:
        raise RuntimeError("Manifest generation did not complete successfully.")

    return output_jsonl


def parse_clip_ids(text: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("No clip IDs parsed.")
    return values


def run_standalone_worker(
    *,
    python_exe: str,
    model_dir: Path,
    worker_script: Path,
    manifest_path: Path,
    output_dir: Path,
    run_mode: str,
    test_clip_ids: List[int],
    shard_index: int,
    num_shards: int,
    dtype: str,
    force: bool,
    dry_run: bool,
    state_backend: str,
    state_db_path: Optional[Path],
    dynamodb_table: Optional[str],
    aws_region: Optional[str],
    aws_profile: Optional[str],
    run_id: str,
    lease_seconds: int,
    max_attempts: int,
) -> None:
    if not worker_script.exists():
        raise FileNotFoundError(f"Standalone worker not found: {worker_script}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    cmd = [
        python_exe,
        str(worker_script),
        "--model-path",
        str(model_dir),
        "--manifest-path",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
        "--dtype",
        dtype,
        "--state-backend",
        state_backend,
        "--run-id",
        run_id,
        "--lease-seconds",
        str(lease_seconds),
        "--max-attempts",
        str(max_attempts),
    ]

    if state_backend == "sqlite":
        if state_db_path is None:
            raise ValueError("state_db_path is required for sqlite state.")
        cmd.extend(["--state-db-path", str(state_db_path)])
    elif state_backend == "dynamodb":
        if not dynamodb_table:
            raise ValueError("dynamodb_table is required for dynamodb state.")
        cmd.extend(["--dynamodb-table", dynamodb_table])
        if aws_region:
            cmd.extend(["--aws-region", aws_region])
        if aws_profile:
            cmd.extend(["--aws-profile", aws_profile])

    if run_mode == "test":
        cmd.extend(
            [
                "--clip-ids",
                ",".join(str(value) for value in test_clip_ids),
            ]
        )

    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")

    run_cmd(cmd, cwd=model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an episode and invoke one standalone VideoThinker inference shard "
            "without modifying source code at runtime."
        )
    )

    parser.add_argument("--video", default=None, help="Original episode video path.")
    parser.add_argument("--episode", required=True, help="Episode ID, for example ep02.")
    parser.add_argument(
        "--existing-manifest",
        default=None,
        help=(
            "Use an existing manifest and skip clip, keyframe, and manifest preparation. "
            "This is recommended for the current optical-flow manifest."
        ),
    )
    parser.add_argument("--work-root", default=DEFAULT_WORK_ROOT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--worker-script",
        default=None,
        help=(
            "Standalone worker path. Defaults to "
            "<model-dir>\\src\\workers\\inference_worker.py."
        ),
    )
    parser.add_argument(
        "--worker-output-dir",
        default=None,
        help="Optional explicit output directory for this worker run.",
    )

    parser.add_argument("--segment-seconds", type=float, default=20.0)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--candidate-fps", type=float, default=2.0)
    parser.add_argument("--black-threshold", type=float, default=18.0)
    parser.add_argument("--min-gap-sec", type=float, default=1.0)

    parser.add_argument("--run-mode", choices=["test", "all"], default="test")
    parser.add_argument(
        "--test-clip-ids",
        default="1,3,14,20,31,40,48,52,60,72",
    )
    parser.add_argument("--max-clips", type=int, default=None)

    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--dry-run-infer",
        action="store_true",
        help="Validate worker assignment without loading the model.",
    )

    parser.add_argument(
        "--state-backend",
        choices=["output", "sqlite", "dynamodb"],
        default="sqlite",
        help=(
            "Worker state backend. sqlite is for local concurrency tests; "
            "dynamodb is for shared AWS execution."
        ),
    )
    parser.add_argument(
        "--state-db-path",
        default=None,
        help=(
            "SQLite state database path. Defaults to "
            "<worker-output-dir>\\job_state.db."
        ),
    )
    parser.add_argument(
        "--dynamodb-table",
        default="videothinker-clip-jobs",
        help="DynamoDB table used when --state-backend dynamodb.",
    )
    parser.add_argument(
        "--aws-region",
        default="us-east-2",
        help="AWS region for DynamoDB.",
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
        help=(
            "Optional local AWS CLI profile. Omit in AWS Batch so the job role "
            "is used."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Logical run identifier shared by every shard. Defaults to a "
            "deterministic value derived from episode, mode, dtype, and shard count."
        ),
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=7200,
        help="How long one worker owns a claimed clip before it may be reclaimed.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per clip before FINAL_FAILED.",
    )

    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--force-keyframes", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument(
        "--force-infer-from-start",
        action="store_true",
        help="Delete only this shard's prior worker output and rerun it.",
    )
    parser.add_argument("--copy-split", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be between 0 and {args.num_shards - 1}."
        )
    if args.existing_manifest is None and not args.video:
        raise ValueError(
            "--video is required unless --existing-manifest is supplied."
        )
    if args.lease_seconds < 1:
        raise ValueError("--lease-seconds must be positive.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive.")
    if args.state_backend != "output" and args.force_infer_from_start:
        raise ValueError(
            "--force-infer-from-start is not supported with shared state. "
            "Use a new --run-id for a clean multi-worker rerun."
        )
    if args.state_backend == "dynamodb" and not args.dynamodb_table:
        raise ValueError(
            "--dynamodb-table is required for dynamodb state."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    python_exe = sys.executable
    model_dir = Path(args.model_dir)
    work_root = Path(args.work_root)
    episode_dir = work_root / args.episode
    clips_dir = episode_dir / "clips"
    keyframes_dir = episode_dir / "keyframes"
    outputs_dir = episode_dir / "outputs"

    worker_script = (
        Path(args.worker_script)
        if args.worker_script
        else model_dir / DEFAULT_WORKER_RELATIVE_PATH
    )

    episode_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    test_clip_ids = parse_clip_ids(args.test_clip_ids)

    print("Episode keyframe pipeline v5")
    print("Python:", python_exe)
    print("Episode:", args.episode)
    print("Episode dir:", episode_dir)
    print("Model dir:", model_dir)
    print("Worker script:", worker_script)

    if args.existing_manifest:
        manifest_path = Path(args.existing_manifest)
        print("Using existing manifest:", manifest_path)
        print("Skipping clip, keyframe, and manifest preparation.")
    else:
        video_path = Path(args.video)
        print("Video:", video_path)

        split_video_exact(
            video_path=video_path,
            clips_dir=clips_dir,
            episode=args.episode,
            segment_seconds=args.segment_seconds,
            force_clips=args.force_clips,
            max_clips=args.max_clips,
            reencode=not args.copy_split,
        )

        run_sampler_if_needed(
            python_exe=python_exe,
            model_dir=model_dir,
            clips_dir=clips_dir,
            keyframes_dir=keyframes_dir,
            num_frames=args.num_frames,
            candidate_fps=args.candidate_fps,
            black_threshold=args.black_threshold,
            min_gap_sec=args.min_gap_sec,
            force_keyframes=args.force_keyframes,
        )

        manifest_path = run_manifest_if_needed(
            python_exe=python_exe,
            model_dir=model_dir,
            clips_dir=clips_dir,
            keyframes_dir=keyframes_dir,
            force_manifest=args.force_manifest,
        )

    if args.worker_output_dir:
        worker_output_dir = Path(args.worker_output_dir)
    else:
        run_name = (
            f"{args.episode}_{args.run_mode}_{args.dtype}_"
            f"{args.num_shards:03d}shards"
        )
        worker_output_dir = outputs_dir / "worker_runs" / run_name

    run_id = (
        args.run_id
        or (
            f"{args.episode}-{args.run_mode}-{args.dtype}-"
            f"{args.num_shards}shards"
        )
    )

    state_db_path = (
        Path(args.state_db_path)
        if args.state_db_path
        else worker_output_dir / "job_state.db"
    )

    print("State backend:", args.state_backend)
    print("Run ID:", run_id)
    if args.state_backend == "sqlite":
        print("State database:", state_db_path)
    elif args.state_backend == "dynamodb":
        print("DynamoDB table:", args.dynamodb_table)
        print("AWS region:", args.aws_region)
        print("AWS profile:", args.aws_profile or "default credential chain")
    print("Lease seconds:", args.lease_seconds)
    print("Max attempts:", args.max_attempts)

    if args.skip_infer:
        print("\nSkipping inference.")
    else:
        run_standalone_worker(
            python_exe=python_exe,
            model_dir=model_dir,
            worker_script=worker_script,
            manifest_path=manifest_path,
            output_dir=worker_output_dir,
            run_mode=args.run_mode,
            test_clip_ids=test_clip_ids,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            dtype=args.dtype,
            force=args.force_infer_from_start,
            dry_run=args.dry_run_infer,
            state_backend=args.state_backend,
            state_db_path=state_db_path,
            dynamodb_table=args.dynamodb_table,
            aws_region=args.aws_region,
            aws_profile=args.aws_profile,
            run_id=run_id,
            lease_seconds=args.lease_seconds,
            max_attempts=args.max_attempts,
        )

    print("\nPipeline finished.")
    print("Manifest:", manifest_path)
    print("Worker output dir:", worker_output_dir)
    print("State backend:", args.state_backend)
    print("Run ID:", run_id)
    if args.state_backend == "sqlite":
        print("State database:", state_db_path)
    elif args.state_backend == "dynamodb":
        print("DynamoDB table:", args.dynamodb_table)
        print("AWS region:", args.aws_region)
    print("Shard:", f"{args.shard_index}/{args.num_shards}")


if __name__ == "__main__":
    main()