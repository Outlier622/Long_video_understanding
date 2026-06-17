
"""
run_episode_keyframe_pipeline_v2.py

One-command, resumable pipeline for a new episode such as ep02.

Main behavior:
    - Clips are generated once. Existing clips are reused unless --force-clips is set.
    - Content-aware keyframes are generated once. Existing complete keyframes are reused unless --force-keyframes is set.
    - clip_manifest.jsonl is generated once. Existing complete manifest is reused unless --force-manifest is set.
    - Inference is resumable. Existing successful clip results are skipped unless --force-infer-from-start is set.

Pipeline:
    original episode video
    -> fixed-length clips
    -> content-aware selected keyframes
    -> clip_manifest.jsonl
    -> keyframe-based VideoThinker inference

Example test run:
    python .\run_episode_keyframe_pipeline_v2.py `
      --video "D:\projects\longvideo\raw\ep02.mp4" `
      --episode ep02 `
      --run-mode test

Example all-mode resume run:
    python .\run_episode_keyframe_pipeline_v2.py `
      --video "D:\projects\longvideo\raw\ep02.mp4" `
      --episode ep02 `
      --run-mode all

Regenerate keyframes only when needed:
    python .\run_episode_keyframe_pipeline_v2.py `
      --video "D:\projects\longvideo\raw\ep02.mp4" `
      --episode ep02 `
      --run-mode test `
      --force-keyframes `
      --force-manifest

Requirements:
    - ffmpeg and ffprobe in PATH.
    - content_aware_sampler.py in MODEL_DIR.
    - make_clip_manifest.py in MODEL_DIR.
    - batch_infer_clips_keyframes.py in MODEL_DIR.
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


DEFAULT_WORK_ROOT = r"D:\projects\longvideo\episodes"
DEFAULT_MODEL_DIR = r"D:\projects\VideoThinker-R1-3B"


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("\nRunning command:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def check_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"{name} was not found in PATH. Install FFmpeg and make sure {name}.exe is available in PowerShell."
        )


def get_video_duration_sec(video_path: Path) -> float:
    check_executable("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
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


def expected_clip_count(video_path: Path, segment_seconds: float, max_clips: Optional[int]) -> int:
    duration = get_video_duration_sec(video_path)
    count = int(math.ceil(duration / segment_seconds))
    if max_clips is not None:
        count = min(count, max_clips)
    return count


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
            # Re-encoding is slower but gives more accurate segment durations.
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video_path),
                "-t", f"{clip_duration:.3f}",
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "128k",
                "-reset_timestamps", "1",
                str(out_path),
            ]
        else:
            # Faster, but segment boundaries may follow keyframes.
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video_path),
                "-t", f"{clip_duration:.3f}",
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c", "copy",
                "-reset_timestamps", "1",
                str(out_path),
            ]

        print(f"\nCreating clip {clip_id:04d}: start={start:.3f}s duration={clip_duration:.3f}s")
        run_cmd(cmd)

    return total_clips


def read_jsonl(path: Path) -> List[dict]:
    records = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON in {path} line {line_num}; ignoring that line.")
    return records


def clip_file_set_from_clips(clips_dir: Path) -> Set[str]:
    return {p.name for p in list_clip_files(clips_dir)}


def summary_is_complete(summary_path: Path, clips_dir: Path) -> Tuple[bool, str]:
    expected = clip_file_set_from_clips(clips_dir)
    if not expected:
        return False, "No clip mp4 files found."

    if not summary_path.exists():
        return False, "sampling_summary.jsonl does not exist."

    records = read_jsonl(summary_path)
    if not records:
        return False, "sampling_summary.jsonl is empty or invalid."

    completed = set()
    errors = []

    for r in records:
        clip_file = r.get("clip_file")
        error = r.get("error")
        selected_count = int(r.get("selected_count", 0) or 0)
        if error:
            errors.append((clip_file, error))
            continue
        if clip_file and selected_count > 0:
            completed.add(clip_file)

    missing = sorted(expected - completed)

    if missing:
        return False, f"Keyframe sampling incomplete. Missing {len(missing)} clips, e.g. {missing[:5]}."

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

    completed = set()

    for r in records:
        clip_file = r.get("clip_file")
        selected_count = int(r.get("selected_frame_count", 0) or 0)
        if clip_file and selected_count > 0:
            completed.add(clip_file)

    missing = sorted(expected - completed)

    if missing:
        return False, f"Manifest incomplete. Missing {len(missing)} clips, e.g. {missing[:5]}."

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
        "--input", str(clips_dir),
        "--output", str(keyframes_dir),
        "--num-frames", str(num_frames),
        "--candidate-fps", str(candidate_fps),
        "--black-threshold", str(black_threshold),
        "--min-gap-sec", str(min_gap_sec),
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
        "--summary", str(summary_path),
        "--output-jsonl", str(output_jsonl),
        "--output-csv", str(output_csv),
    ]
    run_cmd(cmd, cwd=model_dir)

    complete_after, reason_after = manifest_is_complete(output_jsonl, clips_dir)
    print("Manifest status after run:", reason_after)
    if not complete_after:
        raise RuntimeError("Manifest generation did not complete successfully.")

    return output_jsonl


def replace_constant_string(source: str, name: str, new_value: str) -> str:
    pattern_path = rf'^{name}\s*=\s*Path\(r?["\'].*?["\']\)\s*$'
    replacement_path = f'{name} = Path(r"{new_value}")'
    source, count_path = re.subn(
        pattern_path,
        lambda m: replacement_path,
        source,
        flags=re.MULTILINE,
    )
    if count_path > 0:
        return source

    pattern_str = rf'^{name}\s*=\s*["\'].*?["\']\s*$'
    replacement_str = f'{name} = "{new_value}"'
    source, count_str = re.subn(
        pattern_str,
        lambda m: replacement_str,
        source,
        flags=re.MULTILINE,
    )
    if count_str > 0:
        return source

    raise RuntimeError(f"Could not replace constant: {name}")


def replace_test_clip_ids(source: str, clip_ids: List[int]) -> str:
    clip_ids_text = "{" + ", ".join(str(x) for x in clip_ids) + "}"
    pattern = r'^TEST_CLIP_IDS\s*=\s*\{.*?\}\s*$'
    replacement = f"TEST_CLIP_IDS = {clip_ids_text}"
    source, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError("Could not replace TEST_CLIP_IDS")
    return source


def successful_clip_ids_from_output(output_path: Path) -> Set[int]:
    completed = set()
    for r in read_jsonl(output_path):
        clip_id = r.get("clip_id")
        if clip_id is None:
            continue

        # Only skip successful clips. Failed or invalid clips will be retried.
        if r.get("error") is None and r.get("json_parse_ok") is True:
            completed.add(int(clip_id))

    return completed


def filter_manifest_for_run_mode(
    manifest_path: Path,
    run_mode: str,
    test_clip_ids: List[int],
) -> List[dict]:
    records = read_jsonl(manifest_path)
    if not records:
        raise RuntimeError(f"Manifest is empty or invalid: {manifest_path}")

    records = sorted(records, key=lambda r: int(r.get("clip_id", 0)))

    if run_mode == "test":
        test_set = set(test_clip_ids)
        return [r for r in records if int(r.get("clip_id", -1)) in test_set]

    if run_mode == "all":
        return records

    raise ValueError(f"Unsupported run mode: {run_mode}")


def prepare_remaining_manifest(
    manifest_path: Path,
    output_final_path: Path,
    keyframes_dir: Path,
    episode: str,
    run_mode: str,
    test_clip_ids: List[int],
    force_infer_from_start: bool,
) -> Tuple[Optional[Path], int, int]:
    if force_infer_from_start and output_final_path.exists():
        print("\nForce inference from start. Removing old output:", output_final_path)
        output_final_path.unlink()

    target_records = filter_manifest_for_run_mode(
        manifest_path=manifest_path,
        run_mode=run_mode,
        test_clip_ids=test_clip_ids,
    )

    completed_ids = successful_clip_ids_from_output(output_final_path)
    remaining_records = [
        r for r in target_records
        if int(r.get("clip_id", -1)) not in completed_ids
    ]

    print("\nInference resume status")
    print("Target clips:", len(target_records))
    print("Already completed successful clips:", len(completed_ids))
    print("Remaining clips:", len(remaining_records))

    if not remaining_records:
        return None, len(target_records), 0

    remaining_manifest = keyframes_dir / f"clip_manifest_{episode}_{run_mode}_remaining.jsonl"

    with remaining_manifest.open("w", encoding="utf-8") as f:
        for r in remaining_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return remaining_manifest, len(target_records), len(remaining_records)


def patch_inference_script(
    model_dir: Path,
    episode: str,
    remaining_manifest_path: Path,
    temp_output_path: Path,
    run_mode: str,
    test_clip_ids: List[int],
) -> Path:
    original_script = model_dir / "batch_infer_clips_keyframes.py"
    if not original_script.exists():
        raise FileNotFoundError(f"Missing script: {original_script}")

    source = original_script.read_text(encoding="utf-8")

    # Both output constants point to the same temporary output file. The pipeline
    # will append this temp output to the final resumable output afterward.
    source = replace_constant_string(source, "MANIFEST_PATH", str(remaining_manifest_path))
    source = replace_constant_string(source, "OUTPUT_TEST_PATH", str(temp_output_path))
    source = replace_constant_string(source, "OUTPUT_ALL_PATH", str(temp_output_path))
    source = replace_constant_string(source, "RUN_MODE", run_mode)
    source = replace_constant_string(source, "VERSION", f"{episode}_keyframe_v2_content_aware_sampling")
    source = replace_test_clip_ids(source, test_clip_ids)
    source = source.replace(
        'with output_path.open("w", encoding="utf-8") as f:',
        'with output_path.open("a", encoding="utf-8") as f:',
    )
    patched_script = model_dir / f"batch_infer_clips_keyframes_{episode}_{run_mode}_resume_autorun.py"
    patched_script.write_text(source, encoding="utf-8")

    print("\nCreated patched inference script:", patched_script)
    print("Remaining manifest:", remaining_manifest_path)
    print("Temp output:", temp_output_path)
    print("Run mode:", run_mode)

    return patched_script


def append_temp_output_to_final(temp_output_path: Path, final_output_path: Path) -> None:
    if not temp_output_path.exists():
        raise FileNotFoundError(f"Temp inference output not found: {temp_output_path}")

    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_output_path.open("r", encoding="utf-8") as src, final_output_path.open("a", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(line if line.endswith("\n") else line + "\n")

    print("\nAppended temp output to final output:")
    print("Temp:", temp_output_path)
    print("Final:", final_output_path)


def run_resumable_inference(
    python_exe: str,
    model_dir: Path,
    episode: str,
    manifest_path: Path,
    keyframes_dir: Path,
    outputs_dir: Path,
    run_mode: str,
    test_clip_ids: List[int],
    force_infer_from_start: bool,
) -> None:
    final_output_path = outputs_dir / (
        f"{episode}_results_keyframes_test.jsonl" if run_mode == "test"
        else f"{episode}_results_keyframes.jsonl"
    )

    remaining_manifest_path, target_count, remaining_count = prepare_remaining_manifest(
        manifest_path=manifest_path,
        output_final_path=final_output_path,
        keyframes_dir=keyframes_dir,
        episode=episode,
        run_mode=run_mode,
        test_clip_ids=test_clip_ids,
        force_infer_from_start=force_infer_from_start,
    )

    if remaining_manifest_path is None:
        print("Inference is already complete for this run mode. Nothing to do.")
        print("Final output:", final_output_path)
        return

    temp_output_path = final_output_path

    patched_script = patch_inference_script(
        model_dir=model_dir,
        episode=episode,
        remaining_manifest_path=remaining_manifest_path,
        temp_output_path=temp_output_path,
        run_mode=run_mode,
        test_clip_ids=test_clip_ids,
    )

    run_cmd([python_exe, str(patched_script)], cwd=model_dir)

    #append_temp_output_to_final(temp_output_path, final_output_path)

    completed_after = successful_clip_ids_from_output(final_output_path)
    print("\nInference final status")
    print("Target clips:", target_count)
    print("Successful clips in final output:", len(completed_after))
    print("Final output:", final_output_path)


def parse_clip_ids(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("No clip ids parsed.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare clips, one-time keyframes, manifest, and resumable keyframe inference for a new episode."
    )

    parser.add_argument("--video", required=True, help="Path to the original episode video.")
    parser.add_argument("--episode", required=True, help="Episode id, for example ep02.")
    parser.add_argument("--work-root", default=DEFAULT_WORK_ROOT, help="Root directory for episode outputs.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory containing VideoThinker model and scripts.")

    parser.add_argument("--segment-seconds", type=float, default=20.0, help="Clip length in seconds.")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of selected keyframes per clip.")
    parser.add_argument("--candidate-fps", type=float, default=2.0, help="Candidate frame sampling rate.")
    parser.add_argument("--black-threshold", type=float, default=18.0, help="Brightness threshold for near-black frames.")
    parser.add_argument("--min-gap-sec", type=float, default=1.0, help="Preferred minimum time gap between selected frames.")

    parser.add_argument("--run-mode", choices=["test", "all"], default="test", help="Inference run mode.")
    parser.add_argument("--test-clip-ids", default="1,3,14,20,31,40,48,52,60,73", help="Comma-separated clip ids for test mode.")
    parser.add_argument("--max-clips", type=int, default=None, help="Only split the first N clips. Useful for debugging.")

    parser.add_argument("--skip-infer", action="store_true", help="Prepare clips/keyframes/manifest only, do not run inference.")
    parser.add_argument("--force-clips", action="store_true", help="Delete and regenerate clips.")
    parser.add_argument("--force-keyframes", action="store_true", help="Delete and regenerate selected keyframes.")
    parser.add_argument("--force-manifest", action="store_true", help="Regenerate clip_manifest.jsonl.")
    parser.add_argument("--force-infer-from-start", action="store_true", help="Delete existing inference output and rerun from start.")

    parser.add_argument("--copy-split", action="store_true", help="Use ffmpeg stream copy for faster splitting. Default re-encodes for accurate segments.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = sys.executable

    video_path = Path(args.video)
    model_dir = Path(args.model_dir)
    work_root = Path(args.work_root)
    episode_dir = work_root / args.episode

    clips_dir = episode_dir / "clips"
    keyframes_dir = episode_dir / "keyframes"
    outputs_dir = episode_dir / "outputs"

    episode_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    test_clip_ids = parse_clip_ids(args.test_clip_ids)

    print("Episode keyframe pipeline v2")
    print("Python:", python_exe)
    print("Video:", video_path)
    print("Episode:", args.episode)
    print("Episode dir:", episode_dir)
    print("Model dir:", model_dir)

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

    if not args.skip_infer:
        run_resumable_inference(
            python_exe=python_exe,
            model_dir=model_dir,
            episode=args.episode,
            manifest_path=manifest_path,
            keyframes_dir=keyframes_dir,
            outputs_dir=outputs_dir,
            run_mode=args.run_mode,
            test_clip_ids=test_clip_ids,
            force_infer_from_start=args.force_infer_from_start,
        )
    else:
        print("\nSkipping inference.")

    print("\nPipeline finished.")
    print("Clips:", clips_dir)
    print("Keyframes:", keyframes_dir)
    print("Manifest:", manifest_path)
    print("Outputs:", outputs_dir)


if __name__ == "__main__":
    main()
