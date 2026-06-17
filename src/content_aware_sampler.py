"""
content_aware_sampler.py

Purpose:
    Select informative keyframes from video clips for long-video event extraction.

Why:
    Uniform video frame sampling can miss short but important events. This script
    samples candidate frames, scores them using lightweight OpenCV signals, and
    selects a fixed number of visually informative frames in chronological order.

Main signals:
    - Motion score: grayscale frame difference
    - Scene-change score: color histogram difference
    - Sharpness score: Laplacian variance
    - Brightness score: used to avoid black / near-black frames
    - Temporal diversity: avoid selecting many adjacent frames

Typical usage:
    python content_aware_sampler.py --input "D:\projects\longvideo\clips" --output "D:\projects\longvideo\keyframes" --num-frames 12

Single clip:
    python content_aware_sampler.py --input "D:\projects\longvideo\clips\ep01_clip_0052.mp4" --output "D:\projects\longvideo\keyframes" --num-frames 12

Dependencies:
    pip install opencv-python numpy
"""

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameCandidate:
    frame_index: int
    timestamp_sec: float
    brightness: float
    sharpness: float
    motion_score: float
    hist_diff_score: float
    combined_score: float
    selected_reason: str = ""


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_motion_score(prev_frame: Optional[np.ndarray], frame: np.ndarray) -> float:
    if prev_frame is None:
        return 0.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to reduce noise and cost.
    prev_gray = cv2.resize(prev_gray, (160, 90))
    curr_gray = cv2.resize(curr_gray, (160, 90))

    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def compute_hist_diff(prev_frame: Optional[np.ndarray], frame: np.ndarray) -> float:
    if prev_frame is None:
        return 0.0

    # HSV histogram is usually more robust for scene-change detection.
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])

    cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

    # Correlation is higher when frames are similar. Convert to difference.
    corr = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    return float(max(0.0, 1.0 - corr))


def normalize_values(values: List[float]) -> List[float]:
    if not values:
        return []

    arr = np.array(values, dtype=np.float32)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if math.isclose(max_v, min_v):
        return [0.0 for _ in values]

    return ((arr - min_v) / (max_v - min_v)).astype(float).tolist()


def is_near_black(candidate: FrameCandidate, black_threshold: float) -> bool:
    return candidate.brightness < black_threshold


def sample_candidates(
    video_path: Path,
    candidate_fps: float,
    black_threshold: float,
) -> Tuple[List[FrameCandidate], List[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / source_fps if source_fps and source_fps > 0 else 0.0

    if source_fps <= 0:
        source_fps = 25.0

    step = max(1, int(round(source_fps / candidate_fps)))

    raw_candidates = []
    raw_frames = []

    prev_sampled_frame = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            timestamp_sec = frame_index / source_fps

            brightness = compute_brightness(frame)
            sharpness = compute_sharpness(frame)
            motion_score = compute_motion_score(prev_sampled_frame, frame)
            hist_diff_score = compute_hist_diff(prev_sampled_frame, frame)

            raw_candidates.append(
                FrameCandidate(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    brightness=brightness,
                    sharpness=sharpness,
                    motion_score=motion_score,
                    hist_diff_score=hist_diff_score,
                    combined_score=0.0,
                )
            )
            raw_frames.append(frame.copy())
            prev_sampled_frame = frame.copy()

        frame_index += 1

    cap.release()

    if not raw_candidates:
        raise RuntimeError(f"No candidate frames extracted from: {video_path}")

    # Normalize each signal before combining.
    motion_norm = normalize_values([c.motion_score for c in raw_candidates])
    hist_norm = normalize_values([c.hist_diff_score for c in raw_candidates])
    sharp_norm = normalize_values([c.sharpness for c in raw_candidates])
    bright_norm = normalize_values([c.brightness for c in raw_candidates])

    for idx, c in enumerate(raw_candidates):
        # Penalize near-black frames unless almost the entire clip is black.
        black_penalty = 0.45 if is_near_black(c, black_threshold) else 0.0

        # Weights are intentionally simple and interpretable.
        # Motion and scene changes matter most for event detection.
        c.combined_score = (
            0.40 * motion_norm[idx]
            + 0.35 * hist_norm[idx]
            + 0.15 * sharp_norm[idx]
            + 0.10 * bright_norm[idx]
            - black_penalty
        )

    metadata = {
        "video_path": str(video_path),
        "source_fps": source_fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "candidate_fps": candidate_fps,
        "candidate_count": len(raw_candidates),
        "black_threshold": black_threshold,
    }

    return raw_candidates, raw_frames, metadata


def temporal_distance_ok(
    candidate: FrameCandidate,
    selected: List[FrameCandidate],
    min_gap_sec: float,
) -> bool:
    for existing in selected:
        if abs(candidate.timestamp_sec - existing.timestamp_sec) < min_gap_sec:
            return False
    return True


def find_nearest_candidate(
    candidates: List[FrameCandidate],
    target_time: float,
    black_threshold: float,
    prefer_non_black: bool = True,
) -> FrameCandidate:
    sorted_candidates = sorted(candidates, key=lambda c: abs(c.timestamp_sec - target_time))

    if prefer_non_black:
        for c in sorted_candidates:
            if not is_near_black(c, black_threshold):
                return c

    return sorted_candidates[0]


def select_keyframes(
    candidates: List[FrameCandidate],
    num_frames: int,
    duration_sec: float,
    black_threshold: float,
    min_gap_sec: float,
) -> List[FrameCandidate]:
    if len(candidates) <= num_frames:
        selected = list(candidates)
        for c in selected:
            c.selected_reason = "all_candidates_kept"
        return sorted(selected, key=lambda c: c.timestamp_sec)

    selected: List[FrameCandidate] = []

    # Always preserve rough temporal context: beginning, middle, end.
    anchors = [
        (0.0, "temporal_anchor_start"),
        (duration_sec * 0.5, "temporal_anchor_middle"),
        (max(0.0, duration_sec - 0.05), "temporal_anchor_end"),
    ]

    for target_time, reason in anchors:
        c = find_nearest_candidate(candidates, target_time, black_threshold)
        if c not in selected:
            c.selected_reason = reason
            selected.append(c)

    # Pick top-scoring frames while enforcing temporal diversity.
    ranked = sorted(candidates, key=lambda c: c.combined_score, reverse=True)

    for c in ranked:
        if len(selected) >= num_frames:
            break
        if c in selected:
            continue
        if temporal_distance_ok(c, selected, min_gap_sec):
            c.selected_reason = "high_content_score"
            selected.append(c)

    # If min_gap was too strict, fill remaining slots without it.
    for c in ranked:
        if len(selected) >= num_frames:
            break
        if c in selected:
            continue
        c.selected_reason = "fill_remaining_best_score"
        selected.append(c)

    return sorted(selected, key=lambda c: c.timestamp_sec)


def save_selected_frames(
    video_path: Path,
    candidates: List[FrameCandidate],
    frames: List[np.ndarray],
    selected: List[FrameCandidate],
    output_root: Path,
    jpeg_quality: int,
) -> dict:
    clip_name = video_path.stem
    clip_output_dir = output_root / clip_name
    safe_mkdir(clip_output_dir)

    # Map frame_index to actual image array.
    frame_by_index = {
        c.frame_index: frames[i]
        for i, c in enumerate(candidates)
    }

    selected_records = []

    for order, c in enumerate(selected):
        frame = frame_by_index[c.frame_index]

        filename = f"{order:02d}_t{c.timestamp_sec:07.3f}_f{c.frame_index}.jpg"
        output_path = clip_output_dir / filename

        cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )

        record = asdict(c)
        record["timestamp"] = format_timestamp(c.timestamp_sec)
        record["image_file"] = filename
        record["image_path"] = str(output_path)
        selected_records.append(record)

    return {
        "clip_file": video_path.name,
        "clip_stem": video_path.stem,
        "output_dir": str(clip_output_dir),
        "selected_count": len(selected_records),
        "selected_frames": selected_records,
    }


def write_contact_sheet(
    selected_info: dict,
    output_root: Path,
    thumb_width: int = 240,
    label_height: int = 35,
) -> Optional[str]:
    selected_frames = selected_info["selected_frames"]
    if not selected_frames:
        return None

    images = []
    labels = []

    for item in selected_frames:
        img = cv2.imread(item["image_path"])
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_width / max(1, w)
        thumb_height = int(h * scale)
        thumb = cv2.resize(img, (thumb_width, thumb_height))

        label = f'{item["timestamp"]} | {item["selected_reason"]}'
        images.append(thumb)
        labels.append(label)

    if not images:
        return None

    max_h = max(img.shape[0] for img in images)
    tile_h = max_h + label_height
    tile_w = thumb_width

    tiles = []
    for img, label in zip(images, labels):
        canvas = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img

        cv2.putText(
            canvas,
            label[:34],
            (8, max_h + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        tiles.append(canvas)

    # Build a simple horizontal sheet. For 12 frames this is still readable.
    sheet = np.hstack(tiles)

    sheet_path = output_root / f'{selected_info["clip_stem"]}_contact_sheet.jpg'
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(sheet_path)


def process_video(
    video_path: Path,
    output_root: Path,
    num_frames: int,
    candidate_fps: float,
    black_threshold: float,
    min_gap_sec: float,
    jpeg_quality: int,
    make_contact_sheet: bool,
) -> dict:
    candidates, frames, meta = sample_candidates(
        video_path=video_path,
        candidate_fps=candidate_fps,
        black_threshold=black_threshold,
    )

    selected = select_keyframes(
        candidates=candidates,
        num_frames=num_frames,
        duration_sec=meta["duration_sec"],
        black_threshold=black_threshold,
        min_gap_sec=min_gap_sec,
    )

    selected_info = save_selected_frames(
        video_path=video_path,
        candidates=candidates,
        frames=frames,
        selected=selected,
        output_root=output_root,
        jpeg_quality=jpeg_quality,
    )

    result = {
        "sampler_version": "content_aware_sampler_v1",
        "selection_strategy": {
            "num_frames": num_frames,
            "candidate_fps": candidate_fps,
            "min_gap_sec": min_gap_sec,
            "signals": {
                "motion_score": "mean grayscale frame difference",
                "hist_diff_score": "HSV histogram difference",
                "sharpness": "Laplacian variance",
                "brightness": "mean grayscale brightness",
            },
            "combined_score": "0.40 motion + 0.35 hist_diff + 0.15 sharpness + 0.10 brightness - black penalty",
        },
        "video_metadata": meta,
        **selected_info,
    }

    if make_contact_sheet:
        sheet_path = write_contact_sheet(selected_info, output_root / video_path.stem)
        result["contact_sheet_path"] = sheet_path

    # Save per-clip metadata.
    clip_output_dir = Path(selected_info["output_dir"])
    metadata_path = clip_output_dir / "selected_frames.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    result["metadata_path"] = str(metadata_path)

    return result


def discover_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mp4":
            raise ValueError(f"Input file is not an mp4: {input_path}")
        return [input_path]

    if input_path.is_dir():
        videos = sorted(input_path.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No mp4 files found in directory: {input_path}")
        return videos

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Content-aware keyframe sampler for long-video event extraction."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\projects\longvideo\clips",
        help="Input mp4 file or directory containing mp4 clips.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\projects\longvideo\keyframes",
        help="Output directory for selected keyframes and metadata.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=12,
        help="Number of keyframes to select per clip.",
    )
    parser.add_argument(
        "--candidate-fps",
        type=float,
        default=2.0,
        help="Candidate frame sampling rate before content-aware selection.",
    )
    parser.add_argument(
        "--black-threshold",
        type=float,
        default=18.0,
        help="Frames below this mean brightness are treated as near-black.",
    )
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=1.0,
        help="Minimum preferred time gap between selected frames.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for saved frames.",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Disable contact sheet generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    safe_mkdir(output_root)

    videos = discover_videos(input_path)

    print("Content-aware keyframe sampling")
    print("Input:", input_path)
    print("Output:", output_root)
    print("Video count:", len(videos))
    print("Frames per clip:", args.num_frames)
    print("Candidate FPS:", args.candidate_fps)

    all_results = []

    for idx, video_path in enumerate(videos, start=1):
        print(f"\nProcessing {idx}/{len(videos)}: {video_path.name}")

        try:
            result = process_video(
                video_path=video_path,
                output_root=output_root,
                num_frames=args.num_frames,
                candidate_fps=args.candidate_fps,
                black_threshold=args.black_threshold,
                min_gap_sec=args.min_gap_sec,
                jpeg_quality=args.jpeg_quality,
                make_contact_sheet=not args.no_contact_sheet,
            )

            all_results.append(result)

            print("Selected frames:", result["selected_count"])
            print("Metadata:", result["metadata_path"])
            if result.get("contact_sheet_path"):
                print("Contact sheet:", result["contact_sheet_path"])

        except Exception as e:
            error_result = {
                "sampler_version": "content_aware_sampler_v1",
                "clip_file": video_path.name,
                "error": repr(e),
            }
            all_results.append(error_result)
            print("Error:", repr(e))

    summary_path = output_root / "sampling_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\nDone.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
