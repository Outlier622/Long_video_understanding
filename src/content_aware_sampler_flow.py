"""
content_aware_sampler_flow.py

Content-aware keyframe sampler with optical-flow-based motion scoring.

Purpose:
    Select high-information keyframes from video clips for long-video MLLM inference.

Compatible with:
    make_clip_manifest.py
    batch_infer_clips_keyframes.py

Main difference from previous sampler:
    Adds optical_flow_score into combined_score.

Recommended first run:
    python .\content_aware_sampler_flow.py `
      --input "D:\projects\longvideo\episodes\ep02\clips" `
      --output "D:\projects\longvideo\episodes\ep02\keyframes" `
      --num-frames 8 `
      --candidate-fps 2.0 `
      --use-optical-flow `
      --flow-weight 0.25 `
      --flow-resize-width 384 `
      --flow-mag-threshold 0.6 `
      --flow-clip-value 5.0 `
      --subtract-camera-motion `
      --ignore-bottom-ratio 0.12
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class CandidateFrame:
    frame_index: int
    timestamp_sec: float
    timestamp: str
    image: np.ndarray

    brightness_raw: float = 0.0
    sharpness_raw: float = 0.0
    frame_diff_raw: float = 0.0
    hist_diff_raw: float = 0.0
    flow_mean_raw: float = 0.0
    flow_p90_raw: float = 0.0
    flow_motion_area_raw: float = 0.0
    flow_score_raw: float = 0.0

    brightness_score: float = 0.0
    sharpness_score: float = 0.0
    motion_score: float = 0.0
    hist_diff_score: float = 0.0
    optical_flow_score: float = 0.0
    combined_score: float = 0.0

    selected_reason: str = ""


def format_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def normalize_values(values: List[float], clip_value: Optional[float] = None) -> List[float]:
    if not values:
        return []

    arr = np.array(values, dtype=np.float32)

    if clip_value is not None and clip_value > 0:
        arr = np.clip(arr, 0, clip_value)
        return (arr / clip_value).clip(0, 1).astype(float).tolist()

    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if max_v - min_v < 1e-8:
        return [0.0 for _ in values]

    return ((arr - min_v) / (max_v - min_v)).clip(0, 1).astype(float).tolist()


def brightness_score_from_raw(brightness: float, black_threshold: float, ideal_brightness: float) -> float:
    if brightness < black_threshold:
        return 0.0
    distance = abs(brightness - ideal_brightness)
    score = 1.0 - min(distance / max(ideal_brightness, 1e-6), 1.0)
    return float(max(score, 0.0))


def resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    if w == width:
        return frame
    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def crop_ignore_bottom(gray: np.ndarray, ignore_bottom_ratio: float) -> np.ndarray:
    if ignore_bottom_ratio <= 0:
        return gray
    h = gray.shape[0]
    keep_h = int(round(h * (1.0 - ignore_bottom_ratio)))
    keep_h = max(1, min(h, keep_h))
    return gray[:keep_h, :]


def compute_laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_hist_diff(prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    hist_curr = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)
    return float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA))


def compute_frame_diff(prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def compute_optical_flow_score(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    flow_resize_width: int,
    flow_mag_threshold: float,
    flow_clip_value: float,
    subtract_camera_motion: bool,
    ignore_bottom_ratio: float,
) -> Tuple[float, float, float, float]:
    if prev_gray is None:
        return 0.0, 0.0, 0.0, 0.0

    prev = resize_keep_aspect(prev_gray, flow_resize_width)
    curr = resize_keep_aspect(curr_gray, flow_resize_width)

    prev = crop_ignore_bottom(prev, ignore_bottom_ratio)
    curr = crop_ignore_bottom(curr, ignore_bottom_ratio)

    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)

    flow = cv2.calcOpticalFlowFarneback(
        prev,
        curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    dx = flow[..., 0]
    dy = flow[..., 1]

    if subtract_camera_motion:
        dx = dx - np.median(dx)
        dy = dy - np.median(dy)

    mag = np.sqrt(dx * dx + dy * dy)

    mean_flow = float(np.mean(mag))
    p90_flow = float(np.percentile(mag, 90))
    motion_area_ratio = float(np.mean(mag > flow_mag_threshold))

    p90_norm = min(p90_flow / max(flow_clip_value, 1e-6), 1.0)
    mean_norm = min(mean_flow / max(flow_clip_value, 1e-6), 1.0)

    flow_score_raw = 0.5 * p90_norm + 0.3 * mean_norm + 0.2 * motion_area_ratio
    flow_score_raw = float(max(0.0, min(flow_score_raw, 1.0)))

    return flow_score_raw, mean_flow, p90_flow, motion_area_ratio


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "duration": format_time(duration_sec),
    }


def extract_candidate_frames(
    video_path: Path,
    candidate_fps: float,
    black_threshold: float,
    ideal_brightness: float,
    use_optical_flow: bool,
    flow_resize_width: int,
    flow_mag_threshold: float,
    flow_clip_value: float,
    subtract_camera_motion: bool,
    ignore_bottom_ratio: float,
) -> Tuple[List[CandidateFrame], Dict[str, Any]]:
    metadata = get_video_metadata(video_path)
    fps = metadata["fps"]
    frame_count = metadata["frame_count"]

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    step = max(1, int(round(fps / candidate_fps)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    candidates: List[CandidateFrame] = []
    prev_gray_for_scores = None
    current_index = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if current_index % step != 0:
            current_index += 1
            continue

        timestamp_sec = current_index / fps
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_for_scores = crop_ignore_bottom(gray, ignore_bottom_ratio)

        brightness_raw = float(np.mean(gray))
        sharpness_raw = compute_laplacian_sharpness(gray_for_scores)
        frame_diff_raw = compute_frame_diff(prev_gray_for_scores, gray_for_scores)
        hist_diff_raw = compute_hist_diff(prev_gray_for_scores, gray_for_scores)

        if use_optical_flow:
            flow_score_raw, flow_mean, flow_p90, flow_motion_area = compute_optical_flow_score(
                prev_gray=prev_gray_for_scores,
                curr_gray=gray_for_scores,
                flow_resize_width=flow_resize_width,
                flow_mag_threshold=flow_mag_threshold,
                flow_clip_value=flow_clip_value,
                subtract_camera_motion=subtract_camera_motion,
                ignore_bottom_ratio=0.0,
            )
        else:
            flow_score_raw, flow_mean, flow_p90, flow_motion_area = 0.0, 0.0, 0.0, 0.0

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        candidates.append(
            CandidateFrame(
                frame_index=current_index,
                timestamp_sec=timestamp_sec,
                timestamp=format_time(timestamp_sec),
                image=frame_rgb,
                brightness_raw=brightness_raw,
                sharpness_raw=sharpness_raw,
                frame_diff_raw=frame_diff_raw,
                hist_diff_raw=hist_diff_raw,
                flow_mean_raw=flow_mean,
                flow_p90_raw=flow_p90,
                flow_motion_area_raw=flow_motion_area,
                flow_score_raw=flow_score_raw,
            )
        )

        prev_gray_for_scores = gray_for_scores
        current_index += 1

    cap.release()
    return candidates, metadata


def assign_scores(
    candidates: List[CandidateFrame],
    black_threshold: float,
    ideal_brightness: float,
    motion_clip_value: float,
    scene_clip_value: float,
    sharpness_clip_value: float,
    scene_weight: float,
    motion_weight: float,
    flow_weight: float,
    sharpness_weight: float,
    brightness_weight: float,
) -> None:
    if not candidates:
        return

    motion_scores = normalize_values([c.frame_diff_raw for c in candidates], clip_value=motion_clip_value)
    hist_scores = normalize_values([c.hist_diff_raw for c in candidates], clip_value=scene_clip_value)
    sharpness_scores = normalize_values([c.sharpness_raw for c in candidates], clip_value=sharpness_clip_value)

    for idx, c in enumerate(candidates):
        c.motion_score = motion_scores[idx]
        c.hist_diff_score = hist_scores[idx]
        c.sharpness_score = sharpness_scores[idx]
        c.brightness_score = brightness_score_from_raw(
            brightness=c.brightness_raw,
            black_threshold=black_threshold,
            ideal_brightness=ideal_brightness,
        )
        c.optical_flow_score = c.flow_score_raw

        c.combined_score = (
            scene_weight * c.hist_diff_score
            + motion_weight * c.motion_score
            + flow_weight * c.optical_flow_score
            + sharpness_weight * c.sharpness_score
            + brightness_weight * c.brightness_score
        )

        if c.brightness_raw < black_threshold:
            c.combined_score *= 0.2


def pick_anchor_frame(candidates: List[CandidateFrame], target_sec: float, used_indices: set, reason: str) -> Optional[CandidateFrame]:
    available = [c for c in candidates if c.frame_index not in used_indices]
    if not available:
        return None
    selected = min(available, key=lambda c: abs(c.timestamp_sec - target_sec))
    selected.selected_reason = reason
    used_indices.add(selected.frame_index)
    return selected


def select_keyframes(
    candidates: List[CandidateFrame],
    num_frames: int,
    duration_sec: float,
    min_gap_sec: float,
    anchor_start: bool,
    anchor_middle: bool,
    anchor_end: bool,
) -> List[CandidateFrame]:
    if not candidates:
        return []

    if len(candidates) <= num_frames:
        selected = sorted(candidates, key=lambda c: c.timestamp_sec)
        for c in selected:
            if not c.selected_reason:
                c.selected_reason = "all_candidates_used"
        return selected

    selected: List[CandidateFrame] = []
    used_indices = set()

    anchors = []
    if anchor_start:
        anchors.append((0.0, "temporal_anchor_start"))
    if anchor_middle:
        anchors.append((duration_sec / 2.0, "temporal_anchor_middle"))
    if anchor_end:
        anchors.append((max(0.0, duration_sec - 0.2), "temporal_anchor_end"))

    for target_sec, reason in anchors:
        if len(selected) >= num_frames:
            break
        anchor = pick_anchor_frame(candidates, target_sec, used_indices, reason)
        if anchor is not None:
            selected.append(anchor)

    remaining = sorted(candidates, key=lambda c: c.combined_score, reverse=True)

    for candidate in remaining:
        if len(selected) >= num_frames:
            break
        if candidate.frame_index in used_indices:
            continue
        too_close = any(abs(candidate.timestamp_sec - s.timestamp_sec) < min_gap_sec for s in selected)
        if too_close:
            continue
        candidate.selected_reason = "high_content_score"
        selected.append(candidate)
        used_indices.add(candidate.frame_index)

    if len(selected) < num_frames:
        for candidate in remaining:
            if len(selected) >= num_frames:
                break
            if candidate.frame_index in used_indices:
                continue
            candidate.selected_reason = "high_content_score_gap_relaxed"
            selected.append(candidate)
            used_indices.add(candidate.frame_index)

    return sorted(selected, key=lambda c: c.timestamp_sec)


def save_frame_image(frame_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).save(output_path, quality=95)


def make_contact_sheet(selected: List[CandidateFrame], output_path: Path, title: str, thumb_width: int = 320) -> None:
    if not selected:
        return

    thumbs = []
    label_height = 58

    for idx, c in enumerate(selected):
        img = Image.fromarray(c.image)
        w, h = img.size
        scale = thumb_width / float(w)
        thumb_height = max(1, int(round(h * scale)))
        img = img.resize((thumb_width, thumb_height))

        canvas = Image.new("RGB", (thumb_width, thumb_height + label_height), "white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        label = (
            f"{idx:02d} | {c.timestamp}\n"
            f"{c.selected_reason}\n"
            f"score={c.combined_score:.3f} flow={c.optical_flow_score:.3f}"
        )
        draw.text((6, thumb_height + 4), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    cell_w = thumb_width
    cell_h = max(t.size[1] for t in thumbs)
    title_h = 40

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))

    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = title_h + (i // cols) * cell_h
        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def candidate_to_record(c: CandidateFrame, image_file: Optional[str], image_path: Optional[Path]) -> Dict[str, Any]:
    return {
        "frame_index": c.frame_index,
        "timestamp_sec": c.timestamp_sec,
        "timestamp": c.timestamp,
        "image_file": image_file,
        "image_path": str(image_path) if image_path is not None else None,
        "selected_reason": c.selected_reason,
        "brightness": c.brightness_raw,
        "sharpness": c.sharpness_raw,
        "motion_score": c.motion_score,
        "hist_diff_score": c.hist_diff_score,
        "optical_flow_score": c.optical_flow_score,
        "flow_mean_raw": c.flow_mean_raw,
        "flow_p90_raw": c.flow_p90_raw,
        "flow_motion_area_raw": c.flow_motion_area_raw,
        "combined_score": c.combined_score,
    }


def process_video(video_path: Path, output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    clip_stem = video_path.stem
    clip_output_dir = output_root / f"{clip_stem}{args.output_suffix}"

    if clip_output_dir.exists() and args.overwrite:
        shutil.rmtree(clip_output_dir)
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    candidates, metadata = extract_candidate_frames(
        video_path=video_path,
        candidate_fps=args.candidate_fps,
        black_threshold=args.black_threshold,
        ideal_brightness=args.ideal_brightness,
        use_optical_flow=args.use_optical_flow,
        flow_resize_width=args.flow_resize_width,
        flow_mag_threshold=args.flow_mag_threshold,
        flow_clip_value=args.flow_clip_value,
        subtract_camera_motion=args.subtract_camera_motion,
        ignore_bottom_ratio=args.ignore_bottom_ratio,
    )

    assign_scores(
        candidates=candidates,
        black_threshold=args.black_threshold,
        ideal_brightness=args.ideal_brightness,
        motion_clip_value=args.motion_clip_value,
        scene_clip_value=args.scene_clip_value,
        sharpness_clip_value=args.sharpness_clip_value,
        scene_weight=args.scene_weight,
        motion_weight=args.motion_weight,
        flow_weight=args.flow_weight if args.use_optical_flow else 0.0,
        sharpness_weight=args.sharpness_weight,
        brightness_weight=args.brightness_weight,
    )

    selected = select_keyframes(
        candidates=candidates,
        num_frames=args.num_frames,
        duration_sec=float(metadata.get("duration_sec", 0.0) or 0.0),
        min_gap_sec=args.min_gap_sec,
        anchor_start=args.anchor_start,
        anchor_middle=args.anchor_middle,
        anchor_end=args.anchor_end,
    )

    selected_records = []
    for order, frame in enumerate(selected):
        image_file = f"{order:02d}_t{frame.timestamp_sec:07.3f}_f{frame.frame_index}.jpg"
        image_path = clip_output_dir / image_file
        save_frame_image(frame.image, image_path)
        selected_records.append(candidate_to_record(frame, image_file, image_path))

    contact_sheet_path = clip_output_dir / f"{clip_stem}{args.output_suffix}_contact_sheet.jpg"
    make_contact_sheet(
        selected=selected,
        output_path=contact_sheet_path,
        title=f"{clip_stem}{args.output_suffix} | content-aware + optical flow",
        thumb_width=args.contact_sheet_thumb_width,
    )

    metadata_path = clip_output_dir / "selected_frames.json"
    result = {
        "sampler_version": "content_aware_sampler_flow_v1",
        "clip_file": video_path.name,
        "clip_stem": clip_stem,
        "output_dir": str(clip_output_dir),
        "video_metadata": metadata,
        "num_candidates": len(candidates),
        "selected_count": len(selected_records),
        "selected_frames": selected_records,
        "contact_sheet_path": str(contact_sheet_path),
        "metadata_path": str(metadata_path),
        "scoring_config": {
            "num_frames": args.num_frames,
            "candidate_fps": args.candidate_fps,
            "min_gap_sec": args.min_gap_sec,
            "black_threshold": args.black_threshold,
            "ideal_brightness": args.ideal_brightness,
            "use_optical_flow": args.use_optical_flow,
            "flow_weight": args.flow_weight,
            "flow_resize_width": args.flow_resize_width,
            "flow_mag_threshold": args.flow_mag_threshold,
            "flow_clip_value": args.flow_clip_value,
            "subtract_camera_motion": args.subtract_camera_motion,
            "ignore_bottom_ratio": args.ignore_bottom_ratio,
            "scene_weight": args.scene_weight,
            "motion_weight": args.motion_weight,
            "sharpness_weight": args.sharpness_weight,
            "brightness_weight": args.brightness_weight,
        },
        "error": None,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def find_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Input file is not a supported video: {input_path}")
        return [input_path]

    if input_path.is_dir():
        videos = []
        for p in sorted(input_path.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(p)
        return videos

    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select content-aware keyframes with optical-flow-based motion scoring."
    )

    parser.add_argument("--input", required=True, help="Video file or directory of video clips.")
    parser.add_argument("--output", required=True, help="Output directory for selected keyframes.")
    parser.add_argument("--output-suffix", default="_flow", help="Suffix added to each per-clip output folder, e.g. ep02_clip_0000_flow.")
    parser.add_argument("--summary-name", default="sampling_summary_flow.jsonl", help="Summary JSONL filename.")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of selected frames per clip.")
    parser.add_argument("--candidate-fps", type=float, default=2.0, help="Candidate sampling FPS.")
    parser.add_argument("--min-gap-sec", type=float, default=1.0, help="Preferred minimum gap between selected frames.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite per-clip output folders.")

    parser.add_argument("--anchor-start", action="store_true", default=True, help="Include a frame near the start.")
    parser.add_argument("--no-anchor-start", dest="anchor_start", action="store_false")
    parser.add_argument("--anchor-middle", action="store_true", default=True, help="Include a frame near the middle.")
    parser.add_argument("--no-anchor-middle", dest="anchor_middle", action="store_false")
    parser.add_argument("--anchor-end", action="store_true", default=True, help="Include a frame near the end.")
    parser.add_argument("--no-anchor-end", dest="anchor_end", action="store_false")

    parser.add_argument("--black-threshold", type=float, default=18.0, help="Brightness below this is treated as near-black.")
    parser.add_argument("--ideal-brightness", type=float, default=110.0, help="Brightness target for brightness_score.")
    parser.add_argument("--ignore-bottom-ratio", type=float, default=0.12, help="Ignore bottom area to reduce subtitle influence.")

    parser.add_argument("--use-optical-flow", action="store_true", help="Enable optical-flow-based scoring.")
    parser.add_argument("--flow-weight", type=float, default=0.25, help="Weight of optical_flow_score in combined_score.")
    parser.add_argument("--flow-resize-width", type=int, default=384, help="Resize width for optical flow computation.")
    parser.add_argument("--flow-mag-threshold", type=float, default=0.6, help="Magnitude threshold for motion area ratio.")
    parser.add_argument("--flow-clip-value", type=float, default=5.0, help="Normalization clip value for flow magnitude.")
    parser.add_argument("--subtract-camera-motion", action="store_true", help="Subtract median flow as approximate camera motion.")

    parser.add_argument("--motion-clip-value", type=float, default=40.0, help="Frame diff normalization clip value.")
    parser.add_argument("--scene-clip-value", type=float, default=0.8, help="Histogram diff normalization clip value.")
    parser.add_argument("--sharpness-clip-value", type=float, default=800.0, help="Sharpness normalization clip value.")

    parser.add_argument("--scene-weight", type=float, default=0.30, help="Scene change score weight.")
    parser.add_argument("--motion-weight", type=float, default=0.25, help="Frame difference score weight.")
    parser.add_argument("--sharpness-weight", type=float, default=0.10, help="Sharpness score weight.")
    parser.add_argument("--brightness-weight", type=float, default=0.10, help="Brightness score weight.")

    parser.add_argument("--contact-sheet-thumb-width", type=int, default=320)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_path)

    print("Content-aware keyframe sampling with optical flow")
    print("Input:", input_path)
    print("Output:", output_root)
    print("Video count:", len(videos))
    print("Frames per clip:", args.num_frames)
    print("Candidate FPS:", args.candidate_fps)
    print("Use optical flow:", args.use_optical_flow)
    print("Flow weight:", args.flow_weight if args.use_optical_flow else 0.0)

    summary_path = output_root / args.summary_name

    with summary_path.open("w", encoding="utf-8") as summary_file:
        for idx, video_path in enumerate(videos, start=1):
            print(f"\nProcessing {idx}/{len(videos)}: {video_path.name}")
            try:
                result = process_video(video_path, output_root, args)
                print("Selected frames:", result["selected_count"])
                print("Metadata:", result["metadata_path"])
                print("Contact sheet:", result["contact_sheet_path"])
            except Exception as e:
                print("Error:", repr(e))
                result = {
                    "sampler_version": "content_aware_sampler_flow_v1",
                    "clip_file": video_path.name,
                    "clip_stem": video_path.stem,
                    "output_dir": str(output_root / video_path.stem),
                    "video_metadata": {"video_path": str(video_path)},
                    "selected_count": 0,
                    "selected_frames": [],
                    "contact_sheet_path": None,
                    "metadata_path": None,
                    "error": repr(e),
                }
            summary_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary_file.flush()

    print("\nDone.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
