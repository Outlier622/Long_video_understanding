import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def limit_frames_per_clip(df: pd.DataFrame, max_frames: Optional[int]) -> pd.DataFrame:
    if max_frames is None or max_frames <= 0:
        return df.copy()

    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values(["clip_id", "timestamp", "frame_id"])
    return df.groupby("clip_id", group_keys=False).head(max_frames).reset_index(drop=True)


def load_image_gray(path: str) -> Optional[np.ndarray]:
    if not isinstance(path, str) or not os.path.exists(path):
        return None
    try:
        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None


def image_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def image_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def image_blur_score(gray: np.ndarray) -> float:
    if cv2 is None:
        return np.nan
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def vector_for_similarity(gray: np.ndarray, size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    img = Image.fromarray(gray).resize(size)
    arr = np.array(img).astype(np.float32)
    arr = arr - arr.mean()
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.flatten()
    return (arr / norm).flatten()


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def evaluate_keyframes(
    df: pd.DataFrame,
    version: str,
    blur_threshold: float,
    entropy_threshold: float,
    dark_threshold: float,
    bright_threshold: float,
    duplicate_threshold: float,
) -> pd.DataFrame:
    rows = []

    for clip_id, group in df.groupby("clip_id"):
        prev_vec = None
        group = group.sort_values(["timestamp", "frame_id"])

        for _, row in group.iterrows():
            frame_path = str(row.get("frame_path", ""))
            gray = load_image_gray(frame_path)

            if gray is None:
                rows.append({
                    "version": version,
                    "clip_id": clip_id,
                    "frame_id": row.get("frame_id", ""),
                    "timestamp": row.get("timestamp", np.nan),
                    "frame_path": frame_path,
                    "image_found": 0,
                    "blur_score": np.nan,
                    "entropy": np.nan,
                    "brightness": np.nan,
                    "duplicate_similarity": np.nan,
                    "is_blurry": 1,
                    "is_low_entropy": 1,
                    "is_too_dark_or_bright": 1,
                    "is_duplicate": 0,
                    "low_info": 1,
                    "low_info_reason": "missing image",
                })
                continue

            blur = image_blur_score(gray)
            ent = image_entropy(gray)
            bright = image_brightness(gray)

            vec = vector_for_similarity(gray)
            dup_sim = cosine_similarity(prev_vec, vec)
            prev_vec = vec

            is_blurry = int(not np.isnan(blur) and blur < blur_threshold)
            is_low_entropy = int(ent < entropy_threshold)
            is_too_dark_or_bright = int(bright < dark_threshold or bright > bright_threshold)
            is_duplicate = int(dup_sim >= duplicate_threshold)

            reasons = []
            if is_blurry:
                reasons.append("blurry")
            if is_low_entropy:
                reasons.append("low entropy")
            if is_too_dark_or_bright:
                reasons.append("too dark/bright")
            if is_duplicate:
                reasons.append("near duplicate")

            rows.append({
                "version": version,
                "clip_id": clip_id,
                "frame_id": row.get("frame_id", ""),
                "timestamp": row.get("timestamp", np.nan),
                "frame_path": frame_path,
                "image_found": 1,
                "blur_score": blur,
                "entropy": ent,
                "brightness": bright,
                "duplicate_similarity": dup_sim,
                "is_blurry": is_blurry,
                "is_low_entropy": is_low_entropy,
                "is_too_dark_or_bright": is_too_dark_or_bright,
                "is_duplicate": is_duplicate,
                "low_info": int(len(reasons) > 0),
                "low_info_reason": "; ".join(reasons),
            })

    return pd.DataFrame(rows)


def evaluate_sampler_failures(df: pd.DataFrame, version: str, expected_frames_per_clip: int) -> pd.DataFrame:
    rows = []

    for clip_id, group in df.groupby("clip_id"):
        generated = len(group)
        valid_paths = group["frame_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p)).sum()
        failed = int(generated < expected_frames_per_clip or valid_paths < expected_frames_per_clip)

        rows.append({
            "version": version,
            "clip_id": clip_id,
            "expected_keyframes": expected_frames_per_clip,
            "generated_keyframes": generated,
            "valid_frame_paths": int(valid_paths),
            "sampler_failed": failed,
        })

    return pd.DataFrame(rows)


def compute_motion_timestamps(video_path: str, sample_fps: float, motion_percentile: float) -> pd.DataFrame:
    if cv2 is None:
        raise ImportError("opencv-python is required for motion coverage.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return pd.DataFrame(columns=["timestamp", "motion_score", "is_motion_event"])

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    frame_interval = max(1, int(round(fps / sample_fps)))
    rows = []
    prev_gray = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        if prev_gray is None:
            motion_score = 0.0
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = float(np.mean(mag))

        rows.append({"timestamp": timestamp, "motion_score": motion_score})
        prev_gray = gray
        frame_idx += 1

    cap.release()

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "motion_score", "is_motion_event"])

    threshold = float(np.percentile(out["motion_score"], motion_percentile))
    out["is_motion_event"] = (out["motion_score"] >= threshold).astype(int)
    return out


def build_clip_path_map(clips_manifest: Optional[str], clips_dir: Optional[str]) -> Dict[str, str]:
    clip_map = {}

    if clips_manifest and os.path.exists(clips_manifest):
        df = pd.read_csv(clips_manifest)
        if "clip_id" in df.columns and "clip_path" in df.columns:
            for _, row in df.iterrows():
                clip_map[str(row["clip_id"])] = str(row["clip_path"])

    if clips_dir and os.path.exists(clips_dir):
        for p in Path(clips_dir).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
                clip_map[p.stem.lower()] = str(p)

    return clip_map


def evaluate_motion_coverage(
    keyframes_df: pd.DataFrame,
    version: str,
    clip_path_map: Dict[str, str],
    sample_fps: float,
    motion_percentile: float,
    tolerance_seconds: float,
) -> pd.DataFrame:
    if cv2 is None:
        return pd.DataFrame([{
            "version": version,
            "clip_id": "ALL",
            "motion_events": np.nan,
            "covered_motion_events": np.nan,
            "motion_event_coverage": np.nan,
            "note": "opencv-python not installed",
        }])

    rows = []

    for clip_id, group in keyframes_df.groupby("clip_id"):
        video_path = clip_path_map.get(str(clip_id))

        if not video_path or not os.path.exists(video_path):
            rows.append({
                "version": version,
                "clip_id": clip_id,
                "motion_events": np.nan,
                "covered_motion_events": np.nan,
                "motion_event_coverage": np.nan,
                "note": "clip video not found",
            })
            continue

        motion_df = compute_motion_timestamps(video_path, sample_fps, motion_percentile)

        if motion_df.empty:
            rows.append({
                "version": version,
                "clip_id": clip_id,
                "motion_events": 0,
                "covered_motion_events": 0,
                "motion_event_coverage": np.nan,
                "note": "no motion data",
            })
            continue

        motion_times = motion_df.loc[motion_df["is_motion_event"] == 1, "timestamp"].astype(float).tolist()
        keyframe_times = pd.to_numeric(group["timestamp"], errors="coerce").dropna().astype(float).tolist()

        if not motion_times:
            coverage = np.nan
            covered = 0
        else:
            covered = sum(
                1 for mt in motion_times
                if any(abs(mt - kt) <= tolerance_seconds for kt in keyframe_times)
            )
            coverage = covered / len(motion_times)

        rows.append({
            "version": version,
            "clip_id": clip_id,
            "motion_events": len(motion_times),
            "covered_motion_events": covered,
            "motion_event_coverage": coverage,
            "note": "",
        })

    return pd.DataFrame(rows)


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{100 * float(x):.1f}%"


def summarize_metrics(keyframe_eval: pd.DataFrame, failure_eval: pd.DataFrame, motion_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for version in sorted(keyframe_eval["version"].unique()):
        kf = keyframe_eval[keyframe_eval["version"] == version]
        fail = failure_eval[failure_eval["version"] == version]
        mot = motion_eval[motion_eval["version"] == version] if not motion_eval.empty else pd.DataFrame()

        rows.append({
            "version": version,
            "clips": int(kf["clip_id"].nunique()),
            "selected_keyframes": int(len(kf)),
            "image_found_rate": float(kf["image_found"].mean()) if len(kf) else np.nan,
            "low_information_frame_ratio": float(kf["low_info"].mean()) if len(kf) else np.nan,
            "duplicate_frame_ratio": float(kf["is_duplicate"].mean()) if len(kf) else np.nan,
            "blurry_frame_ratio": float(kf["is_blurry"].mean()) if len(kf) else np.nan,
            "low_entropy_frame_ratio": float(kf["is_low_entropy"].mean()) if len(kf) else np.nan,
            "average_blur_score": float(kf["blur_score"].mean()) if len(kf) else np.nan,
            "average_entropy": float(kf["entropy"].mean()) if len(kf) else np.nan,
            "average_brightness": float(kf["brightness"].mean()) if len(kf) else np.nan,
            "sampler_failure_rate": float(fail["sampler_failed"].mean()) if len(fail) else np.nan,
            "sampler_errors": int(fail["sampler_failed"].sum()) if len(fail) else np.nan,
            "motion_event_coverage": float(mot["motion_event_coverage"].mean(skipna=True)) if len(mot) else np.nan,
        })

    return pd.DataFrame(rows)


def build_resume_values(metrics: pd.DataFrame) -> pd.DataFrame:
    fixed = metrics[metrics["version"] == "fixed"]
    improved = metrics[metrics["version"] == "improved"]

    rows = []

    if fixed.empty or improved.empty:
        return pd.DataFrame(columns=["metric", "baseline", "final", "resume_phrase"])

    f = fixed.iloc[0]
    im = improved.iloc[0]
    clips = int(im.get("clips", 0))
    frames = int(im.get("selected_keyframes", 0))

    rows.append({
        "metric": "low-information frame ratio",
        "baseline": pct(f.get("low_information_frame_ratio")),
        "final": pct(im.get("low_information_frame_ratio")),
        "resume_phrase": f"reduced low-information frame ratio from {pct(f.get('low_information_frame_ratio'))} to {pct(im.get('low_information_frame_ratio'))} across {clips} clips",
    })

    rows.append({
        "metric": "motion-event coverage",
        "baseline": pct(f.get("motion_event_coverage")),
        "final": pct(im.get("motion_event_coverage")),
        "resume_phrase": f"improved motion-event coverage from {pct(f.get('motion_event_coverage'))} to {pct(im.get('motion_event_coverage'))} across {clips} clips",
    })

    rows.append({
        "metric": "sampler failure rate",
        "baseline": pct(f.get("sampler_failure_rate")),
        "final": pct(im.get("sampler_failure_rate")),
        "resume_phrase": f"reduced sampler failure rate from {pct(f.get('sampler_failure_rate'))} to {pct(im.get('sampler_failure_rate'))} across {clips} clips",
    })

    rows.append({
        "metric": "processed keyframe inputs",
        "baseline": str(int(f.get("selected_keyframes", 0))),
        "final": str(frames),
        "resume_phrase": f"processed {clips} clips and evaluated {frames} selected keyframe inputs with {int(im.get('sampler_errors', 0))} sampler errors",
    })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed vs improved keyframe sampling for long-video pipeline.")

    parser.add_argument("--fixed-keyframes", required=True)
    parser.add_argument("--improved-keyframes", required=True)
    parser.add_argument("--clips-manifest", default=None)
    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--output-dir", default="evaluation/outputs")
    parser.add_argument("--output-xlsx", default="long_video_eval_results.xlsx")

    parser.add_argument("--expected-frames", type=int, default=12)
    parser.add_argument("--limit-frames-per-clip", type=int, default=12)

    parser.add_argument("--blur-threshold", type=float, default=80.0)
    parser.add_argument("--entropy-threshold", type=float, default=4.2)
    parser.add_argument("--dark-threshold", type=float, default=25.0)
    parser.add_argument("--bright-threshold", type=float, default=235.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.96)

    parser.add_argument("--motion-sample-fps", type=float, default=2.0)
    parser.add_argument("--motion-percentile", type=float, default=80.0)
    parser.add_argument("--motion-tolerance-seconds", type=float, default=1.0)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    fixed_df = safe_read_csv(args.fixed_keyframes)
    improved_df = safe_read_csv(args.improved_keyframes)

    required_cols = {"clip_id", "frame_id", "timestamp", "frame_path"}
    for name, df in [("fixed", fixed_df), ("improved", improved_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} keyframes missing columns: {missing}")

    raw_counts = pd.DataFrame([
        {"version": "fixed_raw", "rows": len(fixed_df), "clips": fixed_df["clip_id"].nunique()},
        {"version": "improved_raw", "rows": len(improved_df), "clips": improved_df["clip_id"].nunique()},
    ])

    fixed_df_limited = limit_frames_per_clip(fixed_df, args.limit_frames_per_clip)
    improved_df_limited = limit_frames_per_clip(improved_df, args.limit_frames_per_clip)

    limited_counts = pd.DataFrame([
        {"version": "fixed_limited", "rows": len(fixed_df_limited), "clips": fixed_df_limited["clip_id"].nunique()},
        {"version": "improved_limited", "rows": len(improved_df_limited), "clips": improved_df_limited["clip_id"].nunique()},
    ])

    print("Evaluating keyframe quality...")
    fixed_kf_eval = evaluate_keyframes(
        fixed_df_limited,
        "fixed",
        args.blur_threshold,
        args.entropy_threshold,
        args.dark_threshold,
        args.bright_threshold,
        args.duplicate_threshold,
    )
    improved_kf_eval = evaluate_keyframes(
        improved_df_limited,
        "improved",
        args.blur_threshold,
        args.entropy_threshold,
        args.dark_threshold,
        args.bright_threshold,
        args.duplicate_threshold,
    )
    keyframe_eval = pd.concat([fixed_kf_eval, improved_kf_eval], ignore_index=True)

    print("Evaluating sampler failures...")
    failure_eval = pd.concat([
        evaluate_sampler_failures(fixed_df_limited, "fixed", args.expected_frames),
        evaluate_sampler_failures(improved_df_limited, "improved", args.expected_frames),
    ], ignore_index=True)

    print("Evaluating motion-event coverage...")
    clip_map = build_clip_path_map(args.clips_manifest, args.clips_dir)
    motion_eval = pd.concat([
        evaluate_motion_coverage(
            fixed_df_limited,
            "fixed",
            clip_map,
            args.motion_sample_fps,
            args.motion_percentile,
            args.motion_tolerance_seconds,
        ),
        evaluate_motion_coverage(
            improved_df_limited,
            "improved",
            clip_map,
            args.motion_sample_fps,
            args.motion_percentile,
            args.motion_tolerance_seconds,
        ),
    ], ignore_index=True)

    print("Summarizing metrics...")
    metrics = summarize_metrics(keyframe_eval, failure_eval, motion_eval)
    resume_values = build_resume_values(metrics)

    output_xlsx = os.path.join(args.output_dir, args.output_xlsx)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics_summary", index=False)
        resume_values.to_excel(writer, sheet_name="resume_fill_values", index=False)
        raw_counts.to_excel(writer, sheet_name="raw_counts", index=False)
        limited_counts.to_excel(writer, sheet_name="limited_counts", index=False)
        keyframe_eval.to_excel(writer, sheet_name="keyframe_eval", index=False)
        failure_eval.to_excel(writer, sheet_name="sampler_failures", index=False)
        motion_eval.to_excel(writer, sheet_name="motion_coverage", index=False)

    print(f"\nDone. Results saved to: {output_xlsx}")

    print("\nMetrics summary:")
    display_cols = [
        "version",
        "clips",
        "selected_keyframes",
        "low_information_frame_ratio",
        "motion_event_coverage",
        "sampler_failure_rate",
        "sampler_errors",
    ]
    print(metrics[display_cols].to_string(index=False))

    print("\nResume fill values:")
    if resume_values.empty:
        print("No resume values generated.")
    else:
        print(resume_values[["metric", "baseline", "final", "resume_phrase"]].to_string(index=False))

    print("\nNote:")
    print(f"  Raw fixed rows={len(fixed_df)}, raw improved rows={len(improved_df)}.")
    print(f"  Evaluation used at most {args.limit_frames_per_clip} frames per clip for fair comparison.")


if __name__ == "__main__":
    main()
