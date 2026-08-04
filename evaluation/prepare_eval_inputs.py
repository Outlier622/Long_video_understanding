import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
TEXT_EXTS = {".txt", ".md"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_clip_id(raw: str) -> str:
    """
    Keep episode-aware clip IDs such as:
      ep01_clip_0000
      ep02_clip_0044

    Also supports:
      clip_000
      clip_001
    """
    raw = str(raw).replace("\\", "/").lower()

    # Prefer episode-aware clip pattern.
    m = re.search(r"(ep\d+[_-]clip[_-]\d+)", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace("-", "_").lower()

    # Fallback: generic clip pattern.
    m = re.search(r"(clip[_-]\d+)", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace("-", "_").lower()

    # Last fallback: use basename-like string.
    return Path(raw).stem.lower()


def extract_clip_id(path: Path) -> str:
    """
    Extract clip id from either folder name or file path.
    Examples:
      D:/.../ep01_clip_0000/frame.jpg -> ep01_clip_0000
      D:/.../ep01_clip_0000.mp4       -> ep01_clip_0000
      D:/.../clip_000/frame.jpg       -> clip_000
    """
    text = str(path).replace("\\", "/").lower()
    return normalize_clip_id(text)


def extract_frame_id(path: Path, index: int) -> str:
    text = path.stem.lower()

    patterns = [
        r"frame[_-]?(\d+)",
        r"keyframe[_-]?(\d+)",
        r"img[_-]?(\d+)",
        r"image[_-]?(\d+)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return f"frame_{int(m.group(1)):05d}"

    return f"frame_{index:05d}"


def extract_timestamp(path: Path, fallback_index: int, expected_per_clip: int = 12) -> float:
    """
    Try to read timestamp from filename.
    Supported names:
      00_t003.250_f78.jpg
      frame_t3.25.jpg
      frame_3.25s.jpg

    If timestamp does not exist, use frame order as a stable fallback.
    """
    text = path.stem.lower()

    patterns = [
        r"t[_-]?(\d+(?:\.\d+)?)",
        r"time[_-]?(\d+(?:\.\d+)?)",
        r"ts[_-]?(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)s",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Fallback only preserves order, not real video time.
    return float(fallback_index)


def scan_keyframes(keyframe_dir: str, expected_per_clip: int = 12) -> pd.DataFrame:
    root = Path(keyframe_dir)

    if not root.exists():
        raise FileNotFoundError(f"Keyframe directory not found: {keyframe_dir}")

    image_paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    rows = []
    grouped: Dict[str, List[Path]] = {}

    for p in image_paths:
        clip_id = extract_clip_id(p)
        grouped.setdefault(clip_id, []).append(p)

    for clip_id, paths in sorted(grouped.items()):
        paths = sorted(paths)

        for i, p in enumerate(paths, start=1):
            rows.append({
                "clip_id": clip_id,
                "frame_id": extract_frame_id(p, i),
                "timestamp": extract_timestamp(p, i, expected_per_clip),
                "frame_path": str(p)
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[WARN] No keyframe images found in {keyframe_dir}")
        return pd.DataFrame(columns=["clip_id", "frame_id", "timestamp", "frame_path"])

    df = df.sort_values(["clip_id", "timestamp", "frame_id"]).reset_index(drop=True)
    return df


def read_summary_from_txt_folder(summary_dir: str) -> pd.DataFrame:
    root = Path(summary_dir)
    rows = []

    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
            continue

        clip_id = extract_clip_id(p)

        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            text = ""

        rows.append({
            "clip_id": clip_id,
            "summary": text
        })

    return pd.DataFrame(rows)


def read_summary_from_json(path: str) -> pd.DataFrame:
    p = Path(path)

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    rows = []

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            clip_raw = (
                item.get("clip_id")
                or item.get("clip")
                or item.get("id")
                or item.get("name")
                or item.get("video")
                or item.get("file")
                or item.get("filename")
            )

            summary = (
                item.get("summary")
                or item.get("caption")
                or item.get("description")
                or item.get("output")
                or item.get("answer")
                or item.get("response")
                or ""
            )

            if clip_raw:
                rows.append({
                    "clip_id": normalize_clip_id(str(clip_raw)),
                    "summary": str(summary)
                })

    elif isinstance(data, dict):
        for k, v in data.items():
            clip_id = normalize_clip_id(str(k))

            if isinstance(v, dict):
                summary = (
                    v.get("summary")
                    or v.get("caption")
                    or v.get("description")
                    or v.get("output")
                    or v.get("answer")
                    or v.get("response")
                    or ""
                )
            else:
                summary = str(v)

            rows.append({
                "clip_id": clip_id,
                "summary": str(summary)
            })

    return pd.DataFrame(rows)


def read_summary_from_jsonl(path: str) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            if not isinstance(item, dict):
                continue

            clip_raw = (
                item.get("clip_id")
                or item.get("clip")
                or item.get("id")
                or item.get("name")
                or item.get("video")
                or item.get("file")
                or item.get("filename")
                or item.get("video_path")
                or item.get("clip_path")
            )

            summary = (
                item.get("summary")
                or item.get("caption")
                or item.get("description")
                or item.get("output")
                or item.get("answer")
                or item.get("response")
                or item.get("result")
                or ""
            )

            if clip_raw:
                rows.append({
                    "clip_id": normalize_clip_id(str(clip_raw)),
                    "summary": str(summary)
                })

    return pd.DataFrame(rows)


def read_summary_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "clip_id" not in df.columns:
        possible_clip_cols = ["clip", "id", "name", "file", "filename", "video", "video_path", "clip_path"]
        for col in possible_clip_cols:
            if col in df.columns:
                df["clip_id"] = df[col].apply(lambda x: normalize_clip_id(str(x)))
                break

    if "summary" not in df.columns:
        possible_summary_cols = ["caption", "description", "output", "answer", "text", "response", "result"]
        for col in possible_summary_cols:
            if col in df.columns:
                df["summary"] = df[col]
                break

    if "clip_id" not in df.columns or "summary" not in df.columns:
        raise ValueError(
            f"Cannot find clip_id and summary columns in {path}. "
            f"Existing columns: {list(df.columns)}"
        )

    out = df[["clip_id", "summary"]].copy()
    out["clip_id"] = out["clip_id"].apply(lambda x: normalize_clip_id(str(x)))
    out["summary"] = out["summary"].fillna("").astype(str)
    return out


def scan_summaries(summary_path: Optional[str]) -> pd.DataFrame:
    if not summary_path:
        return pd.DataFrame(columns=["clip_id", "summary"])

    p = Path(summary_path)

    if not p.exists():
        raise FileNotFoundError(f"Summary path not found: {summary_path}")

    if p.is_dir():
        df = read_summary_from_txt_folder(summary_path)
    elif p.suffix.lower() == ".json":
        df = read_summary_from_json(summary_path)
    elif p.suffix.lower() == ".jsonl":
        df = read_summary_from_jsonl(summary_path)
    elif p.suffix.lower() == ".csv":
        df = read_summary_from_csv(summary_path)
    else:
        raise ValueError(f"Unsupported summary format: {summary_path}")

    if df.empty:
        print(f"[WARN] No summaries found in {summary_path}")
        return pd.DataFrame(columns=["clip_id", "summary"])

    df["clip_id"] = df["clip_id"].apply(lambda x: normalize_clip_id(str(x)))
    df["summary"] = df["summary"].fillna("").astype(str)
    df = df.sort_values("clip_id").reset_index(drop=True)

    return df


def scan_clips(clips_dir: Optional[str]) -> pd.DataFrame:
    if not clips_dir:
        return pd.DataFrame(columns=["clip_id", "clip_path"])

    root = Path(clips_dir)

    if not root.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    rows = []

    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rows.append({
                "clip_id": extract_clip_id(p),
                "clip_path": str(p)
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[WARN] No video clips found in {clips_dir}")
        return pd.DataFrame(columns=["clip_id", "clip_path"])

    return df.sort_values("clip_id").reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path} rows={len(df)}")


def print_basic_check(name: str, df: pd.DataFrame):
    print(f"\n{name}:")
    print(f"  rows: {len(df)}")

    if "clip_id" in df.columns:
        print(f"  clips: {df['clip_id'].nunique()}")

    if len(df) > 0:
        print("  sample:")
        print(df.head(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation input CSV files for long-video pipeline evaluation."
    )

    parser.add_argument("--fixed-keyframes-dir", required=True)
    parser.add_argument("--improved-keyframes-dir", required=True)

    parser.add_argument("--isolated-summary-path", default=None)
    parser.add_argument("--cross-summary-path", default=None)

    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--output-dir", default="evaluation/inputs")
    parser.add_argument("--expected-per-clip", type=int, default=12)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    fixed_df = scan_keyframes(args.fixed_keyframes_dir, args.expected_per_clip)
    improved_df = scan_keyframes(args.improved_keyframes_dir, args.expected_per_clip)

    isolated_df = scan_summaries(args.isolated_summary_path)
    cross_df = scan_summaries(args.cross_summary_path)

    clips_df = scan_clips(args.clips_dir)

    fixed_out = os.path.join(args.output_dir, "fixed_keyframes.csv")
    improved_out = os.path.join(args.output_dir, "improved_keyframes.csv")
    isolated_out = os.path.join(args.output_dir, "isolated_summaries.csv")
    cross_out = os.path.join(args.output_dir, "cross_clip_summaries.csv")
    clips_out = os.path.join(args.output_dir, "clips_manifest.csv")

    save_csv(fixed_df, fixed_out)
    save_csv(improved_df, improved_out)
    save_csv(isolated_df, isolated_out)
    save_csv(cross_df, cross_out)

    if not clips_df.empty:
        save_csv(clips_df, clips_out)

    print_basic_check("fixed_keyframes", fixed_df)
    print_basic_check("improved_keyframes", improved_df)
    print_basic_check("isolated_summaries", isolated_df)
    print_basic_check("cross_clip_summaries", cross_df)
    print_basic_check("clips_manifest", clips_df)

    print("\nDone. Evaluation input CSV files are ready.")


if __name__ == "__main__":
    main()
