import argparse
import os
from pathlib import Path

import cv2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sample_fixed_keyframes(video_path, output_dir, num_frames=12):
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        print(f"[ERROR] Empty video: {video_path}")
        cap.release()
        return False

    if fps <= 0:
        fps = 24.0

    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = [
            round(i * (total_frames - 1) / (num_frames - 1))
            for i in range(num_frames)
        ]

    for out_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            print(f"[WARN] Failed to read frame {frame_idx} from {video_path}")
            continue

        timestamp = frame_idx / fps
        output_name = f"{out_idx:02d}_t{timestamp:07.3f}_f{frame_idx}.jpg"
        output_path = Path(output_dir) / output_name
        cv2.imwrite(str(output_path), frame)

    cap.release()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=12)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    clips = sorted(clips_dir.glob("ep01_clip_*.mp4"))

    print(f"Found {len(clips)} clips")

    success = 0
    failed = 0

    for clip_path in clips:
        clip_id = clip_path.stem
        clip_output_dir = output_dir / clip_id

        ok = sample_fixed_keyframes(
            video_path=clip_path,
            output_dir=clip_output_dir,
            num_frames=args.num_frames,
        )

        if ok:
            success += 1
        else:
            failed += 1

    print(f"Done. success={success}, failed={failed}")


if __name__ == "__main__":
    main()