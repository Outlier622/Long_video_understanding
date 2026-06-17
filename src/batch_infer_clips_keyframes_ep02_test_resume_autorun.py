"""
batch_infer_clips_keyframes.py

Purpose:
    Run VideoThinker / Qwen2.5-VL style inference using content-aware selected
    keyframes instead of raw video input.

Pipeline:
    clips
    -> content_aware_sampler.py
    -> make_clip_manifest.py
    -> clip_manifest.jsonl
    -> this script
    -> ep01_results_keyframes_test.jsonl or ep01_results_keyframes.jsonl

Default mode:
    RUN_MODE = "test"
    It only runs selected difficult clips:
        1, 3, 14, 20, 31, 40, 48, 52, 60, 73

After the test result looks better:
    change RUN_MODE = "all"

Expected manifest path:
    D:\projects\longvideo\keyframes\clip_manifest.jsonl

Dependencies:
    pip install torch transformers qwen-vl-utils
"""

import json
from platform import processor
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =========================
# Paths and run mode
# =========================

MODEL_PATH = r"D:\projects\VideoThinker-R1-3B"

MANIFEST_PATH = Path(r"D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_ep02_test_remaining.jsonl")
OUTPUT_TEST_PATH = Path(r"D:\projects\longvideo\episodes\ep02\outputs\ep02_results_keyframes_test.jsonl")
OUTPUT_ALL_PATH = Path(r"D:\projects\longvideo\episodes\ep02\outputs\ep02_results_keyframes_test.jsonl")
# "test" or "all"
RUN_MODE = "test"
TEST_CLIP_IDS = {1, 3, 14, 20, 31, 40, 48, 52, 60, 73}
VERSION = "ep02_keyframe_v2_content_aware_sampling"
# =========================
# Generation settings
# =========================

MAX_NEW_TOKENS = 500

# Image pixel limits.
# If CUDA OOM happens, reduce MAX_PIXELS to 96 * 28 * 28 or 64 * 28 * 28.
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 128 * 28 * 28

# If one clip contains too many selected frames, only use this many.
# Your sampler normally creates 12 frames, so 12 is fine.
MAX_KEYFRAMES_PER_CLIP = 12

# If true, choose the top N frames in chronological order when more than
# MAX_KEYFRAMES_PER_CLIP frames exist. If false, use first N frames.
USE_TOP_SCORE_WHEN_TOO_MANY_FRAMES = True


# =========================
# Prompt
# =========================

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


# =========================
# Utilities
# =========================

def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in manifest line {line_num}: {e}") from e

            records.append(record)

    if not records:
        raise ValueError(f"No records found in manifest: {path}")

    return records


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
    """
    For image-only inputs, qwen_vl_utils can return {"fps": []}.
    Empty list values should be removed instead of indexing [0].
    """
    if not video_kwargs:
        return {}

    cleaned = {}

    for key, value in video_kwargs.items():
        if isinstance(value, list):
            if len(value) == 0:
                continue
            if len(value) == 1:
                cleaned[key] = value[0]
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value

    return cleaned


def choose_keyframes(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    frames = list(record.get("selected_frames") or [])

    # Keep only frames whose image file exists.
    existing = []
    missing = []

    for frame in frames:
        image_path = frame.get("image_path")
        if image_path and Path(image_path).exists():
            existing.append(frame)
        else:
            missing.append(image_path)

    if missing:
        print("Warning: missing keyframe images:")
        for item in missing[:5]:
            print("  ", item)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    frames = existing

    if len(frames) <= MAX_KEYFRAMES_PER_CLIP:
        return sorted(frames, key=lambda x: float(x.get("local_timestamp_sec", 0.0) or 0.0))

    if USE_TOP_SCORE_WHEN_TOO_MANY_FRAMES:
        frames = sorted(
            frames,
            key=lambda x: float(x.get("combined_score", 0.0) or 0.0),
            reverse=True,
        )[:MAX_KEYFRAMES_PER_CLIP]

    else:
        frames = frames[:MAX_KEYFRAMES_PER_CLIP]

    return sorted(frames, key=lambda x: float(x.get("local_timestamp_sec", 0.0) or 0.0))


def build_frame_metadata_text(frames: List[Dict[str, Any]]) -> str:
    lines = []

    for idx, frame in enumerate(frames):
        frame_label = f"Frame {idx:02d}"
        local_ts = frame.get("local_timestamp") or ""
        global_ts = frame.get("global_timestamp") or ""
        reason = frame.get("selected_reason") or ""
        score = frame.get("combined_score")

        if isinstance(score, (int, float)):
            score_text = f"{score:.3f}"
        else:
            score_text = "unknown"

        lines.append(
            f"{frame_label}: local_time={local_ts}, global_time={global_ts}, "
            f"selection_reason={reason}, content_score={score_text}"
        )

    return "\n".join(lines)


def build_prompt(record: Dict[str, Any], frames: List[Dict[str, Any]]) -> str:
    clip_id = record.get("clip_id")
    clip_file = record.get("clip_file")
    actual_start_time = record.get("actual_start_time")
    actual_end_time = record.get("actual_end_time")
    duration_sec = record.get("duration_sec")

    frame_metadata_text = build_frame_metadata_text(frames)

    metadata = f"""
Clip metadata:
clip_id: {clip_id}
clip_file: {clip_file}
actual_start_time: {actual_start_time}
actual_end_time: {actual_end_time}
duration_sec: {duration_sec}
selected_keyframe_count: {len(frames)}

Keyframe order and timestamps:
{frame_metadata_text}

""".strip()

    return metadata + "\n\n" + KEYFRAME_PROMPT_BODY


def build_messages(record: Dict[str, Any], frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = build_prompt(record, frames)

    content = []

    for frame in frames:
        content.append(
            {
                "type": "image",
                "image": frame["image_path"],
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def infer_record(
    processor,
    model,
    record: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    frames = choose_keyframes(record)

    if not frames:
        raise RuntimeError(f"No existing keyframe images for clip {record.get('clip_id')}")

    messages = build_messages(record, frames)

    # add_vision_id=True makes the rendered prompt label images as Picture 1, Picture 2, etc.
    # Some transformer versions support this argument. If not, fall back.
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

    processor_kwargs = {
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output, frames


def output_path_for_run_mode() -> Path:
    if RUN_MODE == "test":
        return OUTPUT_TEST_PATH
    if RUN_MODE == "all":
        return OUTPUT_ALL_PATH
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


def filter_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = sorted(records, key=lambda x: int(x.get("clip_id", 0)))

    if RUN_MODE == "test":
        return [r for r in records if int(r.get("clip_id", -1)) in TEST_CLIP_IDS]

    if RUN_MODE == "all":
        return records

    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


def build_output_record(
    record: Dict[str, Any],
    raw_output: str,
    used_frames: List[Dict[str, Any]],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    output_record = {
        "version": VERSION,
        "input_mode": "content_aware_keyframes_as_images",
        "run_mode": RUN_MODE,
        "clip_id": record.get("clip_id"),
        "clip_file": record.get("clip_file"),
        "actual_start_time": record.get("actual_start_time"),
        "actual_end_time": record.get("actual_end_time"),
        "duration_sec": record.get("duration_sec"),
        "selected_frame_count": len(used_frames),
        "selected_frames_used": [
            {
                "frame_order": idx,
                "image_path": frame.get("image_path"),
                "local_timestamp": frame.get("local_timestamp"),
                "global_timestamp": frame.get("global_timestamp"),
                "selected_reason": frame.get("selected_reason"),
                "combined_score": frame.get("combined_score"),
            }
            for idx, frame in enumerate(used_frames)
        ],
        "max_keyframes_per_clip": MAX_KEYFRAMES_PER_CLIP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "raw_output": raw_output,
        "cleaned_output": None,
        "parsed_json": None,
        "json_parse_ok": False,
        "error": error,
    }

    if error is not None:
        return output_record

    cleaned_output = try_extract_json(raw_output)
    output_record["cleaned_output"] = cleaned_output

    try:
        parsed = json.loads(cleaned_output)
        output_record["parsed_json"] = parsed
        output_record["json_parse_ok"] = True
    except json.JSONDecodeError:
        output_record["parsed_json"] = None
        output_record["json_parse_ok"] = False

    return output_record


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    output_path = output_path_for_run_mode()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading manifest:", MANIFEST_PATH)
    all_records = load_manifest(MANIFEST_PATH)
    records = filter_records(all_records)

    if not records:
        raise RuntimeError("No records selected for inference.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("Model loaded.")
    print("CUDA available:", torch.cuda.is_available())
    print("Version:", VERSION)
    print("Run mode:", RUN_MODE)
    print("Selected clip count:", len(records))
    print("MAX_KEYFRAMES_PER_CLIP:", MAX_KEYFRAMES_PER_CLIP)
    print("MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
    print("MAX_PIXELS:", MAX_PIXELS)
    print("Output path:", output_path)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            clip_id = record.get("clip_id")
            clip_file = record.get("clip_file")
            start_time = record.get("actual_start_time")
            end_time = record.get("actual_end_time")

            print(f"\nProcessing {idx}/{len(records)}")
            print(f"clip_id: {clip_id}")
            print(f"clip_file: {clip_file}")
            print(f"time: {start_time} to {end_time}")

            used_frames: List[Dict[str, Any]] = []

            try:
                raw_output, used_frames = infer_record(
                    processor=processor,
                    model=model,
                    record=record,
                )

                output_record = build_output_record(
                    record=record,
                    raw_output=raw_output,
                    used_frames=used_frames,
                    error=None,
                )

                print("Raw output:")
                print(raw_output)
                print("JSON parse ok:", output_record["json_parse_ok"])
                print("Frames used:", len(used_frames))

            except torch.OutOfMemoryError as e:
                print("CUDA out of memory on:", clip_file)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                output_record = build_output_record(
                    record=record,
                    raw_output="",
                    used_frames=used_frames,
                    error="CUDA out of memory",
                )

            except Exception as e:
                print("Error on:", clip_file)
                print(repr(e))

                output_record = build_output_record(
                    record=record,
                    raw_output="",
                    used_frames=used_frames,
                    error=repr(e),
                )

            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            f.flush()

    print("\nDone.")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
