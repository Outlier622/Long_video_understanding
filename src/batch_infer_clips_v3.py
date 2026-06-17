import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =========================
# Basic paths
# =========================
MODEL_PATH = r"D:\projects\VideoThinker-R1-3B"
CLIP_DIR = Path(r"D:\projects\longvideo\clips")
OUTPUT_DIR = Path(r"D:\projects\longvideo\outputs")

# First run with RUN_MODE = "test".
# After checking the 10 test clips, change RUN_MODE to "all" and run again.
RUN_MODE = "test"  # "test" or "all"
TEST_CLIP_IDS = {1, 3, 14, 20, 31, 40, 48, 52, 60, 73}

VERSION = "v3_grounded_event_prompt"
OUTPUT_PATH = OUTPUT_DIR / ("ep01_results_v3_test.jsonl" if RUN_MODE == "test" else "ep01_results_v3.jsonl")

# If RESUME_OUTPUT is True, clips already completed in OUTPUT_PATH will be skipped.
# If you want a clean rerun, set OVERWRITE_OUTPUT = True.
RESUME_OUTPUT = True
OVERWRITE_OUTPUT = False

# =========================
# Inference settings
# =========================
SEGMENT_SECONDS = 20
NFRAMES = 12
FALLBACK_NFRAMES = 8
MAX_NEW_TOKENS = 800
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 64 * 28 * 28


V3_PROMPT_BODY = """
You are analyzing a short clip from a sci-fi / tokusatsu-style episode.

Your highest priority is visual grounding. Describe only what is directly visible in this current clip.
Do not infer events from previous clips, future clips, the episode story, or common genre patterns.
Do not write a fluent story if the visual evidence is unclear.
Analyze the clip chronologically from beginning to end.
Mention major visible scene changes and major visible actions.
Do not summarize only one frame.

STRICT ANTI-HALLUCINATION RULES:
1. Do not invent characters, monsters, screens, holograms, explosions, vehicles, locations, or objects.
2. If a person, monster, aircraft, screen, statue, beam, or explosion is not clearly visible, do not mention it.
3. If an object is unclear, use "unclear object" or "unknown object" instead of naming it.
4. If a clip is mostly dark, black screen, credits, subtitles, transition frames, or low-information footage, say that directly.
5. If the visible content is too limited, keep the summary short and set visibility_level to "low".
6. Do not use the vocabulary list below as evidence. It is only a naming aid when the object is clearly visible.
7. When uncertain, prefer a cautious description over a specific label.

Important domain vocabulary that may appear in this episode:
helmeted team member, white-red uniform, fighter aircraft, yellow aircraft, cockpit,
control room, command room, laboratory, unknown artifact, metallic cone-shaped object,
holographic female figure, public screen, news screen, golden pyramid, giant stone statue,
giant humanoid statue, giant hero, Ultraman-like hero, dark monster, flying monster,
pterosaur-like creature, energy beam, explosion, smoke, forest, mountain, city street,
ending credits, black screen with text.

EVENT PRIORITY RULES:
- Prioritize the most visually important event in the clip over background details.
- If a major visible action occurs, such as a creature appearing, a statue or structure falling,
  an aircraft moving, a beam firing, an explosion, smoke, a fight, or a character entering/leaving,
  it must be included in the events list.
- If there are multiple visible events, include them in chronological order.
- If there is no clear action, the events list may be empty.

OUTPUT RULES:
Return valid JSON only. Do not use markdown. Return all fields in English only.
Keep every field concise and evidence-based.

Use this exact JSON schema:
{
  "clip_id": number,
  "start_time": string,
  "end_time": string,
  "visibility_level": "clear | partial | low",
  "summary": string,
  "setting": string,
  "main_subjects": [string],
  "events": [
    {
      "action": string,
      "objects": [string],
      "scene": string
    }
  ],
  "evidence": [string],
  "uncertain_parts": [string],
  "possible_hallucination_risks": [string]
}
""".strip()


def build_prompt(clip_id: int, start_time: str, end_time: str) -> str:
    # Do not use str.format() here because the JSON schema contains raw braces.
    metadata = (
        f"The clip_id is {clip_id}.\n"
        f"The clip time range is {start_time} to {end_time}.\n\n"
    )
    return metadata + V3_PROMPT_BODY


def format_time(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def build_messages(clip_path: Path, clip_id: int, start_time: str, end_time: str, nframes: int):
    prompt = build_prompt(
        clip_id=clip_id,
        start_time=start_time,
        end_time=end_time,
    )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(clip_path),
                    "nframes": nframes,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def normalize_video_kwargs(video_kwargs: Dict) -> Dict:
    # qwen_vl_utils may return fps as a list. The processor expects a scalar fps.
    if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
        video_kwargs["fps"] = video_kwargs["fps"][0]
    return video_kwargs


def extract_clip_number_from_name(path: Path) -> Optional[int]:
    # Supports names such as ep01_clip_0003.mp4, clip_0003.mp4, etc.
    match = re.search(r"(\d+)(?=\.mp4$)", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_completed_clip_ids(output_path: Path) -> set:
    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            clip_id = record.get("clip_id")
            has_error = "error" in record
            parse_ok = record.get("json_parse_ok") is True

            # Skip only clips that produced a valid parsed JSON record.
            # Error records or parse-failed records can be retried.
            if isinstance(clip_id, int) and parse_ok and not has_error:
                completed.add(clip_id)

    return completed


def get_selected_clips(all_clips: List[Path]) -> List[Tuple[int, Path]]:
    selected = []

    for index, clip_path in enumerate(all_clips):
        # Use sorted-list index as the default clip_id because your current pipeline does that.
        # If filename contains a number, it should normally match the index.
        clip_id = index

        if RUN_MODE == "test" and clip_id not in TEST_CLIP_IDS:
            continue

        selected.append((clip_id, clip_path))

    return selected


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
print("V3 inference settings:")
print("  VERSION:", VERSION)
print("  RUN_MODE:", RUN_MODE)
print("  TEST_CLIP_IDS:", sorted(TEST_CLIP_IDS) if RUN_MODE == "test" else "N/A")
print("  NFRAMES:", NFRAMES)
print("  FALLBACK_NFRAMES:", FALLBACK_NFRAMES)
print("  MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
print("  OUTPUT_PATH:", OUTPUT_PATH)

all_clips = sorted(CLIP_DIR.glob("*.mp4"))

if not all_clips:
    raise FileNotFoundError(f"No mp4 clips found in: {CLIP_DIR}")

selected_clips = get_selected_clips(all_clips)

if not selected_clips:
    raise RuntimeError("No clips selected. Check RUN_MODE and TEST_CLIP_IDS.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if OVERWRITE_OUTPUT and OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

completed_clip_ids = load_completed_clip_ids(OUTPUT_PATH) if RESUME_OUTPUT else set()

print(f"Total clips found: {len(all_clips)}")
print(f"Clips selected for this run: {len(selected_clips)}")
print(f"Completed clips already in output: {len(completed_clip_ids)}")


def infer_clip(clip_path: Path, clip_id: int, start_time: str, end_time: str, nframes: int) -> str:
    messages = build_messages(
        clip_path=clip_path,
        clip_id=clip_id,
        start_time=start_time,
        end_time=end_time,
        nframes=nframes,
    )

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

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

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

    return output


def run_inference_with_fallback(clip_path: Path, clip_id: int, start_time: str, end_time: str):
    try:
        return infer_clip(
            clip_path=clip_path,
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            nframes=NFRAMES,
        ), NFRAMES, None
    except torch.OutOfMemoryError as first_oom:
        print(f"CUDA out of memory with {NFRAMES} frames. Retrying with {FALLBACK_NFRAMES} frames...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            return infer_clip(
                clip_path=clip_path,
                clip_id=clip_id,
                start_time=start_time,
                end_time=end_time,
                nframes=FALLBACK_NFRAMES,
            ), FALLBACK_NFRAMES, f"OOM at {NFRAMES} frames; retried with {FALLBACK_NFRAMES} frames"
        except torch.OutOfMemoryError:
            raise first_oom


with OUTPUT_PATH.open("a", encoding="utf-8") as f:
    for run_index, (clip_id, clip_path) in enumerate(selected_clips, start=1):
        if clip_id in completed_clip_ids:
            print(f"\nSkipping completed clip {clip_id}: {clip_path.name}")
            continue

        start_seconds = clip_id * SEGMENT_SECONDS
        end_seconds = (clip_id + 1) * SEGMENT_SECONDS

        start_time = format_time(start_seconds)
        end_time = format_time(end_seconds)

        print(f"\nProcessing selected clip {run_index}/{len(selected_clips)}")
        print(f"Clip id: {clip_id}")
        print(f"Clip file: {clip_path.name}")
        print(f"Time range: {start_time} to {end_time}")

        try:
            raw_output, used_nframes, fallback_note = run_inference_with_fallback(
                clip_path=clip_path,
                clip_id=clip_id,
                start_time=start_time,
                end_time=end_time,
            )

            cleaned_output = try_extract_json(raw_output)

            record = {
                "version": VERSION,
                "run_mode": RUN_MODE,
                "clip_id": clip_id,
                "clip_file": clip_path.name,
                "start_time": start_time,
                "end_time": end_time,
                "nframes": used_nframes,
                "fallback_note": fallback_note,
                "raw_output": raw_output,
                "cleaned_output": cleaned_output,
            }

            try:
                parsed = json.loads(cleaned_output)
                record["parsed_json"] = parsed
                record["json_parse_ok"] = True
            except json.JSONDecodeError:
                record["parsed_json"] = None
                record["json_parse_ok"] = False

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print("Raw output:")
            print(raw_output)
            print("JSON parse ok:", record["json_parse_ok"])
            print("Frames used:", used_nframes)

        except torch.OutOfMemoryError:
            print("CUDA out of memory on:", clip_path.name)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            record = {
                "version": VERSION,
                "run_mode": RUN_MODE,
                "clip_id": clip_id,
                "clip_file": clip_path.name,
                "start_time": start_time,
                "end_time": end_time,
                "nframes": None,
                "error": "CUDA out of memory",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        except Exception as e:
            print("Error on:", clip_path.name)
            print(repr(e))

            record = {
                "version": VERSION,
                "run_mode": RUN_MODE,
                "clip_id": clip_id,
                "clip_file": clip_path.name,
                "start_time": start_time,
                "end_time": end_time,
                "nframes": None,
                "error": repr(e),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

print("\nDone.")
print("Saved to:", OUTPUT_PATH)
