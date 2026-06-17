import json
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH = r"D:\projects\VideoThinker-R1-3B"
CLIP_DIR = Path(r"D:\projects\longvideo\clips")
OUTPUT_PATH = Path(r"D:\projects\longvideo\outputs\ep01_results_v2.jsonl")

SEGMENT_SECONDS = 20
NFRAMES = 12
FALLBACK_NFRAMES = 8
MAX_NEW_TOKENS = 700
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 64 * 28 * 28


V2_PROMPT_BODY = """
You are analyzing a short clip from a sci-fi / tokusatsu-style episode.

Your goal is visual accuracy. Do not write a fluent story if the visual evidence is unclear.
Analyze the clip chronologically from beginning to end.
Mention all major scene changes and major visible events.
Do not summarize only one frame.

Important domain objects that may appear in this video include:
helmeted team member, white-red uniform, fighter aircraft, yellow aircraft, cockpit,
control room, command room, laboratory, unknown artifact, metallic cone-shaped object,
holographic female figure, public screen, news screen, golden pyramid, giant stone statue,
giant humanoid statue, giant hero, Ultraman-like hero, dark monster, flying monster,
pterosaur-like creature, energy beam, explosion, smoke, forest, mountain, city street,
ending credits, black screen with text.

Do not replace sci-fi objects with ordinary daily-life objects unless they are clearly visible.
Avoid unsupported guesses such as car, smartphone, laptop, flower arrangement, couch,
remote control, horse, coffee cup, office, living room, or toy airplane unless they are
clearly shown in the clip.

If an object is unclear, write "unclear object" instead of inventing a specific object.
If the clip contains subtitles, credits, or black-screen text, mention that directly.
If the clip shows multiple events, include multiple events in chronological order.

Return valid JSON only. Do not use markdown. Return all fields in English only.
Keep each field concise but accurate.

Use this exact JSON schema:
{
  "clip_id": number,
  "start_time": string,
  "end_time": string,
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
  "uncertain_parts": [string],
  "possible_hallucination_risks": [string]
}
""".strip()


def build_prompt(clip_id: int, start_time: str, end_time: str) -> str:
    """
    Build the prompt without str.format(), because the JSON schema contains raw braces.
    Using f-string only for the metadata block avoids KeyError caused by JSON braces.
    """
    metadata = (
        f"The clip_id is {clip_id}.\n"
        f"The clip time range is {start_time} to {end_time}.\n\n"
    )
    return metadata + V2_PROMPT_BODY


def format_time(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def clean_json_text(text: str) -> str:
    """
    Remove markdown code fences such as:
    ```json
    {...}
    ```
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()

    if text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text.strip()


def try_extract_json(text: str) -> str:
    """
    If the model returns extra text before or after JSON,
    try to extract the content between the first { and the last }.
    """
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


def normalize_video_kwargs(video_kwargs: dict) -> dict:
    """
    qwen_vl_utils may return fps as a list. The processor expects a scalar fps.
    """
    if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
        video_kwargs["fps"] = video_kwargs["fps"][0]
    return video_kwargs


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
print("V2 inference settings:")
print("  NFRAMES:", NFRAMES)
print("  FALLBACK_NFRAMES:", FALLBACK_NFRAMES)
print("  MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
print("  OUTPUT_PATH:", OUTPUT_PATH)

clips = sorted(CLIP_DIR.glob("*.mp4"))

if not clips:
    raise FileNotFoundError(f"No mp4 clips found in: {CLIP_DIR}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def infer_clip(clip_path: Path, clip_id: int, start_time: str, end_time: str, nframes: int):
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


with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for i, clip_path in enumerate(clips):
        start_seconds = i * SEGMENT_SECONDS
        end_seconds = (i + 1) * SEGMENT_SECONDS

        start_time = format_time(start_seconds)
        end_time = format_time(end_seconds)

        print(f"\nProcessing clip {i + 1}/{len(clips)}: {clip_path.name}")
        print(f"Time range: {start_time} to {end_time}")

        try:
            raw_output, used_nframes, fallback_note = run_inference_with_fallback(
                clip_path=clip_path,
                clip_id=i,
                start_time=start_time,
                end_time=end_time,
            )

            cleaned_output = try_extract_json(raw_output)

            record = {
                "version": "v2_domain_accuracy_prompt_fixed",
                "clip_id": i,
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
                "version": "v2_domain_accuracy_prompt_fixed",
                "clip_id": i,
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
                "version": "v2_domain_accuracy_prompt_fixed",
                "clip_id": i,
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
