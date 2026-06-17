import json
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH = r"D:\projects\VideoThinker-R1-3B"
CLIP_DIR = Path(r"D:\projects\longvideo\clips")
OUTPUT_PATH = Path(r"D:\projects\longvideo\outputs\ep01_results.jsonl")

SEGMENT_SECONDS = 20
NFRAMES = 8
MAX_NEW_TOKENS = 300


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

clips = sorted(CLIP_DIR.glob("*.mp4"))

if not clips:
    raise FileNotFoundError(f"No mp4 clips found in: {CLIP_DIR}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def infer_clip(clip_path: Path, clip_id: int, start_time: str, end_time: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(clip_path),
                    "nframes": NFRAMES,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 64 * 28 * 28,
                },
                {
                    "type": "text",
                    "text": (
                        f"Analyze this video clip. The clip_id is {clip_id}. "
                        f"The clip time range is {start_time} to {end_time}. "
                        "Return valid JSON only. Do not use markdown. "
                        "Return all fields in English only. "
                        "Describe only what is visibly shown in the clip. "
                        "Do not guess the story, character names, or hidden meaning. "
                        "Keep every field concise. "
                        "Use this exact schema: "
                        "{"
                        "\"clip_id\": number, "
                        "\"start_time\": string, "
                        "\"end_time\": string, "
                        "\"summary\": string, "
                        "\"setting\": string, "
                        "\"main_subjects\": [string], "
                        "\"events\": ["
                        "{"
                        "\"action\": string, "
                        "\"objects\": [string], "
                        "\"scene\": string"
                        "}"
                        "]"
                        "}"
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )

    if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
        video_kwargs["fps"] = video_kwargs["fps"][0]

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


with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for i, clip_path in enumerate(clips):
        start_seconds = i * SEGMENT_SECONDS
        end_seconds = (i + 1) * SEGMENT_SECONDS

        start_time = format_time(start_seconds)
        end_time = format_time(end_seconds)

        print(f"\nProcessing clip {i + 1}/{len(clips)}: {clip_path.name}")
        print(f"Time range: {start_time} to {end_time}")

        try:
            raw_output = infer_clip(
                clip_path=clip_path,
                clip_id=i,
                start_time=start_time,
                end_time=end_time,
            )

            cleaned_output = try_extract_json(raw_output)

            record = {
                "clip_id": i,
                "clip_file": clip_path.name,
                "start_time": start_time,
                "end_time": end_time,
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

        except torch.OutOfMemoryError:
            print("CUDA out of memory on:", clip_path.name)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            record = {
                "clip_id": i,
                "clip_file": clip_path.name,
                "start_time": start_time,
                "end_time": end_time,
                "error": "CUDA out of memory",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        except Exception as e:
            print("Error on:", clip_path.name)
            print(repr(e))

            record = {
                "clip_id": i,
                "clip_file": clip_path.name,
                "start_time": start_time,
                "end_time": end_time,
                "error": repr(e),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

print("\nDone.")
print("Saved to:", OUTPUT_PATH)