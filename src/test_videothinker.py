import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = r"D:\projects\VideoThinker-R1-3B"
video_uri = r"D:\projects\longvideo\videos\ep01_5s.mp4"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

print("Model loaded successfully.")
print("CUDA available:", torch.cuda.is_available())

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_uri,
                "fps": 0.5,
                "min_pixels": 4 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            },
            {
                "type": "text",
                "text": (
                    "Describe the main events in this video. "
                    "Return a concise answer."
                ),
            },
        ],
    }
]

print("Applying chat template...")
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("Processing video...")
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    return_video_kwargs=True,
)
print("video_kwargs before fix:", video_kwargs)

if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
    video_kwargs["fps"] = video_kwargs["fps"][0]

print("video_kwargs after fix:", video_kwargs)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)

inputs = inputs.to(model.device)

print("Generating...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n===== OUTPUT =====")
print(output[0])