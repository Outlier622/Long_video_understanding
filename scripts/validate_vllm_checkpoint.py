"""Bounded offline vLLM smoke test; CPU offload is not a serving benchmark."""
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

sys.path.insert(0, '/workspace')
from src.workers.inference_worker_v5 import build_prompt

out = Path('/results')
out.mkdir(exist_ok=True)
result = {'scope': 'CPU-offloaded short-generation compatibility smoke', 'measurements': []}
def save():
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

try:
    processor = AutoProcessor.from_pretrained('/model', local_files_only=True)
    started = time.perf_counter()
    llm = LLM(model='/model', dtype='bfloat16', enforce_eager=True,
              cpu_offload_gb=6, gpu_memory_utilization=0.8, swap_space=0,
              max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=4096,
              limit_mm_per_prompt={'image': 8, 'video': 0},
              mm_processor_kwargs={'min_pixels': 4*28*28, 'max_pixels': 128*28*28})
    result['load_seconds'] = time.perf_counter()-started
    result['model_load_count'] = 1
    save()
    records = [json.loads(line) for line in Path('/data/episodes/ep02/keyframes/clip_manifest_flow.jsonl').read_text().splitlines() if line.strip()]
    for clip_id in [1, 3]:
        record = next(r for r in records if int(r['clip_id']) == clip_id)
        frames = sorted(record['selected_frames'], key=lambda f: f['local_timestamp_sec'])
        content = []
        for frame in frames:
            path = frame['image_path'].replace('\\', '/')
            path = '/data/' + path.split('/longvideo/', 1)[1]
            content.append({'type': 'image', 'image': path, 'min_pixels': 4*28*28, 'max_pixels': 128*28*28})
        content.append({'type': 'text', 'text': build_prompt(record, frames)})
        messages = [{'role': 'user', 'content': content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        images, _ = process_vision_info(messages)
        started = time.perf_counter()
        response = llm.generate({'prompt': prompt, 'multi_modal_data': {'image': images}}, SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)[0]
        item = {'clip_id': clip_id, 'seconds': time.perf_counter()-started,
                'input_tokens': len(response.prompt_token_ids),
                'output_tokens': len(response.outputs[0].token_ids), 'text': response.outputs[0].text,
                'finish_reason': response.outputs[0].finish_reason}
        result['measurements'].append(item)
        print(json.dumps(item), flush=True)
        save()
    result['status'] = 'smoke_passed'
except Exception:
    result['status'] = 'failed'
    result['error'] = traceback.format_exc()
    raise
finally:
    save()
