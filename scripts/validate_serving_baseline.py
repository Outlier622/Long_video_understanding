"""Read-only model/input validation and bounded local inference smoke test.

Writes measurements to a new run directory; never changes model configuration.
Short outputs test runtime compatibility, not structured-output quality.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.workers.inference_worker_v5 import build_prompt, normalize_video_kwargs
from qwen_vl_utils import process_vision_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--gpu-memory-gib', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    result = {
        'scope': 'local CPU-offloaded short-generation smoke test, not production benchmark',
        'model': args.model, 'manifest': args.manifest,
        'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in
                     ['torch', 'transformers', 'accelerate', 'qwen-vl-utils']},
        'max_new_tokens': args.max_new_tokens,
        'measurements': [],
    }

    def save():
        (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    try:
        records = [json.loads(line) for line in Path(args.manifest).read_text(
            encoding='utf-8').splitlines() if line.strip()]
        ids = [int(r['clip_id']) for r in records]
        selected = [next(r for r in records if int(r['clip_id']) == i) for i in [1, 3, 14]]
        result['input_validation'] = {
            'record_count': len(records), 'duplicate_ids': len(ids) - len(set(ids)),
            'missing_ids_in_zero_based_range': sorted(set(range(max(ids)+1)) - set(ids)),
            'selected_clip_ids': [r['clip_id'] for r in selected],
            'missing_frame_paths': [f['image_path'] for r in selected
                                    for f in r['selected_frames'] if not Path(f['image_path']).is_file()],
        }
        if result['input_validation']['missing_frame_paths']:
            raise FileNotFoundError('Selected input frames are missing')
        result['gpu'] = torch.cuda.get_device_name(0)
        result['gpu_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
        save()
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map='auto',
            max_memory={0: f'{args.gpu_memory_gib}GiB', 'cpu': '16GiB'},
            local_files_only=True, attn_implementation='sdpa')
        model.eval()
        torch.cuda.synchronize()
        result['model_load_seconds'] = time.perf_counter() - started
        result['model_load_count'] = 1
        result['device_map'] = {k: str(v) for k, v in model.hf_device_map.items()}
        result['config_use_cache'] = getattr(model.config, 'use_cache', None)
        result['text_config_use_cache'] = getattr(model.config.get_text_config(), 'use_cache', None)
        result['generation_config_use_cache'] = model.generation_config.use_cache
        print(json.dumps({'loaded': True, 'seconds': result['model_load_seconds'],
                          'device_map': result['device_map']}, indent=2), flush=True)
        save()

        # First compare cache off/on on identical input, then process two new clips.
        for record, cache in [(selected[0], False), (selected[0], True),
                              (selected[1], True), (selected[2], True)]:
            frames = sorted(record['selected_frames'], key=lambda f: f['local_timestamp_sec'])
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': f['image_path'], 'min_pixels': 4*28*28,
                 'max_pixels': 128*28*28} for f in frames] +
                [{'type': 'text', 'text': build_prompt(record, frames)}]}]
            started = time.perf_counter()
            text = processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, add_vision_id=True)
            images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
            processor_kwargs = dict(text=[text], images=images, padding=True, return_tensors='pt')
            if videos:
                processor_kwargs.update(videos=videos, **normalize_video_kwargs(kwargs))
            inputs = processor(**processor_kwargs).to(model.device)
            torch.cuda.synchronize()
            preprocess_seconds = time.perf_counter() - started
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, use_cache=cache)
            torch.cuda.synchronize()
            generate_seconds = time.perf_counter() - started
            started = time.perf_counter()
            output_ids = generated[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
            decode_seconds = time.perf_counter() - started
            measurement = {
                'clip_id': record['clip_id'], 'use_cache': cache,
                'frame_count': len(frames), 'input_tokens': inputs.input_ids.shape[1],
                'output_tokens': output_ids.shape[1], 'preprocess_seconds': preprocess_seconds,
                'generate_seconds': generate_seconds, 'decode_seconds': decode_seconds,
                'generation_tokens_per_second': output_ids.shape[1]/generate_seconds,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
                'output_text': output_text,
                'hit_token_budget': output_ids.shape[1] == args.max_new_tokens,
            }
            result['measurements'].append(measurement)
            print(json.dumps(measurement), flush=True)
            save()
        result['status'] = 'smoke_passed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        save()
        raise
    save()


if __name__ == '__main__':
    main()
