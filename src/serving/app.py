"""One process, one resident vLLM engine; run uvicorn with exactly one worker."""
import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from src.workers.inference_worker_v5 import build_prompt
from src.serving.recovery import RecoverySupervisor
from src.serving.output_validation import OutputValidationError, validate_output

log = logging.getLogger(__name__)
DATA = Path(os.getenv('VIDEO_DATA_ROOT', '/data')).resolve()
MODEL = os.getenv('MODEL_PATH', '/model')
MANIFESTS = {
    'ep01': 'episodes/ep01/keyframes/clip_manifest.jsonl',
    'ep02': 'episodes/ep02/keyframes/clip_manifest_flow.jsonl',
}


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    episode: Literal['ep01', 'ep02'] = 'ep02'
    clip_id: int = Field(ge=0, strict=True)
    max_new_tokens: int = Field(default=500, ge=1, le=512, strict=True)
    timeout_seconds: float = Field(default=600, ge=0.1, le=600)
    require_json: bool = True


def initialize():
    records = {}
    for episode, relative in MANIFESTS.items():
        path = DATA / relative
        if path.is_file():
            entries = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            records[episode] = {int(r['clip_id']): r for r in entries}
            if len(records[episode]) != len(entries):
                raise ValueError(f'Duplicate clip IDs in {episode}')
    if not records:
        raise ValueError('No configured manifests found')
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=MODEL, dtype='bfloat16', enforce_eager=True,
        cpu_offload_gb=float(os.getenv('CPU_OFFLOAD_GB', '6')),
        gpu_memory_utilization=float(os.getenv('GPU_MEMORY_UTILIZATION', '0.8')),
        swap_space=0, max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=4096, limit_mm_per_prompt={'image': 8, 'video': 0},
        mm_processor_kwargs={'min_pixels': 4*28*28, 'max_pixels': 128*28*28},
        disable_log_requests=True))
    return records, processor, engine


async def load(app):
    started = time.perf_counter()
    try:
        app.state.records, app.state.processor, app.state.engine = await asyncio.to_thread(initialize)
        app.state.recovery.mark_ready(time.perf_counter()-started)
    except Exception:
        log.exception('Model initialization failed')
        app.state.recovery.fatal('Model initialization failed')


@asynccontextmanager
async def lifespan(app):
    app.state.status = 'loading'
    app.state.engine = None
    app.state.model_load_count = 0
    app.state.active = 0
    app.state.completed = 0
    app.state.failed = 0
    app.state.instance_id = str(uuid.uuid4())
    app.state.recovery = RecoverySupervisor(app.state)
    app.state.recovery.start()
    task = asyncio.create_task(load(app))
    monitor = asyncio.create_task(app.state.recovery.monitor())
    try:
        yield
    finally:
        app.state.recovery.close()
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        # Normal shutdown completes the tracked initializer. Fatal failures use
        # process exit instead, avoiding waits on a dead CUDA engine.
        await task
        if app.state.engine is not None:
            app.state.engine.shutdown_background_loop()


app = FastAPI(title='VideoThinker resident inference', lifespan=lifespan)


@app.get('/health/live')
async def live():
    return {'status': 'alive'}


@app.get('/health/ready')
async def ready(request: Request):
    state = request.app.state
    if state.status != 'ready':
        raise HTTPException(503, detail={'status': state.status})
    try:
        async with asyncio.timeout(state.recovery.config.check_timeout):
            await state.engine.check_health()
    except Exception:
        if state.engine.errored or state.engine.is_stopped:
            state.recovery.fatal('Readiness detected a stopped or errored engine')
        raise HTTPException(503, detail={'status': 'unhealthy'})
    return {'status': 'ready', 'instance_id': state.instance_id,
            'model_load_count': state.model_load_count, 'load_seconds': state.load_seconds,
            'active_requests': state.active, 'completed_requests': state.completed,
            'failed_requests': state.failed, 'max_active_requests': 1}


def prepare(state, record):
    frames = sorted(record['selected_frames'], key=lambda f: f['local_timestamp_sec'])
    if not 1 <= len(frames) <= 8:
        raise ValueError('Clip must contain between 1 and 8 frames')
    content = []
    for frame in frames:
        raw = frame['image_path'].replace('\\', '/')
        relative = raw.split('/longvideo/', 1)[1] if '/longvideo/' in raw else raw
        path = (DATA / relative).resolve()
        if not path.is_relative_to(DATA) or not path.is_file():
            raise ValueError('Frame missing or outside data root')
        content.append({'type': 'image', 'image': str(path),
                        'min_pixels': 4*28*28, 'max_pixels': 128*28*28})
    content.append({'type': 'text', 'text': build_prompt(record, frames)})
    messages = [{'role': 'user', 'content': content}]
    prompt = state.processor.apply_chat_template(messages, tokenize=False,
                         add_generation_prompt=True, add_vision_id=True)
    images, _ = process_vision_info(messages)
    return {'prompt': prompt, 'multi_modal_data': {'image': images}}


async def generate(state, record, body, request_id):
    inputs = await asyncio.to_thread(prepare, state, record)
    final = None
    async for output in state.engine.generate(inputs,
            SamplingParams(temperature=0, max_tokens=body.max_new_tokens), request_id):
        final = output
    if final is None or not final.outputs:
        raise RuntimeError('Engine returned no output')
    return final


async def disconnected(request, stopping):
    # Starlette's receive probe uses an AnyIO cancellation scope; it can absorb
    # Task.cancel() during a probe. An explicit stop flag also ends the watcher.
    while not stopping.is_set():
        if await request.is_disconnected():
            return True
        await asyncio.sleep(0.2)
    return False


@app.post('/v1/infer')
async def infer(body: InferenceRequest, request: Request):
    state = request.app.state
    if state.status != 'ready':
        raise HTTPException(503, 'Model is not ready')
    record = state.records.get(body.episode, {}).get(body.clip_id)
    if record is None:
        raise HTTPException(404, 'Episode or clip not found')
    # Atomic within this event loop: no await between check and increment.
    if state.active >= 1:
        raise HTTPException(429, 'Inference capacity reached', headers={'Retry-After': '5'})
    state.active += 1
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    work = asyncio.create_task(generate(state, record, body, request_id))
    disconnect_stopping = asyncio.Event()
    disconnect = asyncio.create_task(disconnected(request, disconnect_stopping))
    try:
        async with asyncio.timeout(body.timeout_seconds):
            done, _ = await asyncio.wait([work, disconnect], return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                raise HTTPException(499, 'Client disconnected')
            output = work.result()
        if state.status != 'ready':
            raise HTTPException(503, 'Engine failed before response completion')
        text = output.outputs[0].text
        parsed = None
        validation_errors = []
        try:
            parsed = validate_output(text, record, len(record['selected_frames']))
        except OutputValidationError as error:
            validation_errors = error.issues
            if body.require_json:
                raise HTTPException(502, detail={'error': 'Model output failed validation',
                    'request_id': request_id, 'finish_reason': output.outputs[0].finish_reason,
                    'validation_errors': validation_errors})
        state.completed += 1
        return {'request_id': request_id, 'instance_id': state.instance_id,
                'clip_id': body.clip_id, 'text': text, 'result': parsed,
                'structured_output_valid': parsed is not None,
                'validation_errors': validation_errors,
                'finish_reason': output.outputs[0].finish_reason,
                'input_tokens': len(output.prompt_token_ids),
                'output_tokens': len(output.outputs[0].token_ids),
                'elapsed_seconds': time.perf_counter()-started}
    except TimeoutError:
        state.failed += 1
        raise HTTPException(504, detail={'error': 'Inference deadline exceeded', 'request_id': request_id})
    except HTTPException:
        state.failed += 1
        raise
    except asyncio.CancelledError:
        state.failed += 1
        raise
    except Exception:
        state.failed += 1
        log.exception('Inference failed: %s', request_id)
        if state.engine.errored or state.engine.is_stopped:
            state.recovery.fatal('Inference detected a stopped or errored engine')
        raise HTTPException(500, detail={'error': 'Inference failed', 'request_id': request_id})
    finally:
        disconnect_stopping.set()
        for task in [work, disconnect]:
            task.cancel()
        await asyncio.gather(work, disconnect, return_exceptions=True)
        if state.engine.is_running:
            with contextlib.suppress(Exception):
                await state.engine.abort(request_id)
        state.active -= 1
