"""Test-only launcher: filesystem markers inject one-shot faults, no HTTP hook.

Mount this file read-only and mount a fresh writable /faults directory. Never
include this launcher in the serving image or its normal entrypoint.
"""
import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import src.serving.app as service

FAULTS = Path('/faults')
original_initialize = service.initialize


def consume(name):
    path = FAULTS / name
    try:
        path.rename(FAULTS / (name+'.consumed'))
        return True
    except FileNotFoundError:
        return False


def initialize():
    if consume('initialization.fail'):
        # Exercise actual local processor/model initialization failure.
        service.MODEL = '/missing-model-for-recovery-test'
    return original_initialize()


async def inject_runtime_fault(app):
    while True:
        if app.state.status == 'ready' and consume('runtime.fail'):
            # Same error state vLLM records when its engine loop crashes.
            app.state.engine.set_errored(RuntimeError('Injected engine failure'))
            print('TEST: vLLM runtime error state injected', flush=True)
        await asyncio.sleep(0.2)


@asynccontextmanager
async def lifespan(app):
    async with service.lifespan(app):
        task = asyncio.create_task(inject_runtime_fault(app))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


service.initialize = initialize
service.app.router.lifespan_context = lifespan
uvicorn.run(service.app, host='0.0.0.0', port=8000, workers=1)
