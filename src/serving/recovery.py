"""Fail closed and exit so the container runtime can replace a broken engine."""
import asyncio
import logging
import math
import os
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoveryConfig:
    check_interval: float = 5
    check_timeout: float = 5
    failure_limit: int = 3
    startup_timeout: float = 600
    unresponsive_timeout: float = 60
    exit_delay: float = 1

    @classmethod
    def from_environment(cls):
        values = dict(
            check_interval=float(os.getenv('ENGINE_CHECK_INTERVAL_SECONDS', '5')),
            check_timeout=float(os.getenv('ENGINE_CHECK_TIMEOUT_SECONDS', '5')),
            failure_limit=int(os.getenv('ENGINE_CHECK_FAILURE_LIMIT', '3')),
            startup_timeout=float(os.getenv('MODEL_STARTUP_TIMEOUT_SECONDS', '600')),
            unresponsive_timeout=float(os.getenv('SERVICE_UNRESPONSIVE_TIMEOUT_SECONDS', '60')),
        )
        config = cls(**values)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('Recovery settings must be finite positive values')
        if config.unresponsive_timeout <= config.check_interval + config.check_timeout:
            raise ValueError('Unresponsive timeout must exceed check interval plus timeout')
        return config


class RecoverySupervisor:
    def __init__(self, state, config=None, exit_process=os._exit):
        self.state = state
        self.config = config or RecoveryConfig.from_environment()
        self.exit_process = exit_process
        self.started = time.monotonic()
        self.heartbeat = self.started
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.exit_timer = None
        self.watchdog = None
        self.failure_count = 0
        self.state.fatal_reason = None

    def fatal(self, reason):
        with self.lock:
            if self.closed.is_set() or self.state.fatal_reason is not None:
                return
            self.state.fatal_reason = reason
            self.state.status = 'failed'
            log.critical('Fatal serving failure; exiting with code 1: %s', reason)
            # A separate timer can terminate even if the asyncio loop is stuck.
            # Hard exit deliberately abandons a broken engine and releases CUDA
            # resources through process teardown instead of awaiting its tasks.
            self.exit_timer = threading.Timer(self.config.exit_delay, self.exit_process, args=(1,))
            self.exit_timer.daemon = True
            self.exit_timer.start()

    def mark_alive(self):
        with self.lock:
            self.heartbeat = time.monotonic()

    def mark_ready(self, load_seconds):
        with self.lock:
            if self.closed.is_set() or self.state.fatal_reason is not None:
                return False
            self.state.model_load_count = 1
            self.state.load_seconds = load_seconds
            self.state.status = 'ready'
            self.heartbeat = time.monotonic()
            return True

    def check_deadlines(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.state.status == 'loading' and now-self.started > self.config.startup_timeout:
                self.fatal('Model initialization exceeded startup deadline')
            elif self.state.status == 'ready' and now-self.heartbeat > self.config.unresponsive_timeout:
                self.fatal('Service health loop stopped responding')

    def start(self):
        def watch():
            while not self.closed.wait(0.5):
                self.check_deadlines()
        self.watchdog = threading.Thread(target=watch, name='serving-watchdog', daemon=True)
        self.watchdog.start()

    async def check_engine(self):
        if self.state.status != 'ready':
            return
        engine = self.state.engine
        if engine.errored or engine.is_stopped:
            self.fatal('vLLM engine entered a stopped or errored state')
            return
        try:
            async with asyncio.timeout(self.config.check_timeout):
                await engine.check_health()
        except Exception as error:
            self.failure_count += 1
            log.warning('Engine health check failed (%s/%s): %r',
                        self.failure_count, self.config.failure_limit, error)
            if self.failure_count >= self.config.failure_limit:
                self.fatal('Repeated engine health check failures')
        else:
            self.failure_count = 0

    async def monitor(self):
        while not self.closed.is_set():
            self.mark_alive()
            await self.check_engine()
            await asyncio.sleep(self.config.check_interval)

    def close(self):
        with self.lock:
            self.closed.set()
            if self.exit_timer:
                self.exit_timer.cancel()
        if self.watchdog:
            self.watchdog.join(timeout=1)
