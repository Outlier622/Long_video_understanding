"""Bounded recovery-policy tests without loading model weights or CUDA."""
import asyncio
import threading
import unittest
from types import SimpleNamespace

from src.serving.recovery import RecoveryConfig, RecoverySupervisor


class FakeEngine:
    errored = False
    is_stopped = False

    def __init__(self):
        self.error = None
        self.hang = False
        self.checks = 0

    async def check_health(self):
        self.checks += 1
        if self.hang:
            await asyncio.Event().wait()
        if self.error:
            raise self.error


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.state = SimpleNamespace(status='ready', engine=self.engine)
        self.exits = []
        self.exited = threading.Event()

        def exit_process(code):
            self.exits.append(code)
            self.exited.set()

        self.supervisor = RecoverySupervisor(self.state, RecoveryConfig(
            check_interval=0.01, check_timeout=0.02, failure_limit=2,
            startup_timeout=1, unresponsive_timeout=1, exit_delay=0.01), exit_process)

    def tearDown(self):
        self.supervisor.close()

    async def test_healthy_checks_do_not_exit(self):
        await self.supervisor.check_engine()
        self.assertEqual(self.engine.checks, 1)
        self.assertEqual(self.state.status, 'ready')
        self.assertEqual(self.exits, [])

    async def test_stopped_engine_fails_closed_and_exits_once(self):
        self.engine.is_stopped = True
        await self.supervisor.check_engine()
        self.supervisor.fatal('second observer of same failure')
        self.assertEqual(self.state.status, 'failed')
        self.assertTrue(self.exited.wait(0.5))
        self.assertEqual(self.exits, [1])

    async def test_transient_health_failure_resets_after_success(self):
        self.engine.error = RuntimeError('temporary error')
        await self.supervisor.check_engine()
        self.assertEqual(self.state.status, 'ready')
        self.engine.error = None
        await self.supervisor.check_engine()
        self.assertEqual(self.supervisor.failure_count, 0)
        self.assertEqual(self.exits, [])

    async def test_repeated_health_errors_exit(self):
        self.engine.error = RuntimeError('worker unavailable')
        await self.supervisor.check_engine()
        await self.supervisor.check_engine()
        self.assertEqual(self.state.status, 'failed')
        self.assertTrue(self.exited.wait(0.5))

    async def test_hung_async_health_checks_have_deadlines(self):
        self.engine.hang = True
        await self.supervisor.check_engine()
        await self.supervisor.check_engine()
        self.assertEqual(self.state.status, 'failed')
        self.assertTrue(self.exited.wait(0.5))

    async def test_monitor_checks_without_http_requests(self):
        task = asyncio.create_task(self.supervisor.monitor())
        await asyncio.sleep(0.04)
        self.supervisor.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertGreaterEqual(self.engine.checks, 2)
        self.assertEqual(self.exits, [])

    async def test_startup_deadline(self):
        self.state.status = 'loading'
        self.supervisor.check_deadlines(self.supervisor.started+2)
        self.assertIn('startup deadline', self.state.fatal_reason)
        self.assertTrue(self.exited.wait(0.5))

    async def test_late_initializer_cannot_override_fatal_state(self):
        self.state.status = 'loading'
        self.supervisor.check_deadlines(self.supervisor.started+2)
        self.assertFalse(self.supervisor.mark_ready(2))
        self.assertEqual(self.state.status, 'failed')
        self.assertTrue(self.exited.wait(0.5))

    async def test_watchdog_does_not_need_async_loop_progress(self):
        # This deliberately blocks this test's event loop. The watchdog and exit
        # timer must still run independently on their own threads.
        self.supervisor.heartbeat -= 2
        self.supervisor.start()
        self.assertTrue(self.exited.wait(1.5))
        self.assertIn('stopped responding', self.state.fatal_reason)

    async def test_normal_shutdown_suppresses_failure_exit(self):
        self.supervisor.close()
        self.supervisor.fatal('late shutdown exception')
        self.assertEqual(self.exits, [])
        self.assertEqual(self.state.status, 'ready')


if __name__ == '__main__':
    unittest.main()
