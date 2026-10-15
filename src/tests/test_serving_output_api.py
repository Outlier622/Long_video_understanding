"""Real FastAPI route/lifespan checks using synthetic model outputs; no GPU."""
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import src.serving.app as service
from src.tests.test_output_validation import record, valid_output


class FakeEngine:
    errored = False
    is_stopped = False
    is_running = False

    async def check_health(self):
        pass

    def shutdown_background_loop(self):
        pass


class OutputApiTests(unittest.TestCase):
    def setUp(self):
        self.text = json.dumps(valid_output())

        async def generate(*args):
            return SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                text=self.text, token_ids=[1], finish_reason='stop')])

        self.init_patch = patch.object(service, 'initialize', return_value=(
            {'ep02': {1: record()}}, None, FakeEngine()))
        self.generate_patch = patch.object(service, 'generate', generate)
        self.init_patch.start()
        self.generate_patch.start()
        self.addCleanup(self.init_patch.stop)
        self.addCleanup(self.generate_patch.stop)
        self.client = TestClient(service.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        deadline = time.monotonic()+5
        while self.client.get('/health/ready').status_code != 200:
            if time.monotonic() > deadline:
                self.fail('Synthetic engine did not become ready')
            time.sleep(0.01)
        self.instance = self.client.get('/health/ready').json()['instance_id']

    def test_valid_output_success(self):
        response = self.client.post('/v1/infer', json={'clip_id': 1})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body['structured_output_valid'])
        self.assertEqual(body['result'], valid_output())
        self.assertEqual(body['validation_errors'], [])

    def test_strict_rejection_reports_issues_without_restart(self):
        value = valid_output()
        value['summary'] = ' '.join(['word']*36)
        value['events'][0]['frame_references'] = ['Frame 08']
        self.text = json.dumps(value)
        response = self.client.post('/v1/infer', json={'clip_id': 1})
        self.assertEqual(response.status_code, 502)
        detail = response.json()['detail']
        self.assertIn('request_id', detail)
        self.assertEqual({i['code'] for i in detail['validation_errors']},
                         {'word_limit', 'invalid_frame_reference'})
        health = self.client.get('/health/ready')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()['instance_id'], self.instance)
        self.assertEqual(health.json()['active_requests'], 0)

    def test_diagnostic_response_keeps_original_text_and_failure_details(self):
        self.text = '{'
        response = self.client.post('/v1/infer', json={'clip_id': 1, 'require_json': False})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body['structured_output_valid'])
        self.assertIsNone(body['result'])
        self.assertEqual(body['text'], self.text)
        self.assertEqual(body['validation_errors'][0]['code'], 'invalid_json')


if __name__ == '__main__':
    unittest.main()
