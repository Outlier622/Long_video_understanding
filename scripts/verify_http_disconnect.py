"""Verify final deployment defaults, client disconnect cleanup and recovery."""
import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=False)
result = {'status': 'running'}

def call(path, body=None):
    request = urllib.request.Request(args.url+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)

try:
    deadline = time.monotonic()+360
    while True:
        try:
            before = call('/health/ready')
            break
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() > deadline:
                raise TimeoutError('Service did not become ready')
            time.sleep(2)
    result['before'] = before
    spec = call('/openapi.json')
    assert spec['components']['schemas']['InferenceRequest']['properties']['timeout_seconds']['default'] == 600
    result['default_timeout_verified'] = 600
    url = urlsplit(args.url)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=20)
    connection.request('POST', '/v1/infer', body=json.dumps({'clip_id': 1}),
                       headers={'Content-Type': 'application/json'})
    deadline = time.monotonic()+20
    while call('/health/ready')['active_requests'] != 1:
        if time.monotonic() > deadline:
            raise TimeoutError('Request did not become active')
        time.sleep(0.1)
    # Leave time for submission to the engine, then abandon the HTTP connection.
    time.sleep(3)
    started = time.monotonic()
    connection.close()
    deadline = time.monotonic()+30
    while True:
        state = call('/health/ready')
        if state['active_requests'] == 0:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Disconnect did not release admission capacity')
        time.sleep(0.2)
    result['disconnect_release_seconds'] = time.monotonic()-started
    assert state['failed_requests'] == before['failed_requests']+1
    recovered = call('/v1/infer', {'clip_id': 3, 'max_new_tokens': 2, 'require_json': False})
    assert recovered['output_tokens'] == 2
    assert recovered['instance_id'] == before['instance_id']
    after = call('/health/ready')
    assert after['active_requests'] == 0 and after['model_load_count'] == 1
    result.update(after=after, recovered=recovered, status='passed')
    print(json.dumps(result, indent=2), flush=True)
except Exception as error:
    result.update(status='failed', error=repr(error))
    raise
finally:
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
