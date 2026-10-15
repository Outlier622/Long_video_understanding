"""HTTP/GPU recovery checks against the isolated fault-injection container.

Start recovery_fault_runner.py with a fresh /faults mount and an
initialization.fail marker first. Docker must use restart=unless-stopped.
"""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8001')
parser.add_argument('--fault-dir', required=True)
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
faults = Path(args.fault_dir)
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=False)
result = {'status': 'running', 'checks': [], 'observations': []}


def save():
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def call(path, body=None, timeout=15):
    request = urllib.request.Request(args.url+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)
    except (OSError, ConnectionError) as error:
        return 0, {'transport_error': repr(error)}


def check(name, response, expected):
    code, body = response
    result['checks'].append({'name': name, 'status_code': code, 'body': body})
    save()
    assert code == expected, (name, code, body)
    print(f'{name}: {code}', flush=True)
    return body


def wait_ready(previous=None):
    deadline = time.monotonic()+720
    last = None
    while time.monotonic() < deadline:
        code, body = call('/health/ready')
        observation = (code, body.get('detail'), body.get('instance_id'))
        if observation != last:
            result['observations'].append({'status_code': code, 'body': body})
            save()
            last = observation
        if code == 200 and body['instance_id'] != previous:
            return body
        time.sleep(2)
    raise TimeoutError('Service did not recover before deadline')


try:
    started = time.monotonic()
    first = wait_ready()
    assert (faults / 'initialization.fail.consumed').is_file()
    assert first['model_load_count'] == 1
    result['initialization_recovery_seconds'] = time.monotonic()-started
    result['initial_ready'] = first
    save()
    check('inference_after_initialization_recovery', call('/v1/infer', {
        'clip_id': 1, 'max_new_tokens': 2, 'require_json': False}), 200)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pending = pool.submit(call, '/v1/infer', {
            'clip_id': 1, 'timeout_seconds': 600}, 660)
        deadline = time.monotonic()+20
        while call('/health/ready')[1].get('active_requests') != 1:
            if time.monotonic() > deadline:
                raise TimeoutError('Request did not become active')
            time.sleep(0.2)
        time.sleep(3)
        started = time.monotonic()
        (faults / 'runtime.fail').touch(exist_ok=False)
        # No readiness calls during this interval: autonomous monitoring must
        # detect the engine fault even without a probe driving detection.
        time.sleep(8)
        recovered = wait_ready(first['instance_id'])
        result['runtime_recovery_seconds'] = time.monotonic()-started
        interrupted = pending.result()
        assert interrupted[0] != 200, interrupted
        result['interrupted_request'] = {'status_code': interrupted[0], 'body': interrupted[1]}
    assert (faults / 'runtime.fail.consumed').is_file()
    assert recovered['model_load_count'] == 1 and recovered['active_requests'] == 0
    result['recovered_ready'] = recovered
    save()
    response = check('inference_after_runtime_recovery', call('/v1/infer', {
        'clip_id': 1, 'max_new_tokens': 2, 'require_json': False}), 200)
    assert response['instance_id'] == recovered['instance_id']
    assert response['output_tokens'] == 2
    check('ordinary_timeout_is_not_fatal', call('/v1/infer', {
        'clip_id': 3, 'timeout_seconds': 1, 'require_json': False}), 504)
    check('invalid_output_is_not_fatal', call('/v1/infer', {
        'clip_id': 1, 'max_new_tokens': 1}), 502)
    check('invalid_input_is_not_fatal', call('/v1/infer', {'clip_id': -1}), 422)
    final = check('final_ready', call('/health/ready'), 200)
    assert final['instance_id'] == recovered['instance_id']
    assert final['model_load_count'] == 1 and final['active_requests'] == 0
    result['status'] = 'passed'
except Exception as error:
    result.update(status='failed', error=repr(error))
    raise
finally:
    save()
