"""Real HTTP acceptance checks against a running GPU service."""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=False)
results = {'checks': [], 'status': 'running'}

def save():
    (out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(args.url+path, data=data,
                headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=660) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)

def check(name, response, expected):
    code, body = response
    results['checks'].append({'name': name, 'status_code': code, 'body': body})
    save()
    assert code == expected, (name, code, body)
    print(f'{name}: {code}', flush=True)
    return body

try:
    deadline = time.monotonic()+360
    while True:
        try:
            code, body = call('/health/ready')
            if code == 200:
                break
            assert code == 503, (code, body)
        except (urllib.error.URLError, ConnectionError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('Service did not become ready')
        time.sleep(2)
    before = check('ready', (code, body), 200)
    check('live', call('/health/live'), 200)
    check('invalid_request', call('/v1/infer', {'clip_id': -1}), 422)
    check('unknown_field', call('/v1/infer', {'clip_id': 1, 'model_path': '/bad'}), 422)
    check('missing_clip', call('/v1/infer', {'clip_id': 9999}), 404)
    first = check('short_clip_1', call('/v1/infer', {
        'clip_id': 1, 'max_new_tokens': 16, 'require_json': False}), 200)
    assert first['output_tokens'] == 16 and not first['structured_output_valid']
    check('strict_truncated_json', call('/v1/infer', {
        'clip_id': 1, 'max_new_tokens': 16}), 502)
    check('deadline', call('/v1/infer', {
        'clip_id': 3, 'timeout_seconds': 2}), 504)
    recovered = check('post_timeout_clip_3', call('/v1/infer', {
        'clip_id': 3, 'max_new_tokens': 16, 'require_json': False}), 200)
    assert recovered['instance_id'] == before['instance_id']
    with concurrent.futures.ThreadPoolExecutor() as pool:
        full = pool.submit(call, '/v1/infer', {'clip_id': 1, 'timeout_seconds': 600})
        active_deadline = time.monotonic()+20
        while call('/health/ready')[1].get('active_requests') != 1:
            if time.monotonic() > active_deadline or full.done():
                raise AssertionError('Full generation did not become active')
            time.sleep(0.2)
        check('over_capacity', call('/v1/infer', {'clip_id': 3}), 429)
        structured = check('full_structured_output', full.result(), 200)
        assert structured['structured_output_valid'] and structured['result']['clip_id'] == 1
    after = check('ready_after_requests', call('/health/ready'), 200)
    assert after['model_load_count'] == 1 and after['active_requests'] == 0
    assert after['instance_id'] == before['instance_id']
    results['status'] = 'passed'
except Exception as error:
    results['status'] = 'failed'
    results['error'] = repr(error)
    raise
finally:
    save()
