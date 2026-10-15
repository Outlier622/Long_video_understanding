"""Short real-GPU check of strict/diagnostic output validation responses."""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=False)
result = {'status': 'running', 'checks': []}


def call(path, body=None):
    request = urllib.request.Request(args.url+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


try:
    code, before = call('/health/ready')
    assert code == 200 and before['active_requests'] == 0
    result['before'] = before
    for strict, expected in [(False, 200), (True, 502)]:
        code, body = call('/v1/infer', {'clip_id': 1, 'max_new_tokens': 1, 'require_json': strict})
        result['checks'].append({'strict': strict, 'status_code': code, 'body': body})
        assert code == expected, (code, body)
        issues = body['detail']['validation_errors'] if strict else body['validation_errors']
        assert issues and all({'path', 'code', 'message'} <= issue.keys() for issue in issues)
        if not strict:
            assert body['result'] is None and not body['structured_output_valid']
        print(f'strict={strict}: {code}, errors={issues}', flush=True)
    code, after = call('/health/ready')
    assert code == 200 and after['active_requests'] == 0
    assert after['instance_id'] == before['instance_id'] and after['model_load_count'] == 1
    result.update(status='passed', after=after)
except Exception as error:
    result.update(status='failed', error=repr(error))
    raise
finally:
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
