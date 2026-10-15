"""Bounded closed-loop HTTP baseline; Python standard library only.

No automatic retries. Diagnostic token-capped traffic and strict complete-output
traffic are reported separately. This client cannot measure TTFT or TPOT.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def summarize(rows, seconds):
    successful = [r for r in rows if r['status'] == 200]
    latencies = [r['client_seconds'] for r in successful]
    counts = {}
    for row in rows:
        key = str(row['status'])
        counts[key] = counts.get(key, 0) + 1
    return {
        'requests': len(rows), 'successes': len(successful), 'status_counts': counts,
        'wall_seconds': seconds, 'error_rate': 1 - len(successful) / len(rows),
        'successful_requests_per_second': len(successful) / seconds,
        'returned_output_tokens_per_second': sum(r['body'].get('output_tokens', 0) for r in successful) / seconds,
        'success_latency_seconds': {f'p{int(q*100)}': percentile(latencies, q) for q in (.50, .95, .99)},
        'all_response_p95_seconds': percentile([r['client_seconds'] for r in rows], .95),
        'valid_structured_outputs': sum(r['body'].get('structured_output_valid') is True for r in successful),
        'percentiles_are_exploratory': len(successful) < 100,
    }


def call(url, path, body=None, timeout=660):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, payload = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, payload = error.code, error.read()
    except (OSError, TimeoutError) as error:
        return {'status': 0, 'body': {'transport_error': str(error)}, 'client_seconds': time.perf_counter()-started}
    try:
        parsed = json.loads(payload)
    except (ValueError, UnicodeError):
        parsed = {'non_json_response': payload.decode(errors='replace')}
    return {'status': status, 'body': parsed, 'client_seconds': time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('D:/projects/longvideo'))
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--strict-requests', type=int, default=3)
    parser.add_argument('--short-tokens', type=int, default=16)
    args = parser.parse_args()
    if args.rounds < 1 or args.strict_requests < 1 or not 1 <= args.short_tokens <= 512:
        parser.error('positive rounds/strict-requests and token limit 1..512 required')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'configuration': vars(args) | {'data_root': str(args.data_root)},
              'stages': [], 'limitations': [
                  'Closed-loop synchronized waves; not an open-loop arrival-rate capacity test.',
                  'One active request limit; concurrent arrivals may be rejected with 429.',
                  'Short diagnostic output is intentionally truncated, not a quality pass.',
                  'P95/P99 on small successful samples are exploratory, not SLO evidence.',
                  'Non-streaming endpoint: TTFT/TPOT unavailable; token rate includes preprocessing and HTTP.',
                  'GPU samples cover the whole device, including other applications.',
                  'No automatic retries; 502 responses do not expose generated token counts.']}
    stop = threading.Event()

    def save():
        (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    def telemetry():
        executable = shutil.which('nvidia-smi')
        if not executable:
            (out/'gpu.jsonl').write_text(json.dumps({'error': 'nvidia-smi unavailable'}), encoding='utf-8')
            return
        with (out/'gpu.jsonl').open('w', encoding='utf-8') as stream:
            while not stop.is_set():
                try:
                    proc = subprocess.run([executable, '--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
                    stream.write(json.dumps({'unix_time': time.time(), 'csv': proc.stdout.strip(), 'error': proc.stderr.strip()})+'\n')
                    stream.flush()
                except (OSError, subprocess.TimeoutExpired) as error:
                    stream.write(json.dumps({'error': str(error)})+'\n')
                stop.wait(2)

    monitor = threading.Thread(target=telemetry, daemon=True)
    try:
        before = call(args.url, '/health/ready', timeout=10)
        result['health_before'] = before
        if before['status'] != 200 or before['body'].get('active_requests') != 0:
            raise RuntimeError('Service must be ready and idle')
        manifest = args.data_root/'episodes/ep02/keyframes/clip_manifest_flow.jsonl'
        records = {int(r['clip_id']): r for r in map(json.loads, manifest.read_text(encoding='utf-8').splitlines()) if r}
        inputs = []
        for clip in (1, 3, 14):
            record = records[clip]
            files = [Path(f['image_path']) for f in record['selected_frames']]
            inputs.append({'clip_id': clip, 'record': record, 'frame_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
        result['inputs'] = inputs
        result['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        monitor.start()
        warmup = call(args.url, '/v1/infer', {'episode': 'ep02', 'clip_id': 1, 'max_new_tokens': args.short_tokens, 'require_json': False})
        result['warmup_excluded'] = warmup
        if warmup['status'] != 200:
            raise RuntimeError('Warmup failed')
        save()
        with (out/'requests.jsonl').open('w', encoding='utf-8') as raw:
            for kind, concurrency, rounds in [('diagnostic', c, args.rounds) for c in (1, 2, 4, 8, 16)] + [('strict', 1, args.strict_requests)]:
                rows = []
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    for wave in range(rounds):
                        barrier = threading.Barrier(concurrency)
                        clip = (1, 3, 14)[wave % 3]
                        body = {'episode': 'ep02', 'clip_id': clip, 'max_new_tokens': 500 if kind == 'strict' else args.short_tokens,
                                'require_json': kind == 'strict', 'timeout_seconds': 600}

                        def request_one(index):
                            barrier.wait(timeout=30)
                            return call(args.url, '/v1/infer', body) | {'kind': kind, 'concurrency': concurrency, 'wave': wave, 'index': index, 'request': body}

                        futures = [pool.submit(request_one, i) for i in range(concurrency)]
                        for future in futures:
                            row = future.result()
                            rows.append(row)
                            raw.write(json.dumps(row)+'\n')
                            raw.flush()
                        print(f'{kind} concurrency={concurrency} wave={wave+1}/{rounds}', flush=True)
                        if any(r['status'] in (0, 503, 504) for r in rows):
                            raise RuntimeError('Transport/unhealthy/timeout response; stop to avoid contaminating later stages')
                result['stages'].append({'kind': kind, 'concurrency': concurrency, **summarize(rows, time.perf_counter()-started)})
                save()
        after = call(args.url, '/health/ready', timeout=10)
        result['health_after'] = after
        identities = {warmup['body'].get('instance_id')}
        identities.update(r['body'].get('instance_id') for r in map(json.loads, (out/'requests.jsonl').read_text(encoding='utf-8').splitlines()) if r['status'] == 200)
        result['stable_instance'] = identities == {before['body']['instance_id']} and after['body'].get('instance_id') == before['body']['instance_id']
        if after['status'] != 200 or after['body'].get('active_requests') != 0 or after['body'].get('model_load_count') != 1 or not result['stable_instance']:
            raise RuntimeError('Final health/model reuse verification failed')
        result['status'] = 'measured'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join(timeout=10)
        gpu_samples = []
        if (out/'gpu.jsonl').exists():
            for line in (out/'gpu.jsonl').read_text(encoding='utf-8').splitlines():
                sample = json.loads(line)
                for gpu in csv.reader(sample.get('csv', '').splitlines()):
                    try:
                        gpu_samples.append((float(gpu[3]), float(gpu[4])))
                    except (IndexError, ValueError):
                        pass
        result['gpu_summary_including_warmup'] = {
            'valid_samples': len(gpu_samples),
            'mean_utilization_percent': sum(s[0] for s in gpu_samples)/len(gpu_samples) if gpu_samples else None,
            'peak_used_memory_mib': max((s[1] for s in gpu_samples), default=None),
        }
        save()
        lines = ['# Local HTTP performance baseline', '', 'Latency quantiles use successful responses only; small samples are exploratory.', '',
                 '| Mode | Concurrency | Success/total | P50 s | P95 s | P99 s | Success req/s | Output tok/s | Error rate |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for stage in result['stages']:
            p = stage['success_latency_seconds']
            fmt = lambda x: 'N/A' if x is None else f'{x:.3f}'
            lines.append(f"| {stage['kind']} | {stage['concurrency']} | {stage['successes']}/{stage['requests']} | {fmt(p['p50'])} | {fmt(p['p95'])} | {fmt(p['p99'])} | {stage['successful_requests_per_second']:.4f} | {stage['returned_output_tokens_per_second']:.3f} | {stage['error_rate']:.1%} |")
        (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
