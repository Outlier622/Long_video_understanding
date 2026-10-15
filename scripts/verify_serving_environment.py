"""Verify a frozen capsule, source tree and checkpoint using only stdlib + Docker."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def check_file(path, expected, size=None, normalize_lf=False):
    if size is not None and path.stat().st_size != size:
        raise ValueError('File size differs from frozen baseline: '+str(path))
    actual = (hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
              if normalize_lf else sha256(path))
    if actual != expected:
        raise ValueError('SHA256 differs from frozen baseline: '+str(path))


def inside(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Frozen manifest references a path outside its declared root')
    return path


def docker(*args):
    return subprocess.check_output(['docker', *args], cwd=ROOT, text=True, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-dir', default=str(ROOT/'environment/serving-20261014'))
    parser.add_argument('--model-dir', default=os.getenv('VIDEO_MODEL_DIR', 'D:/projects/VideoThinker-R1-3B'))
    parser.add_argument('--archive')
    parser.add_argument('--container', help='Also verify the live container, including its writable layer')
    parser.add_argument('--output')
    args = parser.parse_args()
    snapshot = Path(args.snapshot_dir).resolve()
    manifest = json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['snapshot_files_sha256'].items():
        check_file(inside(snapshot, name), expected)
    for name, expected in manifest['host_files_lf_sha256'].items():
        check_file(inside(ROOT, name), expected, normalize_lf=True)
    inventory = json.loads((snapshot/'runtime.lock.json').read_text(encoding='utf-8'))
    expected_source = inventory['host_source_lf_sha256']
    current_source = {str(path.relative_to(ROOT)).replace('\\', '/') for path in (ROOT/'src').rglob('*.py')}
    if current_source != set(expected_source):
        raise ValueError('Host source file set differs from frozen baseline')
    for name, expected in expected_source.items():
        check_file(inside(ROOT, name), expected, normalize_lf=True)

    for expected in [manifest['runtime_image']['id'], manifest['base_image']['id']]:
        if json.loads(docker('image', 'inspect', expected))[0]['Id'] != expected:
            raise ValueError('Frozen image identity mismatch')
    docker('compose', '-f', str(snapshot/'compose.frozen.json'), 'config', '--quiet')

    model_root = Path(args.model_dir).resolve()
    model = json.loads((snapshot/'model.lock.json').read_text(encoding='utf-8'))
    optional = ['config.json', 'generation_config.json', 'preprocessor_config.json',
                'processor_config.json', 'video_preprocessor_config.json', 'tokenizer.json',
                'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                'merges.txt', 'vocab.json', 'tokenizer.model', 'chat_template.json', 'chat_template.jinja',
                'model.safetensors.index.json']
    present = {name for name in optional if (model_root/name).is_file()}
    present |= {path.name for path in model_root.glob('*.safetensors')}
    if present != set(model['files']):
        raise ValueError('Inference model file set differs from frozen baseline')
    for name, expected in model['files'].items():
        print('Verifying model file: '+name, flush=True)
        check_file(inside(model_root, name), expected['sha256'], expected['size_bytes'])

    report = {'status': 'passed', 'runtime_image_id': manifest['runtime_image']['id'],
              'snapshot_integrity': True, 'host_source_matches': True,
              'full_model_hashes_verified': len(model['files']), 'archive_verified': False,
              'live_container_verified': False}
    if args.archive:
        check_file(Path(args.archive), manifest['archive']['sha256'], manifest['archive']['size_bytes'])
        report['archive_verified'] = True
    if args.container:
        container = json.loads(docker('inspect', args.container))[0]
        if container['Image'] != manifest['runtime_image']['id']:
            raise ValueError('Running container does not use frozen image')
        collector = (ROOT/'scripts/collect_serving_environment.py').read_text(encoding='utf-8')
        collected = subprocess.run(['docker', 'exec', '-i', args.container, 'python', '-'],
            input=collector, text=True, encoding='utf-8', capture_output=True, check=True)
        live = json.loads(collected.stdout)
        for key in ['python', 'packages', 'source_sha256']:
            if live[key] != inventory[key]:
                raise ValueError('Live container differs from frozen inventory: '+key)
        expected_os = (snapshot/'os-packages.lock.txt').read_text(encoding='utf-8').splitlines()
        if live['os_packages'] != expected_os:
            raise ValueError('Live OS package inventory differs')
        actual_env = dict(item.split('=', 1) for item in container['Config']['Env'] if '=' in item)
        for key, value in manifest['serving_environment'].items():
            if actual_env.get(key) != str(value):
                raise ValueError('Live serving environment differs: '+key)
        observed = {
            'entrypoint': container['Config']['Entrypoint'],
            'working_directory': container['Config']['WorkingDir'],
            'restart_policy': container['HostConfig']['RestartPolicy'],
            'shm_size_bytes': container['HostConfig']['ShmSize'],
            'gpu_requests': container['HostConfig']['DeviceRequests'],
            'port_bindings': container['HostConfig']['PortBindings'],
        }
        if observed != manifest['container_configuration']:
            raise ValueError('Live startup configuration differs')
        model_mount = next(m for m in container['Mounts'] if m['Destination'] == '/model')
        if model_mount['RW'] or Path(model_mount['Source']).resolve() != model_root:
            raise ValueError('Live model mount differs from the verified model directory')
        gpu = docker('exec', args.container, 'nvidia-smi',
                     '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader').strip()
        report.update(live_container_verified=True, gpu_baseline_matches=gpu == manifest['gpu_observed'])
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
