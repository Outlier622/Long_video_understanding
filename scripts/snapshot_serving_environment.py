"""Capture a running Docker service; never export the unrestricted environment.

Uses local Docker CLI and standard-library Python. The snapshot includes only
explicit serving configuration, package/OS inventories, source hashes, full
checkpoint hashes and an optional local image archive.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True, encoding='utf-8', cwd=ROOT)


def sha256(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def lf_sha256(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def model_files(root):
    index_name = 'model.safetensors.index.json'
    index = json.loads((root/index_name).read_text(encoding='utf-8'))
    names = set(index['weight_map'].values()) | {index_name}
    names |= {name for name in [
        'config.json', 'generation_config.json', 'preprocessor_config.json',
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'added_tokens.json', 'merges.txt', 'vocab.json', 'tokenizer.model',
        'chat_template.json', 'chat_template.jinja',
    ] if (root/name).is_file()}
    result = {}
    for name in sorted(names):
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Model index references a file outside the model directory')
        print('Hashing model file: '+name, flush=True)
        result[name] = {'size_bytes': path.stat().st_size, 'sha256': sha256(path)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', default='videothinker-serving')
    parser.add_argument('--snapshot-dir', required=True)
    parser.add_argument('--archive-dir', required=True)
    parser.add_argument('--baseline-date', default=date.today().isoformat(), type=lambda value: date.fromisoformat(value).isoformat())
    parser.add_argument('--image-tag', help='Unique release tag; existing tags for another image are rejected')
    args = parser.parse_args()
    snapshot = Path(args.snapshot_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    for directory in [snapshot, archive_dir]:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('Choose a fresh or empty snapshot directory: '+str(directory))
        directory.mkdir(parents=True, exist_ok=True)
    container = json.loads(docker('inspect', args.container))[0]
    image_id = container['Image']
    image = json.loads(docker('image', 'inspect', image_id))[0]
    base_ref = 'videothinker-vllm-validation:0.8.5'
    base = json.loads(docker('image', 'inspect', base_ref))[0]
    if base['Id'] != 'sha256:421bcd51d2db69a916c2bd375367bb51b4f7e8c28200ee4091414f8d539de2df':
        raise ValueError('Local serving base does not match the pinned Dockerfile')
    collector = (ROOT/'scripts/collect_serving_environment.py').read_text(encoding='utf-8')
    collected = subprocess.run(['docker', 'exec', '-i', args.container, 'python', '-'],
        input=collector, text=True, encoding='utf-8', capture_output=True, check=True)
    inventory = json.loads(collected.stdout)
    base_collected = subprocess.run(['docker', 'run', '--rm', '-i', '--entrypoint', 'python', base_ref, '-'],
        input=collector, text=True, encoding='utf-8', capture_output=True, check=True)
    base_inventory = json.loads(base_collected.stdout)
    if inventory['packages'] != base_inventory['packages'] or inventory['os_packages'] != base_inventory['os_packages']:
        raise ValueError('Runtime packages differ from the pinned base; capture a matching base first')
    host_source = {}
    host_source_lf = {}
    inherited_source = {}
    for relative, expected in inventory['source_sha256'].items():
        if (ROOT/relative).is_file():
            if lf_sha256(ROOT/relative) != inventory['source_lf_sha256'][relative]:
                raise ValueError('Host code differs from running image: '+relative)
            host_source[relative] = sha256(ROOT/relative)
            host_source_lf[relative] = lf_sha256(ROOT/relative)
        elif base_inventory['source_sha256'].get(relative) == expected:
            inherited_source[relative] = expected
        else:
            raise ValueError('Unexplained code in running image: '+relative)
    inventory['host_source_sha256'] = host_source
    inventory['host_source_lf_sha256'] = host_source_lf
    inventory['inherited_base_source_sha256'] = inherited_source
    pip_check = docker('exec', args.container, 'python', '-m', 'pip', 'check').strip()
    freeze = docker('exec', args.container, 'python', '-m', 'pip', 'freeze', '--all')
    (snapshot/'pip-freeze.raw.txt').write_text(freeze, encoding='utf-8')
    (snapshot/'requirements.constraints.txt').write_text(
        '# Audit constraints from the exact image; not a standalone pip installation recipe.\n'
        '# Includes Conda/base-owned packages and CUDA-specific torch builds.\n'+
        ''.join(f'{name}=={version}\n' for name, version in inventory['packages'].items()), encoding='utf-8')
    os_packages = inventory.pop('os_packages')
    (snapshot/'os-packages.lock.txt').write_text('\n'.join(os_packages)+'\n', encoding='utf-8')
    write_json(snapshot/'runtime.lock.json', inventory)

    composed = json.loads(docker('compose', '-f', 'compose.serving.yml', 'config', '--format', 'json'))
    service = composed['services']['inference']
    serving_env = service['environment']
    allowed_env = {'MODEL_PATH', 'VIDEO_DATA_ROOT', 'CPU_OFFLOAD_GB', 'GPU_MEMORY_UTILIZATION',
                   'MODEL_STARTUP_TIMEOUT_SECONDS', 'ENGINE_CHECK_INTERVAL_SECONDS',
                   'ENGINE_CHECK_TIMEOUT_SECONDS', 'ENGINE_CHECK_FAILURE_LIMIT',
                   'SERVICE_UNRESPONSIVE_TIMEOUT_SECONDS'}
    if set(serving_env)-allowed_env:
        raise ValueError('Review unexpected Compose environment keys before exporting a snapshot')
    actual_env = dict(item.split('=', 1) for item in container['Config']['Env'] if '=' in item)
    for key, value in serving_env.items():
        if str(value) != actual_env.get(key):
            raise ValueError('Running environment differs from Compose: '+key)
    model_mount = next(m for m in container['Mounts'] if m['Destination'] == '/model')
    if model_mount['RW']:
        raise ValueError('Expected a read-only model mount')
    model_root = Path(model_mount['Source']).resolve()
    checkpoint = model_files(model_root)
    write_json(snapshot/'model.lock.json', {'files': checkpoint,
        'scope': 'Full SHA256 of inference configuration, tokenizer and indexed safetensors shards'})

    runtime_tag = args.image_tag or 'videothinker-serving:frozen-'+args.baseline_date.replace('-', '')
    existing = subprocess.run(['docker', 'image', 'inspect', runtime_tag],
        text=True, encoding='utf-8', capture_output=True)
    if existing.returncode == 0 and json.loads(existing.stdout)[0]['Id'] != image_id:
        raise ValueError('Release tag already points to another image; choose a new --image-tag')
    docker('tag', image_id, runtime_tag)
    service.pop('build', None)
    service['image'] = runtime_tag+'@'+image_id
    service['pull_policy'] = 'never'
    for volume in service['volumes']:
        if volume['target'] == '/model':
            volume['source'] = '${VIDEO_MODEL_DIR:-D:/projects/VideoThinker-R1-3B}'
        if volume['target'] == '/data':
            volume['source'] = '${VIDEO_DATA_DIR:-D:/projects/longvideo}'
    write_json(snapshot/'compose.frozen.json', composed)

    archive = archive_dir/'serving-images.tar'
    print('Exporting exact runtime and base images to '+str(archive), flush=True)
    docker('image', 'save', '-o', str(archive), runtime_tag, base_ref)
    archive_digest = sha256(archive)
    nvidia = docker('exec', args.container, 'nvidia-smi',
        '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader').strip()
    controlled = ['.gitattributes', 'Dockerfile.serving', 'compose.serving.yml',
                  'scripts/collect_serving_environment.py', 'scripts/snapshot_serving_environment.py']
    files = ['runtime.lock.json', 'model.lock.json', 'requirements.constraints.txt',
             'os-packages.lock.txt', 'pip-freeze.raw.txt', 'compose.frozen.json']
    write_json(snapshot/'manifest.json', {
        'schema_version': 1, 'baseline_date': args.baseline_date, 'platform': image['Os']+'/'+image['Architecture'],
        'runtime_image': {'id': image_id, 'tag': runtime_tag, 'rootfs_layers': image['RootFS']['Layers']},
        'base_image': {'reference': base_ref+'@'+base['Id'], 'id': base['Id']},
        'docker_server_version': json.loads(docker('version', '--format', '{{json .Server}}'))['Version'],
        'gpu_observed': nvidia, 'pip_check': pip_check,
        'archive': {'filename': archive.name, 'size_bytes': archive.stat().st_size, 'sha256': archive_digest},
        'snapshot_files_sha256': {name: sha256(snapshot/name) for name in files},
        'host_files_sha256': {name: sha256(ROOT/name) for name in controlled},
        'host_files_lf_sha256': {name: lf_sha256(ROOT/name) for name in controlled},
        'serving_environment': serving_env,
        'container_configuration': {
            'entrypoint': container['Config']['Entrypoint'],
            'working_directory': container['Config']['WorkingDir'],
            'restart_policy': container['HostConfig']['RestartPolicy'],
            'shm_size_bytes': container['HostConfig']['ShmSize'],
            'gpu_requests': container['HostConfig']['DeviceRequests'],
            'port_bindings': container['HostConfig']['PortBindings'],
        },
        'limits': 'Image archive excludes model/data mounts, GPU driver, Docker Desktop and host OS.',
    })
    print('Snapshot complete: '+str(snapshot), flush=True)


if __name__ == '__main__':
    main()
