"""Run inside a container: collect versions and source hashes without CUDA init."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata['Name'].lower().replace('_', '-').replace('.', '-')
    packages[name] = distribution.version
source = {str(path.relative_to('/app')): digest(path)
          for path in sorted(Path('/app/src').rglob('*.py'))}
source_lf = {str(path.relative_to('/app')): hashlib.sha256(
    path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    for path in sorted(Path('/app/src').rglob('*.py'))}
print(json.dumps({
    'python': platform.python_version(), 'executable': sys.executable,
    'architecture': platform.machine(), 'libc': platform.libc_ver(),
    'os_release': Path('/etc/os-release').read_text(),
    'packages': dict(sorted(packages.items())), 'source_sha256': source,
    'source_lf_sha256': source_lf,
    'os_packages': subprocess.check_output(
        ['dpkg-query', '-W', '-f=${binary:Package}\t${Version}\n'], text=True).splitlines(),
}, indent=2))
