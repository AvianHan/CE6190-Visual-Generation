"""Fetch exact native checkpoint files to a NEW directory; never overwrite models."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads((ROOT / 'locks/model_integrity.json').read_text())
    dest = args.destination.expanduser().absolute()
    if dest.is_relative_to(ROOT) and not dest.is_relative_to(ROOT / '.runtime'):
        raise ValueError('Keep weights outside the handoff package, or under ignored .runtime/')
    dest.mkdir(parents=True, exist_ok=False)
    def fetch(item):
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint path')
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.part')
        url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{item['path']}"
        subprocess.run(['curl', '-fL', '--retry', '5', '--retry-delay', '5', '-o', str(tmp), url], check=True)
        if tmp.stat().st_size != item['bytes']:
            raise ValueError('Checkpoint size mismatch: ' + item['path'])
        if item['algorithm'] == 'sha256':
            digest = hashlib.sha256()
        elif item['algorithm'] == 'git_blob_sha1':
            digest = hashlib.sha1(b'blob ' + str(item['bytes']).encode() + b'\0')
        else:
            raise ValueError('Unknown hash algorithm')
        with tmp.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['hash']:
            raise ValueError('Checkpoint hash mismatch: ' + item['path'])
        os.link(tmp, path)  # Atomic no-clobber publication; existing paths are never replaced.
        tmp.unlink()  # Only our own successfully published temporary download.
        return item['path']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for name in pool.map(fetch, spec['files']):
            print('VERIFIED', name, flush=True)
    print('All required checkpoint files verified; no GPU workload started.')

if __name__ == '__main__':
    main()
