"""Parallel range-download of the pinned native Wan checkpoint.

Same verification as handoff tools/fetch_model.py: every file's byte size and
sha256 / git-blob-sha1 from locks/model_integrity.json must match, then the file
is published with an atomic no-clobber hard link. Only the transport differs
(N concurrent HTTP range segments instead of one connection per file).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--integrity', required=True, type=Path)
    p.add_argument('--destination', required=True, type=Path)
    p.add_argument('--segments', type=int, default=16)
    p.add_argument('--base', default='https://huggingface.co')
    a = p.parse_args()
    spec = json.loads(a.integrity.read_text())
    dest = a.destination.absolute()
    dest.mkdir(parents=True, exist_ok=False)
    stage = dest.parent / (dest.name + '.staging')
    stage.mkdir(parents=True, exist_ok=False)

    def seg_download(url, out, start, end):
        for _ in range(8):
            r = subprocess.run(['curl', '-sfL', '--retry', '5', '--retry-delay', '3',
                                '-r', f'{start}-{end}', '-o', str(out), url])
            if r.returncode == 0 and out.exists() and out.stat().st_size == end - start + 1:
                return
        raise RuntimeError(f'segment failed {out}')

    jobs = []
    for item in spec['files']:
        url = f"{a.base}/{spec['model_id']}/resolve/{spec['revision']}/{item['path']}"
        n = a.segments if item['bytes'] > 64 * 2**20 else 1
        size = item['bytes']
        step = -(-size // n)
        for i in range(n):
            s, e = i * step, min(size, (i + 1) * step) - 1
            if s > e:
                continue
            out = stage / (item['path'].replace('/', '__') + f'.seg{i:03d}')
            jobs.append((url, out, s, e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
        list(pool.map(lambda j: seg_download(*j), jobs))
    print('segments downloaded', len(jobs), flush=True)

    for item in spec['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('unsafe path')
        segs = sorted(stage.glob(item['path'].replace('/', '__') + '.seg*'))
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.part')
        if item['algorithm'] == 'sha256':
            d = hashlib.sha256()
        elif item['algorithm'] == 'git_blob_sha1':
            d = hashlib.sha1(b'blob ' + str(item['bytes']).encode() + b'\0')
        else:
            raise ValueError('unknown algorithm')
        with tmp.open('xb') as w:
            for sgm in segs:
                with sgm.open('rb') as r:
                    for block in iter(lambda: r.read(8 * 2**20), b''):
                        d.update(block)
                        w.write(block)
        if tmp.stat().st_size != item['bytes']:
            raise ValueError('size mismatch ' + item['path'])
        if d.hexdigest() != item['hash']:
            raise ValueError('hash mismatch ' + item['path'])
        os.link(tmp, path)
        tmp.unlink()
        for sgm in segs:
            sgm.unlink()
        print('VERIFIED', item['path'], item['bytes'], item['algorithm'], item['hash'], flush=True)
    stage.rmdir()
    print('All required checkpoint files verified.', flush=True)


if __name__ == '__main__':
    main()
