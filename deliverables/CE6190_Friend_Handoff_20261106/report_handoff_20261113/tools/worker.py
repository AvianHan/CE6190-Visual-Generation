"""One GPU worker for the receiver run. Never overwrites, never retries silently.

Scheduling: workers claim whole prompt-seed pairs (largest first, LPT) so the
uncached reference and its cached variants run on the same physical GPU. When
no unclaimed pair is left, idle workers steal single unstarted tasks of other
pairs (same machine, same GPU model). Claims are O_EXCL files.

Each task is one unmodified `python -m cachebench.generate` invocation with
the manifest index, so the executed scientific code is the handoff snapshot.
Existing result directories are only skipped when complete and valid; anything
else is recorded as `attention` and left untouched for manual review.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_run import check  # noqa: E402

METHOD_ORDER = {'baseline': 0, 'teacache': 1, 'magcache': 2, 'seacache': 3, 'sencache': 4}


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write_json(path, data):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def excl(path, data):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f)
    return True


def gpu_info(index):
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in out.strip().splitlines():
        i, uuid, name, used, total = [x.strip() for x in line.split(',')]
        if int(i) == index:
            return {'index': index, 'uuid': uuid, 'name': name, 'memory_used_mib': int(used), 'memory_total_mib': int(total)}
    raise ValueError(f'GPU {index} not found')


def gpu_processes(uuid):
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader'], text=True)
    return [line for line in out.strip().splitlines() if line.startswith(uuid)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--results', type=Path)
    p.add_argument('--only-run-ids', type=Path, help='optional file of run_ids to restrict this worker to')
    p.add_argument('--no-steal', action='store_true')
    a = p.parse_args()
    W = a.work
    manifest = a.manifest or W / 'manifest.jsonl'
    results = a.results or W / 'results'
    tag = manifest.stem
    state = W / 'state' / tag
    for sub in ('claims', 'pairs', 'done', 'workers'):
        (state / sub).mkdir(parents=True, exist_ok=True)
    (W / 'logs' / 'tasks').mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    index = {r['run_id']: i for i, r in enumerate(rows)}
    allowed = None
    if a.only_run_ids:
        allowed = {x.strip() for x in a.only_run_ids.read_text().split() if x.strip()}
    by_pair = defaultdict(list)
    for r in rows:
        if allowed is None or r['run_id'] in allowed:
            by_pair[r['pair_id']].append(r)
    for v in by_pair.values():
        v.sort(key=lambda r: (METHOD_ORDER[r['method']], r.get('variant', 'official'), r.get('threshold') or 0))
    pair_order = sorted(by_pair, key=lambda k: (-len(by_pair[k]), by_pair[k][0]['seed'], by_pair[k][0]['prompt_id']))

    g = gpu_info(a.gpu)
    others = gpu_processes(g['uuid'])
    if others or g['memory_used_mib'] > 1024:
        raise SystemExit(f'GPU {a.gpu} busy: {others} used={g["memory_used_mib"]}MiB; refusing to share')
    worker = f"{socket.gethostname()}:gpu{a.gpu}:{os.getpid()}"
    hb = state / 'workers' / f'gpu{a.gpu}.json'
    info = {'worker': worker, 'gpu': g, 'pid': os.getpid(), 'started': now(), 'status': 'starting', 'manifest': str(manifest)}
    write_json(hb, info)

    env = dict(os.environ)
    # Only the venv's NVIDIA wheels: the host LD_LIBRARY_PATH (/usr/local/cuda/lib64) would otherwise
    # mix host cuDNN 9.19 sub-libraries into the pinned cuDNN 9.1 (see quarantine/README).
    nv = sorted(str(x) for x in (W / 'venv' / 'lib' / 'python3.11' / 'site-packages' / 'nvidia').glob('*/lib'))
    env['LD_LIBRARY_PATH'] = ':'.join(nv)
    env.update({'CUDA_VISIBLE_DEVICES': g['uuid'], 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'})
    python = str(W / 'venv' / 'bin' / 'python')
    current_pair = None
    consecutive_failures = 0

    def next_task():
        nonlocal current_pair
        # 1) continue own pair
        if current_pair:
            for r in by_pair[current_pair]:
                if not (state / 'claims' / r['run_id']).exists():
                    return r
            current_pair = None
        # 2) claim a new pair
        for pid in pair_order:
            if excl(state / 'pairs' / pid, {'worker': worker, 'gpu_uuid': g['uuid'], 'time': now()}):
                current_pair = pid
                for r in by_pair[pid]:
                    if not (state / 'claims' / r['run_id']).exists():
                        return r
                current_pair = None
        if a.no_steal:
            return None
        # 3) steal from the pair with most unclaimed tasks
        best, best_n = None, 0
        for pid in pair_order:
            free = [r for r in by_pair[pid] if not (state / 'claims' / r['run_id']).exists()]
            if len(free) > best_n:
                best, best_n = free[0], len(free)
        return best

    while True:
        if (W / 'state' / 'STOP').exists() or (state / 'STOP').exists():
            info.update(status='stopped_by_flag', time=now()); write_json(hb, info); break
        r = next_task()
        if r is None:
            info.update(status='finished_no_work', time=now()); write_json(hb, info); break
        rid = r['run_id']
        claim = {'worker': worker, 'gpu_uuid': g['uuid'], 'gpu_index': a.gpu, 'pid': os.getpid(), 'time': now(),
                 'method': r['method'], 'pair_id': r['pair_id']}
        if not excl(state / 'claims' / rid, claim):
            continue
        out = results / rid
        if out.exists():
            try:
                problems, m = check(out)
            except Exception as e:  # noqa: BLE001
                problems, m = [repr(e)], {}
            status = 'complete_existing' if not problems else 'attention'
            write_json(state / 'done' / f'{rid}.json', {**claim, 'status': status, 'problems': problems,
                                                         'note': 'pre-existing output; not touched'})
            continue
        log = W / 'logs' / 'tasks' / f'{rid}.log'
        if log.exists():
            write_json(state / 'done' / f'{rid}.json', {**claim, 'status': 'attention',
                       'problems': ['task log already exists without a result directory'], 'note': 'not run'})
            continue
        # Never share the GPU: wait (do not kill) while any foreign compute process is present.
        waited = 0
        while gpu_processes(g['uuid']):
            info.update(status='waiting_foreign_gpu_process', foreign=gpu_processes(g['uuid']), time=now())
            write_json(hb, info); time.sleep(60); waited += 60
        if waited:
            info.pop('foreign', None)
        cmd = [python, '-m', 'cachebench.generate', '--manifest', str(manifest), '--index', str(index[rid]),
               '--output-dir', str(results), '--vendor-dir', str(W / 'vendor'),
               '--checkpoint-dir', str(W / 'models' / 'Wan2.1-T2V-1.3B'), '--device', '0']
        info.update(status='running', run_id=rid, method=r['method'], task_started=now(), time=now())
        write_json(hb, info)
        t0 = time.time()
        with log.open('x') as lf:
            lf.write(json.dumps({'cmd': cmd, 'claim': claim}) + '\n'); lf.flush()
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(W))
            info.update(child_pid=proc.pid); write_json(hb, info)
            rc = proc.wait()
        wall = time.time() - t0
        rec = {**claim, 'returncode': rc, 'wall_seconds': wall, 'finished': now(), 'log': str(log)}
        try:
            problems, m = check(out)
            rec.update(problems=problems, warnings=m.get('_check_warnings'), generation_seconds=m.get('generation_seconds'),
                       generation_seconds_repeats=m.get('generation_seconds_repeats'),
                       refresh_ratio=m.get('refresh_ratio'), peak_memory_gb=m.get('peak_memory_gb'),
                       model_load_seconds=m.get('model_load_seconds'))
            rec['status'] = 'complete' if (rc == 0 and not problems) else ('invalid' if rc == 0 else 'failed')
        except Exception as e:  # noqa: BLE001
            rec.update(status='failed', problems=[repr(e)])
        write_json(state / 'done' / f'{rid}.json', rec)
        consecutive_failures = 0 if rec['status'] == 'complete' else consecutive_failures + 1
        info.update(status='idle', last_run_id=rid, last_status=rec['status'], time=now(), child_pid=None)
        write_json(hb, info)
        if consecutive_failures >= 2:
            info.update(status='halted_after_failures', time=now()); write_json(hb, info)
            break


if __name__ == '__main__':
    main()
