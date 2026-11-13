"""Progress summary for a worker-run manifest (machine-readable + human table)."""
import argparse
import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--manifest', type=Path)
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    W = a.work
    manifest = a.manifest or W / 'manifest.jsonl'
    state = W / 'state' / manifest.stem
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    done = {}
    for f in (state / 'done').glob('*.json'):
        try:
            done[f.stem] = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            pass
    claims = {f.name for f in (state / 'claims').iterdir()} if (state / 'claims').exists() else set()
    total = Counter(r['method'] for r in rows)
    st = defaultdict(Counter)
    for r in rows:
        d = done.get(r['run_id'])
        s = d['status'] if d else ('running' if r['run_id'] in claims else 'pending')
        st[r['method']][s] += 1
    # durations per method (wall seconds incl. load) for ETA
    wall = defaultdict(list)
    gen = defaultdict(list)
    for r in rows:
        d = done.get(r['run_id'])
        if d and d.get('status') == 'complete' and d.get('wall_seconds'):
            wall[r['method']].append(d['wall_seconds'])
            if d.get('generation_seconds'):
                gen[(r['method'], r.get('threshold'))].append(d['generation_seconds'])
    workers = []
    for f in sorted((state / 'workers').glob('*.json')):
        try:
            workers.append(json.loads(f.read_text()))
        except Exception:  # noqa: BLE001
            pass
    active = sum(1 for w in workers if w.get('status') in ('running', 'idle', 'starting'))
    allwall = [x for v in wall.values() for x in v]
    remaining_s = 0.0
    for m in total:
        n_left = st[m]['pending'] + st[m]['running'] * 0.5
        mean = statistics.mean(wall[m]) if wall[m] else (statistics.mean(allwall) if allwall else 900)
        remaining_s += n_left * mean
    eta_h = remaining_s / max(active, 1) / 3600
    complete = sum(st[m]['complete'] + st[m]['complete_existing'] for m in total)
    bad = [(rid, d['status'], d.get('problems')) for rid, d in done.items() if d['status'] not in ('complete', 'complete_existing')]
    out = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'manifest': str(manifest), 'total': len(rows), 'complete': complete,
           'by_method': {m: dict(st[m]) | {'total': total[m]} for m in sorted(total)},
           'active_workers': active, 'eta_hours': round(eta_h, 2),
           'median_generation_seconds': {f'{k[0]}@{k[1]}': round(statistics.median(v), 1) for k, v in sorted(gen.items(), key=str)},
           'mean_task_wall_seconds': {m: round(statistics.mean(v), 1) for m, v in wall.items() if v},
           'problems': bad[:20],
           'workers': [{k: w.get(k) for k in ('worker', 'status', 'run_id', 'method', 'task_started', 'last_status')} for w in workers]}
    if a.json:
        print(json.dumps(out, indent=2)); return
    print(f"[{out['time']}] {manifest.name}: {complete}/{len(rows)} complete, active workers {active}, ETA ~{eta_h:.1f} h")
    for m in sorted(total):
        print(f"  {m:9s} {dict(st[m])} / {total[m]}")
    if out['median_generation_seconds']:
        print('  median generation s:', out['median_generation_seconds'])
    if bad:
        print('  PROBLEMS:', bad[:10])
    for w in out['workers']:
        print('  ', w)
    try:
        print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw',
                                       '--format=csv,noheader'], text=True).strip())
    except Exception:  # noqa: BLE001
        pass


if __name__ == '__main__':
    main()
