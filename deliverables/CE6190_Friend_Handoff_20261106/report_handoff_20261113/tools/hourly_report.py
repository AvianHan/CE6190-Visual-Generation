"""Hourly health + progress summary for the grid run (read-only)."""
import collections, glob, json, statistics as st, subprocess, sys, time
W = '/home/ecs-user/ce6190-runtime/a100x8-20261110'
man = [json.loads(l) for l in open(W + '/manifest.jsonl')]
tot = collections.Counter(r['method'] for r in man)
done = collections.Counter(); wall = collections.defaultdict(list); bad = []
for f in glob.glob(W + '/state/manifest/done/*.json'):
    d = json.load(open(f)); done[d['method']] += 1
    if d['status'] == 'complete': wall[d['method']].append(d['wall_seconds'])
    elif d['status'] != 'complete_existing': bad.append((d['method'], d['status'], d.get('gpu_index'), d.get('problems')))
print(time.strftime('%F %T %Z'), f"done {sum(done.values())}/{len(man)}")
rem = 0
for m in tot:
    mw = st.mean(wall[m]) if wall[m] else 0; rem += (tot[m] - done[m]) * mw
    print(f"  {m:9s} {done[m]:4d}/{tot[m]:4d} mean_wall={mw:5.0f}s")
h = rem / 8 / 3600
print(f"remaining ~{h:.1f} h -> ETA {time.strftime('%m-%d %H:%M', time.localtime(time.time() + rem / 8))}")
print('non-complete:', bad)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print('workers:', sh("ps -eo cmd | grep -c '^/[^ ]*python [^ ]*tools/worker.py --work'"))
print('tracebacks:', sh(f"grep -l Traceback {W}/logs/tasks/*.log 2>/dev/null | wc -l"))
print('gpu:', sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | tr '\\n' ';'"))
print('eval_loop:', sh(f"grep -o '\"evaluated\": [0-9]*, \"errors\": [0-9]*' {W}/logs/eval_loop.log | tail -n 3 | tr '\\n' ';'"),
      'errlines', sh(f"grep -ciE 'traceback|error:' {W}/logs/eval_loop.log"))
print('disk:', sh(f"df -h {W} | tail -1"), '| results', sh(f"du -sh {W}/results | cut -f1"))

# ---- Phase D (ablations) ----
def phase(label, manifest_rel, stem):
    """Progress of a batch_worker phase: per-method counts, ETA from per-method mean wall, failures, fidelity."""
    rows = [json.loads(l) for l in open(W + '/' + manifest_rel)]
    tot_m = collections.Counter(r['method'] for r in rows)
    dd = collections.Counter(); bad = []; walls = collections.defaultdict(list)
    for f in glob.glob(f"{W}/state/{stem}/done/*.json"):
        d = json.load(open(f)); dd[d['method']] += 1
        if d['status'] == 'complete': walls[d['method']].append(d.get('wall_seconds') or 0)
        else: bad.append((d['method'], d['status'], d.get('problems')))
    allw = [x for v in walls.values() for x in v]
    rem = sum((tot_m[m] - dd[m]) * (st.mean(walls[m]) if walls[m] else (st.mean(allw) if allw else 0)) for m in tot_m)
    nd = sum(dd.values())
    eta = time.strftime('%m-%d %H:%M', time.localtime(time.time() + rem / 8)) if allw else 'n/a'
    print(f"{label} done {nd}/{len(rows)} " + ' '.join(f"{m}:{dd[m]}/{tot_m[m]}" for m in sorted(tot_m)) +
          f" | remaining ~{rem / 8 / 3600:.1f} h -> ETA {eta} | non-complete: {bad}")
    print(f"{label} fidelity records: {len(glob.glob(f'{W}/eval/{stem}/records/*.jsonl'))} errors: {len(glob.glob(f'{W}/eval/{stem}/errors/*'))}")


print('batch_workers:', sh("ps -eo cmd | grep -c '^/[^ ]*python [^ ]*tools/batch_worker.py'"))
phase('phaseD', 'phaseD/manifest_ablation.jsonl', 'manifest_ablation')
phase('phaseC', 'phaseC/manifest_test.jsonl', 'manifest_test')
