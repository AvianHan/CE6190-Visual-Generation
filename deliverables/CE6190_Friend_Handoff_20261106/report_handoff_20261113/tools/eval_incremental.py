"""Incremental paired fidelity using the official cachebench.evaluate.evaluate_manifest.

For every completed cached run whose uncached reference is complete, a 2-row
sub-manifest [reference, candidate] is evaluated with the unmodified official
function (same checks, same PSNR/SSIM/LPIPS protocol, same record format).
Records land in <work>/eval/<manifest stem>/records/<run_id>.jsonl; `--assemble`
concatenates them in manifest order into one JSONL usable by cachebench.aggregate.
Runs at low priority on CPU so it can overlap GPU generation.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path


def _eval_one(args):
    sub_manifest, root, out, device = args
    os.nice(19)
    import torch
    torch.set_num_threads(2)
    from cachebench.evaluate import evaluate_manifest
    tmp = Path(str(out) + '.tmp')
    try:
        recs = evaluate_manifest(sub_manifest, root, tmp, ('psnr', 'ssim', 'lpips'), device, None)
        assert len(recs) == 1
        os.replace(tmp, out)
        return str(out), None
    except Exception as e:  # noqa: BLE001
        return str(out), repr(e)
    finally:
        Path(sub_manifest).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--manifest', type=Path)
    p.add_argument('--results', type=Path)
    p.add_argument('--procs', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--assemble', type=Path, help='write all records in manifest order to this JSONL')
    p.add_argument('--ref-manifest', type=Path, help='manifest holding the reference (uncached) rows, if not in --manifest')
    a = p.parse_args()
    W = a.work
    manifest = a.manifest or W / 'manifest.jsonl'
    root = a.results or W / 'results'
    state = W / 'state' / manifest.stem
    base = W / 'eval' / manifest.stem
    (base / 'records').mkdir(parents=True, exist_ok=True)
    (base / 'sub').mkdir(parents=True, exist_ok=True)
    (base / 'errors').mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    by_id = {r['run_id']: r for r in rows}
    ref_state = {}
    if a.ref_manifest:
        for x in a.ref_manifest.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r['run_id'] not in by_id:
                    by_id[r['run_id']] = r
                    ref_state[r['run_id']] = W / 'state' / a.ref_manifest.stem

    def complete(rid):
        f = ref_state.get(rid, state) / 'done' / f'{rid}.json'
        if not f.exists():
            return False
        return json.loads(f.read_text()).get('status') in ('complete', 'complete_existing')

    if a.assemble:
        out, missing = [], []
        for r in rows:
            if r['method'] == 'baseline':
                continue
            f = base / 'records' / f"{r['run_id']}.jsonl"
            if f.exists():
                out.append(f.read_text().strip())
            else:
                missing.append(r['run_id'])
        a.assemble.parent.mkdir(parents=True, exist_ok=True)
        a.assemble.write_text(''.join(x + '\n' for x in out))
        print(json.dumps({'assembled': len(out), 'missing': len(missing), 'output': str(a.assemble)}))
        return

    jobs = []
    for r in rows:
        if r['method'] == 'baseline':
            continue
        rid = r['run_id']
        out = base / 'records' / f'{rid}.jsonl'
        if out.exists() or (base / 'errors' / f'{rid}.txt').exists():
            continue
        if not (complete(rid) and complete(r['reference_run_id'])):
            continue
        sub = base / 'sub' / f'{rid}.jsonl'
        sub.write_text(json.dumps(by_id[r['reference_run_id']], sort_keys=True) + '\n' + json.dumps(r, sort_keys=True) + '\n')
        jobs.append((str(sub), str(root), str(out), a.device))
    if not jobs:
        print(json.dumps({'evaluated': 0, 'errors': 0}))
        return
    errors = 0
    with mp.get_context('spawn').Pool(a.procs) as pool:
        for out, err in pool.imap_unordered(_eval_one, jobs):
            if err:
                errors += 1
                rid = Path(out).name.split('.')[0]
                (base / 'errors' / f'{rid}.txt').write_text(err)
    print(json.dumps({'evaluated': len(jobs) - errors, 'errors': errors}))


if __name__ == '__main__':
    main()
