"""Parallel acceptance (tools/check_run.check, with MP4 decode) over every run_id of a manifest."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_run import check

W = Path('/home/ecs-user/ce6190-runtime/a100x8-20261110')

def one(rid):
    try:
        p, m = check(W / 'results' / rid, decode_video=True)
        c = m.get('config', {})
        return {'run_id': rid, 'ok': not p, 'problems': p, 'warnings': m.get('_check_warnings'), 'method': c.get('method'),
                'threshold': c.get('threshold'), 'pair_id': c.get('pair_id'), 'refresh_ratio': m.get('refresh_ratio'),
                'generation_seconds': m.get('generation_seconds'), 'gpu_uuid': m.get('gpu_uuid')}
    except Exception as e:  # noqa: BLE001
        return {'run_id': rid, 'ok': False, 'problems': [f'check crashed: {e!r}']}

if __name__ == '__main__':
    manifest, out = Path(sys.argv[1]), Path(sys.argv[2])
    rids = [json.loads(l)['run_id'] for l in manifest.read_text().splitlines() if l.strip()]
    with Pool(32) as pool, open(out, 'w') as f:
        for i, r in enumerate(pool.imap_unordered(one, rids, chunksize=4), 1):
            f.write(json.dumps(r) + '\n')
            if i % 200 == 0:
                print(i, flush=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    bad = [r for r in rows if not r['ok']]
    warn = [r for r in rows if r.get('warnings')]
    print(json.dumps({'checked': len(rows), 'ok': len(rows) - len(bad), 'bad': len(bad), 'with_warnings': len(warn)}))
