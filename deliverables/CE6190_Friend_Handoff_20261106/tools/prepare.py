"""Verify read-only handoff and create a NEW receiver-local manifest directory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args()
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Handoff checksum mismatch: ' + name)
    rows = [json.loads(x) for x in (ROOT / 'pending.jsonl').read_text().splitlines()]
    completed = {json.loads(x)['run_id'] for x in (ROOT / 'local_completed.jsonl').read_text().splitlines()}
    extra = [json.loads(x) for x in (ROOT / 'new_hardware_baselines.jsonl').read_text().splitlines()]
    identifiers = {r['run_id'] for r in rows}
    if len(rows) != 1409 or len(identifiers) != len(rows) or identifiers & completed:
        raise ValueError('Invalid or overlapping assignment')
    if len(extra) != 2 or any(r['method'] != 'baseline' or r['run_id'] not in completed for r in extra):
        raise ValueError('Invalid receiver-only baseline supplement')
    rows = extra + rows
    if len({r['run_id'] for r in rows}) != 1411:
        raise ValueError('Duplicate execution task')
    refs = {r['run_id'] for r in rows if r['method'] == 'baseline'}
    if not {r['reference_run_id'] for r in rows} <= refs:
        raise ValueError('Missing baseline assignment')
    work = args.work_dir.expanduser().absolute()
    if work.is_relative_to(ROOT) and not work.is_relative_to(ROOT / '.runtime'):
        raise ValueError('Use an external new work directory, or .runtime/ under this package')
    # Exclusive creation is intentional: never reuse or clean an existing directory.
    work.mkdir(parents=True, exist_ok=False)
    mapping = {}
    for row in rows:
        if row.get('calibration_path'):
            old = row['calibration_path']
            new = str(ROOT / 'calibration' / (row['method'] + '.json'))
            mapping[old] = new
            row['calibration_path'] = new
    (work / 'manifest.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    (work / 'handoff.json').write_text(json.dumps({'package': str(ROOT), 'tasks': len(rows),
        'path_mapping': mapping, 'hardware_group': 'RECORD_RECEIVER_HARDWARE_BEFORE_RUN',
        'original_manifest_sha256': hashlib.sha256((ROOT / 'pending.jsonl').read_bytes()).hexdigest()}, indent=2))
    print(json.dumps({'status': 'prepared_only', 'work_dir': str(work), 'tasks': len(rows)}))

if __name__ == '__main__':
    main()
