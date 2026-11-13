"""Phase C (frozen test) paired fidelity summary.

Main table = official protocol: records with sample_index < 5 only (evaluate_manifest default samples_per_prompt=5);
extra temporal-flickering seeds 5..24 are written separately. Aggregation uses the official cachebench.aggregate.summarize
(bootstrap over prompt means) unchanged. --provisional allows unequal pair sets while the run is incomplete.
Also writes a denoise-only speedup column (records carry denoise_seconds; baseline denoise from the reference run).
"""
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path
W = Path('/home/ecs-user/ce6190-runtime/a100x8-20261110')

from cachebench.aggregate import summarize, save_summary

p = argparse.ArgumentParser(); p.add_argument('--provisional', action='store_true'); a = p.parse_args()
recs = [json.loads(l) for f in sorted((W / 'eval' / 'manifest_test' / 'records').glob('*.jsonl'))
        for l in f.read_text().splitlines() if l.strip()]
main = [r for r in recs if r['config']['sample_index'] < 5]
extra = [r for r in recs if r['config']['sample_index'] >= 5]
tag = '_provisional' if a.provisional else ''
(W / 'eval' / f'phaseC_fidelity{tag}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in main))
(W / 'eval' / f'phaseC_fidelity_tfextra{tag}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in extra))
s = summarize(main, allow_unpaired=a.provisional)
save_summary(s, W / 'analysis' / f'phaseC_summary{tag}.json')
# denoise-only speedup per method (reference denoise from the baseline run's metrics.json)
base = {}
for r in main:
    rid = r['config']['reference_run_id']
    if rid not in base:
        base[rid] = json.loads((W / 'results' / rid / 'metrics.json').read_text()).get('denoise_seconds')
by = defaultdict(list)
for r in main:
    by[r['config']['method']].append((base[r['config']['reference_run_id']], r['denoise_seconds'], r['reference_generation_seconds'], r['generation_seconds']))
rows = {m: {'n': len(v), 'denoise_speedup_ratio_of_totals': sum(x[0] for x in v) / sum(x[1] for x in v),
            'e2e_speedup_ratio_of_totals': sum(x[2] for x in v) / sum(x[3] for x in v)} for m, v in sorted(by.items())}
(W / 'analysis' / f'phaseC_denoise_speedup{tag}.json').write_text(json.dumps(rows, indent=2))
print(json.dumps({'main_records': len(main), 'tf_extra_records': len(extra), 'speedups': rows}, indent=1))
for x in s:
    m = x['metrics']
    print(x['policy']['method'], x['n_videos'], 'e2e %.3f' % x['ratio_of_total_times'], 'lpips %.4f' % m['lpips']['mean'],
          'ssim %.4f' % m['ssim']['mean'], 'psnr %.2f' % m['psnr_db']['mean'])
