"""Qualitative frame grids for the frozen test (Phase C), selected by a pre-declared rule, not by eye.

Rule (fixed before looking at any frame): among prompt-seed pairs (sample_index < 5) where all four cached methods have a
fidelity record, the SUCCESS case is the pair with the lowest worst-method LPIPS and the FAILURE case the pair with the
highest mean LPIPS over the four methods. Frames 1, 41 and 81 are shown uncropped for uncached + 4 methods.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

W = Path('/home/ecs-user/ce6190-runtime/a100x8-20261110')
OUT = W / 'analysis' / 'figures'
METHODS = ['teacache', 'magcache', 'seacache', 'sencache']
NAME = {'baseline': 'Uncached', 'teacache': 'TeaCache', 'magcache': 'MagCache', 'seacache': 'SeaCache', 'sencache': 'SenCache'}
FRAMES = [0, 40, 80]
INK2, SURFACE = '#52514e', '#fcfcfb'


def load_records():
    by_pair = defaultdict(dict)
    for f in (W / 'eval' / 'manifest_test' / 'records').glob('*.jsonl'):
        txt = f.read_text().strip()
        if not txt:
            continue
        r = json.loads(txt)
        c = r['config']
        if c['sample_index'] >= 5:
            continue
        by_pair[c['pair_id']][c['method']] = r
    return {k: v for k, v in by_pair.items() if all(m in v for m in METHODS)}


def grid(pair, recs, tag, title):
    c0 = recs[METHODS[0]]['config']
    rows = [('baseline', c0['reference_run_id'], None)] + [(m, recs[m]['config']['run_id'], recs[m]) for m in METHODS]
    fig, axes = plt.subplots(len(rows), len(FRAMES), figsize=(7.0, 1.0 + 1.15 * len(rows)))
    for i, (m, rid, rec) in enumerate(rows):
        fr = np.load(W / 'results' / rid / 'frames.npy', mmap_mode='r')
        for j, k in enumerate(FRAMES):
            ax = axes[i, j]
            ax.imshow(np.asarray(fr[k]))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f'frame {k + 1}', fontsize=8, color=INK2)
            if j == 0:
                lab = NAME[m] if rec is None else f"{NAME[m]}\nLPIPS {rec['lpips']:.3f}"
                ax.set_ylabel(lab, fontsize=7.5, color=INK2, rotation=0, ha='right', va='center', labelpad=6)
    fig.suptitle(f"{title}: “{c0['prompt']}” (seed {c0['seed']})", fontsize=8.5, x=0.02, ha='left')
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'qualitative_{tag}.{ext}', dpi=150 if ext == 'png' else 200, bbox_inches='tight')
    plt.close(fig)


def main():
    pairs = load_records()
    worst = {k: max(v[m]['lpips'] for m in METHODS) for k, v in pairs.items()}
    mean = {k: float(np.mean([v[m]['lpips'] for m in METHODS])) for k, v in pairs.items()}
    s = min(worst, key=worst.get)
    f = max(mean, key=mean.get)
    sel = {'n_candidate_pairs': len(pairs), 'rule': __doc__.split('Rule')[1].strip(),
           'success': {'pair_id': s, 'prompt': pairs[s]['teacache']['config']['prompt'], 'seed': pairs[s]['teacache']['config']['seed'],
                       'lpips': {m: pairs[s][m]['lpips'] for m in METHODS}},
           'failure': {'pair_id': f, 'prompt': pairs[f]['teacache']['config']['prompt'], 'seed': pairs[f]['teacache']['config']['seed'],
                       'lpips': {m: pairs[f][m]['lpips'] for m in METHODS}}}
    (W / 'analysis' / 'qualitative_selection.json').write_text(json.dumps(sel, indent=2))
    grid(s, pairs[s], 'success', 'Success case (lowest worst-method LPIPS)')
    grid(f, pairs[f], 'failure', 'Failure case (highest mean LPIPS)')
    print(json.dumps({k: sel[k] for k in ('n_candidate_pairs', 'success', 'failure')}, indent=1))


if __name__ == '__main__':
    main()
