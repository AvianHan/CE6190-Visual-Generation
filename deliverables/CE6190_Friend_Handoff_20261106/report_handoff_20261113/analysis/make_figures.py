"""Report figures from measured A100 results (grid tune split + Phase D ablations).

Colors: documented validated categorical palette (dataviz reference), fixed slot order by method;
identity is never color-alone (marker shape + direct label + legend). Light mode only (PDF report).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

W = Path('/home/ecs-user/ce6190-runtime/a100x8-20261110')
OUT = W / 'analysis' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # categorical slots 1-4, fixed order
METHODS = ['teacache', 'magcache', 'seacache', 'sencache']
NAME = {'teacache': 'TeaCache', 'magcache': 'MagCache', 'seacache': 'SeaCache', 'sencache': 'SenCache'}
COLOR = dict(zip(METHODS, SLOT))
MARKER = {'teacache': 'o', 'magcache': 's', 'seacache': 'D', 'sencache': '^'}
FROZEN = json.load(open(W / 'analysis' / 'frozen_2.0x_denoise.json'))['selection']['decisions']
FROZEN = {d['method']: float(d['threshold']) for d in FROZEN}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 9,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def pareto():
    summ = json.load(open(W / 'analysis' / 'grid_summary.json'))
    den = {(r['method'], float(r['threshold'])): float(r['denoise_speedup']) for r in csv.DictReader(open(W / 'analysis' / 'denoise_speedup.csv'))}
    pts = defaultdict(list)
    for s in summ:
        m, t = s['policy']['method'], float(s['policy']['threshold'])
        lp = s['metrics']['lpips']
        pts[m].append((t, s['ratio_of_total_times'], den[(m, t)], lp['mean'], lp['ci95'][0], lp['ci95'][1]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, xi, xlabel, title in ((axes[0], 1, 'End-to-end speedup (×, incl. VAE decode)', '(a) End-to-end'),
                                  (axes[1], 2, 'Denoising-only speedup (×)', '(b) Denoising loop only')):
        for m in METHODS:
            v = sorted(pts[m], key=lambda r: r[xi])
            x = [r[xi] for r in v]; y = [r[3] for r in v]
            err = [[r[3] - r[4] for r in v], [r[5] - r[3] for r in v]]
            ax.errorbar(x, y, yerr=err, color=COLOR[m], lw=1.6, marker=MARKER[m], ms=5, mec=SURFACE, mew=1.0,
                        elinewidth=0.9, capsize=2, label=NAME[m], zorder=3)
            f = [r for r in v if r[0] == FROZEN[m]][0]
            ax.scatter([f[xi]], [f[3]], s=110, facecolors='none', edgecolors=INK, linewidths=1.0, zorder=4)
            dy = {'seacache': -8, 'sencache': 7}.get(m, 0)
            ax.annotate(NAME[m], (x[-1], y[-1]), xytext=(5, dy), textcoords='offset points', color=INK2,
                        fontsize=7, va='center')
        if xi == 2:
            ax.axvline(2.0, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
            ax.text(2.0, 0.005, ' 2.0× target', color=INK2, fontsize=7, va='bottom')
        ax.set_xlabel(xlabel); ax.set_title(title, loc='left')
    axes[0].set_ylabel('LPIPS vs. uncached (lower is better)')
    axes[0].set_ylim(bottom=0)
    axes[1].legend(loc='upper left', fontsize=7, handlelength=1.6)
    fig.text(0.01, -0.04, 'Points: thresholds per method; bars: 95% prompt-cluster bootstrap CI (86 prompt–seed pairs, '
             'A100). Ringed: frozen setting (2.0× denoising-only budget).', fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, 'pareto_speed_lpips')


def refresh_schedule():
    rec = [json.loads(l) for l in open(W / 'eval' / 'grid_fidelity_common.jsonl')]
    prob = {}
    for m in METHODS:
        runs = [r['config']['run_id'] for r in rec if r['config']['method'] == m and float(r['config']['threshold']) == FROZEN[m]]
        acc = np.zeros(50); n = 0
        for rid in runs:
            t = json.load(open(W / 'results' / rid / 'trace.json'))['repeats'][0]
            step = np.zeros(50)
            for e in t:
                step[e['step']] += e['refresh'] / 2.0          # two CFG branches per step
            acc += step; n += 1
        prob[m] = (acc / n, n)
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    mat = np.vstack([prob[m][0] for m in METHODS])
    im = ax.imshow(mat, aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([f"{NAME[m]} ({FROZEN[m]:g})" for m in METHODS])
    ax.set_xticks([0, 9, 19, 29, 39, 49]); ax.set_xticklabels(['1', '10', '20', '30', '40', '50'])
    ax.set_xlabel('Denoising step (UniPC, 50 steps)')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('P(full compute)', color=INK2); cb.outline.set_visible(False)
    ax.set_title('Refresh probability per step at the frozen thresholds (86 pairs, both CFG branches)', loc='left')
    save(fig, 'refresh_schedule')
    return {m: float(prob[m][0].mean()) for m in METHODS}


def ablation():
    rows = list(csv.DictReader(open(W / 'analysis' / 'ablation_table.csv')))
    order = {'seacache': ['official', 'no_gain_normalization', 'raw_proxy'], 'sencache': ['official', 'time_only', 'latent_only']}
    label = {'official': 'official', 'no_gain_normalization': 'no gain normalization', 'raw_proxy': 'raw proxy',
             'time_only': 'time term only', 'latent_only': 'latent term only'}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, m in zip(axes, ('seacache', 'sencache')):
        for i, v in enumerate(order[m]):
            r = sorted([x for x in rows if x['method'] == m and x['variant'] == v], key=lambda x: float(x['speedup']))
            x = [float(a['speedup']) for a in r]; y = [float(a['lpips']) for a in r]
            ax.plot(x, y, color=SLOT[i], lw=1.6, marker='osD'[i], ms=5, mec=SURFACE, mew=1.0, label=label[v], zorder=3)
            off = {('seacache', 'official'): (-10, 9, 'right'), ('seacache', 'no_gain_normalization'): (5, -8, 'left'),
                   ('sencache', 'official'): (0, 10, 'center'), ('sencache', 'time_only'): (0, -11, 'center')}.get((m, v), (5, 0, 'left'))
            ax.annotate(label[v], (x[-1], y[-1]), xytext=off[:2], textcoords='offset points', color=INK2, fontsize=7,
                        va='center', ha=off[2])
        n = rows[[x['method'] for x in rows].index(m)]['n_pairs']
        ax.set_title(f"({'ab'[m == 'sencache']}) {NAME[m]} variants ({n} pairs)", loc='left')
        ax.set_xlabel('End-to-end speedup (×)')
        ax.legend(loc='upper left', fontsize=7)
    axes[1].text(0.98, 0.32, 'official = time term only\nfor thresholds ≥ 1 (identical runs)', transform=axes[1].transAxes,
                 ha='right', fontsize=6.5, color=INK2)
    axes[0].set_ylabel('LPIPS vs. uncached')
    axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, 'ablation')


if __name__ == '__main__':
    pareto()
    print('mean refresh at frozen:', refresh_schedule())
    ablation()
    print(sorted(p.name for p in OUT.iterdir()))
