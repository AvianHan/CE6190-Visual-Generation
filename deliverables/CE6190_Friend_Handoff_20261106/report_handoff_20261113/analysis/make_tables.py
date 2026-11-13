"""LaTeX table snippets for the report, generated only from measured result files.

Inputs: analysis/grid_summary.json (official cachebench.aggregate, 86 common A100 pairs, tune split),
analysis/denoise_speedup.csv, analysis/ablation_table.csv, analysis/frozen_2.0x_denoise.json,
per-run results/<run_id>/metrics.json. Output: report/tables/*.tex.
"""
import csv
import json
import statistics as st
from pathlib import Path

W = Path('/home/ecs-user/ce6190-runtime/a100x8-20261110')
T = W / 'report' / 'tables'
T.mkdir(parents=True, exist_ok=True)
NAME = {'baseline': 'Uncached', 'teacache': 'TeaCache', 'magcache': 'MagCache', 'seacache': 'SeaCache', 'sencache': 'SenCache'}
ORDER = ['teacache', 'magcache', 'seacache', 'sencache']
FROZEN = {d['method']: float(d['threshold']) for d in json.load(open(W / 'analysis' / 'frozen_2.0x_denoise.json'))['selection']['decisions']}


def ci(m, k, f='{:.3f}'):
    x = m[k]
    return f"{f.format(x['mean'])} [{f.format(x['ci95'][0])}, {f.format(x['ci95'][1])}]"


def write(name, body):
    (T / f'{name}.tex').write_text(body)
    print('wrote', name)


def grid():
    summ = json.load(open(W / 'analysis' / 'grid_summary.json'))
    den = {(r['method'], float(r['threshold'])): r for r in csv.DictReader(open(W / 'analysis' / 'denoise_speedup.csv'))}
    by = {(s['policy']['method'], float(s['policy']['threshold'])): s for s in summ}
    n_pairs = {s['n_videos'] for s in summ}
    assert len(n_pairs) == 1
    n = n_pairs.pop()
    # Fidelity / speed over all evaluated thresholds.
    lines = []
    for m in ORDER:
        ths = sorted(t for (mm, t) in by if mm == m)
        for i, t in enumerate(ths):
            s = by[(m, t)]; me = s['metrics']; d = den[(m, t)]
            mark = r'$^\star$' if t == FROZEN[m] else ''
            lines.append(f"{NAME[m] if i == 0 else ''} & {t:g}{mark} & {s['ratio_of_total_times']:.2f} & {float(d['denoise_speedup']):.2f} & "
                         f"{me['refresh_ratio']['mean']:.2f} & {me['psnr_db']['mean']:.2f} & {me['ssim']['mean']:.3f} & {ci(me, 'lpips')} \\\\")
        lines.append(r'\addlinespace')
    write('grid_tradeoff', r"""\begin{table}[htbp]\centering\small
\begin{tabular}{@{}llrrrrrl@{}}\toprule
Method & Thr. & E2E $\times$ & Denoise $\times$ & Refresh & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ [95\% CI]\\\midrule
""" + '\n'.join(lines[:-1]) + rf"""
\bottomrule\end{{tabular}}
\caption{{Validation (tune) grid on A100-SXM4-80GB: {n} prompt--seed pairs common to all policies (29 development prompts
$\times$ seeds 0--2). Speedups are ratios of summed times against the paired uncached runs (E2E includes text encoding,
denoising and VAE decoding; ``Denoise'' is the denoising loop only). Refresh is the fraction of the 100 CFG-branch calls
that ran the full block stack. Agreement metrics compare against the same-seed uncached frames; means over prompts with
prompt-cluster bootstrap CIs (2000 draws). $^\star$: frozen setting (2.0$\times$ denoising-only budget).}}
\label{{tab:grid}}\end{{table}}
""")
    # Efficiency at the frozen settings.
    rec = [json.loads(l) for l in open(W / 'eval' / 'grid_fidelity_common.jsonl')]
    base_ids = sorted({r['config']['reference_run_id'] for r in rec})
    bm = [json.load(open(W / 'results' / b / 'metrics.json')) for b in base_ids]
    rows = [f"Uncached & -- & {st.mean(x['generation_seconds'] for x in bm):.1f} & {st.mean(x['denoise_seconds'] for x in bm):.1f} & "
            f"{st.mean(x['vae_decode_seconds'] for x in bm):.1f} & 1.00 / 1.00 & 100 & {st.mean(x['peak_memory_gb'] for x in bm):.1f} & n/a \\\\"]
    for m in ORDER:
        t = FROZEN[m]
        ids = [r['config']['run_id'] for r in rec if r['config']['method'] == m and float(r['config']['threshold']) == t]
        mm = [json.load(open(W / 'results' / i / 'metrics.json')) for i in ids]
        s = by[(m, t)]; d = den[(m, t)]
        calls = st.mean(x['refresh_ratio'] for x in mm) * 100
        cal = 'reused' if m in ('magcache', 'sencache') else 'none'
        rows.append(f"{NAME[m]} & {t:g} & {st.mean(x['generation_seconds'] for x in mm):.1f} & {st.mean(x['denoise_seconds'] for x in mm):.1f} & "
                    f"{st.mean(x['vae_decode_seconds'] for x in mm):.1f} & {s['ratio_of_total_times']:.2f} / {float(d['denoise_speedup']):.2f} & "
                    f"{calls:.0f} & {st.mean(x['peak_memory_gb'] for x in mm):.1f} & {cal} \\\\")
    write('efficiency', r"""\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrcrrl@{}}\toprule
Method & Thr. & Total (s) & Denoise (s) & VAE (s) & Speedup E2E / Den. & Full calls & Peak GB & Calib.\\\midrule
""" + '\n'.join(rows) + rf"""
\bottomrule\end{{tabular}}
\caption{{Latency at the frozen settings on the validation grid ({n} pairs; per-run time is the mean of 3 timed repeats after
one warm-up, GPU-synchronised, model loading excluded). Full calls: full block-stack evaluations per video out of 100
(50 UniPC steps $\times$ 2 CFG branches). The FP32 VAE decode ($\approx$98.5\,s, identical for all methods) bounds the
end-to-end speedup on this GPU. Calibration: MagCache ratios and SenCache sensitivity profiles were reused from the
handoff package (same checkpoint fingerprint); their one-off calibration time was not re-measured here.}}
\label{{tab:efficiency}}\end{{table}}
""")


def ablation():
    rows = list(csv.DictReader(open(W / 'analysis' / 'ablation_table.csv')))
    label = {'official': 'official', 'no_gain_normalization': 'no gain normalisation', 'raw_proxy': 'raw (unfiltered) proxy',
             'time_only': 'time term only', 'latent_only': 'latent term only'}
    out = []
    for m in ('seacache', 'sencache'):
        for v in (('official', 'no_gain_normalization', 'raw_proxy') if m == 'seacache' else ('official', 'time_only', 'latent_only')):
            rr = sorted([r for r in rows if r['method'] == m and r['variant'] == v], key=lambda r: float(r['threshold']))
            for i, r in enumerate(rr):
                lp = float(r['lpips']); c = json.loads(r['lpips_ci95'].replace("'", '"')) if r['lpips_ci95'].startswith('[') else None
                psnr = r['psnr'] if r['psnr'] not in ('', 'None') else r'$\infty$'
                out.append(f"{NAME[m] + ' ' + label[v] if i == 0 else ''} & {float(r['threshold']):g} & {r['n_pairs']} & {float(r['refresh']):.2f} & "
                           f"{float(r['speedup']):.2f} & {psnr} & {float(r['ssim']):.3f} & {lp:.3f} \\\\")
            out.append(r'\addlinespace')
    write('ablation', r"""\begin{table}[htbp]\centering\small
\begin{tabular}{@{}llrrrrrr@{}}\toprule
Variant & Thr. & Pairs & Refresh & E2E $\times$ & PSNR & SSIM & LPIPS\\\midrule
""" + '\n'.join(out[:-1]) + r"""
\bottomrule\end{tabular}
\caption{Component ablations on the validation prompts (A100). Each variant is compared with the official policy on the
same prompt--seed pairs whose official runs were measured on this GPU (SeaCache 21, SenCache 24). Ablation runs used one
timed generation in a warm process; official rows come from the 3-repeat grid. A PSNR of $\infty$ means every frame was
identical to the uncached output (the policy never reused).}
\label{tab:ablation}\end{table}
""")


if __name__ == '__main__':
    grid()
    ablation()
