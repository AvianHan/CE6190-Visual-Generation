"""Render the report's tables from the audited, retained summary records.

The output uses helper macros defined in report/preamble.tex (method swatches, \\best,
\\bestlp, \\gain, \\loss, \\rel, \\grouprow). Numbers are formatted exactly as before; only the
emphasis (bold, cell shading, colour) is derived here, by the rules stated in each table caption.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'analysis/report_data'
OUT = ROOT / 'report/tables'

METHOD_MACRO = {'Uncached': r'\mUnc', 'TeaCache': r'\mTea', 'MagCache': r'\mMag',
                'SeaCache': r'\mSea', 'SenCache': r'\mSen', 'Fixed': r'\mFix'}


def rows(name):
    return json.loads((DATA / (name + '.json')).read_text())


def save(name, header, body, spec):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + '.tex')).write_text(
        '\\renewcommand{\\tabularxcolumn}[1]{m{#1}}%\n\\begin{tabularx}{\\linewidth}{' + spec + '}\n\\toprule\n' + header +
        '\n\\midrule\n' + '\n'.join(body) + '\n\\bottomrule\n\\end{tabularx}\n')


def ranks(values, higher_is_better):
    """Return {index: 1 or 2} for the best and second-best distinct values; ties share a rank."""
    distinct = sorted(set(values), reverse=higher_is_better)
    out = {}
    for i, v in enumerate(values):
        if v == distinct[0]:
            out[i] = 1
        elif len(distinct) > 1 and v == distinct[1]:
            out[i] = 2
    return out


def emph(text, rank):
    return r'\best{' + text + '}' if rank == 1 else text


def rel_change(new, ref):
    return f"{100 * (new - ref) / ref:+.1f}\\%".replace('+', '$+$').replace('-', '$-$')


def wan_primary():
    data = rows('wan_primary')
    cached = [r for r in data if r['method'] != 'Uncached']
    tea = next(r for r in cached if r['method'] == 'TeaCache')
    # Displayed (rounded) values decide the ranks, so ties in the table are ranked as ties.
    cols = {
        'time': ([f"{r['generation_s']:.2f}" for r in cached], False),
        'gen': ([f"{r['e2e_speedup_ratio_of_totals']:.3f}" for r in cached], True),
        'den': ([f"{r['denoise_speedup_ratio_of_totals']:.3f}" for r in cached], True),
        'cmp': ([f"{100 * r['refresh_ratio']:.1f}" for r in cached], False),
        'psnr': ([f"{r['psnr_db']:.2f}" for r in cached], True),
        'ssim': ([f"{r['ssim']:.3f}" for r in cached], True),
        'lpips': ([f"{r['lpips']:.4f}" for r in cached], False),
    }
    rank = {k: ranks([float(v) for v in vals], hib) for k, (vals, hib) in cols.items()}
    body = []
    for r in data:
        if r['method'] == 'Uncached':
            body.append(r'\rowcolor{refrow}' + ' & '.join([
                METHOD_MACRO['Uncached'], '--', f"{r['generation_s']:.2f}",
                f"{r['e2e_speedup_ratio_of_totals']:.3f}", f"{r['denoise_speedup_ratio_of_totals']:.3f}",
                f"{100 * r['refresh_ratio']:.1f}", '--', '--', '--']) + r' \\')
            body.append(r'\midrule')
            continue
        i = cached.index(r)
        cell = {k: emph(vals[i], rank[k].get(i)) for k, (vals, _) in cols.items()}
        lp = cols['lpips'][0][i]
        lp = r'\bestlp{' + lp + '}' if rank['lpips'].get(i) == 1 else lp
        if r['method'] != 'TeaCache':
            lp += r'\rel{' + rel_change(r['lpips'], tea['lpips']) + '}'
        body.append(' & '.join([METHOD_MACRO[r['method']], f"{r['threshold']:g}", cell['time'], cell['gen'],
                                cell['den'], cell['cmp'], cell['psnr'], cell['ssim'], lp]) + r' \\')
    header = (r'\rowcolor{tabhead} & & \multicolumn{4}{c}{\thead{Efficiency}} & '
              r'\multicolumn{3}{c}{\thead{Fidelity to Uncached}} \\' + '\n'
              r'\cmidrule(lr){3-6}\cmidrule(l){7-9}' + '\n'
              r'\rowcolor{tabhead}Method & $\epsilon$ & Time $\downarrow$ & Generation\newline speedup $\uparrow$ & '
              r'Denoising\newline speedup $\uparrow$ & Compute $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & '
              r'LPIPS $\downarrow$ \\')
    save('wan_primary', header, body, 'll' + 'R' * 7)


def flux_budgets():
    names = {'teacache': 'TeaCache', 'magcache': 'MagCache', 'seacache': 'SeaCache',
             'sencache': 'SenCache', 'fixed': 'Fixed'}
    groups = [('1.7', ['seacache_0.3', 'sencache_0.5', 'fixed_16', 'teacache_0.25']),
              ('1.96', ['fixed_14', 'sencache_0.8']),
              ('2.27', ['sencache_1.0', 'fixed_12', 'seacache_0.45', 'magcache_0.09']),
              ('2.45', ['sencache_2', 'seacache_0.6', 'magcache_0.12', 'teacache_0.5'])]
    byid = {r['config_id']: r for r in rows('flux_all_configs')}
    metrics = [('psnr', '{:.2f}', True), ('ssim', '{:.3f}', True), ('lpips', '{:.4f}', False),
               ('clip_score', '{:.2f}', True), ('image_reward', '{:.3f}', True)]
    body = []
    for g, (speed, ids) in enumerate(groups):
        if g:
            body.append(r'\midrule')
        body.append(r'\grouprow{8}{Near $' + speed + r'\times$ generation speedup}')
        rs = [byid[c] for c in ids]
        shown = {m: [fmt.format(r[m]) for r in rs] for m, fmt, _ in metrics}
        best = {m: {i for i, v in enumerate(shown[m])
                    if float(v) == (max if hib else min)(float(x) for x in shown[m])}
                for m, _, hib in metrics}
        order = sorted(range(len(rs)), key=lambda i: rs[i]['lpips'])
        winner, runner_up = order[0], order[1]
        for i, r in enumerate(rs):
            cells = []
            for m, _, _ in metrics:
                v = shown[m][i]
                if m == 'lpips' and i == winner:
                    v = r'\bestlp{' + v + '}' + r'\rel{' + rel_change(r['lpips'], rs[runner_up]['lpips']) + '}'
                elif i in best[m]:
                    v = r'\best{' + v + '}'
                cells.append(v)
            label = METHOD_MACRO[names[r['method']]] + '{} ' + f"{r['threshold']:g}"
            body.append(' & '.join([label, f"{r['median_speedup']:.3f}", f"{100 * r['mean_refresh_ratio']:.1f}",
                                    *cells]) + r' \\')
    header = (r'\rowcolor{tabhead} & \multicolumn{2}{c}{\thead{Efficiency}} & '
              r'\multicolumn{3}{c}{\thead{Fidelity to Uncached}} & \multicolumn{2}{c}{\thead{Content quality}} \\'
              + '\n' r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}' + '\n'
              r'\rowcolor{tabhead}Configuration & Speedup $\uparrow$ & Compute $\downarrow$ & PSNR $\uparrow$ & '
              r'SSIM $\uparrow$ & LPIPS $\downarrow$ & CLIP $\uparrow$ & ImageReward $\uparrow$ \\')
    save('flux_budgets', header, body, 'l' + 'R' * 6 + 'r')


VBENCH_GROUPS = [('Video quality', ['subject_consistency', 'background_consistency', 'temporal_flickering',
                                    'motion_smoothness', 'dynamic_degree', 'aesthetic_quality',
                                    'imaging_quality']),
                 ('Video--condition consistency', ['object_class', 'multiple_objects', 'human_action', 'color',
                                                   'spatial_relationship', 'scene', 'appearance_style',
                                                   'temporal_style', 'overall_consistency'])]


def vbench():
    methods = ['Uncached', 'TeaCache', 'MagCache', 'SeaCache', 'SenCache']
    by_dim = {r['dimension']: r for r in rows('vbench_16dims_scores')}
    assert sorted(by_dim) == sorted(d for _, ds in VBENCH_GROUPS for d in ds)
    body = []
    for g, (title, dims) in enumerate(VBENCH_GROUPS):
        if g:
            body.append(r'\midrule')
        body.append(r'\grouprow{6}{' + title + '}')
        for d in dims:
            r = by_dim[d]
            ref = float(f"{r['Uncached']:.4f}")
            cells = [r'\cellcolor{refrow}' + f"{ref:.4f}"]
            for m in methods[1:]:
                v = f"{r[m]:.4f}"
                diff = float(v) - ref
                # Dynamic degree measures motion prevalence, so it has no better direction to colour.
                if d != 'dynamic_degree' and diff >= 0.01 - 1e-9:
                    v = r'\gain{' + v + '}'
                elif d != 'dynamic_degree' and diff <= -0.01 + 1e-9:
                    v = r'\loss{' + v + '}'
                cells.append(v)
            body.append(' & '.join([d.replace('_', ' ').capitalize(), *cells]) + r' \\')
    header = r'\rowcolor{tabhead}Dimension & ' + ' & '.join(METHOD_MACRO[m] for m in methods) + r' \\'
    save('vbench', header, body, 'l' + 'R' * 5)


def ablations():
    labels = {'official': 'Full', 'no_gain_normalization': 'No gain norm.', 'raw': 'Raw proxy',
              'raw_proxy': 'Raw proxy', 'latent_only': 'Latent only', 'time_only': 'Time only'}
    keep = {'SeaCache': [.1, .2], 'SenCache': [.5, 2]}
    data = [r for r in rows('wan_component_ablations') if r['threshold'] in keep.get(r['method'], [])]
    body = []
    previous_method, previous_eps = None, None
    full = {}
    for r in data:
        if r['variant'] == 'official':
            full[(r['method'], r['threshold'])] = r
    for r in data:
        if r['method'] != previous_method:
            if previous_method:
                body.append(r'\midrule')
            body.append(r'\grouprow{5}{' + METHOD_MACRO[r['method']] + '}')
        elif r['threshold'] != previous_eps:
            body.append(r'\addlinespace[2pt]')
        previous_method, previous_eps = r['method'], r['threshold']
        speed, lp = f"{r['e2e_speedup_ratio_of_totals']:.3f}", f"{r['lpips']:.4f}"
        name = labels.get(r['variant'], r['variant'].replace('_', ' '))
        if r['variant'] == 'official':
            prefix = r'\rowcolor{refrow}'
        else:
            prefix = ''
            ref = full[(r['method'], r['threshold'])]
            # Colour a change only when it exceeds 5% of the Full value at the same threshold.
            ds = (r['e2e_speedup_ratio_of_totals'] - ref['e2e_speedup_ratio_of_totals']) / ref['e2e_speedup_ratio_of_totals']
            dl = (r['lpips'] - ref['lpips']) / ref['lpips']
            speed = r'\gain{' + speed + '}' if ds > .05 else r'\loss{' + speed + '}' if ds < -.05 else speed
            lp = r'\gain{' + lp + '}' if dl < -.05 else r'\loss{' + lp + '}' if dl > .05 else lp
        body.append(prefix + ' & '.join([name, f"{r['threshold']:g}", speed, lp,
                                         f"{100 * r['refresh_ratio']:.1f}"]) + r' \\')
    header = (r'\rowcolor{tabhead}Variant & $\epsilon$ & Speedup $\uparrow$ & LPIPS $\downarrow$ & '
              r'Compute $\downarrow$ \\')
    save('ablations', header, body, 'lr' + 'R' * 3)


def main():
    wan_primary()
    flux_budgets()
    vbench()
    ablations()


if __name__ == '__main__':
    main()
