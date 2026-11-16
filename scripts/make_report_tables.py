"""Render the report's tables from the audited, retained summary records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'analysis/report_data'
OUT = ROOT / 'report/tables'


def rows(name):
    return json.loads((DATA / (name + '.json')).read_text())


def save(name, header, body, spec):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + '.tex')).write_text(
        '\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}' + spec + '@{}}\n\\toprule\n' + header + r' \\' +
        '\n\\midrule\n' + '\n'.join(body) + '\n\\bottomrule\n\\end{tabular*}\n')


def main():
    body = []
    for r in rows('wan_primary'):
        base = r['method'] == 'Uncached'
        eps = '--' if base else f"{r['threshold']:g}"
        fidelity = ['--', '--', '--'] if base else [f"{r['psnr_db']:.2f}", f"{r['ssim']:.3f}", f"{r['lpips']:.4f}"]
        if base:
            name = 'Uncached'
        else:
            name = r['method']
        if r['method'] == 'SeaCache':
            fidelity[-1] = r'\textbf{' + fidelity[-1] + '}'
        body.append(' & '.join([name, eps, f"{r['generation_s']:.2f}",
            f"{r['e2e_speedup_ratio_of_totals']:.3f}", f"{r['denoise_speedup_ratio_of_totals']:.3f}",
            *fidelity, f"{100*r['refresh_ratio']:.1f}"]) + r' \\')
        if base:
            body.append(r'\midrule')
    save('wan_primary', r'Method & $\epsilon$ & \shortstack{Time\\(s)} & \shortstack{Gen.\\speedup} & \shortstack{Denoise\\speedup} & \shortstack{PSNR $\uparrow$\\(dB)} & SSIM $\uparrow$ & LPIPS $\downarrow$ & \shortstack{Compute\\(\%)}', body, 'lrrrrrrrr')
    names = {'teacache':'TeaCache', 'magcache':'MagCache','seacache':'SeaCache','sencache':'SenCache','fixed':'Fixed'}
    wanted = ['seacache_0.3','sencache_0.5','fixed_16','teacache_0.25',
              'fixed_14','sencache_0.8','sencache_1.0','fixed_12','seacache_0.45','magcache_0.09',
              'sencache_2','seacache_0.6','magcache_0.12','teacache_0.5']
    byid = {r['config_id']:r for r in rows('flux_all_configs')}
    body=[]
    for i,c in enumerate(wanted):
        r=byid[c]
        if i in [4,6,10]: body.append(r'\midrule')
        if i in [0,4,6,10]:
            group = {0:'1.7',4:'1.96',6:'2.27',10:'2.45'}[i]
            body.append(r'\multicolumn{7}{@{}l}{\textit{Near $'+group+r'\times$ generation speedup}} \\[2pt]')
        label=names[r['method']]+' '+f"{r['threshold']:g}"
        lpips=f"{r['lpips']:.4f}"
        if c in ['seacache_0.3','fixed_14','sencache_1.0','sencache_2']:
            lpips=r'\textbf{'+lpips+'}'
        body.append(' & '.join([label, f"{r['median_speedup']:.3f}", f"{r['psnr']:.2f}",
          f"{r['ssim']:.3f}", lpips,f"{r['clip_score']:.2f}",f"{r['image_reward']:.3f}"]) + r' \\')
    save('flux_budgets',r'Configuration & Speedup & \shortstack{PSNR $\uparrow$\\(dB)} & SSIM $\uparrow$ & LPIPS $\downarrow$ & CLIP $\uparrow$ & ImageReward $\uparrow$',body,'lrrrrrr')
    body=[]
    for r in rows('vbench_16dims_scores'):
        name=r['dimension'].replace('_',' ').capitalize()
        body.append(' & '.join([name]+[f"{r[m]:.4f}" for m in ['Uncached','TeaCache','MagCache','SeaCache','SenCache']])+r' \\')
    save('vbench', 'Dimension & Uncached & TeaCache & MagCache & SeaCache & SenCache', body, 'lrrrrr')
    body=[]
    labels={'official':'Full', 'no_gain_normalization':'No gain norm.','raw':'Raw proxy', 'raw_proxy':'Raw proxy','latent_only':'Latent only','time_only':'Time only'}
    previous_method = None
    for r in rows('wan_component_ablations'):
        keep=(r['method']=='SeaCache' and r['threshold'] in [.1,.2]) or (r['method']=='SenCache' and r['threshold'] in [.5,2])
        if not keep: continue
        if previous_method and previous_method != r['method']:
            body.append(r'\midrule')
        previous_method = r['method']
        label=r['method']+' / '+labels.get(r['variant'],r['variant'].replace('_',' '))
        body.append(' & '.join([label,f"{r['threshold']:g}",f"{r['e2e_speedup_ratio_of_totals']:.3f}",f"{r['lpips']:.4f}",f"{100*r['refresh_ratio']:.1f}"])+r' \\')
    save('ablations',r'Variant & $\epsilon$ & Generation speedup & LPIPS $\downarrow$ & Compute (\%)',body,'lrrrr')


if __name__ == '__main__':
    main()
