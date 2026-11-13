"""Phase C VBench pipeline (GUIDE protocol, custom 110-prompt subset). Resume-safe: finished jobs are skipped.

Steps
 1. stage (venv, official `cachebench.evaluate stage-vbench`):
      vbench_stage/<m>          --samples 5 --allow-subset            (15 non-flicker dimensions)
      vbench_stage/<m>_flicker  --dimension temporal_flickering --samples 25 --allow-subset
 2. official vendor/vbench/static_filter.py on each <m>_flicker (GPU) -> <m>_flicker/static/filtered_videos
 3. tools/run_vbench.py per (method, dimension) -> vbench_out/<m>/<dim>/ ; flicker on the filtered videos.
 Jobs run concurrently, one per GPU (CUDA_VISIBLE_DEVICES); a job is done when its *_eval_results.json exists.
Usage: venv/bin/python tools/vbench_phaseC.py [--gpus 0 1 ...] [--stage-only]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path('/home/ecs-user/ce6190-runtime/a100x8-20261110')
MAN = W / 'phaseC' / 'manifest_test.jsonl'
INFO = Path('/home/ecs-user/CE6190-Visual-Generation/benchmarks/vbench/VBench_full_info.json')
METHODS = ['baseline', 'teacache', 'magcache', 'seacache', 'sencache']
DIMS = ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality',
        'imaging_quality', 'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene',
        'temporal_style', 'appearance_style', 'overall_consistency']
ST, OUT, LOG = W / 'vbench_stage', W / 'vbench_out', W / 'logs' / 'vbench'
VENV, VBV = W / 'venv' / 'bin' / 'python', W / 'venv_vbench' / 'bin' / 'python'


def run(cmd, **kw):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def stage():
    for m in METHODS:
        for d, extra in ((ST / m, ['--samples', '5']),
                         (ST / f'{m}_flicker', ['--dimension', 'temporal_flickering', '--samples', '25'])):
            if (d / 'staging.json').exists():
                continue
            run([VENV, '-m', 'cachebench.evaluate', 'stage-vbench', '--manifest', MAN, '--root', W / 'results',
                 '--output', d, '--official-info', INFO, '--method', m, '--allow-subset', *extra])


def jobs():
    env = dict(os.environ, VBENCH_CACHE_DIR=str(W / 'vbench_cache'), TORCH_HOME=str(W / 'vbench_cache' / 'torch_home'))
    out = []
    for m in METHODS:
        fl = ST / f'{m}_flicker'
        filt = fl / 'static' / 'filtered_videos'
        out.append((f'{m}_static_filter', fl / 'static' / 'filtered_static_video.json',
                    [VBV, W / 'vendor' / 'vbench' / 'static_filter.py', '--videos_path', fl / 'videos',
                     '--result_path', fl / 'static', '--filter_scope', 'temporal_flickering'], None))
        out.append((f'{m}_temporal_flickering', OUT / m / 'temporal_flickering' / f'{m}_temporal_flickering_eval_results.json',
                    [VBV, W / 'tools' / 'run_vbench.py', '--stage-dir', fl, '--dimension', 'temporal_flickering',
                     '--videos-override', filt, '--output', OUT / m / 'temporal_flickering',
                     '--name', f'{m}_temporal_flickering'], f'{m}_static_filter'))
        for d in DIMS:
            out.append((f'{m}_{d}', OUT / m / d / f'{m}_{d}_eval_results.json',
                        [VBV, W / 'tools' / 'run_vbench.py', '--stage-dir', ST / m, '--dimension', d,
                         '--output', OUT / m / d, '--name', f'{m}_{d}'], None))
    return out, env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--stage-only', action='store_true')
    a = p.parse_args()
    stage()
    if a.stage_only:
        return
    LOG.mkdir(parents=True, exist_ok=True)
    todo, env = jobs()
    done = {n for n, marker, _, _ in todo if marker.exists()}
    pending = [j for j in todo if j[0] not in done]
    free, running, failed = list(a.gpus), {}, set()
    while pending or running:
        for g, (n, proc) in list(running.items()):
            if proc.poll() is not None:
                del running[g]; free.append(g)
                (done if proc.returncode == 0 else failed).add(n)
                print(time.strftime('%T'), n, 'rc', proc.returncode, flush=True)
        for j in list(pending):
            if not free:
                break
            n, marker, cmd, dep = j
            if dep in failed:
                pending.remove(j); failed.add(n); continue
            if dep and dep not in done:
                continue
            g = free.pop(0); pending.remove(j)
            e = dict(env, CUDA_VISIBLE_DEVICES=str(g), PYTHONPATH=str(W / 'vendor' / 'vbench'))
            running[g] = (n, subprocess.Popen(list(map(str, cmd)), env=e, cwd=str(W / 'vendor' / 'vbench'),
                                              stdout=open(LOG / f'{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%T'), 'start', n, 'gpu', g, flush=True)
        time.sleep(10)
    print(json.dumps({'done': len(done), 'failed': sorted(failed)}))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
