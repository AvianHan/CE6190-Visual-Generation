"""Thin wrapper around the pinned official VBench evaluator (vendor/vbench @ fd18b3d).

The GUIDE refers to scripts/run_vbench.py, which is absent from the handoff checkout; this replaces it without changing
any VBench code: it points VBENCH_CACHE_DIR at the pre-fetched weights, loads the stage directory's VBench_info.json
(written by `cachebench.evaluate stage-vbench`, a custom subset), and calls VBench.evaluate in vbench_standard mode with
local=True (no downloads). Run with the separate VBench venv:
  venv_vbench/bin/python tools/run_vbench.py --stage-dir S --dimension d1 d2 --output O [--videos-override DIR]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

W = Path('/home/ecs-user/ce6190-runtime/a100x8-20261110')
VB = W / 'vendor' / 'vbench'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage-dir', required=True, type=Path)
    p.add_argument('--dimension', nargs='+', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--videos-override', type=Path)
    p.add_argument('--name')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    os.environ.setdefault('VBENCH_CACHE_DIR', str(W / 'vbench_cache'))
    # torch.hub checkpoints (e.g. DINO ViT-B/16) are served from pre-fetched files; no downloads at evaluation time.
    os.environ.setdefault('TORCH_HOME', str(W / 'vbench_cache' / 'torch_home'))
    sys.path.insert(0, str(VB))
    import torch
    from vbench import VBench

    stage = json.loads((a.stage_dir / 'staging.json').read_text())
    info = a.stage_dir / 'VBench_info.json'
    videos = a.videos_override or (a.stage_dir / 'videos')
    name = a.name or f"{stage['method']}{'_' + stage['dimension'] if stage.get('dimension') else ''}"
    a.output.mkdir(parents=True, exist_ok=True)
    run_info = {'started': datetime.datetime.now().astimezone().isoformat(timespec='seconds'), 'stage': stage,
                'stage_dir': str(a.stage_dir), 'videos': str(videos), 'n_videos': len(list(Path(videos).glob('*.mp4'))),
                'dimensions': a.dimension, 'vbench_commit': subprocess.run(['git', '-C', str(VB), 'rev-parse', 'HEAD'],
                                                                           capture_output=True, text=True).stdout.strip(),
                'vbench_cache': os.environ['VBENCH_CACHE_DIR'], 'torch_home': os.environ['TORCH_HOME'], 'device': a.device,
                'torch': torch.__version__, 'protocol_note': 'custom 110-prompt subset; not a leaderboard score'}
    (a.output / f'{name}_run_info.json').write_text(json.dumps(run_info, indent=2))
    bench = VBench(torch.device(a.device), str(info), str(a.output))
    bench.evaluate(videos_path=str(videos), name=name, dimension_list=a.dimension, local=True, mode='vbench_standard')
    run_info['finished'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    (a.output / f'{name}_run_info.json').write_text(json.dumps(run_info, indent=2))
    print(json.dumps({'output': str(a.output / f'{name}_eval_results.json'), 'n_videos': run_info['n_videos']}))


if __name__ == '__main__':
    main()
