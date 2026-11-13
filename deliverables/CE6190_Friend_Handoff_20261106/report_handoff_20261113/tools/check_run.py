"""Post-run acceptance checks required by the handoff README (section 3).

metrics.status == complete; each of the timed traces has 100 CFG-branch calls;
uncached refreshes every call; frames are uint8 RGB (81,480,832,3); the MP4
decodes; provenance fields present; model fingerprint matches the lock;
calibration profile hash matches the handoff evidence.
"""
import json
import sys
from pathlib import Path

import numpy as np

FINGERPRINT = 'b6725fb6e8b1b4869a82655cb9e9eec6730f51199ed91f7e7230be01cbd4eba4'
PROFILE_SHA = {
    'magcache': '7f145477115af1ec2f13b338071ec0556543d9ab1721898eadaad70ef10a877a',
    'sencache': 'a480e3f18a8a1f2057b4e54c8915c729ef51ee835ed00877ab4490d381678600',
}


def check(run_dir: Path, decode_video=True):
    problems = []
    m = json.loads((run_dir / 'metrics.json').read_text())
    cfg = m.get('config', {})
    if m.get('status') != 'complete':
        return [f"status={m.get('status')}"], m
    steps = cfg.get('num_inference_steps', 50)
    trace = json.loads((run_dir / 'trace.json').read_text())['repeats']
    if len(trace) != m.get('timing_repeats'):
        problems.append(f'trace repeats {len(trace)} != {m.get("timing_repeats")}')
    for i, t in enumerate(trace):
        if len(t) != 2 * steps:
            problems.append(f'trace[{i}] has {len(t)} calls')
        if cfg.get('method') == 'baseline' and not all(e['refresh'] for e in t):
            problems.append(f'baseline trace[{i}] has a reuse step')
    f = np.load(run_dir / 'frames.npy', mmap_mode='r', allow_pickle=False)
    want = (cfg.get('num_frames', 81), cfg.get('height', 480), cfg.get('width', 832), 3)
    if f.dtype != np.uint8 or tuple(f.shape) != want:
        problems.append(f'frames {f.dtype} {f.shape}')
    if decode_video:
        n = None
        try:
            import imageio.v2 as imageio
            reader = imageio.get_reader(str(run_dir / 'video.mp4'))
            n = sum(1 for _ in reader)
            reader.close()
        except Exception as e:  # noqa: BLE001
            problems.append(f'video decode error {e!r}')
        if n is not None and n != want[0]:
            problems.append(f'video has {n} frames')
    if m.get('checkpoint_fingerprint') != FINGERPRINT:
        problems.append('checkpoint fingerprint mismatch')
    for key in ('gpu_name', 'versions', 'actual_source_revisions', 'code_sha256', 'generation_seconds'):
        if not m.get(key):
            problems.append(f'missing {key}')
    method = cfg.get('method')
    if method in PROFILE_SHA and m.get('calibration_sha256') != PROFILE_SHA[method]:
        problems.append('calibration profile hash mismatch')
    revs = {k: v.get('commit') for k, v in (m.get('actual_source_revisions') or {}).items()}
    if revs != SOURCE_COMMITS:
        problems.append(f'source revisions differ from lock: {revs}')
    if m.get('cudnn_version') != CUDNN_VERSION:
        problems.append(f'cudnn_version {m.get("cudnn_version")} != locked {CUDNN_VERSION}')
    if m.get('versions') and m['versions'] != VERSIONS:
        problems.append(f'dependency versions differ from handoff lock: {m["versions"]}')
    # Non-fatal observations, kept in metrics for the analysis.
    warnings = []
    patterns = [[e['refresh'] for e in t] for t in trace]
    if any(pt != patterns[0] for pt in patterns[1:]):
        warnings.append('refresh schedule differs between timed repeats (nondeterministic decision)')
    if method != 'baseline' and m.get('refresh_ratio') == 1.0:
        warnings.append('cached method never reused a residual')
    m['_check_warnings'] = warnings
    return problems, m


SOURCE_COMMITS = {
    'wan': '9737cba9c1c3c4d04b33fcad41c111989865d315',
    'teacache': '7c10efc4702c6b619f47805f7abe4a7a08085aa0',
    'magcache': 'df81cb181776c2c61477c08e1d21f87fda1cd938',
    'seacache': '8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2',
    'sencache': '3fb96a41511ede1e38f3555c863b1cb7cddfe567',
}
CUDNN_VERSION = 90100
VERSIONS = {'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124', 'numpy': '1.26.4', 'diffusers': '0.32.2',
            'transformers': '4.49.0', 'accelerate': '1.4.0', 'flash-attn': '2.7.4.post1', 'imageio': '2.37.0',
            'Pillow': '11.1.0'}


if __name__ == '__main__':
    bad = 0
    for d in sys.argv[1:]:
        p, m = check(Path(d))
        print(json.dumps({'run': d, 'ok': not p, 'problems': p, 'warnings': m.get('_check_warnings'),
                          'method': m.get('config', {}).get('method'), 'threshold': m.get('config', {}).get('threshold'),
                          'generation_seconds': m.get('generation_seconds'), 'refresh_ratio': m.get('refresh_ratio'),
                          'gpu': m.get('gpu_name'), 'peak_memory_gb': m.get('peak_memory_gb')}))
        bad += bool(p)
    sys.exit(1 if bad else 0)
