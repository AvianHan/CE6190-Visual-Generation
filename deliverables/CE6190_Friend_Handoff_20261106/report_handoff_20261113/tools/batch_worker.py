"""Persistent-backend GPU worker for the post-grid phases (frozen test subset, ablations).

Identical per-task generation and output format to `cachebench.generate` (same
NativeWanBackend.generate call, same frames/MP4/metrics/trace files), except
that the Wan backend is loaded ONCE per process and one process-level warmup
generation is done at start-up instead of a per-task warmup. Rows are expected
to carry warmup=0, timing_repeats>=1. This saves ~4x GPU time versus the grid
protocol; the grid (Phase A) remains the authoritative timing measurement.

Claiming/scheduling/no-overwrite rules are the same as tools/worker.py.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import METHOD_ORDER, excl, gpu_info, gpu_processes, now, write_json  # noqa: E402
from check_run import check  # noqa: E402


def run_task(backend, row, out, base_meta):
    import numpy as np
    import torch
    from cachebench.policies import validate_row
    validate_row(row)
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'status': 'running', 'config': row, 'python': sys.version, 'platform': platform.platform(),
              **base_meta}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    try:
        result['implementation_notes'] = {'variant': row.get('variant', 'official'),
                                          'sencache_scale': row.get('sencache_scale', 'official_81f'),
                                          'teacache_proxy': 'e0' if row.get('use_ret_steps', False) else 'e',
                                          'precision': 'bfloat16 autocast with native FP32 operations',
                                          'attention_backend': 'flash_attention_2',
                                          'runner': 'tools/batch_worker.py persistent backend; process-level warmup'}
        warmup = int(row.get('warmup', 0)); repeats = int(row.get('timing_repeats', 1))
        for _ in range(warmup):
            warm, _, _ = backend.generate(row); del warm
        backend.sync(); torch.cuda.reset_peak_memory_stats(0)
        timings = []; all_trace = []; video = None
        for _ in range(repeats):
            if video is not None:
                del video
            video, timing, trace = backend.generate(row); timings.append(timing); all_trace.append(trace)
        result['peak_memory_gb'] = torch.cuda.max_memory_allocated(0) / 1024**3
        frames = ((video.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        np.save(out / 'frames.npy', frames, allow_pickle=False)
        import imageio.v2 as imageio
        imageio.mimwrite(str(out / 'video.mp4'), frames, fps=float(row.get('fps', 16)), codec='libx264', quality=8,
                         macro_block_size=1)
        for key in timings[0]:
            result[key] = statistics.median(t[key] for t in timings)
            result[key + '_repeats'] = [t[key] for t in timings]
        if row.get('calibration_path'):
            cp = Path(row['calibration_path']); profile = json.loads(cp.read_text())
            result['calibration_sha256'] = hashlib.sha256(cp.read_bytes()).hexdigest()
            result['calibration_provenance'] = {k: profile.get(k) for k in ('sample_ids', 'config', 'source_revisions',
                                                                            'aggregation', 'calibration_wall_seconds')}
        result['warmup'] = warmup; result['timing_repeats'] = repeats; result['process_warm'] = True
        result['forward_calls'] = len(all_trace[-1]); result['refresh_calls'] = sum(x['refresh'] for x in all_trace[-1])
        result['refresh_ratio'] = result['refresh_calls'] / result['forward_calls']
        result['code_sha256'] = backend.code_hashes; result['status'] = 'complete'
        result['timing_scope'] = ('text encoding + noise creation + denoising + VAE decode; excludes load, '
                                  'scheduler/policy initialization, profile-file reading, transfer to CPU and video/file encoding')
        result['frames_shape'] = list(frames.shape); result['backend'] = 'seacache_wan_official_loop'
        (out / 'trace.json').write_text(json.dumps({'repeats': all_trace}, indent=2))
    except Exception as error:  # noqa: BLE001
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        (out / 'metrics.json').write_text(json.dumps(result, indent=2))
        return False
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--results', required=True, type=Path)
    p.add_argument('--no-steal', action='store_true')
    a = p.parse_args()
    W = a.work
    g = gpu_info(a.gpu)
    if gpu_processes(g['uuid']) or g['memory_used_mib'] > 1024:
        raise SystemExit(f'GPU {a.gpu} busy; refusing to share')
    os.environ['CUDA_VISIBLE_DEVICES'] = g['uuid']
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    torch.set_num_threads(4)
    # Fail closed on the host-cuDNN mixing problem (launch via tools/gpu_env.sh).
    if torch.backends.cudnn.version() != 90100:
        raise SystemExit(f'cuDNN {torch.backends.cudnn.version()} != locked 90100; launch with tools/gpu_env.sh')
    from cachebench.wan_adapter import NativeWanBackend

    manifest = a.manifest; results = a.results
    state = W / 'state' / manifest.stem
    for sub in ('claims', 'pairs', 'done', 'workers'):
        (state / sub).mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r['pair_id']].append(r)
    for v in by_pair.values():
        v.sort(key=lambda r: (METHOD_ORDER[r['method']], r.get('variant', 'official'), r.get('threshold') or 0))
    pair_order = sorted(by_pair, key=lambda k: (-len(by_pair[k]), by_pair[k][0]['seed'], by_pair[k][0]['prompt_id']))
    worker = f"{socket.gethostname()}:gpu{a.gpu}:{os.getpid()}"
    hb = state / 'workers' / f'gpu{a.gpu}.json'
    info = {'worker': worker, 'gpu': g, 'pid': os.getpid(), 'started': now(), 'status': 'loading', 'runner': 'batch_worker'}
    write_json(hb, info)

    t0 = time.perf_counter()
    backend = NativeWanBackend(W / 'vendor', W / 'models' / 'Wan2.1-T2V-1.3B', 0)
    backend.sync()
    base_meta = {'model_load_seconds': time.perf_counter() - t0, 'checkpoint_fingerprint': backend.fingerprint,
                 'source_revisions': backend.revisions, 'actual_source_revisions': backend.revisions,
                 'gpu_name': torch.cuda.get_device_name(0),
                 'versions': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'diffusers', 'transformers',
                                                                        'accelerate', 'flash-attn', 'imageio', 'Pillow')},
                 'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
                 'gpu_capability': list(torch.cuda.get_device_capability(0)), 'gpu_uuid': g['uuid']}
    base_meta['environment'] = {k: base_meta[k] for k in ('gpu_name', 'versions', 'cuda_version', 'cudnn_version', 'gpu_capability')}
    base_meta['environment'].update(python=sys.version, platform=platform.platform(), generation_backend='seacache_wan_official_loop')
    base_meta['checkpoint_fingerprint_kind'] = 'sha256 of relative filenames and complete file contents; cached on unchanged size/mtime_ns'
    base_meta['source_validation'] = ('pinned clean git commits; unchanged official cache forward bodies and SeaCache t2v_generate '
                                      'denoising loop; project-owned setup and measurement')
    # Process-level warmup: one full uncached generation of the first manifest row's problem, discarded.
    warm_row = dict(rows[0]); warm_row.update(method='baseline')
    for k in ('threshold', 'calibration_path', 'variant'):
        warm_row.pop(k, None)
    info.update(status='warming', time=now()); write_json(hb, info)
    v, _, _ = backend.generate(warm_row); del v; backend.sync()

    current_pair = None
    fails = 0

    def next_task():
        nonlocal current_pair
        if current_pair:
            for r in by_pair[current_pair]:
                if not (state / 'claims' / r['run_id']).exists():
                    return r
            current_pair = None
        for pid in pair_order:
            if excl(state / 'pairs' / pid, {'worker': worker, 'gpu_uuid': g['uuid'], 'time': now()}):
                current_pair = pid
                for r in by_pair[pid]:
                    if not (state / 'claims' / r['run_id']).exists():
                        return r
                current_pair = None
        if a.no_steal:
            return None
        best, best_n = None, 0
        for pid in pair_order:
            free = [r for r in by_pair[pid] if not (state / 'claims' / r['run_id']).exists()]
            if len(free) > best_n:
                best, best_n = free[0], len(free)
        return best

    while True:
        if (W / 'state' / 'STOP').exists() or (state / 'STOP').exists():
            info.update(status='stopped_by_flag', time=now()); write_json(hb, info); break
        r = next_task()
        if r is None:
            info.update(status='finished_no_work', time=now()); write_json(hb, info); break
        rid = r['run_id']
        claim = {'worker': worker, 'gpu_uuid': g['uuid'], 'gpu_index': a.gpu, 'pid': os.getpid(), 'time': now(),
                 'method': r['method'], 'pair_id': r['pair_id']}
        if not excl(state / 'claims' / rid, claim):
            continue
        out = results / rid
        if out.exists():
            try:
                problems, _ = check(out)
            except Exception as e:  # noqa: BLE001
                problems = [repr(e)]
            write_json(state / 'done' / f'{rid}.json', {**claim, 'status': 'complete_existing' if not problems else 'attention',
                                                         'problems': problems, 'note': 'pre-existing output; not touched'})
            continue
        info.update(status='running', run_id=rid, method=r['method'], task_started=now(), time=now()); write_json(hb, info)
        t1 = time.time()
        ok = run_task(backend, r, out, base_meta)
        rec = {**claim, 'wall_seconds': time.time() - t1, 'finished': now()}
        try:
            problems, m = check(out)
            rec.update(problems=problems, generation_seconds=m.get('generation_seconds'), refresh_ratio=m.get('refresh_ratio'),
                       peak_memory_gb=m.get('peak_memory_gb'))
            rec['status'] = 'complete' if ok and not problems else ('invalid' if ok else 'failed')
        except Exception as e:  # noqa: BLE001
            rec.update(status='failed', problems=[repr(e)])
        write_json(state / 'done' / f'{rid}.json', rec)
        fails = 0 if rec['status'] == 'complete' else fails + 1
        info.update(status='idle', last_run_id=rid, last_status=rec['status'], time=now()); write_json(hb, info)
        if fails >= 2:
            info.update(status='halted_after_failures', time=now()); write_json(hb, info); break


if __name__ == '__main__':
    main()
