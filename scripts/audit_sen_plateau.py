#!/usr/bin/env python3
"""Read-only audit of the delivered Sen-FLUX threshold-2 decision logs.

Standard library only. Does not import or execute project/model code.
The reconstructed K/start threshold come from the delivery provenance file;
the original image JSONs do not independently record these parameters.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = 'CE6190_FLUX_Supplement_20261113/'
PREFIX = ROOT + 'run_records/drawbench_flux_28step_sencache/sencache_2/'
N, K, G, EPS, START = 28, 3, 6, 2.0, 0.045


def cap_schedule(n, k, g):
    decisions = [1] * g
    skips = 0
    for _ in range(g, n):
        compute = skips == k
        decisions.append(int(compute))
        skips = 0 if compute else skips + 1
    return decisions


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(zip_path):
    violations = []
    check_counts = collections.Counter()
    schedules = collections.Counter()
    compute_counts = collections.Counter()
    seeds, gpus, thresholds = collections.Counter(), collections.Counter(), collections.Counter()
    prompt_ids = []
    early_errors, reused_errors, uncapped_refresh_errors = [], [], []
    max_reuse = None
    cap_refreshes = 0
    longest_reuse = 0
    records_digest = hashlib.sha256()

    def check(ok, name, record=None, step=None):
        check_counts[name] += 1
        if not ok:
            violations.append({'check': name, 'record': record, 'step_1based': step})

    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist()
                       if n.startswith(PREFIX) and n.endswith('.json')
                       and '/' not in n[len(PREFIX):])
        check(len(names) == 200, 'exactly_200_original_records')
        provenance = json.loads(z.read(ROOT + 'Sen_FLUX_parameters_and_provenance.json'))
        check(provenance['sen_k'] == K, 'declared_K_is_3')
        check(provenance['sen_thresh_start'] == START, 'declared_start_is_0_045')
        check(provenance['steps'] == N, 'declared_steps_is_28')
        source_bytes = z.read(ROOT + 'previous_handoff/code/src/flux_cache.py')
        profile_bytes = z.read(ROOT + 'calibration/sensitivity_flux_dev_28step_1024.json')
        check(digest(profile_bytes) == provenance['profile_sha256'], 'delivered_profile_hash_matches')

        for name in names:
            raw = z.read(name)
            records_digest.update(name.encode('utf-8') + b'\0' + raw + b'\0')
            r = json.loads(raw)
            pid = r['prompt_id']
            prompt_ids.append(pid)
            seeds[str(r['seed'])] += 1
            gpus[r['gpu']] += 1
            thresholds[str(r['threshold'])] += 1
            d, e = r['decisions'], r['sen_errors']
            check(r['method'] == 'sencache' and r['config_id'] == 'sencache_2'
                  and r['threshold'] == EPS, 'method_and_threshold_identity', pid)
            check(r['seed'] == 0 and r['gpu'] == 'NVIDIA H200', 'seed_and_gpu', pid)
            check(len(d) == len(e) == N, '28_decisions_and_errors', pid)
            if len(d) != N or len(e) != N:
                continue
            check(all(type(v) is int and v in (0, 1) for v in d), 'binary_decisions', pid)
            check(d[0] == 1 and e[0] is None, 'initial_compute_without_cached_error', pid, 1)
            check(all(v == 1 for v in d[:G]), 'observed_first_six_compute', pid)
            check(sum(d) == r['compute_steps'] == 11, 'eleven_compute_steps', pid)
            check(abs(r['refresh_ratio'] - sum(d)/N) < 1e-12, 'refresh_ratio_matches', pid)
            check(d == cap_schedule(N, K, G), 'equals_cap_schedule_after_observed_prefix', pid)
            schedules[''.join(map(str, d))] += 1
            compute_counts[str(sum(d))] += 1

            skips = 0
            for i, (decision, error) in enumerate(zip(d, e)):
                if i == 0:
                    skips = 0
                    continue
                valid = isinstance(error, (int, float)) and math.isfinite(error) and error >= 0
                check(valid, 'finite_nonnegative_normalized_error', pid, i + 1)
                if not valid:
                    continue
                threshold = START if i < G else EPS
                predicted = int(not (error < threshold and skips < K))
                check(predicted == decision, 'decision_matches_logged_error_and_declared_gate', pid, i + 1)
                if i < G:
                    early_errors.append(error)
                else:
                    if decision == 0:
                        reused_errors.append(error)
                        if max_reuse is None or error > max_reuse['normalized_error']:
                            max_reuse = {'normalized_error': error, 'prompt_id': pid,
                                         'step_1based': i + 1, 'record_path': name}
                    elif skips == K:
                        cap_refreshes += 1
                    else:
                        uncapped_refresh_errors.append(error)
                skips = 0 if decision else skips + 1
                longest_reuse = max(longest_reuse, skips)
                check(skips <= K, 'no_more_than_three_consecutive_reuses', pid, i + 1)

    check(sorted(prompt_ids) == [f'db_{i:03d}' for i in range(200)], 'exact_prompt_coverage_without_duplicates')
    check(not uncapped_refresh_errors, 'no_uncapped_refreshes_after_step_six')
    lower = max(reused_errors) if reused_errors else None
    upper = min(uncapped_refresh_errors) if uncapped_refresh_errors else None
    return {
        'status': 'pass' if not violations else 'fail',
        'scope': 'Existing 200 Sen-FLUX threshold-2 records; no model execution',
        'zip_path': str(Path(zip_path).resolve()),
        'source_member': ROOT + 'previous_handoff/code/src/flux_cache.py',
        'source_sha256': digest(source_bytes),
        'profile_sha256': digest(profile_bytes),
        'records_sha256_sorted_paths_and_bytes': records_digest.hexdigest(),
        'record_count': len(names), 'unique_prompts': len(set(prompt_ids)),
        'seeds': dict(seeds), 'gpus': dict(gpus), 'thresholds': dict(thresholds),
        'declared_gate_parameters': {'steps': N, 'K': K, 'threshold_start': START,
                                     'threshold_main': EPS, 'switch_after_steps': G},
        'schedules': dict(schedules), 'compute_counts': dict(compute_counts),
        'longest_reuse_run': longest_reuse,
        'observed_prefix_compute_steps': G,
        'early_noninitial_error_min': min(early_errors),
        'early_noninitial_error_max': max(early_errors),
        'post_prefix_reuse_events': len(reused_errors),
        'post_prefix_cap_refresh_events': cap_refreshes,
        'post_prefix_uncapped_refresh_events': len(uncapped_refresh_errors),
        'max_reuse_error_witness': max_reuse,
        'schedule_preserving_main_threshold_interval': {
            'lower_exclusive': lower, 'upper_inclusive': upper,
            'upper_unbounded': upper is None,
            'interpretation': 'epsilon > lower, while early threshold, K, source, profile, inputs, and deterministic execution stay fixed'},
        'threshold_2_margin_above_lower': EPS - lower,
        'compute_count_bounds': {
            'first_step_only': math.ceil(N/(K+1)),
            'conditional_on_observed_prefix': G + (N-G)//(K+1),
            'conditional_bound_attained': all(int(c) == G+(N-G)//(K+1) for c in compute_counts)},
        'limitations': [
            'First six computes are observed and explained by tight-threshold comparisons, not unconditionally forced by SenGate.',
            'K and threshold_start are reconstructed from delivered provenance/source; original per-image JSON does not bind them or the profile hash at runtime.',
            'The platform proof is conditional on fixed inputs/profile/source and deterministic forward/sampler execution, and concerns these 200 prompts at seed0 only.',
            'Identical schedules under those assumptions do not prove best quality, global generalization, or that early steps are intrinsically more important.',
            'No changed threshold or model generation was actually executed.'
        ],
        'check_counts': dict(check_counts), 'violations': violations
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('zip_path')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit(a.zip_path)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if a.output:
        a.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
