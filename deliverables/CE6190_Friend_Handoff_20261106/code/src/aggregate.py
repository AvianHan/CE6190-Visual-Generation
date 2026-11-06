"""Paired aggregation with prompt-cluster bootstrap and Pareto frontiers."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .manifest import digest

POLICY_KEYS = ("method", "variant", "threshold", "threshold_start", "max_skip", "retention_ratio", "calibration_path", "use_ret_steps", "use_ref_steps", "sencache_scale")


def policy(config):
    return {k: config[k] for k in POLICY_KEYS if k in config}


def bootstrap_prompt_means(rows, key, draws=2000, seed=0):
    """Seeds are nested in prompts; resample prompts, not individual video frames."""
    clusters = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            clusters[row["config"]["prompt_id"]].append(float(value))
    values = np.array([np.mean(x) for _, x in sorted(clusters.items())], dtype=np.float64)
    if not len(values):
        return {"mean": None, "ci95": None, "n_prompts": 0}
    mean = float(values.mean())
    if len(values) == 1:
        return {"mean": mean, "ci95": None, "n_prompts": 1}
    rng = np.random.default_rng(seed)
    estimates = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)])
    return {"mean": mean, "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])], "n_prompts": len(values)}


def pareto_indices(points):
    """Indices nondominated under smaller latency and smaller LPIPS."""
    front = []
    for i, (latency, error) in enumerate(points):
        if not any(t <= latency and e <= error and (t < latency or e < error)
                   for j, (t, e) in enumerate(points) if j != i):
            front.append(i)
    return front


def summarize(records, draws=2000, allow_unpaired=False):
    if len({r["config"]["split"] for r in records}) > 1:
        raise ValueError("Mixed tune/test/smoke splits: aggregate each split separately")
    for key in ("gpu_name", "checkpoint_fingerprint", "environment", "actual_source_revisions"):
        if len({json.dumps(r.get(key), sort_keys=True) for r in records}) > 1:
            raise ValueError(f"Mixed actual provenance across policies: {key}")
    groups = defaultdict(list)
    for row in records:
        if row.get("status") != "evaluated":
            raise ValueError("Only measured evaluated records may be aggregated")
        groups[digest(policy(row["config"]))].append(row)
    pair_sets = [{r["config"]["pair_id"] for r in group} for group in groups.values()]
    if not allow_unpaired and pair_sets and any(s != pair_sets[0] for s in pair_sets[1:]):
        raise ValueError("Methods/settings have different successful pairs; complete missing runs before comparison")
    summaries = []
    for group_id, rows in sorted(groups.items()):
        if len({r.get("calibration_sha256") for r in rows}) > 1:
            raise ValueError("Calibration profile changed within one policy; do not merge overwritten profiles")
        ids = [r["config"]["pair_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate paired results in one policy")
        hardware = {r.get("gpu_name") for r in rows}
        if len(hardware) != 1:
            raise ValueError("Mixed hardware within one policy; split the analysis")
        inf_count = sum(bool(r.get("psnr_db_is_infinite")) for r in rows)
        summary = {"policy_id": group_id, "policy": policy(rows[0]["config"]),
                   "n_videos": len(rows), "n_prompts": len({r['config']['prompt_id'] for r in rows}),
                   "splits": sorted({r['config']['split'] for r in rows}), "gpu_name": next(iter(hardware)),
                   "psnr_infinite_count": inf_count,
                   "calibration_sha256": rows[0].get("calibration_sha256"),
                   "metrics": {key: bootstrap_prompt_means(rows, key, draws=draws) for key in
                               ("generation_seconds", "reference_generation_seconds", "speedup", "psnr_db", "ssim", "lpips", "refresh_ratio", "peak_memory_gb")}}
        # Two speedup definitions are intentionally both preserved.
        summary["ratio_of_total_times"] = sum(r["reference_generation_seconds"] for r in rows) / sum(r["generation_seconds"] for r in rows)
        if inf_count:
            summary["metrics"]["psnr_db"]["warning"] = "Finite-only mean; exact matches have infinite PSNR. Do not treat this as the all-video arithmetic mean."
        summaries.append(summary)
    valid = [i for i, s in enumerate(summaries) if s["metrics"]["lpips"]["mean"] is not None]
    front = pareto_indices([(summaries[i]["metrics"]["generation_seconds"]["mean"], summaries[i]["metrics"]["lpips"]["mean"]) for i in valid])
    front_global = {valid[j] for j in front}
    for i, summary in enumerate(summaries):
        summary["pareto_latency_lpips"] = i in front_global if i in valid else None
    return summaries


def freeze_at_speedup(summaries, original_config, target_speedup):
    """Choose lowest LPIPS satisfying a latency budget on TUNE only.

    This freezes threshold choices before test, preventing test-set best-picking.
    """
    if target_speedup <= 1:
        raise ValueError("Target speedup must exceed one")
    if any(s["splits"] != ["tune"] for s in summaries):
        raise ValueError("Threshold selection is permitted on tune data only")
    if any(s["policy"].get("variant", "official") != "official" for s in summaries):
        raise ValueError("Main-table threshold selection must use official variants only; analyze ablations separately")
    by_method = defaultdict(list)
    for summary in summaries:
        if summary["metrics"]["lpips"]["mean"] is None:
            raise ValueError("LPIPS is required for quality-aware validation selection")
        by_method[summary["policy"]["method"]].append(summary)
    chosen = []
    decisions = []
    wanted = {s["method"] for s in original_config["methods"]} - {"baseline"}
    if set(by_method) != wanted:
        raise ValueError("All requested methods must have completed validation results")
    for method in sorted(by_method):
        candidates = [s for s in by_method[method] if s["ratio_of_total_times"] >= target_speedup]
        if not candidates:
            raise ValueError(f"{method} cannot reach measured {target_speedup}x; extend validation sweep or report not reached")
        selected = min(candidates, key=lambda s: (s["metrics"]["lpips"]["mean"], s["metrics"]["generation_seconds"]["mean"]))
        chosen.append({"method": method, "parameters": {k: v for k, v in selected["policy"].items() if k != "method"}})
        decisions.append({"method": method, "policy_id": selected["policy_id"], "validation_speedup": selected["ratio_of_total_times"],
                          "validation_lpips": selected["metrics"]["lpips"]["mean"]})
    output = dict(original_config)
    output["methods"] = [{"method": "baseline"}] + chosen
    output["default_split"] = "test"
    output["selection"] = {"split": "tune", "target_speedup": target_speedup, "rule": "minimum LPIPS among measured speedup >= target", "decisions": decisions}
    return output


def save_summary(summaries, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))
    columns = ["policy_id", "method", "variant", "n_prompts", "n_videos", "seconds", "speedup", "lpips", "ssim", "finite_psnr", "pareto"]
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for s in summaries:
            writer.writerow({"policy_id": s["policy_id"], "method": s["policy"]["method"], "n_prompts": s["n_prompts"],
                             "variant": s["policy"].get("variant", "official"),
                             "n_videos": s["n_videos"], "seconds": s["metrics"]["generation_seconds"]["mean"],
                             "speedup": s["ratio_of_total_times"], "lpips": s["metrics"]["lpips"]["mean"],
                             "ssim": s["metrics"]["ssim"]["mean"], "finite_psnr": s["metrics"]["psnr_db"]["mean"],
                             "pareto": s["pareto_latency_lpips"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--freeze-config"); p.add_argument("--target-speedup", type=float, default=2.0)
    p.add_argument("--frozen-output")
    args = p.parse_args()
    records = [json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    summaries = summarize(records, draws=args.bootstrap_draws)
    save_summary(summaries, args.output)
    if args.freeze_config:
        if not args.frozen_output:
            p.error("--frozen-output is required with --freeze-config")
        original = json.loads(Path(args.freeze_config).read_text())
        frozen = freeze_at_speedup(summaries, original, args.target_speedup)
        Path(args.frozen_output).write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    print(json.dumps({"policies": len(summaries), "output": args.output}))


if __name__ == "__main__":
    main()
