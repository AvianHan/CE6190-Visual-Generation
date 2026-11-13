"""Deterministic experiment manifests. No model imports or network access.

The manifest is the authoritative experiment plan. It never invents measured results.
VBench metadata duplicates are merged by exact prompt text, preserving dimensions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

METHODS = {"baseline", "teacache", "magcache", "seacache", "sencache"}
GENERATION_KEYS = (
    "model_id", "checkpoint_revision", "num_frames", "height", "width",
    "num_inference_steps", "guidance_scale", "flow_shift", "sample_solver",
    "dtype", "negative_prompt", "fps", "attention_backend", "offload", "generation_backend",
)


def digest(value, length=20):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def canonical_prompt(prompt):
    return " ".join(prompt.split()).casefold()


def load_prompts(path):
    """Load JSON array, JSONL, or plain text; VBench full metadata is accepted."""
    path = Path(path)
    if path.suffix == ".json":
        values = json.loads(path.read_text())
    elif path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not isinstance(values, list):
        raise ValueError("Prompt input must contain a list")
    unique = {}
    for value in values:
        row = {"prompt": value} if isinstance(value, str) else dict(value)
        prompt = row.get("prompt", row.get("prompt_en", ""))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Empty/non-string prompt")
        prompt = prompt.strip()
        dimensions = row.get("dimensions", row.get("dimension", []))
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        if prompt not in unique:
            unique[prompt] = {
                "prompt_id": row.get("prompt_id", digest(prompt, 16)),
                "prompt": prompt,
                "dimensions": [],
                "split": row.get("split"),
            }
        out = unique[prompt]
        if row.get("split") and out["split"] and out["split"] != row["split"]:
            raise ValueError(f"Prompt assigned to more than one split: {prompt}")
        out["split"] = out["split"] or row.get("split")
        out["dimensions"] = sorted(set(out["dimensions"]) | set(dimensions))
    if len({r["prompt_id"] for r in unique.values()}) != len(unique):
        raise ValueError("Different prompts share a prompt_id")
    return list(unique.values())


def assert_disjoint(*collections):
    seen = {}
    for i, rows in enumerate(collections):
        for row in rows:
            key = canonical_prompt(row["prompt"])
            if key in seen and seen[key] != i:
                raise ValueError(f"Prompt leakage between sets {seen[key]} and {i}: {row['prompt']}")
            seen[key] = i


def split_prompts(rows, fractions=(0.10, 0.20, 0.70), salt="ce6190-v1"):
    """Optional hash split at prompt level; never split seeds of one prompt."""
    if len(fractions) != 3 or any(x < 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("calibration/tune/test fractions must sum to one")
    out = []
    for row in rows:
        copied = dict(row)
        u = int(digest([salt, canonical_prompt(row["prompt"])], 16), 16) / 2**64
        copied["split"] = "calibration" if u < fractions[0] else "tune" if u < sum(fractions[:2]) else "test"
        out.append(copied)
    return out


def policy_variants(spec):
    method = spec["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    grid = spec.get("grid", {})
    keys = sorted(grid)
    if any(not isinstance(grid[k], list) or not grid[k] for k in keys):
        raise ValueError("Every grid parameter must be a nonempty list")
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield {"method": method, **spec.get("parameters", {}), **dict(zip(keys, vals))}


def build_manifest(prompts, config):
    generation = config["generation"]
    missing = [k for k in ("model_id", "num_frames", "height", "width", "num_inference_steps") if k not in generation]
    if missing:
        raise ValueError(f"Missing generation fields {missing}")
    if generation["num_frames"] % 4 != 1:
        raise ValueError("Wan frame count must be 4n+1")
    if generation["height"] % 16 or generation["width"] % 16:
        raise ValueError("Resolution must be divisible by 16")
    specs = config["methods"]
    if sum(s["method"] == "baseline" for s in specs) != 1:
        raise ValueError("Exactly one baseline method specification required")
    variants = [v for s in specs for v in policy_variants(s)]
    if sum(v["method"] == "baseline" for v in variants) != 1:
        raise ValueError("Baseline must not have a policy grid")
    baseline_policy = next(v for v in variants if v["method"] == "baseline")
    default_split = config.get("default_split", "test")
    rows = []
    for prompt in prompts:
        split = prompt.get("split") or default_split
        if split not in {"calibration", "tune", "test", "smoke"}:
            raise ValueError(f"Unknown split {split}")
        seeds = list(config.get("seeds", [0, 1, 2, 3, 4]))
        if "temporal_flickering" in prompt["dimensions"] and config.get("temporal_flickering_25", False):
            # Independent predetermined seeds, still identical across methods.
            seeds = list(range(25))
        for sample_index, seed in enumerate(seeds):
            base = {**generation, **prompt, "split": split, "seed": seed, "sample_index": sample_index}
            pair_id = digest({k: base.get(k) for k in GENERATION_KEYS + ("prompt_id", "prompt", "seed")})
            ref_id = digest([pair_id, baseline_policy])
            for policy in variants:
                # Explicitly prevent a method-specific grid changing the generated problem.
                if set(policy) & set(GENERATION_KEYS):
                    raise ValueError("Generation settings belong in generation, shared by all methods")
                run_id = digest([pair_id, policy])
                rows.append({**base, **policy, "pair_id": pair_id, "run_id": run_id,
                             "reference_run_id": ref_id, "schema_version": 1,
                             "source_revisions": config.get("source_revisions", {}),
                             "experiment_id": config.get("experiment_id", "ce6190-cache")})
    if len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate runs: duplicate policy variants or seeds")
    return rows


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate run ids")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--exclude-prompts", action="append", default=[], help="Fail if any prompt overlaps a development/calibration set")
    p.add_argument("--hash-split", action="store_true", help="Use a documented 10/20/70 custom split; not the full VBench protocol")
    args = p.parse_args()
    prompts = load_prompts(args.prompts)
    for exclusion in args.exclude_prompts:
        assert_disjoint(prompts, load_prompts(exclusion))
    if args.hash_split:
        prompts = split_prompts(prompts)
    config = json.loads(Path(args.config).read_text())
    rows = build_manifest(prompts, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {"prompts": len(prompts), "runs": len(rows), "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "prompt_source": str(Path(args.prompts).resolve()), "config": config, "contains_measured_results": False}
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"manifest": str(output), "prompts": len(prompts), "runs": len(rows)}))


if __name__ == "__main__":
    main()
