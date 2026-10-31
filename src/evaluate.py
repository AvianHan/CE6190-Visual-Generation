"""Paired fidelity evaluation on pre-encoding RGB arrays and VBench staging.

Fidelity is similarity to the uncached result, not a substitute for video quality.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np

from .manifest import GENERATION_KEYS, read_manifest


@lru_cache(maxsize=4)
def lpips_network(device):
    """Load pretrained LPIPS once per device, not once per video."""
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("LPIPS requires torch and lpips with pretrained weights; no random-weight fallback") from error
    return lpips.LPIPS(net="alex", version="0.1").eval().to(device)


def validate_frames(array, name="frames"):
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 [T,H,W,3], got {array.shape}/{array.dtype}")
    if not array.size:
        raise ValueError("Empty video")


def fidelity_metrics(reference, candidate, include=("psnr", "ssim"), lpips_device="cpu"):
    """No resizing or temporal alignment: matched jobs must have identical frames."""
    validate_frames(reference, "reference")
    validate_frames(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate frame shapes differ")
    include = set(include)
    if include - {"psnr", "ssim", "lpips"}:
        raise ValueError(f"Unsupported metrics: {include}")
    result = {"frames": len(reference), "height": int(reference.shape[1]), "width": int(reference.shape[2]),
              "rgb_data_range": 255, "reference_type": "same_seed_uncached_rgb_before_video_encoding"}
    if "psnr" in include:
        per_frame_mse = [float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)) for a, b in zip(reference, candidate)]
        mse = float(np.mean(per_frame_mse))
        result["mse_rgb"] = mse
        result["psnr_db"] = None if mse == 0 else 10 * math.log10(255**2 / mse)
        result["psnr_db_is_infinite"] = mse == 0
        psnrs = [None if x == 0 else 10 * math.log10(255**2 / x) for x in per_frame_mse]
        result["mean_frame_psnr_db"] = None if any(x is None for x in psnrs) else float(np.mean(psnrs))
        result["identical_frame_count"] = sum(x == 0 for x in per_frame_mse)
    if "ssim" in include:
        try:
            from skimage.metrics import structural_similarity
        except ImportError as error:
            raise RuntimeError("SSIM requires scikit-image; install the metrics dependencies") from error
        if min(reference.shape[1:3]) < 11:
            raise ValueError("SSIM protocol uses an 11x11 window; frame dimensions must be >=11")
        scores = [structural_similarity(a, b, channel_axis=-1, data_range=255, gaussian_weights=True,
                                        sigma=1.5, use_sample_covariance=False, win_size=11)
                  for a, b in zip(reference, candidate)]
        result["ssim"] = float(np.mean(scores))
        result["ssim_protocol"] = "RGB channel mean; Gaussian 11x11 sigma1.5; population covariance; frame mean"
    if "lpips" in include:
        try:
            import torch
            import lpips
        except ImportError as error:
            raise RuntimeError("LPIPS requires torch and lpips with pretrained weights; no random-weight fallback") from error
        net = lpips_network(lpips_device)
        scores = []
        with torch.inference_mode():
            for a, b in zip(reference, candidate):
                aa = torch.from_numpy(np.array(a)).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device) / 127.5 - 1
                bb = torch.from_numpy(np.array(b)).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device) / 127.5 - 1
                scores.append(float(net(aa, bb).item()))
        result["lpips"] = float(np.mean(scores))
        result["lpips_protocol"] = "AlexNet v0.1 pretrained; RGB [-1,1]; original resolution; frame mean"
    return result


def load_finished(root, run_id):
    directory = Path(root) / run_id
    metadata_file = directory / "metrics.json"
    frame_file = directory / "frames.npy"
    if not metadata_file.is_file() or not frame_file.is_file():
        raise FileNotFoundError(f"Incomplete run {run_id}: need metrics.json and frames.npy")
    metadata = json.loads(metadata_file.read_text())
    if metadata.get("status") not in {"ok", "success", "completed", "complete"}:
        raise ValueError(f"Run {run_id} not successful: {metadata.get('status')}")
    frames = np.load(frame_file, mmap_mode="r", allow_pickle=False)
    validate_frames(frames)
    return metadata, frames


def evaluate_manifest(manifest_path, root, output, include=("psnr", "ssim", "lpips"), lpips_device="cpu", samples_per_prompt=5):
    rows = read_manifest(manifest_path)
    if samples_per_prompt is not None:
        if samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be positive")
        rows = [r for r in rows if r.get("sample_index", 0) < samples_per_prompt]
    expected = {r["run_id"]: r for r in rows}
    # Fail before writing a misleading partial aggregate.
    for row in rows:
        if row["reference_run_id"] not in expected:
            raise ValueError(f"Missing baseline manifest row for {row['run_id']}")
        load_finished(root, row["run_id"])
    records = []
    for row in rows:
        if row["method"] == "baseline":
            continue
        ref_row = expected[row["reference_run_id"]]
        if ref_row["method"] != "baseline":
            raise ValueError("Reference must be the uncached baseline")
        for key in GENERATION_KEYS + ("prompt_id", "prompt", "seed", "pair_id"):
            if row.get(key) != ref_row.get(key):
                raise ValueError(f"Mismatched paired configuration: {key}")
        metadata, candidate = load_finished(root, row["run_id"])
        ref_meta, reference = load_finished(root, row["reference_run_id"])
        for key in ("checkpoint_fingerprint", "gpu_name", "environment", "actual_source_revisions"):
            if not metadata.get(key) or not ref_meta.get(key):
                raise ValueError(f"Missing actual run provenance: {key}")
            if metadata[key] != ref_meta[key]:
                raise ValueError(f"Paired runs have incompatible actual provenance: {key}")
        for meta, spec in [(metadata, row), (ref_meta, ref_row)]:
            stored = meta.get("config", {})
            for key in spec:
                if stored.get(key) != spec.get(key):
                    raise ValueError(f"Metadata does not match manifest: {spec['run_id']} {key}")
        metrics = fidelity_metrics(reference, candidate, include, lpips_device)
        seconds = metadata.get("generation_seconds")
        reference_seconds = ref_meta.get("generation_seconds")
        if seconds is None or reference_seconds is None or seconds <= 0 or reference_seconds <= 0:
            raise ValueError("Positive synchronized generation timings are required")
        records.append({"config": row, **metrics, "generation_seconds": seconds,
                        "reference_generation_seconds": reference_seconds,
                        "speedup": reference_seconds / seconds,
                        "denoise_seconds": metadata.get("denoise_seconds"),
                        "peak_memory_gb": metadata.get("peak_memory_gb"),
                        "refresh_ratio": metadata.get("refresh_ratio"),
                        "gpu_name": metadata.get("gpu_name"), "checkpoint_fingerprint": metadata["checkpoint_fingerprint"],
                        "environment": metadata["environment"], "actual_source_revisions": metadata["actual_source_revisions"],
                        "calibration_sha256": metadata.get("calibration_sha256"),
                        "status": "evaluated"})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in records))
    return records


def stage_vbench(manifest, root, output, official_info, method, policy_id=None, samples=5, allow_subset=False, dimension=None):
    """Stage exact official filenames and preserve semantic auxiliary metadata.

    This is preparation only. Run official evaluate.py separately. For flickering,
    stage 25 candidates and apply official static_filter.py before evaluation.
    """
    rows = [r for r in read_manifest(manifest) if r["method"] == method]
    if dimension:
        rows = [r for r in rows if dimension in r.get("dimensions", [])]
    if policy_id:
        rows = [r for r in rows if r.get("policy_id") == policy_id]
    # There must be one policy variant per prompt/seed; grid outputs cannot mix.
    pairs = [(r["prompt_id"], r["sample_index"]) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Several policy settings found; supply a manifest containing only the frozen setting")
    original = json.loads(Path(official_info).read_text())
    if dimension:
        original = [r for r in original if dimension in r["dimension"]]
    prompt_set = {r["prompt_en"] for r in original}
    present = {r["prompt"] for r in rows}
    if present - prompt_set:
        raise ValueError("Non-VBench prompts present; use custom-input metrics instead")
    if present != prompt_set and not allow_subset:
        raise ValueError("Incomplete official prompt suite; --allow-subset records a custom subset, not standard VBench")
    dst = Path(output)
    videos = dst / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    selected = {(r["prompt"], r["sample_index"]): r for r in rows}
    expected_filenames = set()
    for prompt in sorted(present):
        if "/" in prompt or "\\" in prompt or len((prompt + "-24.mp4").encode()) > 255:
            raise ValueError("Prompt incompatible with official filename convention")
        for i in range(samples):
            row = selected.get((prompt, i))
            if row is None:
                raise ValueError(f"Missing sample {i} for {prompt}")
            meta, _ = load_finished(root, row["run_id"])
            if any(meta.get("config", {}).get(key) != value for key, value in row.items()):
                raise ValueError(f"Staged video metadata does not match manifest: {row['run_id']}")
            source = (Path(root) / row["run_id"] / "video.mp4").resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            target = videos / f"{prompt}-{i}.mp4"
            expected_filenames.add(target.name)
            if target.is_symlink() and target.resolve() == source:
                continue
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {target}")
            target.symlink_to(source)
    stale = {p.name for p in videos.iterdir()} - expected_filenames
    if stale:
        raise ValueError(f"Unexpected stale files in staging directory: {sorted(stale)[:3]}")
    metadata = [r for r in original if r["prompt_en"] in present]
    (dst / "VBench_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    details = {"protocol": "custom_subset" if present != prompt_set else "standard_prompt_coverage",
               "unique_prompts": len(present), "metadata_records": len(metadata), "samples_per_prompt": samples,
               "warning": "Coverage alone does not establish official temporal-flickering/static-filter protocol compliance",
               "generation_manifest": str(Path(manifest).resolve()), "method": method}
    details["dimension"] = dimension
    (dst / "staging.json").write_text(json.dumps(details, indent=2))
    return details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fidelity")
    f.add_argument("--manifest", required=True); f.add_argument("--root", required=True); f.add_argument("--output", required=True)
    f.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips"])
    f.add_argument("--device", default="cpu")
    f.add_argument("--samples-per-prompt", type=int, default=5, help="Fidelity uses first five common seeds; extra flickering candidates are reserved for static filtering")
    v = sub.add_parser("stage-vbench")
    v.add_argument("--manifest", required=True); v.add_argument("--root", required=True); v.add_argument("--output", required=True)
    v.add_argument("--official-info", required=True); v.add_argument("--method", required=True)
    v.add_argument("--samples", type=int, default=5); v.add_argument("--allow-subset", action="store_true")
    v.add_argument("--dimension")
    a = p.parse_args()
    if a.command == "fidelity":
        result = evaluate_manifest(a.manifest, a.root, a.output, a.metrics, a.device, a.samples_per_prompt)
        print(json.dumps({"evaluated_pairs": len(result), "output": a.output}))
    else:
        print(json.dumps(stage_vbench(a.manifest, a.root, a.output, a.official_info, a.method,
                                     samples=a.samples, allow_subset=a.allow_subset, dimension=a.dimension)))


if __name__ == "__main__":
    main()
