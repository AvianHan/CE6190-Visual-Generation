"""Evaluate DrawBench x FLUX runs and write per-image + summary tables.

Metrics per image:
* fidelity vs same-prompt/seed uncached PNG: PSNR, SSIM, LPIPS (cachebench.evaluate)
* CLIP score: openai/clip-vit-large-patch14, 100 * max(cos(img, text), 0)
* ImageReward-v1.0 (THUDM) on the prompt
Summary per config: paired speedup = median over prompts of uncached_s / cached_s
(same GPU); quality = mean over prompts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cachebench.evaluate import fidelity_metrics


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


class ClipScorer:
    def __init__(self, device, name="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).eval().to(device)
        self.proc = CLIPProcessor.from_pretrained(name)
        self.device = device

    @torch.inference_mode()
    def __call__(self, image, prompt):
        x = self.proc(text=[prompt], images=[image], return_tensors="pt", padding=True, truncation=True,
                      max_length=77).to(self.device)
        out = self.model(**x)
        i = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        t = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return float(100 * (i * t).sum().clamp(min=0))


class ImageRewardScorer:
    def __init__(self, device):
        import ImageReward as RM
        self.model = RM.load("ImageReward-v1.0", device=device)

    @torch.inference_mode()
    def __call__(self, image, prompt):
        return float(self.model.score(prompt, image))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--metrics", nargs="+", default=["fidelity", "clip", "imagereward"])
    args = p.parse_args()
    run = Path(args.run)
    configs = sorted(d.name for d in run.iterdir() if d.is_dir() and any(d.glob("*.json")))
    if "uncached" not in configs:
        raise SystemExit("uncached reference missing")
    clip = ClipScorer(args.device) if "clip" in args.metrics else None
    ir = ImageRewardScorer(args.device) if "imagereward" in args.metrics else None

    ref = {j.stem: json.loads(j.read_text()) for j in (run / "uncached").glob("*.json")}
    rows = []
    for cfg in configs:
        for j in sorted((run / cfg).glob("*.json")):
            rec = json.loads(j.read_text())
            png = j.with_suffix(".png")
            row = {k: rec[k] for k in ("prompt_id", "category", "prompt", "config_id", "method", "threshold",
                                       "gpu", "generation_s", "denoise_s", "compute_steps", "refresh_ratio",
                                       "peak_mem_gib")}
            base = ref.get(rec["prompt_id"])
            if base:
                row["speedup"] = base["generation_s"] / rec["generation_s"]
                row["denoise_speedup"] = base["denoise_s"] / rec["denoise_s"]
            img = Image.open(png).convert("RGB")
            if "fidelity" in args.metrics and base and cfg != "uncached":
                f = fidelity_metrics(load_rgb(run / "uncached" / png.name)[None], np.asarray(img)[None],
                                     include=("psnr", "ssim", "lpips"), lpips_device=args.device)
                row.update(psnr=f["psnr_db"], ssim=f["ssim"], lpips=f["lpips"])
            if clip:
                row["clip_score"] = clip(img, rec["prompt"])
            if ir:
                row["image_reward"] = ir(img, rec["prompt"])
            rows.append(row)
        print(f"evaluated {cfg}", flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(run / "per_image_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def stat(vals, fn):
        vals = [v for v in vals if v is not None]
        return float(fn(vals)) if vals else None

    order = {"uncached": 0, "teacache": 1, "magcache": 2, "seacache": 3, "sencache": 4}
    summary = []
    for cfg in sorted(configs, key=lambda c: (order.get(c.split("_")[0], 9), c)):
        rs = [r for r in rows if r["config_id"] == cfg]
        summary.append({
            "config_id": cfg, "method": rs[0]["method"], "threshold": rs[0]["threshold"], "n": len(rs),
            "gpu": ",".join(sorted({r["gpu"] for r in rs})),
            "median_generation_s": stat([r["generation_s"] for r in rs], np.median),
            "median_speedup": stat([r.get("speedup") for r in rs], np.median),
            "median_denoise_speedup": stat([r.get("denoise_speedup") for r in rs], np.median),
            "mean_refresh_ratio": stat([r["refresh_ratio"] for r in rs], np.mean),
            "median_peak_mem_gib": stat([r["peak_mem_gib"] for r in rs], np.median),
            **{f"mean_{m}": stat([r.get(m) for r in rs], np.mean)
               for m in ("psnr", "ssim", "lpips", "clip_score", "image_reward")},
        })
    with open(run / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
