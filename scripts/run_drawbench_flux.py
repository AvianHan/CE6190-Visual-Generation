"""DrawBench x FLUX.1-dev: Uncached / TeaCache / MagCache / SeaCache / SenCache generation.

Each shard runs *all* configs for its prompts on one GPU, so paired speedups
(uncached / cached, same prompt, same GPU) are well defined. Outputs per job:
``<out>/<config_id>/<prompt_id>.png`` (lossless) + ``.json`` (timing, refresh
schedule, peak memory). ``--resume`` skips jobs whose json already exists.

Timing scope = text encoding + latent init + denoising + VAE decode +
postprocess on GPU (``output_type='pt'``); excludes model load, PNG encoding.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cachebench import flux_cache as fc

DEFAULT_CONFIGS = ["uncached", "teacache:0.25", "teacache:0.4", "teacache:0.6", "teacache:0.8",
                   "magcache:0.06", "magcache:0.12", "magcache:0.24",
                   "seacache:0.15", "seacache:0.3", "seacache:0.6"]
WARMUP_PROMPT = "a photo of a cat sitting on a wooden table"


def config_id(spec):
    return spec if spec == "uncached" else spec.replace(":", "_")


def parse_spec(spec):
    if spec == "uncached":
        return "uncached", None
    method, thresh = spec.split(":")
    if method not in fc.METHODS:
        raise ValueError(spec)
    return method, float(thresh)


def read_prompts(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return [{"prompt_id": f"db_{i:03d}", "prompt": r["Prompts"].strip(), "category": r["Category"].strip()}
            for i, r in enumerate(rows)]


class Timer:
    """Synchronised wall-clock accumulation around a callable."""

    def __init__(self, fn):
        self.fn, self.total = fn, 0.0

    def __call__(self, *a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = self.fn(*a, **k)
        torch.cuda.synchronize()
        self.total += time.perf_counter() - t
        return out


def generate(pipe, args, spec, prompt, seed):
    method, thresh = parse_spec(spec)
    tr = pipe.transformer
    fc.uninstall(tr)
    gate = None
    if method != "uncached":
        gate = fc.make_gate(method, thresh, args.steps, pipe.scheduler, Path(args.vendor) / "seacache",
                            sensitivity=args.sen_calibration, sen_thresh_start=args.sen_thresh_start,
                            sen_k=args.sen_k)
        fc.install(tr, gate)
    enc, dec = Timer(type(pipe).encode_prompt.__get__(pipe)), Timer(type(pipe.vae).decode.__get__(pipe.vae))
    pipe.encode_prompt, pipe.vae.decode = enc, dec
    calls = [0]
    hook = tr.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = pipe(prompt=prompt, height=args.height, width=args.width, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, max_sequence_length=512, output_type="pt",
                     generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        hook.remove()
        del pipe.encode_prompt
        del pipe.vae.decode
        fc.uninstall(tr)
    decisions = gate.decisions if gate else [True] * calls[0]
    if len(decisions) != args.steps or calls[0] != args.steps:
        raise RuntimeError(f"{spec}: expected {args.steps} transformer calls, got {calls[0]}/{len(decisions)}")
    rgb = (image.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return rgb, {"generation_s": total, "text_encode_s": enc.total, "vae_decode_s": dec.total,
                 "denoise_s": total - enc.total - dec.total, "compute_steps": int(sum(decisions)),
                 "refresh_ratio": sum(decisions) / len(decisions), "decisions": [int(d) for d in decisions],
                 "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                 **({"sen_errors": gate.errors} if method == "sencache" else {})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--revision", default=None)
    p.add_argument("--prompts", default="benchmarks/drawbench/drawbench_prompts.csv")
    p.add_argument("--vendor", required=True, help="pinned upstream checkouts (cachebench.sources)")
    p.add_argument("--output", required=True)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--shard", default="0/1", help="i/n: prompts[i::n]")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=1, help="warmup generations per config (not saved)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sen-calibration", default=None, help="scripts/calibrate_sencache_flux.py --merge output")
    p.add_argument("--sen-thresh-start", type=float, default=fc.SEN_THRESH_START)
    p.add_argument("--sen-k", type=int, default=fc.SEN_K)
    args = p.parse_args()

    from diffusers import FluxPipeline
    import diffusers
    i, n = map(int, args.shard.split("/"))
    prompts = read_prompts(args.prompts)[i::n][: args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.configs:
        parse_spec(spec)
        (out / config_id(spec)).mkdir(exist_ok=True)

    pipe = FluxPipeline.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    env = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "diffusers": diffusers.__version__, "python": platform.python_version(), "argv": sys.argv,
           "settings": {k: getattr(args, k) for k in ("model", "revision", "seed", "steps", "guidance", "height",
                                                       "width", "sen_calibration", "sen_thresh_start",
                                                       "sen_k")}}
    (out / f"env_shard{i}of{n}.json").write_text(json.dumps(env, indent=1))

    for spec in args.configs:
        for _ in range(args.warmup):
            generate(pipe, args, spec, WARMUP_PROMPT, 1234)
        print(f"[shard {i}/{n}] warmed up {spec}", flush=True)

    todo = [(r, s) for r in prompts for s in args.configs]
    for k, (row, spec) in enumerate(todo):
        stem = out / config_id(spec) / row["prompt_id"]
        if args.resume and stem.with_suffix(".json").is_file():
            continue
        rgb, rec = generate(pipe, args, spec, row["prompt"], args.seed)
        Image.fromarray(rgb).save(stem.with_suffix(".png"))
        method, thresh = parse_spec(spec)
        rec.update(row, config_id=config_id(spec), method=method, threshold=thresh, seed=args.seed,
                   gpu=env["gpu"], shard=args.shard)
        stem.with_suffix(".json").write_text(json.dumps(rec))
        print(f"[shard {i}/{n}] {k + 1}/{len(todo)} {row['prompt_id']} {config_id(spec)} "
              f"{rec['generation_s']:.2f}s steps={rec['compute_steps']}", flush=True)


if __name__ == "__main__":
    main()
