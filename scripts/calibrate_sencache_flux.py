"""SenCache sensitivity calibration for FLUX.1-dev (our port of SenCache
``Wan2.1/sensitivity_calculation.py``; upstream has no FLUX version).

Per real image x0 (VAE-encoded, packed) and per sampling step i of the same
28-step FlowMatchEuler schedule used for generation (dynamic shift, 1024^2):

* x_t = (1 - sigma_i) x0 + sigma_i * noise          (fresh noise per step, as upstream)
* J_x = ||f(x_next, t_i) - f(x_t, t_i)|| / ||x_next - x_t||,  x_next = Euler step of x_t
                                                     (``calculate_solver_step_sensitivity``)
* J_t = ||(f(x_t, t_{i+1}) - f(x_t, t_i)) / (t_{i+1} - t_i)||,  t_28 = 0
                                                     (``calculate_jacobian_norm_T``)

f is the stock (uncached) transformer called exactly as FluxPipeline calls it
(bf16, guidance 3.5, T5 max_sequence_length 512); t is the network timestep in
[0, 1000] after bf16 rounding (``flux_cache.network_timestep``), the same value
SenGate measures. J_t reuses f(x_t, t_i) from the J_x pass (upstream recomputes
it; identical value). Per-sample raw values are written to ``<out>/samples``;
``--merge`` averages over samples (arithmetic mean: project convention, as in
``cachebench.calibrate``) into the json SenGate reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cachebench import flux_cache as fc


def read_samples(descriptions, images_dir, split, offset, num):
    rows = [json.loads(l) for l in open(descriptions)]
    rows = [r for r in rows if r["split"] == split and (Path(images_dir) / r["image_file"]).is_file()]
    rows.sort(key=lambda r: r["example_id"])
    return rows[offset: offset + num]


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    img = img.crop((l, t, l + size, t + size))
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1
    return x[None]


def schedule(pipe, steps, image_seq_len, device):
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    sch = pipe.scheduler
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(image_seq_len, sch.config.get("base_image_seq_len", 256),
                         sch.config.get("max_image_seq_len", 4096), sch.config.get("base_shift", 0.5),
                         sch.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(sch, steps, device, sigmas=sigmas, mu=mu)
    return timesteps.float(), sch.sigmas.float().to(device)


@torch.inference_mode()
def calibrate_one(pipe, row, args, device):
    tr, dtype = pipe.transformer, torch.bfloat16
    g = torch.Generator("cpu").manual_seed(args.seed + int(row["example_id"].split("_")[-1]))
    x = load_image(Path(args.images_dir) / row["image_file"], args.size).to(device, pipe.vae.dtype)
    lat = pipe.vae.encode(x).latent_dist.sample(g)
    lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    b, c, h, w = lat.shape
    x0 = pipe._pack_latents(lat, b, c, h, w).to(dtype)
    _, img_ids = pipe.prepare_latents(1, c, args.size, args.size, dtype, device, None, latents=x0)
    pe, pooled, txt_ids = pipe.encode_prompt(prompt=row["description"], prompt_2=None, device=device,
                                             num_images_per_prompt=1, max_sequence_length=512)
    guidance = torch.full([1], args.guidance, device=device, dtype=torch.float32)
    timesteps, sigmas = schedule(pipe, args.steps, x0.shape[1], device)

    def f(z, t):
        ts = t.expand(1).to(dtype)  # FluxPipeline: timestep = t.expand(B).to(latents.dtype)
        return tr(hidden_states=z, timestep=ts / 1000, guidance=guidance, pooled_projections=pooled,
                  encoder_hidden_states=pe, txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]

    def net_t(t):
        return fc.network_timestep(t.expand(1).to(dtype) / 1000, dtype).item()

    raw, tq = [], []
    for i in range(args.steps):
        t = timesteps[i]
        t_next = timesteps[i + 1] if i + 1 < args.steps else torch.zeros_like(t)
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        noise = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(device)
        x_t = ((1 - sigma) * x0.float() + sigma * noise).to(dtype)
        v = f(x_t, t)
        x_next = (x_t.float() + (sigma_next - sigma) * v.float()).to(dtype)  # FlowMatchEulerDiscrete.step
        v_next = f(x_next, t)
        dx = torch.norm(x_next.float() - x_t.float()).item()
        j_x = 0.0 if dx < 1e-8 else torch.norm(v_next.float() - v.float()).item() / dx
        a, b_ = net_t(t), net_t(t_next)
        v_t = f(x_t, t_next)
        j_t = torch.norm((v_t.float() - v.float()) / (b_ - a)).item()
        if not (np.isfinite(j_x) and np.isfinite(j_t)) or j_x <= 0 or j_t <= 0:
            raise RuntimeError(f"{row['example_id']} step {i}: invalid sensitivity {j_x}, {j_t}")
        raw.append([j_x, j_t])
        tq.append(a)
    return raw, tq, timesteps.cpu().tolist(), sigmas.cpu().tolist()


def merge(out):
    recs = [json.loads(p.read_text()) for p in sorted((out / "samples").glob("*.json"))]
    if not recs:
        raise SystemExit("no samples")
    tq = recs[0]["network_timesteps"]
    if any(r["network_timesteps"] != tq for r in recs):
        raise SystemExit("timestep mismatch across samples")
    raw = np.array([r["raw"] for r in recs])  # (N, steps, 2)
    res = {
        "method": "sencache", "model": "FLUX.1-dev", "port": "ours (no upstream FLUX implementation)",
        "settings": recs[0]["settings"], "num_samples": len(recs), "sample_ids": [r["example_id"] for r in recs],
        "timesteps": tq, "scheduler_timesteps": recs[0]["scheduler_timesteps"], "sigmas": recs[0]["sigmas"],
        "J_x_norm": raw[:, :, 0].mean(0).tolist(), "J_t_norm": raw[:, :, 1].mean(0).tolist(),
        "J_x_std": raw[:, :, 0].std(0).tolist(), "J_t_std": raw[:, :, 1].std(0).tolist(),
        "raw_data": raw.tolist(),
        "aggregation": "arithmetic mean over images (project convention; upstream converter absent)",
        "calibration_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = out / "sensitivity_flux_dev_28step_1024.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path} from {len(recs)} samples")
    for i, (t, jx, jt) in enumerate(zip(tq, res["J_x_norm"], res["J_t_norm"])):
        print(f"{i:2d} t={t:7.2f} J_x={jx:.4f} J_t={jt:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--revision", default=None)
    p.add_argument("--descriptions", help="DOCCI docci_descriptions.jsonlines")
    p.add_argument("--images-dir")
    p.add_argument("--split", default="train")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    if args.merge:
        return merge(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    from diffusers import FluxPipeline
    i, n = map(int, args.shard.split("/"))
    rows = read_samples(args.descriptions, args.images_dir, args.split, args.offset, args.num_samples)[i::n]
    pipe = FluxPipeline.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    settings = {k: getattr(args, k) for k in ("model", "revision", "split", "steps", "guidance", "size", "seed")}
    for k, row in enumerate(rows):
        dst = out / "samples" / f"{row['example_id']}.json"
        if dst.is_file():
            continue
        raw, tq, ts, sig = calibrate_one(pipe, row, args, "cuda")
        dst.write_text(json.dumps({"example_id": row["example_id"], "image_file": row["image_file"],
                                   "description": row["description"], "settings": settings, "raw": raw,
                                   "network_timesteps": tq, "scheduler_timesteps": ts, "sigmas": sig}))
        print(f"[shard {i}/{n}] {k + 1}/{len(rows)} {row['example_id']} "
              f"J_x[0,-1]={raw[0][0]:.3f},{raw[-1][0]:.3f} J_t[0,-1]={raw[0][1]:.4f},{raw[-1][1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
