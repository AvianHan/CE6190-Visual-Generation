"""FLUX.1-dev (T2I) cache integration: one shared forward, author WHEN rules.

Why not run the upstream forwards directly: TeaCache4FLUX and MagCache4FLUX are
written against the diffusers <=0.34 FLUX API (single blocks consume the
concatenated text+image stream), while SeaCache/FLUX targets the newer API
(single blocks return ``(encoder_hidden_states, hidden_states)``). No single
diffusers version runs all three unchanged. So this module keeps one compute
path -- the stock diffusers 0.32.2 ``FluxTransformer2DModel.forward`` without
ControlNet/LoRA/IP-Adapter/gradient-checkpointing branches -- and swaps only the
WHEN gate:

* HOW (shared): cache ``r = h_out - h_in`` over the full double+single block
  stack on image tokens; on reuse compute ``h_in + r``. Embeddings, rotary,
  norm_out and proj_out are always computed. Same as all three upstreams.
* WHEN (per method, kept verbatim from pinned upstream source):
  - TeaCache: rescaled rel-L1 of block-0 ``norm1`` modulated input, accumulated;
    first and last step forced. ``TeaCache4FLUX/teacache_flux.py``.
  - MagCache: calibrated magnitude ratios, accumulated error, K-limit,
    retention ratio, hard-coded forced step 11 (of 28).
    ``MagCache4FLUX/magcache_flux.py``.
  - SeaCache: same accumulator as TeaCache but on a spectral (Wiener) filtered
    modulated input; filter functions are AST-loaded from the pinned
    ``FLUX/util_seacache.py``. ``FLUX/seacache_generate.py``.

  - SenCache (our port; upstream has Wan2.1/CogVideoX/LTX-Video only): the
    decision rule of ``Wan2.1/sencache.py`` -- sensitivity error
    ``J_z*||z - z_c|| + J_t*|t - t_c|`` against a dual threshold (tighter for
    the first 20% of calls), at most K consecutive reuses, J looked up at the
    nearest calibrated timestep when the cache is refreshed. FLUX-dev has no
    CFG, so the cond/uncond pairing collapses to one call per step. ``z`` is
    the packed latent fed to the transformer, ``t`` the (bf16) timestep in
    [0, 1000] as the network sees it. Thresholds are given per element and
    multiplied by ``sqrt(numel(z))``; upstream's hard-coded 1447.97790038 is
    exactly sqrt(16*21*60*104), the 81-frame 832x480 Wan latent, and for FLUX
    1024^2 (4096 tokens x 64) the factor is 512. J norms come from
    ``scripts/calibrate_sencache_flux.py`` (FLUX port of
    ``Wan2.1/sensitivity_calculation.py``).

``tests/test_flux_cache.py`` checks bitwise equality against each upstream
forward on a tiny random FLUX model (SeaCache via a new-API single-block shim)
and SenCache decision parity against the upstream Wan forward on a stub model.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import types
from pathlib import Path

import numpy as np
import torch

SEA_UTIL_FILE = "FLUX/util_seacache.py"
SEA_FLUX_FILE = "FLUX/seacache_generate.py"
TEA_FLUX_FILE = "TeaCache4FLUX/teacache_flux.py"
MAG_FLUX_FILE = "MagCache4FLUX/magcache_flux.py"

# TeaCache4FLUX/teacache_flux.py (pinned 7c10efc), inside teacache_forward.
TEA_FLUX_COEFFICIENTS = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
# MagCache4FLUX/magcache_flux.py (pinned df81cb1): calibrated for 28 steps.
MAG_FLUX_RATIOS = [1.0] + [1.21094, 1.11719, 1.07812, 1.0625, 1.03906, 1.03125, 1.03906, 1.02344, 1.03125,
                           1.02344, 0.98047, 1.01562, 1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.0, 0.99609,
                           0.99609, 0.98047, 0.98828, 0.96484, 0.95703, 0.93359, 0.89062]
MAG_FLUX_K = 5
MAG_FLUX_RETENTION = 0.1
# SenCache Wan2.1 README example / project configs: K=3, start threshold 0.045.
SEN_K = 3
SEN_THRESH_START = 0.045
SEN_SWITCH_FRACTION = 0.2  # sencache.py: threshold_switch_step = round(total_calls * 0.2)
METHODS = ("uncached", "teacache", "magcache", "seacache", "sencache", "fixed")


def load_functions(path, names=None, extra_namespace=None):
    """Exec only imports + selected top-level function defs from a source file."""
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            and not (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("util_seacache"))]
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    if names is not None and {f.name for f in funcs} != set(names):
        raise ValueError(f"{path}: missing {set(names) - {f.name for f in funcs}}")
    namespace = dict(extra_namespace or {})
    exec(compile(ast.Module(body=body + funcs, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def nearest_interp(src_array, target_length):
    """MagCache4FLUX ``nearest_interp`` (only used when steps != 28)."""
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])
    scale = (src_length - 1) / (target_length - 1)
    return src_array[np.round(np.arange(target_length) * scale).astype(int)]


class Gate:
    """Per-trajectory WHEN state. ``decide`` is called once per transformer call."""
    stores_residual = True

    def __init__(self, num_steps):
        self.num_steps = int(num_steps)
        self.residual = None
        self.cnt = 0
        self.decisions = []

    def record(self, compute):
        self.decisions.append(bool(compute))
        return compute


class TeaGate(Gate):
    def __init__(self, num_steps, thresh):
        super().__init__(num_steps)
        self.thresh = float(thresh)
        self.rescale = np.poly1d(TEA_FLUX_COEFFICIENTS)
        self.acc = 0
        self.prev = None

    def decide(self, model, hidden_states, temb, img_ids, **_):
        mod = model.transformer_blocks[0].norm1(hidden_states.clone(), emb=temb.clone())[0]
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            compute = True
            self.acc = 0
        else:
            self.acc += self.rescale(((mod - self.prev).abs().mean() / self.prev.abs().mean()).cpu().item())
            compute = not (self.acc < self.thresh)
            if compute:
                self.acc = 0
        self.prev = mod
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
        return self.record(compute)


class SeaGate(Gate):
    def __init__(self, num_steps, thresh, scheduler, sea_util_path):
        super().__init__(num_steps)
        ns, self.source_sha256 = load_functions(sea_util_path)
        self.apply_sea_with_scheduler, self.rel_l1 = ns["apply_sea_with_scheduler"], ns["rel_l1"]
        self.thresh = float(thresh)
        self.scheduler = scheduler
        self.acc = 0.0
        self.prev = None

    def decide(self, model, hidden_states, temb, img_ids, **_):
        mod = model.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]
        if self.cnt == 0 or self.cnt == self.num_steps - 1 or self.prev is None:
            compute = True
            self.acc = 0.0
        else:
            mod = mod.reshape(mod.shape[0], int(img_ids[:, 1].max().item() + 1),
                              int(img_ids[:, 2].max().item() + 1), mod.shape[-1])
            mod = self.apply_sea_with_scheduler(mod, self.scheduler, self.cnt, power_exp=2.0,
                                                dims=(-2, -3), norm_mode="mean")
            mod = mod.reshape(mod.shape[0], -1, mod.shape[-1])
            self.acc += self.rel_l1(mod, self.prev)
            compute = not (self.acc < self.thresh)
            if compute:
                self.acc = 0.0
        self.prev = mod
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
        # Upstream reuses only if a residual exists.
        return self.record(compute or self.residual is None)


class MagGate(Gate):
    def __init__(self, num_steps, thresh, k=MAG_FLUX_K, retention=MAG_FLUX_RETENTION):
        super().__init__(num_steps)
        ratios = np.array(MAG_FLUX_RATIOS)
        self.ratios = ratios if len(ratios) == self.num_steps else nearest_interp(ratios, self.num_steps)
        self.thresh, self.k, self.retention = float(thresh), int(k), float(retention)
        self.acc_ratio, self.acc_steps, self.acc_err = 1.0, 0, 0

    def decide(self, model, hidden_states, temb, img_ids, **_):
        skip = False
        if self.cnt >= int(self.retention * self.num_steps + 0.5):
            self.acc_ratio = self.acc_ratio * self.ratios[self.cnt]
            self.acc_steps += 1
            self.acc_err += np.abs(1 - self.acc_ratio)
            if (self.acc_err <= self.thresh and self.acc_steps <= self.k
                    and np.round(self.cnt * ((28 - 1) / (self.num_steps - 1))).astype(int) != 11):
                skip = True
            else:
                self.acc_ratio, self.acc_steps, self.acc_err = 1.0, 0, 0
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
            self.acc_ratio, self.acc_steps, self.acc_err = 1.0, 0, 0
        return self.record(not skip)


def network_timestep(timestep, dtype):
    """The t in [0, 1000] the FLUX forward computes (``timestep.to(dtype) * 1000``), as float32."""
    return (timestep.to(dtype) * 1000).float()


def load_sensitivity(path):
    data = json.loads(Path(path).read_text())
    return {k: np.asarray(data[k], dtype=np.float64) for k in ("timesteps", "J_x_norm", "J_t_norm")}


class SenGate(Gate):
    """SenCache WHEN rule (``Wan2.1/sencache.py``), one call per step (no CFG)."""

    def __init__(self, num_steps, thresh, sensitivity, thresh_start=SEN_THRESH_START, k=SEN_K):
        super().__init__(num_steps)
        sens = load_sensitivity(sensitivity) if isinstance(sensitivity, (str, Path)) else sensitivity
        self.J_z_norm, self.J_t_norm = list(sens["J_x_norm"]), list(sens["J_t_norm"])
        self.timesteps = np.asarray(sens["timesteps"])
        self.thresh, self.thresh_start, self.k = float(thresh), float(thresh_start), int(k)
        self.switch = int(round(self.num_steps * SEN_SWITCH_FRACTION))
        self.errors = []
        self._reset()

    def _reset(self):
        self.cnt, self.skips = 0, 0
        self.cached_z = self.cached_t = self.cached_J_z = self.cached_J_t = None

    def decide(self, model, hidden_states, temb, img_ids, z=None, t=None, **_):
        if z.shape[0] != 1:
            raise ValueError("SenGate: batch size 1 only (norms are per sample, as upstream)")
        scale = math.sqrt(z[0].numel())
        thresh = (self.thresh_start if self.cnt < self.switch else self.thresh) * scale
        compute, err = True, None
        if self.cached_z is not None:
            err = (self.cached_J_z * torch.norm(z.float() - self.cached_z.float()).item()
                   + self.cached_J_t * torch.abs(t - self.cached_t).item())
            if err < thresh and self.skips < self.k:
                compute = False
                self.skips += 1
            else:
                self.skips = 0
        else:
            self.skips = 0
        if compute:
            self.cached_z, self.cached_t = z.detach(), t.detach()
            i = np.argmin(np.abs(self.timesteps - t.item()))
            self.cached_J_z, self.cached_J_t = self.J_z_norm[i], self.J_t_norm[i]
        self.errors.append(None if err is None else err / scale)
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self._reset()
        return self.record(compute)


def fixed_schedule(num_steps, compute_steps, warmup=None):
    """Static control (ours, not a published method): compute the first ``warmup``
    steps (default round(0.2*num_steps), SenCache's strict-threshold phase), then
    spread the remaining compute steps evenly, ending on the last step."""
    warmup = int(round(num_steps * SEN_SWITCH_FRACTION)) if warmup is None else int(warmup)
    n = int(compute_steps) - warmup
    if not 0 <= n <= num_steps - warmup:
        raise ValueError(f"fixed schedule: {compute_steps} compute steps out of range")
    rest = set(np.round(np.linspace(warmup - 1, num_steps - 1, n + 1))[1:].astype(int).tolist()) if n else set()
    out = [i < warmup or i in rest for i in range(num_steps)]
    assert sum(out) == compute_steps
    return out


class FixedGate(Gate):
    """Prompt-independent schedule; same residual reuse as the other gates."""

    def __init__(self, num_steps, compute_steps):
        super().__init__(num_steps)
        self.schedule = fixed_schedule(num_steps, compute_steps)

    def decide(self, model, hidden_states, temb, img_ids, **_):
        compute = self.schedule[self.cnt]
        self.cnt = (self.cnt + 1) % self.num_steps
        return self.record(compute)


class AlwaysGate(Gate):
    """Full compute through the shared forward (threshold-0 / parity path)."""
    stores_residual = False

    def decide(self, model, hidden_states, temb, img_ids, **_):
        return self.record(True)


def cached_flux_forward(self, hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None,
                        img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None,
                        controlnet_block_samples=None, controlnet_single_block_samples=None, return_dict=True,
                        controlnet_blocks_repeat=False):
    """diffusers 0.32.2 FluxTransformer2DModel.forward + residual-cache gate."""
    if joint_attention_kwargs or controlnet_block_samples is not None or controlnet_single_block_samples is not None:
        raise ValueError("cached_flux_forward: LoRA/IP-Adapter/ControlNet are not supported")
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        raise ValueError("cached_flux_forward: gradient checkpointing is not supported")
    gate = self.cache_gate
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    # SenCache inputs: packed latent and the timestep (in [0, 1000]) as the network sees it.
    sen_z, sen_t = hidden_states, network_timestep(timestep, hidden_states.dtype)
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    guidance = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None
    temb = (self.time_text_embed(timestep, pooled_projections) if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections))
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    image_rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids), dim=0))

    if not gate.decide(self, hidden_states, temb, img_ids, z=sen_z, t=sen_t):
        hidden_states = hidden_states + gate.residual
    else:
        ori_hidden_states = hidden_states
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states=hidden_states,
                                                         encoder_hidden_states=encoder_hidden_states, temb=temb,
                                                         image_rotary_emb=image_rotary_emb,
                                                         joint_attention_kwargs=None)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb,
                                  joint_attention_kwargs=None)
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        if gate.stores_residual:
            gate.residual = hidden_states - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def make_gate(method, thresh, num_steps, scheduler=None, seacache_dir=None, sensitivity=None,
              sen_thresh_start=SEN_THRESH_START, sen_k=SEN_K):
    if method == "teacache":
        return TeaGate(num_steps, thresh)
    if method == "magcache":
        return MagGate(num_steps, thresh)
    if method == "seacache":
        return SeaGate(num_steps, thresh, scheduler, Path(seacache_dir) / SEA_UTIL_FILE)
    if method == "sencache":
        if sensitivity is None:
            raise ValueError("sencache needs a FLUX sensitivity calibration file")
        return SenGate(num_steps, thresh, sensitivity, sen_thresh_start, sen_k)
    if method == "fixed":
        return FixedGate(num_steps, int(thresh))
    if method == "always":
        return AlwaysGate(num_steps)
    raise ValueError(f"unknown method {method}")


def install(transformer, gate):
    """Bind the shared forward to this transformer instance with a fresh gate."""
    transformer.cache_gate = gate
    transformer.forward = types.MethodType(cached_flux_forward, transformer)


def uninstall(transformer):
    transformer.__dict__.pop("forward", None)
    transformer.__dict__.pop("cache_gate", None)
