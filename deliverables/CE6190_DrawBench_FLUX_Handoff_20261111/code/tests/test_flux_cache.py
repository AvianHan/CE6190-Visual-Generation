"""Bitwise parity of the shared FLUX forward against pinned upstream forwards.

Tiny random FluxTransformer2DModel on CPU/fp32, 28-step flow-matching loop.
Needs diffusers==0.32.2 and pinned vendor checkouts (env CE6190_VENDOR,
default ./vendor): teacache, magcache, seacache.
"""
import copy
import math
import os
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel  # noqa: E402

from cachebench import flux_cache as fc  # noqa: E402

VENDOR = Path(os.environ.get("CE6190_VENDOR", "vendor"))
pytestmark = pytest.mark.skipif(not (VENDOR / "seacache" / fc.SEA_UTIL_FILE).is_file(),
                                reason="pinned vendor checkouts not available")
STEPS = 28
H = W = 8
THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]


def tiny_model():
    torch.manual_seed(0)
    return FluxTransformer2DModel(patch_size=1, in_channels=8, num_layers=2, num_single_layers=2,
                                  attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
                                  pooled_projection_dim=12, guidance_embeds=True, axes_dims_rope=(4, 6, 6)).eval()


def inputs():
    g = torch.Generator().manual_seed(1)
    img_ids = torch.zeros(H, W, 3)
    img_ids[..., 1] = torch.arange(H)[:, None]
    img_ids[..., 2] = torch.arange(W)[None, :]
    return dict(latents=torch.randn(1, H * W, 8, generator=g), txt=torch.randn(1, 5, 24, generator=g),
                pooled=torch.randn(1, 12, generator=g), img_ids=img_ids.reshape(-1, 3), txt_ids=torch.zeros(5, 3))


def run(model, scheduler):
    x = inputs()
    lat, outs = x["latents"], []
    with torch.inference_mode():
        for t in scheduler.timesteps:
            out = model(hidden_states=lat, encoder_hidden_states=x["txt"], pooled_projections=x["pooled"],
                        timestep=(t / 1000).expand(1), img_ids=x["img_ids"], txt_ids=x["txt_ids"],
                        guidance=torch.full((1,), 3.5), return_dict=False)[0]
            outs.append(out.clone())
            lat = scheduler.step(out, t, lat, return_dict=False)[0]
    return torch.stack(outs)


def fresh_scheduler():
    s = FlowMatchEulerDiscreteScheduler(shift=3.0)
    s.set_timesteps(STEPS)
    return s


def upstream(model, path, name, extra=None):
    ns, _ = fc.load_functions(path, [name], extra)
    model.forward = types.MethodType(ns[name], model)
    return model


def ours(model, method, thresh, scheduler):
    fc.install(model, fc.make_gate(method, thresh, STEPS, scheduler, VENDOR / "seacache"))
    return model


def test_always_gate_equals_stock_forward():
    base = tiny_model()
    ref = run(base, fresh_scheduler())
    m = ours(copy.deepcopy(base), "always", 0, None)
    assert torch.equal(run(m, fresh_scheduler()), ref)


def check_parity(method, make_upstream):
    base = tiny_model()
    full = run(copy.deepcopy(base), fresh_scheduler())
    mixed = 0
    for thresh in THRESHOLDS:
        s1 = fresh_scheduler()
        m = ours(copy.deepcopy(base), method, thresh, s1)
        got = run(m, s1)
        s2 = fresh_scheduler()
        ref = run(make_upstream(copy.deepcopy(base), thresh, s2), s2)
        assert torch.equal(got, ref), f"{method} thresh={thresh} differs from upstream"
        d = m.cache_gate.decisions
        assert len(d) == STEPS
        if 0 < sum(d) < STEPS:
            mixed += 1
            assert not torch.equal(got, full)
    assert mixed, f"{method}: no threshold produced a mixed compute/reuse schedule"


def test_teacache_matches_upstream():
    def make(m, thresh, s):
        m.enable_teacache, m.cnt, m.num_steps, m.rel_l1_thresh = True, 0, STEPS, thresh
        m.accumulated_rel_l1_distance, m.previous_modulated_input, m.previous_residual = 0, None, None
        return upstream(m, VENDOR / "teacache" / fc.TEA_FLUX_FILE, "teacache_forward")
    check_parity("teacache", make)


def test_magcache_matches_upstream():
    def make(m, thresh, s):
        m.cnt, m.num_steps, m.mag_ratios = 0, STEPS, np.array(fc.MAG_FLUX_RATIOS)
        m.K, m.magcache_thresh, m.retention_ratio = 5, thresh, 0.1
        m.accumulated_ratio, m.accumulated_err, m.accumulated_steps = 1, 0, 0
        return upstream(m, VENDOR / "magcache" / fc.MAG_FLUX_FILE, "magcache_forward")
    check_parity("magcache", make)


class NewApiSingleBlock(torch.nn.Module):
    """diffusers>=0.35 single-block calling convention over a 0.32.2 block."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs=None):
        n = encoder_hidden_states.shape[1]
        x = self.block(hidden_states=torch.cat([encoder_hidden_states, hidden_states], dim=1), temb=temb,
                       image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
        return x[:, :n], x[:, n:]


def test_seacache_matches_upstream():
    util, _ = fc.load_functions(VENDOR / "seacache" / fc.SEA_UTIL_FILE)

    def make(m, thresh, s):
        m.single_transformer_blocks = torch.nn.ModuleList(NewApiSingleBlock(b) for b in m.single_transformer_blocks)
        m.scheduler, m.enable_seacache, m.seacache_thresh = s, True, thresh
        m.cnt, m.num_steps, m.accumulated_rel_l1_distance = 0, STEPS, 0.0
        m.previous_modulated_input, m.previous_residual = None, None
        return upstream(m, VENDOR / "seacache" / fc.SEA_FLUX_FILE, "seacache_forward",
                        {"rel_l1": util["rel_l1"], "apply_sea_with_scheduler": util["apply_sea_with_scheduler"]})
    check_parity("seacache", make)


def test_constants_match_pinned_source():
    tea = (VENDOR / "teacache" / fc.TEA_FLUX_FILE).read_text()
    mag = (VENDOR / "magcache" / fc.MAG_FLUX_FILE).read_text()
    assert "4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01" in tea
    assert ", ".join(str(r) for r in fc.MAG_FLUX_RATIOS[1:]) in mag
    assert "K = 5" in mag and "retention_ratio = 0.1" in mag


# ---------------------------------------------------------------------------
# SenCache: upstream has no FLUX forward, so check the WHEN rule against the
# pinned Wan2.1 ``sencache_forward`` on a stub Wan model (CFG: cond+uncond calls),
# then feed SenGate the same (z, t) sequence and compare decisions.
SEN_FILE = "Wan2.1/sencache.py"


def upstream_sencache_forward():
    import ast
    import contextlib
    import logging
    path = VENDOR / "sencache" / SEN_FILE
    tree = ast.parse(path.read_text())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sencache_forward"]
    ns = {"torch": torch, "np": np, "logging": logging,
          "amp": types.SimpleNamespace(autocast=lambda **_: contextlib.nullcontext()),
          "sinusoidal_embedding_1d": lambda dim, t: (t.float() / 1000).reshape(1, 1).expand(1, dim)}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(path), "exec"), ns)
    return ns["sencache_forward"]


class StubPatch:
    weight = torch.zeros(1)

    def __call__(self, u):
        return u


class StubWan:
    """Just enough of WanModel for sencache_forward; blocks are a fixed nonlinear map."""
    model_type, freq_dim, dim, text_len = "t2v", 4, 4, 8

    def __init__(self):
        g = torch.Generator().manual_seed(2)
        self.w = torch.randn(4, 4, generator=g) * 0.5
        self.patch_embedding = StubPatch()  # (1, C, F, H, W) passthrough; grid sizes from shape[2:]
        self.freqs = torch.zeros(1)
        self.compute_calls = 0

    def blocks_fn(self, x, e=None, **_):
        self.compute_calls += 1
        return torch.tanh(x @ self.w) + 0.3 * x * e.mean()


def make_stub(forward, sens, thresh, thresh_start, k, steps):
    m = StubWan()
    m.time_embedding = lambda x: x
    m.time_projection = lambda e: e.mean(1, keepdim=True).expand(1, 6 * m.dim)
    m.text_embedding = lambda c: c
    m.blocks = [lambda x, **kw: m.blocks_fn(x, **kw)]
    m.head = lambda x, e: x
    m.unpatchify = lambda x, grid: [x[0].transpose(0, 1).reshape(4, *grid[0].tolist())]
    m.J_z_norm, m.J_t_norm = list(sens["J_x_norm"]), list(sens["J_t_norm"])
    m.sencache_timesteps_array = sens["timesteps"]
    m.sencache_threshold_start, m.sencache_threshold_main, m.sencache_K = thresh_start, thresh, k
    m.cached_z = m.cached_t = m.cached_residual_cond = m.cached_residual_uncond = None
    m.cached_J_z_norm = m.cached_J_t_norm = None
    m.accumulated_skips, m.skip_this_step, m.cnt, m.current_skip_count, m.all_skip_counts = 0, False, 0, 0, []
    m.num_steps = 2 * steps
    m.threshold_switch_step = int(round(2 * steps * 0.2))
    m.ret_steps, m.cutoff_steps = 0, 2 * steps
    m.forward = types.MethodType(forward, m)
    return m


@pytest.mark.skipif(not (VENDOR / "sencache" / SEN_FILE).is_file(), reason="sencache checkout missing")
def test_sencache_gate_matches_upstream_decisions():
    forward = upstream_sencache_forward()
    sched = fresh_scheduler()
    timesteps = sched.timesteps.clone()
    g = torch.Generator().manual_seed(3)
    sens = {"timesteps": timesteps.numpy()[::-1][::2].copy(),  # coarser, unsorted-agnostic lookup grid
            "J_x_norm": np.abs(np.random.default_rng(0).normal(1.0, 0.5, 14)) + 0.1,
            "J_t_norm": np.abs(np.random.default_rng(1).normal(0.02, 0.01, 14)) + 1e-3}
    z0 = torch.randn(4, 1, 2, 3, generator=g)
    numel = z0.numel()
    mixed = 0
    for raw in [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        for raw_start in [0.0, 0.045, 1.0]:
            m = make_stub(forward, sens, raw * math.sqrt(numel), raw_start * math.sqrt(numel), 3, STEPS)
            s = fresh_scheduler()
            z, zs, ts, up = z0.clone(), [], [], []
            for t in s.timesteps:
                tt = t.reshape(1)
                zs.append(z.clone())
                ts.append(tt.clone())
                before = m.compute_calls
                vc = m.forward([z], tt, [torch.zeros(1, 1)], seq_len=6)[0]
                m.forward([z], tt, [torch.zeros(1, 1)], seq_len=6)  # uncond follows cond decision
                up.append(m.compute_calls - before == 2)
                assert m.compute_calls - before in (0, 2)
                z = s.step(vc, t, z, return_dict=False)[0]
            gate = fc.SenGate(STEPS, raw, sens, thresh_start=raw_start, k=3)
            ours = [gate.decide(None, None, None, None, z=zz.reshape(1, -1), t=tt) for zz, tt in zip(zs, ts)]
            assert ours == up, f"thresh={raw} start={raw_start}: {ours} vs {up}"
            mixed += 0 < sum(up) < STEPS
            assert gate.cnt == 0 and gate.cached_z is None  # reset after a full trajectory
    assert mixed


def test_sencache_flux_forward_runs_and_skips():
    base = tiny_model()
    full = run(copy.deepcopy(base), fresh_scheduler())
    s = fresh_scheduler()
    tq = [float(fc.network_timestep(t.reshape(1) / 1000, torch.float32)) for t in s.timesteps]
    sens = {"timesteps": np.array(tq), "J_x_norm": np.ones(STEPS), "J_t_norm": np.full(STEPS, 1e-3)}
    m = copy.deepcopy(base)
    fc.install(m, fc.SenGate(STEPS, 0.5, sens, thresh_start=0.05, k=3))
    got = run(m, s)
    d = m.cache_gate.decisions
    assert len(d) == STEPS and d[0] and 0 < sum(d) < STEPS
    assert not torch.equal(got, full)
    m2 = copy.deepcopy(base)
    fc.install(m2, fc.SenGate(STEPS, 0.0, sens, thresh_start=0.0, k=3))
    assert torch.equal(run(m2, fresh_scheduler()), full)


def test_fixed_schedule_control():
    for n in range(6, STEPS + 1):
        sch = fc.fixed_schedule(STEPS, n)
        assert sum(sch) == n and all(sch[:6]) and (n == 6 or sch[-1])
    m = copy.deepcopy(tiny_model())
    fc.install(m, fc.make_gate("fixed", 14, STEPS))
    run(m, fresh_scheduler())
    assert m.cache_gate.decisions == fc.fixed_schedule(STEPS, 14)
