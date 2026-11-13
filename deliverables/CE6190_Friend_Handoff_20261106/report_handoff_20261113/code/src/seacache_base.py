"""Reuse the pinned SeaCache T2V denoising loop without running its CLI.

Only the original ``for ... enumerate(tqdm(timesteps))`` node is compiled.
Its body is unchanged. Model loading, text/noise preparation, cache reset,
timing, and optional decoding remain explicit project integration code.
The caller must first verify the upstream checkout with ``sources.verify``.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

SEACACHE_T2V_FILE = "Wan2.1/seacache_generate.py"
SEACACHE_T2V_FUNCTION = "t2v_generate"


def load_seacache_denoising_loop(path, torch_module):
    """Return the original loop as a callable, plus source/function/loop hashes.

    ``tqdm`` is bound to an identity iterator to suppress progress output in
    timed runs. The model, timestep order, CFG expression, and sampler calls
    inside the upstream loop are not rewritten.
    """
    path = Path(path)
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    module = ast.parse(source, filename=str(path))
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == SEACACHE_T2V_FUNCTION]
    if len(functions) != 1:
        raise ValueError(f"{path}: expected exactly one {SEACACHE_T2V_FUNCTION}")
    original_function = functions[0]
    expected_iterator = ast.dump(
        ast.parse("enumerate(tqdm(timesteps))", mode="eval").body,
        include_attributes=False,
    )
    loops = [node for node in ast.walk(original_function)
             if isinstance(node, ast.For)
             and ast.dump(node.iter, include_attributes=False) == expected_iterator]
    if len(loops) != 1:
        raise ValueError(f"{path}: expected exactly one SeaCache denoising loop")
    original_loop = loops[0]
    loop_ast = ast.dump(original_loop, include_attributes=False)
    wrapper = ast.parse(
        "def run(self, latents, timesteps, arg_c, arg_null, guide_scale, "
        "sample_scheduler, seed_g):\n    pass\n"
    )
    wrapped_loop = copy.deepcopy(original_loop)
    return_latents = ast.Return(value=ast.Name(id="latents", ctx=ast.Load()))
    ast.copy_location(return_latents, original_loop)
    wrapper.body[0].body = [wrapped_loop, return_latents]
    ast.fix_missing_locations(wrapper)
    # This assertion protects against accidental transformations added later.
    assert ast.dump(wrapped_loop, include_attributes=False) == loop_ast
    namespace = {
        "__builtins__": {"enumerate": enumerate},
        "torch": torch_module,
        "tqdm": lambda values: values,
    }
    exec(compile(wrapper, str(path), "exec"), namespace)
    function_source = ast.get_source_segment(source, original_function)
    hashes = {
        "seacache_generation_source": hashlib.sha256(raw).hexdigest(),
        "seacache_t2v_generate": hashlib.sha256(function_source.encode()).hexdigest(),
        "seacache_denoising_loop_ast": hashlib.sha256(loop_ast.encode()).hexdigest(),
    }
    return namespace["run"], hashes


class ObservedScheduler:
    """Notify calibration before each real sampler update; delegate otherwise."""

    def __init__(self, scheduler, observer):
        self._scheduler = scheduler
        self._observer = observer
        self._step_index = 0

    def __getattr__(self, name):
        # In particular, do not hide timesteps or sigmas used by Sea filtering.
        return getattr(self._scheduler, name)

    def step(self, model_output, timestep, sample, *args, **kwargs):
        self._observer(self._step_index, timestep,
                       sample.squeeze(0), model_output.squeeze(0))
        self._step_index += 1
        return self._scheduler.step(model_output, timestep, sample, *args, **kwargs)
