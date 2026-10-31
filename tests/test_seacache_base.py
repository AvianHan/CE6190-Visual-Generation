"""Stdlib mock tests of SeaCache's actual extracted loop, not GPU results."""
import ast
import contextlib
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest

from cachebench.seacache_base import load_seacache_denoising_loop, ObservedScheduler
from cachebench.wan_adapter import NativeWanBackend


SOURCE = Path(os.environ.get(
    "CACHEBENCH_SEACACHE_SOURCE",
    Path(__file__).resolve().parents[1] / "vendor/seacache/Wan2.1/seacache_generate.py",
))


class ScalarTensor:
    """A scalar with a batch-rank marker; no tensor library or GPU involved."""

    def __init__(self, value, rank=0):
        self.value, self.rank = float(value), rank

    def __add__(self, other):
        return ScalarTensor(self.value + other.value, self.rank)

    def __sub__(self, other):
        return ScalarTensor(self.value - other.value, self.rank)

    def __rmul__(self, number):
        return ScalarTensor(number * self.value, self.rank)

    def unsqueeze(self, dimension):
        assert dimension == 0
        return ScalarTensor(self.value, self.rank + 1)

    def squeeze(self, dimension):
        assert dimension == 0 and self.rank > 0
        return ScalarTensor(self.value, self.rank - 1)


class FakeGenerator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"
    Generator = FakeGenerator

    @staticmethod
    def stack(values):
        assert len(values) == 1
        return values[0].unsqueeze(0)

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()

    @staticmethod
    def autocast(*args, **kwargs):
        return contextlib.nullcontext()

    @staticmethod
    def randn(shape, **kwargs):
        return ScalarTensor(1.0)


class FakeModel:
    def __init__(self):
        self.calls = []
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, latents, *, t, context, seq_len):
        self.calls.append((context, latents[0].value, t.value, seq_len))
        offset = 2 if context == "conditional" else -1
        return [ScalarTensor(latents[0].value + offset)]


class FakeScheduler:
    def __init__(self):
        self.timesteps = [ScalarTensor(.8), ScalarTensor(.4)]
        self.sigmas = [1., .5, 0.]
        self.calls = []

    def step(self, prediction, timestep, sample, *, return_dict, generator):
        assert prediction.rank == sample.rank == 1
        assert return_dict is False
        self.calls.append((prediction.value, timestep.value, sample.value, generator))
        return (ScalarTensor(sample.value - .1 * prediction.value, rank=1),)


class ExtractionRejectionTests(unittest.TestCase):
    def extract(self, source):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "source.py"
            file.write_text(source)
            return load_seacache_denoising_loop(file, FakeTorch)

    def test_missing_function_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one t2v_generate"):
            self.extract("def unrelated():\n    pass\n")

    def test_duplicate_function_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one t2v_generate"):
            self.extract("def t2v_generate():\n    pass\n" * 2)

    def test_missing_loop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one SeaCache denoising loop"):
            self.extract("def t2v_generate():\n    for t in timesteps:\n        pass\n")

    def test_multiple_matching_loops_are_rejected(self):
        loop = "    for _, t in enumerate(tqdm(timesteps)):\n        pass\n"
        with self.assertRaisesRegex(ValueError, "exactly one SeaCache denoising loop"):
            self.extract("def t2v_generate():\n" + loop * 2)


@unittest.skipUnless(SOURCE.is_file(), "fetch pinned SeaCache or set CACHEBENCH_SEACACHE_SOURCE")
class OriginalSeaCacheLoopTests(unittest.TestCase):
    def setUp(self):
        self.loop, self.hashes = load_seacache_denoising_loop(SOURCE, FakeTorch)
        self.pipe = types.SimpleNamespace(device="mock-cpu", model=FakeModel())
        self.scheduler = FakeScheduler()
        self.generator = object()

    def execute(self, scheduler=None):
        return self.loop(self.pipe, [ScalarTensor(1.)], self.scheduler.timesteps,
                         {"context": "conditional", "seq_len": 7},
                         {"context": "unconditional", "seq_len": 7},
                         2., scheduler or self.scheduler, self.generator)

    def test_real_loop_orders_cfg_and_updates_current_latent(self):
        result = self.execute()
        self.assertEqual([c[0] for c in self.pipe.model.calls],
                         ["conditional", "unconditional"] * 2)
        # cond=x+2; uncond=x-1; scale=2 -> prediction=x+5.
        # Sampler: x' = x-.1*prediction. 1 -> .4 -> -.14.
        self.assertAlmostEqual(result[0].value, -.14)
        self.assertAlmostEqual(self.scheduler.calls[0][0], 6.)
        self.assertAlmostEqual(self.scheduler.calls[1][0], 5.4)
        self.assertAlmostEqual(self.pipe.model.calls[2][1], .4)
        self.assertTrue(all(c[3] is self.generator for c in self.scheduler.calls))
        self.assertEqual(self.pipe.model.devices, ["mock-cpu", "mock-cpu"])

    def test_observer_receives_preupdate_latent_and_guided_prediction(self):
        observed = []

        def observe(index, timestep, latent, prediction):
            self.assertEqual(len(self.scheduler.calls), index)
            self.assertEqual((latent.rank, prediction.rank), (0, 0))
            observed.append((index, timestep.value, latent.value, prediction.value))

        proxy = ObservedScheduler(self.scheduler, observe)
        self.assertIs(proxy.sigmas, self.scheduler.sigmas)
        self.assertIs(proxy.timesteps, self.scheduler.timesteps)
        self.execute(proxy)
        self.assertEqual([c[0] for c in observed], [0, 1])
        self.assertAlmostEqual(observed[0][2], 1.)
        self.assertAlmostEqual(observed[1][2], .4)
        self.assertAlmostEqual(observed[1][3], 5.4)

    def test_hashes_identify_actual_source_function_and_unmodified_loop(self):
        source = SOURCE.read_text()
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "t2v_generate")
        loop = next(n for n in ast.walk(function) if isinstance(n, ast.For)
                    and ast.unparse(n.iter) == "enumerate(tqdm(timesteps))")
        self.assertEqual(self.hashes["seacache_generation_source"],
                         hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertEqual(self.hashes["seacache_t2v_generate"],
                         hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest())
        self.assertEqual(self.hashes["seacache_denoising_loop_ast"],
                         hashlib.sha256(ast.dump(loop, include_attributes=False).encode()).hexdigest())

    def test_module_cli_and_imports_are_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "sea.py"
            file.write_text("raise RuntimeError('must not execute module')\n" + SOURCE.read_text())
            loop, _ = load_seacache_denoising_loop(file, FakeTorch)
            result = loop(self.pipe, [ScalarTensor(1.)], self.scheduler.timesteps,
                          {"context": "conditional", "seq_len": 7},
                          {"context": "unconditional", "seq_len": 7},
                          2., self.scheduler, self.generator)
            self.assertAlmostEqual(result[0].value, -.14)

    def test_backend_decode_false_preserves_calibration_observer(self):
        backend = NativeWanBackend.__new__(NativeWanBackend)
        backend.torch, backend.pipe = FakeTorch, self.pipe
        backend.denoising_loop = self.loop
        backend.make_scheduler = lambda row: self.scheduler
        backend.patch = lambda row, scheduler: None
        backend.shape = lambda row: ((1,), 7)
        backend.encode_text = lambda prompt, negative: ("conditional", "unconditional")
        backend.sync = lambda: None
        backend.trace = types.SimpleNamespace(events=[])
        decoded = []
        self.pipe.vae = types.SimpleNamespace(decode=lambda latent: decoded.append(latent))
        observed = []
        row = {"seed": 42, "prompt": "A dog runs.", "guidance_scale": 2.}
        video, timing, trace = backend.generate(
            row, decode=False, observer=lambda *args: observed.append(args))
        self.assertIsNone(video)
        self.assertEqual(decoded, [])
        self.assertEqual(len(observed), 2)
        self.assertEqual(len(self.scheduler.calls), 2)
        self.assertEqual(trace, [])
        self.assertEqual(set(timing), {"generation_seconds", "text_encode_seconds",
                                      "denoise_seconds", "vae_decode_seconds"})


if __name__ == "__main__":
    unittest.main()
