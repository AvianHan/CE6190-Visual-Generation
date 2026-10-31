"""Manifest/paired statistics tests with toy inputs, not model evaluations."""
import copy
import json
import math
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from cachebench.manifest import load_prompts, assert_disjoint, split_prompts, build_manifest
from cachebench.evaluate import fidelity_metrics
from cachebench.aggregate import pareto_indices, bootstrap_prompt_means, summarize, freeze_at_speedup


def config():
    return {"generation": {"model_id": "test", "num_frames": 5, "height": 16, "width": 16, "num_inference_steps": 5},
            "seeds": [0, 1], "methods": [{"method": "baseline"}, {"method": "seacache", "grid": {"threshold": [.1, .2]}}]}


class ManifestTests(unittest.TestCase):
    def test_vbench_duplicate_prompts_merge_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompts.json"
            p.write_text(json.dumps([{"prompt_en": "A dog runs.", "dimension": ["motion_smoothness"]},
                                     {"prompt_en": "A dog runs.", "dimension": ["subject_consistency"]}]))
            rows = load_prompts(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dimensions"], ["motion_smoothness", "subject_consistency"])

    def test_leakage_case_and_whitespace(self):
        with self.assertRaises(ValueError):
            assert_disjoint([{"prompt": "A dog runs."}], [{"prompt": "a   DOG runs."}])

    def test_split_deterministic_and_prompt_level(self):
        rows = [{"prompt": "A dog runs."}, {"prompt": "A boat floats."}]
        self.assertEqual(split_prompts(rows), split_prompts(rows))

    def test_baseline_pair_ids_complete(self):
        prompts = [{"prompt": "A dog runs.", "prompt_id": "dog", "dimensions": []}]
        rows = build_manifest(prompts, config())
        refs = {r["run_id"] for r in rows if r["method"] == "baseline"}
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["reference_run_id"] in refs for r in rows))

    def test_generation_change_changes_pair_id(self):
        prompt = [{"prompt": "A dog runs.", "prompt_id": "dog", "dimensions": []}]
        a = config(); b = copy.deepcopy(a); b["generation"]["num_frames"] = 9
        self.assertNotEqual(build_manifest(prompt, a)[0]["pair_id"], build_manifest(prompt, b)[0]["pair_id"])

    def test_duplicate_seeds_fail(self):
        c = config(); c["seeds"] = [0, 0]
        with self.assertRaises(ValueError):
            build_manifest([{"prompt": "A dog runs.", "prompt_id": "dog", "dimensions": []}], c)

    def test_generation_backend_separates_task_identity(self):
        prompts = [{"prompt": "A dog runs.", "prompt_id": "dog", "dimensions": []}]
        before = config()
        after = copy.deepcopy(before)
        after["generation"]["generation_backend"] = "seacache_wan"
        self.assertNotEqual(build_manifest(prompts, before)[0]["run_id"],
                            build_manifest(prompts, after)[0]["run_id"])


class MetricTests(unittest.TestCase):
    def test_ssim_identical_and_distorted(self):
        try:
            import skimage
        except ImportError:
            self.skipTest("scikit-image unavailable; this is not an SSIM pass")
        rng = np.random.default_rng(7)
        a = rng.integers(40, 210, (2, 24, 24, 3), dtype=np.uint8)
        b = np.clip(a.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        self.assertAlmostEqual(fidelity_metrics(a, a, ["ssim"])["ssim"], 1.0, places=12)
        shifted = fidelity_metrics(a, b, ["ssim"])["ssim"]
        self.assertGreater(shifted, 0); self.assertLess(shifted, 1)

    def test_rgba_rejected(self):
        with self.assertRaises(ValueError):
            fidelity_metrics(np.zeros((2, 16, 16, 4), np.uint8), np.zeros((2, 16, 16, 4), np.uint8), ["psnr"])

    def test_uint8_cast_prevents_wraparound(self):
        a = np.zeros((2, 16, 16, 3), np.uint8); b = np.full_like(a, 255)
        self.assertEqual(fidelity_metrics(a, b, ["psnr"])["psnr_db"], 0)

    def test_shape_mismatch_is_not_resized(self):
        with self.assertRaises(ValueError):
            fidelity_metrics(np.zeros((2, 16, 16, 3), np.uint8), np.zeros((3, 16, 16, 3), np.uint8), ["psnr"])

    def test_wrong_dtype_fails(self):
        with self.assertRaises(ValueError):
            fidelity_metrics(np.zeros((2, 16, 16, 3)), np.zeros((2, 16, 16, 3)), ["psnr"])


class AggregationTests(unittest.TestCase):
    def test_pareto_correct_direction_and_ties(self):
        self.assertEqual(pareto_indices([(1, 3), (2, 2), (3, 4), (1, 3)]), [0, 1, 3])

    def test_bootstrap_clusters_prompts_not_frames(self):
        rows = [{"config": {"prompt_id": "a"}, "score": 0}, {"config": {"prompt_id": "a"}, "score": 2},
                {"config": {"prompt_id": "b"}, "score": 3}]
        out = bootstrap_prompt_means(rows, "score", draws=100)
        self.assertEqual(out["mean"], 2)
        self.assertEqual(out["n_prompts"], 2)

    def test_test_set_cannot_select_thresholds(self):
        with self.assertRaises(ValueError):
            freeze_at_speedup([{"splits": ["test"]}], config(), 2)


class LauncherTests(unittest.TestCase):
    def test_resume_requires_complete_matching_files(self):
        file = Path(__file__).resolve().parents[1] / "scripts" / "run_manifest.py"
        spec = importlib.util.spec_from_file_location("run_manifest_under_test", file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        row = {"run_id": "toy", "prompt": "A dog runs."}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.assertEqual(module.existing_status(root, row), "new")
            p = root / "toy"; p.mkdir()
            (p / "metrics.json").write_text(json.dumps({"config": row, "status": "complete"}))
            self.assertEqual(module.existing_status(root, row), "incomplete")
            # These placeholder files test the launcher file-presence contract only.
            (p / "frames.npy").touch(); (p / "video.mp4").touch()
            self.assertEqual(module.existing_status(root, row), "complete")
            with self.assertRaises(ValueError):
                module.existing_status(root, {**row, "prompt": "Different prompt"})


if __name__ == "__main__":
    unittest.main()
