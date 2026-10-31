"""Independent review checks: provenance guards and numerical conventions.

These use tiny synthetic arrays and scalar policies. They are not video-model
experiments and establish no GPU speed or perceptual-quality result.
"""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cachebench.aggregate import policy, summarize
from cachebench.evaluate import evaluate_manifest, fidelity_metrics, stage_vbench
from cachebench.manifest import build_manifest
from cachebench.policies import validate_row, config_signature


def fixture_rows(seeds=None):
    generation = dict(model_id="Wan2.1-T2V-1.3B", checkpoint_revision="local-content-hash",
                      height=16, width=16, num_frames=5, num_inference_steps=5,
                      guidance_scale=5.0, flow_shift=5.0, sample_solver="unipc",
                      dtype="bfloat16", negative_prompt="", fps=16,
                      attention_backend="flash_attention_2", offload=False)
    cfg = dict(generation=generation, seeds=[0] if seeds is None else seeds, default_split="tune",
               methods=[{"method": "baseline"}, {"method": "seacache", "parameters": {"threshold": .2}}])
    return build_manifest([dict(prompt_id="dog-one", prompt="A dog runs.", dimensions=[], split="tune")], cfg)


def write_finished_fixture(root, rows):
    root = Path(root)
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for row in rows:
        directory = root / row["run_id"]
        directory.mkdir()
        value = 0 if row["method"] == "baseline" else 10
        np.save(directory / "frames.npy", np.full((5, 16, 16, 3), value, dtype=np.uint8))
        metadata = dict(config=row, status="complete", generation_seconds=2 if value == 0 else 1,
                        checkpoint_fingerprint="sha256-weights-A", gpu_name="same-test-device",
                        source_revisions={"wan": {"commit": "same-native-code"}},
                        actual_source_revisions={"wan": {"commit": "same-native-code"}},
                        environment={"torch": "test", "cuda": "test", "hardware": "same-test-device"},
                        versions={"torch": "test"}, cuda_version="test",
                        backend="native_wan_pinned_official_forwards", refresh_ratio=.5,
                        peak_memory_gb=.001)
        (directory / "metrics.json").write_text(json.dumps(metadata))
    return manifest


class NumericalReview(unittest.TestCase):
    def test_valid_manifest_row_accepted(self):
        self.assertEqual(validate_row(fixture_rows()[1])["method"], "seacache")

    def test_rgb_psnr_uses_uint8_range_without_overflow(self):
        ref = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        candidate = np.full_like(ref, 10)
        out = fidelity_metrics(ref, candidate, include=["psnr"])
        self.assertAlmostEqual(out["psnr_db"], 20 * math.log10(255 / 10), places=10)
        self.assertEqual(out["mse_rgb"], 100)

    def test_identical_video_psnr_is_explicit_infinity(self):
        ref = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        out = fidelity_metrics(ref, ref.copy(), include=["psnr"])
        self.assertIsNone(out["psnr_db"])
        self.assertTrue(out["psnr_db_is_infinite"])

    def test_psnr_protocol_separates_global_and_frame_mean(self):
        ref = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        candidate = np.stack([np.full((16, 16, 3), 1, np.uint8), np.full((16, 16, 3), 10, np.uint8)])
        out = fidelity_metrics(ref, candidate, include=["psnr"])
        self.assertNotAlmostEqual(out["psnr_db"], out["mean_frame_psnr_db"])

    def test_nonfinite_thresholds_rejected(self):
        row = fixture_rows()[1]
        for threshold in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                validate_row({**row, "threshold": threshold})

    def test_scale_mode_is_part_of_policy_identity(self):
        self.assertNotEqual(policy({"method": "sencache", "sencache_scale": "latent_rms"}),
                            policy({"method": "sencache", "sencache_scale": "official_81f"}))

    def test_ablation_variant_is_part_of_policy_identity(self):
        self.assertNotEqual(policy({"method": "seacache", "variant": "official"}),
                            policy({"method": "seacache", "variant": "raw_proxy"}))

    def test_calibration_signature_covers_negative_prompt(self):
        row = fixture_rows()[1]
        self.assertNotEqual(config_signature(row), config_signature({**row, "negative_prompt": "motion blur"}))

    def test_calibration_signature_separates_generation_backend(self):
        row = fixture_rows()[1]
        self.assertNotEqual(config_signature(row), config_signature({**row, "generation_backend": "previous_custom_loop"}))

    def test_unsupported_generation_backend_rejected(self):
        with self.assertRaises(ValueError):
            validate_row({**fixture_rows()[1], "generation_backend": "previous_custom_loop"})


class ProvenanceReview(unittest.TestCase):
    def evaluate_tampered(self, field, value):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = fixture_rows(); manifest = write_finished_fixture(root, rows)
            file = root / rows[1]["run_id"] / "metrics.json"
            data = json.loads(file.read_text())
            if field.startswith("config."):
                data["config"][field.split(".", 1)[1]] = value
            else:
                data[field] = value
            file.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                evaluate_manifest(manifest, root, root / "evaluated.jsonl", include=["psnr"])

    def test_checkpoint_mismatch_rejected(self):
        self.evaluate_tampered("checkpoint_fingerprint", "sha256-other-weights")

    def test_gpu_mismatch_rejected(self):
        self.evaluate_tampered("gpu_name", "different-test-device")

    def test_policy_metadata_mismatch_rejected(self):
        self.evaluate_tampered("config.threshold", .9)

    def test_method_metadata_mismatch_rejected(self):
        self.evaluate_tampered("config.method", "baseline")

    def test_prompt_metadata_mismatch_rejected(self):
        self.evaluate_tampered("config.prompt", "A different dog sits.")

    def test_unmodified_pair_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = fixture_rows(); manifest = write_finished_fixture(root, rows)
            out = evaluate_manifest(manifest, root, root / "evaluated.jsonl", include=["psnr"])
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["speedup"], 2)

    def test_fidelity_default_excludes_extra_flickering_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = fixture_rows(list(range(6))); manifest = write_finished_fixture(root, rows)
            out = evaluate_manifest(manifest, root, root / "evaluated.jsonl", include=["psnr"])
            self.assertEqual(len(out), 5)
            self.assertEqual({r["config"]["sample_index"] for r in out}, set(range(5)))

    def test_vbench_staging_rejects_wrong_actual_policy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = fixture_rows(); manifest = write_finished_fixture(root, rows)
            file = root / rows[1]["run_id"] / "metrics.json"
            data = json.loads(file.read_text()); data["config"]["threshold"] = .9
            file.write_text(json.dumps(data))
            official = root / "official.json"
            official.write_text(json.dumps([{"prompt_en": "A dog runs.", "dimension": []}]))
            with self.assertRaises(ValueError):
                stage_vbench(manifest, root, root / "stage", official, "seacache", samples=1)


class AggregationReview(unittest.TestCase):
    def record(self, method="seacache", threshold=.2, pair="pair-one", split="tune", gpu="same-test-device"):
        config = {**fixture_rows()[1], "method": method, "threshold": threshold,
                  "pair_id": pair, "split": split}
        return dict(config=config, status="evaluated", gpu_name=gpu, generation_seconds=1,
                    reference_generation_seconds=2, speedup=2, lpips=.1, ssim=.8,
                    psnr_db=25, refresh_ratio=.5, peak_memory_gb=1)

    def test_hardware_must_match_across_policies(self):
        records = [self.record(), self.record(method="teacache", gpu="other-device")]
        with self.assertRaises(ValueError):
            summarize(records, draws=20)

    def test_tune_and_test_cannot_be_pooled(self):
        records = [self.record(pair="one", split="tune"), self.record(pair="two", split="test")]
        with self.assertRaises(ValueError):
            summarize(records, draws=20)


if __name__ == "__main__":
    unittest.main()
