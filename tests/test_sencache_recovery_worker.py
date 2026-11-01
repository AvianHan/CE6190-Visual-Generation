"""Offline decoder/recovery tests; no GPU, network or third-party imports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sencache_recovery_worker_tests", REPO / "scripts/sencache_recovery_worker.py")
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class FakeReader:
    def __init__(self, frames, reported):
        self.frames = frames
        self.reported = reported
        self.closed = False

    def count_frames(self):
        if isinstance(self.reported, Exception):
            raise self.reported
        return self.reported

    def __iter__(self):
        return iter(self.frames)

    def close(self):
        self.closed = True


class DecoderTests(unittest.TestCase):
    def test_actual_stream_count_replaces_wrong_metadata_for_uniform_selection(self):
        reader = FakeReader(range(7), 5)
        observed = []
        def choose(total, wanted):
            observed.append((total, wanted))
            return [0, 3, 6]
        result = WORKER.decoded_plan(Path("dog.webm"), 3, lambda path: reader, choose)
        self.assertEqual(result["metadata_count_frames"], 5)
        self.assertEqual(result["actual_decoded_frames"], 7)
        self.assertEqual(observed, [(7, 3)])
        self.assertEqual(list(WORKER.iter_selected_frames(range(7), 7, result["selected_frame_indices"])),
                         [(0, 0), (3, 3), (6, 6)])
        self.assertTrue(reader.closed)

    def test_correct_metadata_preserves_original_indices_and_frame_values(self):
        reader = FakeReader(["a", "b", "c", "d", "e"], 5)
        details = WORKER.decoded_plan("video.webm", 4, lambda path: reader,
                                      lambda total, wanted: [0, 1, 2, 4])
        result = list(WORKER.iter_selected_frames(reader.frames, 5, details["selected_frame_indices"]))
        self.assertEqual(result, [(0, "a"), (1, "b"), (2, "c"), (4, "e")])

    def test_metadata_failure_does_not_replace_a_valid_actual_stream(self):
        reader = FakeReader(range(2), RuntimeError("metadata unavailable"))
        details = WORKER.decoded_plan("video.webm", 3, lambda path: reader,
                                      lambda total, wanted: [0, 0, 1])
        self.assertIsNone(details["metadata_count_frames"])
        self.assertIn("metadata unavailable", details["metadata_count_error"])
        self.assertEqual(details["selected_frame_indices"], [0, 0, 1])
        self.assertEqual(list(WORKER.iter_selected_frames(range(2), 2, [0, 0, 1])), [(0, 0), (1, 1)])

    def test_second_pass_count_change_is_not_silently_truncated_or_zero_filled(self):
        with self.assertRaisesRegex(ValueError, "more frames"):
            list(WORKER.iter_selected_frames(range(6), 5, [0, 4]))
        with self.assertRaisesRegex(ValueError, "expected 5"):
            list(WORKER.iter_selected_frames(range(4), 5, [0, 4]))

    def test_empty_or_failed_decode_closes_reader_and_rejects_plan(self):
        reader = FakeReader([], 10)
        with self.assertRaisesRegex(ValueError, "no frames"):
            WORKER.decoded_plan("bad.webm", 3, lambda path: reader, lambda *_: [0, 0, 0])
        self.assertTrue(reader.closed)
        def failed():
            yield 0
            raise RuntimeError("decoder failed")
        reader = FakeReader(failed(), 5)
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            WORKER.decoded_plan("bad.webm", 3, lambda path: reader, lambda *_: [0, 0, 0])
        self.assertTrue(reader.closed)

    def test_first_pass_does_not_accumulate_frames(self):
        live = [0]
        peak = [0]
        class Frame:
            def __init__(self):
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            def __del__(self):
                live[0] -= 1
        reader = FakeReader((Frame() for _ in range(300)), 299)
        result = WORKER.count_actual_frames(reader)
        self.assertEqual(result["actual_decoded_frames"], 300)
        self.assertLessEqual(peak[0], 2)
        self.assertEqual(live[0], 0)


class RecoveryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sen worker ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "run"
        self.calibration = self.run / "calibration"
        self.calibration.mkdir(parents=True)
        self.video = self.calibration / "dog.webm"
        self.video.write_bytes(b"pinned video source fixture")
        self.hash = WORKER.file_hash(self.video)
        self.sample = {"prompt_id": "dog", "prompt": "A dog swims.", "split": "calibration",
                       "seed": 0, "video_path": str(self.video), "video_sha256": self.hash}
        samples = json.dumps(self.sample) + "\n"
        (self.calibration / "samples.jsonl").write_text(samples)
        config = {"generation": {"num_frames": 81, "height": 480, "width": 832}}
        (self.calibration / "config.json").write_text(json.dumps(config))
        self.excluded = self.run / "heldout.jsonl"
        self.excluded.write_text('{"prompt_id":"other","prompt":"A boat sails."}\n')
        self.inputs = {"samples_sha256": hashlib.sha256(samples.encode()).hexdigest(),
                       "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                       "exclusions": [{"path": str(self.excluded), "sha256": WORKER.file_hash(self.excluded)}]}
        (self.calibration / "inputs.json").write_text(json.dumps(self.inputs))
        (self.run / "plan.json").write_text(json.dumps({"checkpoint": str(self.root / "model")}))
        self.assets_path = self.run / "sencache_recovery/assets.json"
        self.assets_path.parent.mkdir()
        self.assets = {"status": "passed", "manifest_sha256": self.inputs["samples_sha256"],
                       "indices": [0], "records": {"0": {"sample_id": "dog", "path": str(self.video),
                                                         "sha256": self.hash}}}
        self.assets_path.write_text(json.dumps(self.assets))
        self.output = self.calibration / "sencache/shard-0000.json"
        self.audit = self.root / "attempt/audit.json"

    def args(self):
        return argparse.Namespace(run_dir=self.run, sample_index=0, output=self.output, audit_output=self.audit)

    def test_original_frozen_selection_and_command_are_preserved(self):
        argv, identity = WORKER.original_arguments(self.run, 0, self.output)
        self.assertEqual(argv[argv.index("--sample-index") + 1], "0")
        self.assertEqual(argv[argv.index("--samples") + 1], str(self.calibration / "samples.jsonl"))
        self.assertEqual(argv[argv.index("--exclude-prompts") + 1], str(self.excluded))
        self.assertEqual(identity["source_sha256"], self.hash)
        self.assertEqual(identity["samples_sha256"], self.inputs["samples_sha256"])
        self.assertEqual(identity["assets_audit_sha256"], WORKER.file_hash(self.assets_path))

    def test_unpinned_video_must_still_match_cpu_staging_observed_hash(self):
        self.sample["video_sha256"] = None
        content = json.dumps(self.sample) + "\n"
        (self.calibration / "samples.jsonl").write_text(content)
        digest = hashlib.sha256(content.encode()).hexdigest()
        self.inputs["samples_sha256"] = digest
        (self.calibration / "inputs.json").write_text(json.dumps(self.inputs))
        self.assets["manifest_sha256"] = digest
        self.assets_path.write_text(json.dumps(self.assets))
        WORKER.original_arguments(self.run, 0, self.output)
        self.video.write_bytes(b"source changed after CPU staging")
        with self.assertRaisesRegex(ValueError, "passed asset audit"):
            WORKER.original_arguments(self.run, 0, self.output)

    def test_gpu_work_requires_passed_matching_asset_preparation(self):
        self.assets["status"] = "retry_wait"
        self.assets_path.write_text(json.dumps(self.assets))
        with self.assertRaisesRegex(ValueError, "has not passed"):
            WORKER.original_arguments(self.run, 0, self.output)
        self.assets["status"] = "passed"
        self.assets["manifest_sha256"] = "0" * 64
        self.assets_path.write_text(json.dumps(self.assets))
        with self.assertRaisesRegex(ValueError, "frozen samples"):
            WORKER.original_arguments(self.run, 0, self.output)

    def test_no_gpu_download_fallback_and_no_overwrite_of_successful_shard(self):
        record = WORKER.local_video_record(self.sample)
        self.assertEqual(record["sha256"], self.hash)
        self.assertEqual(record["checksum_status"], "matches_expected")
        self.video.unlink()
        with self.assertRaisesRegex(ValueError, "staged before GPU"):
            WORKER.local_video_record(self.sample)
        with self.assertRaisesRegex(ValueError, "downloaded before"):
            WORKER.original_arguments(self.run, 0, self.output)
        self.video.write_bytes(b"pinned video source fixture")
        self.output.parent.mkdir()
        self.output.write_text('{"successful":"previous"}')
        with self.assertRaisesRegex(FileExistsError, "not be overwritten"):
            WORKER.original_arguments(self.run, 0, self.output)
        self.assertEqual(self.output.read_text(), '{"successful":"previous"}')

    def test_changed_inputs_or_video_or_exclusions_are_rejected(self):
        self.video.write_bytes(b"wrong source")
        with self.assertRaisesRegex(ValueError, "pinned checksum"):
            WORKER.original_arguments(self.run, 0, self.output)
        self.video.write_bytes(b"pinned video source fixture")
        self.excluded.write_text("changed held-out prompts")
        with self.assertRaisesRegex(ValueError, "Held-out"):
            WORKER.original_arguments(self.run, 0, self.output)
        (self.calibration / "samples.jsonl").write_text("changed samples")
        with self.assertRaisesRegex(ValueError, "samples changed"):
            WORKER.original_arguments(self.run, 0, self.output)

    def fake_module(self, fail=False):
        source = self.run / "code/src/calibrate.py"
        source.parent.mkdir(parents=True)
        source.write_text("# immutable original runtime fixture\n")
        module = types.ModuleType("cachebench.calibrate")
        module.__file__ = str(source)
        module.load_video = lambda *args: "original-loader"
        module.fetch_calibration_video = lambda *args: "original-fetcher"
        def writer(path, data):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                json.dump(data, handle)
        module.write_profile_atomic = writer
        def main():
            module.load_video(self.video, {}, "cuda")
            if fail:
                raise RuntimeError("GPU computation failed")
            module.write_profile_atomic(self.output, {"calibration_code_sha256": "a" * 64,
                  "aggregation": "unchanged original aggregation", "geometry_note": "unchanged geometry"})
        module.main = main
        package = types.ModuleType("cachebench")
        package.calibrate = module
        return package, module

    def fake_loader_factory(self, old, audits):
        def loader(path, row, device):
            audits.append({"source_sha256": self.hash, "actual_decoded_frames": 7,
                           "metadata_count_frames": 5, "selected_frame_indices": [0, 3, 6]})
            return "recovered-loader"
        return loader

    def test_publication_keeps_estimator_provenance_and_adds_decoder_audit(self):
        package, module = self.fake_module()
        old_loader, old_writer, old_argv = module.load_video, module.write_profile_atomic, sys.argv
        old_fetch = module.fetch_calibration_video
        with patch.dict(sys.modules, {"cachebench": package, "cachebench.calibrate": module}), \
                patch.object(WORKER, "actual_stream_loader", self.fake_loader_factory):
            WORKER.run_shard(self.args())
        shard = json.loads(self.output.read_text())
        audit = json.loads(self.audit.read_text())
        self.assertEqual(shard["calibration_code_sha256"], "a" * 64)
        self.assertEqual(shard["aggregation"], "unchanged original aggregation")
        self.assertEqual(shard["geometry_note"], "unchanged geometry")
        self.assertEqual(shard["recovery_decoder"], audit["recovery_decoder"])
        self.assertEqual(shard["recovery_decoder"]["decoder"]["selected_frame_indices"], [0, 3, 6])
        self.assertEqual(shard["recovery_decoder"]["worker_sha256"], WORKER.file_hash(WORKER.__file__))
        self.assertIs(module.load_video, old_loader)
        self.assertIs(module.write_profile_atomic, old_writer)
        self.assertIs(module.fetch_calibration_video, old_fetch)
        self.assertIs(sys.argv, old_argv)

    def test_failed_calibration_never_publishes_shard_and_restores_process_state(self):
        package, module = self.fake_module(fail=True)
        old_loader, old_writer, old_argv = module.load_video, module.write_profile_atomic, sys.argv
        old_fetch = module.fetch_calibration_video
        with patch.dict(sys.modules, {"cachebench": package, "cachebench.calibrate": module}), \
                patch.object(WORKER, "actual_stream_loader", self.fake_loader_factory):
            with self.assertRaisesRegex(RuntimeError, "GPU computation failed"):
                WORKER.run_shard(self.args())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.audit.exists())
        self.assertIs(module.load_video, old_loader)
        self.assertIs(module.write_profile_atomic, old_writer)
        self.assertIs(module.fetch_calibration_video, old_fetch)
        self.assertIs(sys.argv, old_argv)


if __name__ == "__main__":
    unittest.main()
