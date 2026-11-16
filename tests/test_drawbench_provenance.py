"""Resume/provenance tests using only the standard library; no model imports."""
import importlib.util
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "drawbench_provenance.py"
SPEC = importlib.util.spec_from_file_location("drawbench_provenance", MODULE_PATH)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def png_bytes(rgb=b"\x01\x02\x03"):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + rgb)) + chunk(b"IEND", b""))


class DrawBenchProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stem = self.root / "sencache_2" / "db_053"
        self.stem.parent.mkdir()
        self.sample = {"prompt_id": "db_053", "prompt": "Three cats.", "category": "Counting"}
        self.settings = {"model": "example/flux", "revision": "frozen-revision", "width": 1, "height": 1,
                         "seed": 0, "method": "sencache", "threshold": 2.0, "steps": 28,
                         "guidance": 3.5, "sen_profile_sha256": "calibration-hash", "sen_k": 3}
        self.environment = {"gpu": "test GPU", "scientific_configs": {"sencache_2": self.settings}}

    def make_result(self):
        invocation = provenance.write_invocation(self.root, self.environment)
        self.stem.with_suffix(".png").write_bytes(png_bytes())
        record = dict(self.sample, config_id="sencache_2", method="sencache", threshold=2.0,
                      seed=0, gpu="test GPU", generation_s=2.6)
        provenance.write_result_metadata(self.stem, record, self.settings, invocation)
        return invocation

    def check(self, **changes):
        options = dict(output_root=self.root, scientific_config=self.settings, sample=self.sample,
                       resume=True, gpu="test GPU")
        options.update(changes)
        return provenance.check_existing_result(self.stem, **options)

    def snapshot(self):
        return {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_each_invocation_is_unique_and_legacy_environment_is_untouched(self):
        old = self.root / "env_shard0of1.json"
        old.write_text('{"original":true}')
        first = provenance.write_invocation(self.root, self.environment)
        original = (self.root / first["invocation_file"]).read_bytes()
        second = provenance.write_invocation(self.root, self.environment)
        self.assertNotEqual(first["invocation_file"], second["invocation_file"])
        self.assertEqual((self.root / first["invocation_file"]).read_bytes(), original)
        self.assertEqual(old.read_text(), '{"original":true}')

    def test_complete_matching_result_is_skipped_without_modification(self):
        self.make_result()
        original = self.snapshot()
        self.assertTrue(self.check())
        self.assertEqual(self.snapshot(), original)

    def test_new_result_is_not_skipped_but_existing_result_requires_resume(self):
        self.assertFalse(self.check(resume=False))
        self.make_result()
        original = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Use --resume"):
            self.check(resume=False)
        self.assertEqual(self.snapshot(), original)

    def test_scientific_settings_and_sample_must_match(self):
        self.make_result()
        original = self.snapshot()
        for key, value in (("seed", 1), ("revision", "other-revision"), ("threshold", 1.0),
                           ("steps", 50), ("guidance", 4.0), ("sen_profile_sha256", "other-profile")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Scientific configuration differs"):
                self.check(scientific_config=dict(self.settings, **{key: value}))
        with self.assertRaisesRegex(ValueError, "Sample prompt differs"):
            self.check(sample=dict(self.sample, prompt="A different prompt."))
        with self.assertRaisesRegex(ValueError, "GPU type differs"):
            self.check(gpu="another GPU")
        self.assertEqual(self.snapshot(), original)

    def test_missing_png_and_legacy_metadata_are_rejected_unchanged(self):
        self.stem.with_suffix(".json").write_text(json.dumps(self.sample))
        original = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Both PNG and JSON"):
            self.check()
        self.assertEqual(self.snapshot(), original)
        self.stem.with_suffix(".png").write_bytes(png_bytes())
        original = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Legacy metadata"):
            self.check()
        self.assertEqual(self.snapshot(), original)

    def test_changed_or_truncated_png_is_rejected(self):
        self.make_result()
        png = self.stem.with_suffix(".png")
        png.write_bytes(png_bytes(b"\xff\x00\x00"))
        with self.assertRaisesRegex(ValueError, "PNG checksum differs"):
            self.check()
        png.write_bytes(png_bytes()[:-12])
        with self.assertRaisesRegex(ValueError, "Incomplete PNG"):
            self.check()

    def test_invocation_tampering_or_missing_record_is_rejected(self):
        invocation = self.make_result()
        path = self.root / invocation["invocation_file"]
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Invocation record checksum differs"):
            self.check()
        path.unlink()
        with self.assertRaises(ValueError):
            self.check()

    def test_invocation_reference_cannot_escape_output_root(self):
        self.make_result()
        path = self.stem.with_suffix(".json")
        record = json.loads(path.read_text())
        record["provenance"]["invocation_file"] = "../outside.json"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "outside this output directory"):
            self.check()

    def test_metadata_writer_never_overwrites_existing_result(self):
        invocation = self.make_result()
        original = self.stem.with_suffix(".json").read_bytes()
        with self.assertRaises(FileExistsError):
            provenance.write_result_metadata(self.stem, {}, self.settings, invocation)
        self.assertEqual(self.stem.with_suffix(".json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
