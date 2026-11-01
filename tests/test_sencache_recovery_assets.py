"""CPU-only asset recovery tests; all HTTP and retry waits are simulated."""
from email.message import Message
from email.utils import formatdate
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sen_assets_tests", REPO / "scripts/sencache_recovery_assets.py")
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)
VIDEO = b"\x1aE\xdf\xa3mock webm source content"


class Response(io.BytesIO):
    def __init__(self, content=VIDEO, kind="video/webm", url="https://example.invalid/original.webm"):
        super().__init__(content)
        self.headers = {"Content-Type": kind, "Content-Length": str(len(content))}
        self.url = url
    def geturl(self): return self.url


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sen assets ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "calibration/videos").mkdir(parents=True)
        self.samples = [{"prompt_id": f"sample-{i}", "split": "calibration", "prompt": f"Dog example {i}",
                         "video_path": str(self.root / "calibration/videos" / f"{i:04d}.webm"),
                         "video_url": f"https://example.invalid/{i}.webm?original=1",
                         "video_sha256": hashlib.sha256(VIDEO).hexdigest() if i == 0 else None}
                        for i in range(8)]
        self.write_samples()
        Path(self.samples[0]["video_path"]).write_bytes(VIDEO)
        self.audit = self.root / "recovery/assets.json"
        self.clock = 1_700_000_000.0
        self.sleeps = []
    def write_samples(self):
        content = "".join(json.dumps(item) + "\n" for item in self.samples).encode()
        (self.root / "calibration/samples.jsonl").write_bytes(content)
        (self.root / "calibration/inputs.json").write_text(json.dumps({"samples_sha256": hashlib.sha256(content).hexdigest()}))
    def sleep(self, seconds):
        self.sleeps.append(seconds); self.clock += seconds
    def prepare(self, opener, **kwargs):
        return ASSETS.prepare_assets(self.root, self.audit, opener=opener,
                                     sleep=self.sleep, now=lambda: self.clock, **kwargs)
    @staticmethod
    def http_error(code, retry=None):
        headers = Message()
        if retry is not None: headers["Retry-After"] = str(retry)
        return HTTPError("https://example.invalid/original.webm", code, "test error", headers, None)

    def test_reuses_pinned_dog_and_downloads_only_original_missing_urls(self):
        calls = []
        def opener(request, timeout):
            calls.append((request.full_url, request.headers, timeout))
            return Response(url=request.full_url)
        original = (self.root / "calibration/samples.jsonl").read_bytes()
        result = self.prepare(opener)
        self.assertEqual(result["status"], "passed")
        self.assertEqual([call[0] for call in calls], [self.samples[i]["video_url"] for i in (2, 7)])
        self.assertTrue(all("github.com/AvianHan/CE6190" in call[1]["User-agent"] for call in calls))
        self.assertEqual(result["records"]["0"]["status"], "existing")
        self.assertEqual(result["records"]["2"]["checksum_status"], "observed_only")
        self.assertEqual((self.root / "calibration/samples.jsonl").read_bytes(), original)
        self.assertEqual(self.sleeps, [])
        self.prepare(lambda *_a, **_kw: self.fail("Existing videos must not redownload"))

    def test_429_honors_retry_after_and_waits_in_short_chunks(self):
        calls = []
        def opener(request, timeout):
            calls.append(request.full_url)
            if len(calls) == 1: raise self.http_error(429, 125)
            return Response(url=request.full_url)
        result = self.prepare(opener, indices=(2,))
        self.assertEqual(calls, [self.samples[2]["video_url"]] * 2)
        self.assertEqual(self.sleeps, [30, 30, 30, 30, 5])
        waits = [e for e in result["events"] if e["status"] == "waiting_retry"]
        self.assertEqual(waits[0]["wait_seconds"], 125)
        self.assertEqual(result["attempts_by_index"]["2"], 2)

    def test_retry_after_date_and_long_server_delay_are_not_capped(self):
        future = formatdate(self.clock + 7200, usegmt=True)
        self.assertEqual(ASSETS.retry_after_seconds(future, self.clock), 7200)
        self.assertEqual(ASSETS.retry_delay(8, {"Retry-After": future}, self.clock), 7200)
        self.assertEqual(ASSETS.retry_delay(8, {}, self.clock), 3600)
        self.assertEqual(ASSETS.retry_delay(1, {"Retry-After": "bad"}, self.clock), 60)

    def test_permanent_403_does_not_loop_or_install_a_file(self):
        calls = []
        def opener(*_a, **_kw):
            calls.append(1); raise self.http_error(403)
        with self.assertRaises(HTTPError): self.prepare(opener, indices=(2,))
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.sleeps, [])
        self.assertFalse(Path(self.samples[2]["video_path"]).exists())
        self.assertEqual(json.loads(self.audit.read_text())["status"], "failed")

    def test_retry_budget_is_bounded_and_remembered_across_restarts(self):
        calls = []
        def opener(*_a, **_kw):
            calls.append(1); raise self.http_error(429)
        with self.assertRaises(HTTPError): self.prepare(opener, indices=(2,), max_attempts=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sum(self.sleeps), 60)
        with self.assertRaisesRegex(RuntimeError, "retry limit exhausted"):
            self.prepare(opener, indices=(2,), max_attempts=2)
        self.assertEqual(len(calls), 2)

    def test_interrupted_retry_wait_is_respected_on_restart(self):
        calls = []
        def opener(*_a, **_kw):
            calls.append(1)
            if len(calls) == 1: raise self.http_error(429, 100)
            return Response()
        with self.assertRaises(KeyboardInterrupt):
            ASSETS.prepare_assets(self.root, self.audit, indices=(2,), opener=opener,
                                  sleep=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()), now=lambda: self.clock)
        self.clock += 20
        result = self.prepare(opener, indices=(2,))
        self.assertEqual(sum(self.sleeps), 80)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(calls), 2)

    def test_wrong_pinned_hash_never_overwrites_existing_or_publishes_download(self):
        Path(self.samples[0]["video_path"]).write_bytes(VIDEO + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare(lambda *_a, **_kw: self.fail("No download"), indices=(0,))
        self.assertEqual(Path(self.samples[0]["video_path"]).read_bytes(), VIDEO + b"changed")
        self.samples[2]["video_sha256"] = "0" * 64; self.write_samples()
        self.audit.unlink()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare(lambda *_a, **_kw: Response(), indices=(2,))
        self.assertFalse(Path(self.samples[2]["video_path"]).exists())
        self.assertFalse(list((self.root / "calibration/videos").glob("*.recovery-download-*")))

    def test_html_json_and_insecure_redirect_are_rejected_before_install(self):
        for response in (lambda: Response(b"<html>error</html>", "text/html"),
                         lambda: Response(b'{"error": 1}', "application/octet-stream"),
                         lambda: Response(url="http://example.invalid/source.webm")):
            with self.subTest(response=response):
                self.audit.unlink(missing_ok=True)
                with self.assertRaises(ValueError):
                    self.prepare(lambda *_a, **_kw: response(), indices=(2,))
                self.assertFalse(Path(self.samples[2]["video_path"]).exists())

    def test_concurrent_publisher_is_verified_without_overwriting(self):
        path = Path(self.samples[2]["video_path"])
        link = ASSETS.os.link
        def publish_first(source, target):
            path.write_bytes(VIDEO)
            return link(source, target)
        with patch.object(ASSETS.os, "link", side_effect=publish_first):
            result = self.prepare(lambda *_a, **_kw: Response(), indices=(2,))
        self.assertEqual(result["records"]["2"]["status"], "existing_concurrent")
        self.assertEqual(path.read_bytes(), VIDEO)

    def test_changed_unpinned_prepared_file_is_not_silently_reused(self):
        self.prepare(lambda *_a, **_kw: Response(), indices=(2,))
        original_audit = self.audit.read_bytes()
        path = Path(self.samples[2]["video_path"]); path.write_bytes(VIDEO + b"changed")
        with self.assertRaisesRegex(ValueError, "Previously prepared source"):
            self.prepare(lambda *_a, **_kw: self.fail("No download"), indices=(2,))
        self.assertEqual(path.read_bytes(), VIDEO + b"changed")
        self.assertEqual(self.audit.read_bytes(), original_audit)

    def test_passed_audit_reentry_is_byte_identical_and_emits_no_new_events(self):
        first = self.prepare(lambda *_a, **_kw: Response())
        original_audit = self.audit.read_bytes()
        original_mtime = self.audit.stat().st_mtime_ns
        self.clock += 500
        second = self.prepare(lambda *_a, **_kw: self.fail("No redownload"),
                              progress_callback=lambda event: self.fail("No provenance rewrite/callback"))
        self.assertEqual(second, first)
        self.assertEqual(self.audit.read_bytes(), original_audit)
        self.assertEqual(self.audit.stat().st_mtime_ns, original_mtime)
        Path(self.samples[7]["video_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Not a regular source video"):
            self.prepare(lambda *_a, **_kw: self.fail("Do not redownload under pinned audit"))
        self.assertEqual(self.audit.read_bytes(), original_audit)

    def test_passed_audit_checks_source_identity_and_keeps_audit_on_rejection(self):
        self.prepare(lambda *_a, **_kw: Response(), indices=(2,))
        passed = json.loads(self.audit.read_text())
        for key, value in (("sample_id", "different"), ("source_url", "https://example.invalid/changed.webm"),
                           ("path", str(self.root / "elsewhere.webm")), ("sha256", "0" * 64),
                           ("expected_sha256", "0" * 64)):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(passed)); changed["records"]["2"][key] = value
                self.audit.write_text(json.dumps(changed))
                tampered = self.audit.read_bytes()
                with self.assertRaises(ValueError):
                    self.prepare(lambda *_a, **_kw: self.fail("No network"), indices=(2,))
                self.assertEqual(self.audit.read_bytes(), tampered)

    def test_manifest_change_and_path_escape_fail_without_network(self):
        source = self.root / "calibration/samples.jsonl"
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            self.prepare(lambda *_a, **_kw: self.fail("No network"))
        self.samples[2]["video_path"] = str(self.root / "outside.webm"); self.write_samples()
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare(lambda *_a, **_kw: self.fail("No network"), indices=(2,))


if __name__ == "__main__": unittest.main()
