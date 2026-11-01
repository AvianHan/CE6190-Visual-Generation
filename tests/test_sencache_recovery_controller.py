"""Offline tests for the partial Sen calibration recovery; no Slurm/GPU/network."""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecoveryControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_module("sen_controller_existing_queue_tests", REPO / "tests/test_orion_queue.py")
        cls.old.QueueTests.setUpClass()
        cls.r = load_module("sen_recovery_controller_tests", REPO / "scripts/recover_sencache.py")

    def setUp(self):
        self.fixture = self.old.QueueTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.q = self.fixture.q
        self.h = self.fixture.h

    def recovery_fixture(self):
        run, rolling_plan, state, rows, checkout = self.fixture.rolling_fixture(("l40sp", "h200p"))
        self.fixture.passed_parent(run)
        write_json(run / "calibration/magcache/check.json", {"status": "passed", "job_id": "185111"})
        state.update(status="running", controller_job="185225",
                     failed_prerequisites={"parent:sencache": "Slurm CANCELLED exit 0:0; prerequisite has no passed artifact"},
                     unknown_prerequisites={})
        for tasks in state["tasks"].values():
            for task in tasks.values():
                if task["method"] == "sencache":
                    task.update(status="blocked", error=state["failed_prerequisites"]["parent:sencache"])
        baseline = next(row["run_id"] for row in rows if row["method"] == "baseline")
        state["tasks"]["l40sp"][baseline].update(status="submitted", job_id="185228_0", batch="existing receipt")
        write_json(run / "rolling/state.json", state)
        for name, identifier in (("000000.json", "185225"), ("after-185225.json", "185226")):
            write_json(run / "rolling/controllers" / name,
                       {"status": "accepted", "job_id": identifier,
                        "identity": {"kind": "controller"},
                        "command": self.q.controller_command(run, rolling_plan)})

        samples = []
        for index in range(8):
            video = run / "calibration/videos" / f"{index:04d}.webm"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"synthetic source video {index}".encode())
            samples.append({"prompt_id": f"calibration_{index}",
                            "prompt": f"A dog performs calibration action {index}.",
                            "split": "calibration", "seed": 619000 + index,
                            "video_path": str(video), "video_url": f"https://example.invalid/{index}.webm",
                            "video_sha256": digest(video)})
        sample_path = run / "calibration/samples.jsonl"
        sample_path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
        config = json.loads((run / "code/configs/correctness_smoke.json").read_text())
        config_path = run / "calibration/config.json"
        write_json(config_path, config)
        exclusion = run / "code/configs/development_prompts.jsonl"
        exclusions = [{"path": str(exclusion), "sha256": digest(exclusion), "prompts": 64}]
        write_json(run / "calibration/inputs.json",
                   {"samples_sha256": digest(sample_path),
                    "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    "exclusions": exclusions})

        # Import only stdlib portions of the scientific module, exactly like the
        # existing calibration tests. No third-party model dependency is loaded.
        package = types.ModuleType("sen_controller_science")
        package.__path__ = [str(run / "code/src")]
        sys.modules[package.__name__] = package
        self.addCleanup(lambda: [sys.modules.pop(key, None) for key in list(sys.modules)
                                 if key == package.__name__ or key.startswith(package.__name__ + ".")])
        calibrate = load_module(package.__name__ + ".calibrate", run / "code/src/calibrate.py")
        generation = dict(config["generation"])
        self.calibration_data = {}
        for index in range(8):
            values = calibrate._profile_header("sencache", generation, samples, [index], sample_path, exclusions)
            values.update(profile_kind="sample_shard", checkpoint_fingerprint="1" * 64,
                          source_revisions={"wan": "fixed-test-revision"},
                          aggregation="arithmetic mean over independent calibration videos",
                          calibration_code_sha256="2" * 64,
                          geometry_note="original spatial and temporal geometry",
                          estimator_note="official finite difference functions",
                          calibration_wall_seconds=100.0,
                          timesteps=list(range(50)), raw_data=[[[1.0, 2.0] for _ in range(50)]],
                          J_x_norm=[1.0] * 50, J_t_norm=[2.0] * 50,
                          sample_video_sha256={samples[index]["prompt_id"]: samples[index]["video_sha256"]},
                          video_downloads={})
            self.calibration_data[index] = values
            if index not in (0, 2, 7):
                write_json(run / "calibration/sencache" / f"shard-{index:04d}.json", values)
        for script in ("recover_sencache.py", "sencache_recovery_worker.py", "sencache_recovery_assets.py"):
            if (REPO / "scripts" / script).exists():
                shutil.copy2(REPO / "scripts" / script, checkout / "scripts" / script)
        return run, rolling_plan, state, rows, checkout

    def prepared(self):
        run, rolling, state, rows, checkout = self.recovery_fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            plan = self.r.prepare_recovery(run, checkout, self.q)
        return run, plan, rolling, state, rows, checkout

    def paused(self, run, plan, cycle="initial"):
        self.r.begin_pause(run, plan, self.q, cycle)
        state = self.q.read_json(run / "rolling/state.json")
        state.update(status="paused", controller_job="185225",
                     error="FileNotFoundError(2, 'No such file or directory')")
        write_json(run / "rolling/state.json", state)
        (run / "rolling/controller-185225.log").write_text(
            "Traceback (most recent call last):\n" +
            "FileNotFoundError: [Errno 2] No such file or directory: " +
            repr(str(run / "rolling/manifest.jsonl")) + "\n")
        return state


    def write_recovered_shard(self, run, index):
        values = copy.deepcopy(self.calibration_data[index])
        values["recovery_decoder"] = {"decoded_frames": 97, "advertised_frames": 96,
                                      "selection_count": 81, "index_policy": "numpy.linspace original"}
        path = run / "calibration/sencache" / f"shard-{index:04d}.json"
        write_json(path, values)
        return path

    def write_merged_profile(self, run):
        for index in (0, 2, 7):
            self.write_recovered_shard(run, index)
        samples_path = run / "calibration/samples.jsonl"
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
        config = self.q.read_json(run / "calibration/config.json")
        exclusions = self.q.read_json(run / "calibration/inputs.json")["exclusions"]
        calibrate = sys.modules["sen_controller_science.calibrate"]
        paths = [run / "calibration/sencache" / f"shard-{index:04d}.json" for index in range(8)]
        profile = calibrate.merge_profiles(paths, "sencache", config["generation"], samples,
                                           samples_path, exclusions)
        write_json(run / "calibration/sencache/profile.json", profile)
        write_json(run / "sencache_recovery/jobs/finalize.json",
                   {"status": "accepted", "job_id": "191500"})
        write_json(run / "calibration/sencache/check.json", {"status": "passed", "job_id": "191500"})
        return profile

    @contextlib.contextmanager
    def scheduler(self, active=None):
        active = [] if active is None else active
        commands = []
        def execute(command, **kwargs):
            values = list(map(str, command))
            if values[0] != "sbatch":
                raise AssertionError("Unexpected command (including job cancellation): " + repr(values))
            commands.append(values)
            return subprocess.CompletedProcess(values, 0, str(190000 + len(commands)) + "\n", "")
        def accounting(ids):
            return {str(identifier): {"state": "FAILED", "exit_code": "1:0"} for identifier in ids}
        with patch.object(self.q, "query_queue", side_effect=lambda _: list(active)), \
                patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "partition_time", return_value={"time": "3-00:00:00", "seconds": 259200}), \
                patch.object(self.q, "query_accounting", side_effect=accounting), \
                patch.object(self.q.subprocess, "run", side_effect=execute):
            yield active, commands

    @staticmethod
    def jobs(count, start=200000):
        return [{"job_id": str(start + index), "qos": "highpriority", "state": "RUNNING",
                 "name": "other-project", "reason": "None"} for index in range(count)]

    def test_plan_targets_only_missing_shards_and_freezes_five_successful_files(self):
        run, plan, _, _, _, checkout = self.prepared()
        self.assertEqual(plan["targets"], [0, 2, 7])
        self.assertEqual(set(plan["successful_shards"]),
                         {f"shard-{i:04d}.json" for i in (1, 3, 4, 5, 6)})
        originals = {name: (run / "calibration/sencache" / name).read_bytes()
                     for name in plan["successful_shards"]}
        repeated = self.r.prepare_recovery(run, checkout, self.q)
        self.assertEqual(plan, repeated)
        self.assertEqual(originals, {name: (run / "calibration/sencache" / name).read_bytes()
                                     for name in originals})
        changed = run / "calibration/sencache/shard-0001.json"
        changed.write_bytes(changed.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "successful shard changed"):
            self.r.validate_frozen(run, plan, self.q)

    def test_unknown_existing_sencache_submission_blocks_recovery(self):
        run, _, _, _, checkout = self.recovery_fixture()
        write_json(run / "rolling/batches/uncertain/receipt.json",
                   {"status": "submitting", "job_id": None,
                    "identity": {"method": "sencache", "run_ids": ["unknown"]}})
        manifest = (run / "rolling/manifest.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "submission already exists"):
            self.r.prepare_recovery(run, checkout, self.q)
        self.assertEqual((run / "rolling/manifest.jsonl").read_bytes(), manifest)

    def test_maintenance_changes_only_top_level_manifest_until_controllers_exit(self):
        run, plan, _, _, _, _ = self.prepared()
        original = (run / "rolling/manifest.jsonl").read_bytes()
        state = (run / "rolling/state.json").read_bytes()
        batch = run / "rolling/batches/already-running/manifest.jsonl"
        batch.parent.mkdir(parents=True)
        batch.write_bytes(b"immutable GPU task inputs\n")
        first = self.r.begin_pause(run, plan, self.q, "initial")
        self.assertEqual(first["status"], "withdrawn")
        self.assertFalse((run / "rolling/manifest.jsonl").exists())
        backup = run / "sencache_recovery/maintenance/initial/manifest.jsonl"
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual((run / "rolling/state.json").read_bytes(), state)
        self.assertEqual(batch.read_bytes(), b"immutable GPU task inputs\n")
        self.r.begin_pause(run, plan, self.q, "initial")
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual((run / "rolling/state.json").read_bytes(), state)

    def test_pause_requires_terminal_controllers_and_exclusive_lock(self):
        run, plan, _, _, _, _ = self.prepared()
        self.paused(run, plan)
        with self.scheduler(self.jobs(1)) as (active, _):
            active.append({"job_id": "185225"})
            self.assertFalse(self.r.pause_ready(run, self.q, "initial"))
            active.pop()
            with patch.object(self.q, "query_accounting", return_value={}):
                self.assertFalse(self.r.pause_ready(run, self.q, "initial"))
            with (run / "rolling/.controller.lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(self.r.pause_ready(run, self.q, "initial"))
            self.assertTrue(self.r.pause_ready(run, self.q, "initial"))

    def test_maintenance_refuses_tampered_backup_or_unrelated_pause(self):
        run, plan, _, _, _, _ = self.prepared()
        self.paused(run, plan)
        backup = run / "sencache_recovery/maintenance/initial/manifest.jsonl"
        original = backup.read_bytes()
        backup.write_bytes(original + b" ")
        with self.scheduler() as (_, commands):
            with self.assertRaisesRegex(ValueError, "backup changed"):
                self.r.resume_original(run, plan, self.q, "initial")
            self.assertEqual(commands, [])
            backup.write_bytes(original)
            state = self.q.read_json(run / "rolling/state.json")
            state["error"] = "Unrelated source validation failed"
            write_json(run / "rolling/state.json", state)
            with self.assertRaisesRegex(RuntimeError, "expected manifest guard"):
                self.r.resume_original(run, plan, self.q, "initial")
            self.assertEqual(commands, [])

    def test_recovery_submission_respects_all_sixty_slots_and_does_not_repeat_shards(self):
        run, plan, _, _, _, _ = self.prepared()
        with self.scheduler(self.jobs(60)) as (active, commands):
            self.assertFalse(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            self.assertEqual(commands, [])
            del active[57:]
            self.assertFalse(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            self.assertEqual(len(commands), 3)
            for index in (0, 2, 7):
                receipt = self.q.read_json(run / "sencache_recovery/jobs" / f"shard-{index}.json")
                active.append({"job_id": receipt["job_id"]})
                self.assertIn("--sample-index " + str(index), receipt["command"][-1])
            self.assertEqual(len(active), 60)
            self.assertFalse(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            self.assertEqual(len(commands), 3)
            active.pop(0)
            self.assertTrue(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            self.assertEqual(len(commands), 4)
            final = self.q.read_json(run / "sencache_recovery/jobs/finalize.json")
            active.append({"job_id": final["job_id"]})
            self.assertTrue(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            self.assertEqual(len(commands), 4)
            self.assertEqual(len(active), 60)
        for command in commands:
            self.assertEqual(command[command.index("--gres") + 1], "gpu:1")
        self.assertTrue(all((run / "calibration/sencache" / name).is_file()
                            for name in plan["successful_shards"]))

    def test_uncertain_retry_receipt_is_never_silently_resubmitted(self):
        run, plan, _, _, _, _ = self.prepared()
        command = self.r.gpu_command
        with self.scheduler() as (_, commands):
            identity = {"stage": "shard", "index": 0,
                        "worker_sha256": plan["recovery_scripts"]["sencache_recovery_worker.py"]}
            write_json(run / "sencache_recovery/jobs/shard-0.json",
                       {"status": "submitting", "job_id": None,
                        "command": command(run, plan, self.q, 0), "identity": identity})
            with self.assertRaisesRegex(RuntimeError, "needs reconciliation"):
                self.r.submit_gpu_tasks_tick(run, plan, self.q)
            self.assertEqual(commands, [])

    def test_resume_preserves_running_tasks_and_only_unblocks_sencache(self):
        run, plan, _, initial, _, _ = self.prepared()
        state = self.paused(run, plan, "unblock")
        state["failed_prerequisites"]["h200p:magcache"] = "unrelated failure"
        write_json(run / "rolling/state.json", state)
        before = copy.deepcopy(state)
        with self.scheduler() as (active, commands), \
                patch.object(self.r, "validate_recovered_profile", return_value={"status": "passed", "profile_sha256": "a" * 64}):
            result = self.r.resume_original(run, plan, self.q, "unblock", unblock=True)
            self.assertEqual(result["status"], "accepted")
            active.append({"job_id": result["job_id"]})
            repeated = self.r.resume_original(run, plan, self.q, "unblock", unblock=True)
            self.assertEqual(repeated["job_id"], result["job_id"])
            self.assertEqual(len(commands), 1)
        after = self.q.read_json(run / "rolling/state.json")
        self.assertEqual(digest(run / "rolling/manifest.jsonl"), plan["manifest_sha256"])
        self.assertNotIn("parent:sencache", after["failed_prerequisites"])
        self.assertEqual(after["failed_prerequisites"]["h200p:magcache"], "unrelated failure")
        for partition, tasks in after["tasks"].items():
            for identifier, task in tasks.items():
                if task["method"] == "sencache":
                    self.assertEqual(task["status"], "unsubmitted")
                    self.assertNotIn("error", task)
                else:
                    self.assertEqual(task, before["tasks"][partition][identifier])
        self.assertEqual(after["batches"], initial["batches"])

    def test_failed_profile_validation_does_not_prevent_healthy_queue_restore(self):
        run, plan, _, _, _, _ = self.prepared()
        self.paused(run, plan, "unblock")
        with self.scheduler() as (_, commands), \
                patch.object(self.r, "validate_recovered_profile", side_effect=ValueError("No matching passed finalizer")):
            with self.assertRaisesRegex(ValueError, "matching passed finalizer"):
                self.r.resume_original(run, plan, self.q, "unblock", unblock=True)
            result = self.r.resume_original(run, plan, self.q, "unblock", unblock=False)
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(len(commands), 1)
        after = self.q.read_json(run / "rolling/state.json")
        self.assertIn("parent:sencache", after["failed_prerequisites"])
        self.assertTrue(all(task["status"] == "blocked" for tasks in after["tasks"].values()
                            for task in tasks.values() if task["method"] == "sencache"))

    def test_manager_asset_failure_resumes_existing_queue(self):
        run, plan, _, _, _, _ = self.prepared()
        self.paused(run, plan)
        failing = types.SimpleNamespace(prepare_assets=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("HTTP429 retry budget exhausted")))
        with self.scheduler() as (_, commands), \
                patch.object(self.r, "import_queue", return_value=self.q), \
                patch.object(self.r, "import_path", return_value=failing), \
                patch.object(self.r, "resume_until_accepted", wraps=self.r.resume_until_accepted) as resume, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "retry budget exhausted"):
                self.r.run_manager(argparse.Namespace(run_dir=run))
        resume.assert_called_once_with(run, plan, self.q, "initial", unblock=False)
        self.assertEqual(self.r.get_state(run, self.q)["status"], "failed")
        self.assertEqual(len(commands), 1)
        self.assertEqual(digest(run / "rolling/manifest.jsonl"), plan["manifest_sha256"])

    def test_resumed_controller_can_acquire_its_lock_immediately_on_sbatch(self):
        run, plan, _, _, _, _ = self.prepared()
        self.paused(run, plan)
        launches = []
        def start_immediately(command, **kwargs):
            # Slurm may start the job before sbatch returns. The replacement
            # controller must not inherit a still-held maintenance lock.
            with (run / "rolling/.controller.lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                launches.append(command)
            return subprocess.CompletedProcess(command, 0, "191000\n", "")
        with self.scheduler(), patch.object(self.q.subprocess, "run", side_effect=start_immediately):
            result = self.r.resume_original(run, plan, self.q, "initial")
        self.assertEqual(result["job_id"], "191000")
        self.assertEqual(len(launches), 1)

    def test_finalizer_depends_only_on_unfinished_recovery_jobs(self):
        run, plan, _, _, _, _ = self.prepared()
        write_json(run / "sencache_recovery/jobs/shard-0.json",
                   {"status": "accepted", "job_id": "180000"})
        self.write_recovered_shard(run, 0)
        write_json(run / "sencache_recovery/jobs/shard-2.json",
                   {"status": "accepted", "job_id": "180002"})
        with self.scheduler([{"job_id": "180002"}]) as (active, commands), \
                patch.object(self.q, "query_accounting", return_value={
                    "180000": {"state": "COMPLETED", "exit_code": "0:0"}}):
            self.assertFalse(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            seventh = self.q.read_json(run / "sencache_recovery/jobs/shard-7.json")["job_id"]
            active.append({"job_id": seventh})
            self.assertTrue(self.r.submit_gpu_tasks_tick(run, plan, self.q))
        self.assertEqual(len(commands), 2)
        seventh = self.q.read_json(run / "sencache_recovery/jobs/shard-7.json")["job_id"]
        final = self.q.read_json(run / "sencache_recovery/jobs/finalize.json")["command"]
        dependency = final[final.index("--dependency") + 1]
        self.assertEqual(set(dependency.split(":")[1:]), {"180002", seventh})
        self.assertNotIn("180000", dependency)

    def test_unresolved_accepted_shard_does_not_create_blind_dependency(self):
        run, plan, _, _, _, _ = self.prepared()
        write_json(run / "sencache_recovery/jobs/shard-0.json",
                   {"status": "accepted", "job_id": "180000"})
        with self.scheduler() as (_, commands), \
                patch.object(self.q, "query_accounting", return_value={}):
            result = self.r.submit_gpu_tasks_tick(run, plan, self.q)
        self.assertFalse(result)
        self.assertFalse((run / "sencache_recovery/jobs/finalize.json").exists())
        self.assertFalse(any("ce6190-sen-recover-finalize" in command for command in commands))


    def test_real_merged_profile_requires_all_shard_hashes_and_matching_passed_job(self):
        run, plan, _, _, _, _ = self.prepared()
        profile = self.write_merged_profile(run)
        self.assertEqual(len(profile["shards"]), 8)
        audit = self.r.validate_recovered_profile(run, plan, self.q)
        self.assertEqual(audit["original_successful_shards"], plan["successful_shards"])
        self.assertEqual(set(audit["recovered_shards"]), {"0", "2", "7"})
        self.assertEqual(audit["profile_sha256"], digest(run / "calibration/sencache/profile.json"))
        write_json(run / "calibration/sencache/check.json", {"status": "passed", "job_id": "185113"})
        with self.assertRaisesRegex(ValueError, "matching passed"):
            self.r.validate_recovered_profile(run, plan, self.q)
        write_json(run / "calibration/sencache/check.json", {"status": "passed", "job_id": "191500"})
        shard = run / "calibration/sencache/shard-0000.json"
        shard.write_bytes(shard.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Merged profile shard changed"):
            self.r.validate_recovered_profile(run, plan, self.q)

    def test_completed_retry_without_valid_artifact_cannot_submit_finalizer(self):
        run, plan, _, _, _, _ = self.prepared()
        jobs = {}
        for index in (0, 2, 7):
            identifier = str(180000 + index)
            write_json(run / "sencache_recovery/jobs" / f"shard-{index}.json",
                       {"status": "accepted", "job_id": identifier})
            jobs[identifier] = {"state": "COMPLETED", "exit_code": "0:0"}
        with self.scheduler() as (_, commands), patch.object(self.q, "query_accounting", return_value=jobs):
            with self.assertRaisesRegex(RuntimeError, "no valid recovered shard"):
                self.r.submit_gpu_tasks_tick(run, plan, self.q)
            self.assertEqual(commands, [])

    def test_quota_rejected_finalizer_can_retry_after_dependencies_finish(self):
        run, plan, _, _, _, _ = self.prepared()
        identifiers = [str(180000 + index) for index in (0, 2, 7)]
        for index, identifier in zip((0, 2, 7), identifiers):
            write_json(run / "sencache_recovery/jobs" / f"shard-{index}.json",
                       {"status": "accepted", "job_id": identifier})
        attempts = []
        def quota_then_accept(command, **kwargs):
            attempts.append(list(command))
            if len(attempts) == 1:
                return subprocess.CompletedProcess(command, 1, "", "QOSMaxSubmitJobPerUserLimit")
            return subprocess.CompletedProcess(command, 0, "191600\n", "")
        with self.scheduler([{"job_id": identifier} for identifier in identifiers]) as (active, _), \
                patch.object(self.q.subprocess, "run", side_effect=quota_then_accept), \
                patch.object(self.q, "query_accounting", return_value={
                    identifier: {"state": "COMPLETED", "exit_code": "0:0"} for identifier in identifiers}):
            self.assertFalse(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            self.assertEqual(self.q.read_json(run / "sencache_recovery/jobs/finalize.json")["status"], "quota_wait")
            active.clear()
            for index in (0, 2, 7):
                self.write_recovered_shard(run, index)
            self.assertTrue(self.r.submit_gpu_tasks_tick(run, plan, self.q))
            self.assertTrue(self.r.submit_gpu_tasks_tick(run, plan, self.q))
        self.assertEqual(len(attempts), 2)
        self.assertIn("--dependency", attempts[0])
        self.assertNotIn("--dependency", attempts[1])
        archives = list((run / "sencache_recovery/jobs/quota_rejections").glob("*.json"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(self.q.read_json(archives[0])["status"], "quota_wait")
        self.assertEqual(self.q.read_json(run / "sencache_recovery/jobs/finalize.json")["job_id"], "191600")

    def test_start_reentry_preserves_durable_accepted_manager_without_duplicate(self):
        run, plan, _, _, _, checkout = self.prepared()
        args = argparse.Namespace(run_dir=run, submit=True)
        def pause(*values):
            self.paused(run, plan)
        with self.scheduler() as (active, commands), \
                patch.object(self.r, "__file__", str(checkout / "scripts/recover_sencache.py")), \
                patch.object(self.r, "import_queue", return_value=self.q), \
                patch.object(self.r, "wait_for_pause", side_effect=pause) as wait, \
                contextlib.redirect_stdout(io.StringIO()):
            self.r.start(args)
            receipt = run / "sencache_recovery/manager-job.json"
            original = receipt.read_bytes()
            identifier = self.q.read_json(receipt)["job_id"]
            self.assertEqual(self.q.read_json(receipt)["status"], "accepted")
            active.append({"job_id": identifier})
            self.r.start(args)
            active.clear()
            self.r.start(args)  # Even missing/terminal accounting cannot imply safe resubmit.
            self.assertEqual(len(commands), 1)
            self.assertEqual(receipt.read_bytes(), original)
            wait.assert_called_once()

    def test_start_refuses_uncertain_manager_outcome(self):
        run, plan, _, _, _, checkout = self.prepared()
        args = argparse.Namespace(run_dir=run, submit=True)
        with self.scheduler() as (_, commands), \
                patch.object(self.r, "__file__", str(checkout / "scripts/recover_sencache.py")), \
                patch.object(self.r, "import_queue", return_value=self.q), \
                patch.object(self.r, "wait_for_pause", side_effect=lambda *args: self.paused(run, plan)):
            command = self.r.manager_command(run, plan, self.q)
            write_json(run / "sencache_recovery/manager-job.json",
                       {"command": command, "status": "uncertain", "job_id": None,
                        "identity": {"stage": "manager", "recovery_scripts": plan["recovery_scripts"]}})
            with self.assertRaisesRegex(RuntimeError, "needs reconciliation"):
                self.r.start(args)
            self.assertEqual(commands, [])

    def test_resumed_maintenance_does_not_wait_on_the_replacement_controller(self):
        run, plan, _, _, _, _ = self.prepared()
        self.paused(run, plan, "calibration")
        with self.scheduler() as (active, _):
            result = self.r.resume_original(run, plan, self.q, "calibration")
            active.append({"job_id": result["job_id"]})
            with patch.object(self.r, "pause_ready", side_effect=AssertionError("Cannot wait on resumed controller")):
                self.r.wait_for_pause(run, plan, self.q, "calibration")
        self.assertTrue((run / "rolling/manifest.jsonl").is_file())

    def test_pause_traceback_must_identify_the_withdrawn_manifest(self):
        run, plan, _, _, _, _ = self.prepared()
        state = self.paused(run, plan)
        log = run / "rolling/controller-185225.log"
        log.write_text("FileNotFoundError: [Errno 2] No such file or directory: '/unrelated/file.json'\n")
        self.assertFalse(self.r.owned_manifest_pause(run, state))
        with self.scheduler() as (_, commands):
            with self.assertRaisesRegex(RuntimeError, "expected manifest guard"):
                self.r.resume_original(run, plan, self.q, "initial")
            self.assertEqual(commands, [])
    def test_manager_scheduler_outage_after_calibration_resume_reenters_without_duplicates(self):
        run, plan, _, _, _, _ = self.prepared()
        self.paused(run, plan)
        self.write_merged_profile(run)
        for index in (0, 2, 7):
            write_json(run / "sencache_recovery/jobs" / f"shard-{index}.json",
                       {"status": "accepted", "job_id": str(180000 + index)})
        receipts = {path: path.read_bytes() for path in (run / "sencache_recovery/jobs").glob("*.json")}
        result_path = run / "existing-generation-result.mp4"
        result_path.write_bytes(b"existing generation result must remain untouched")
        original_wait = self.r.wait_for_pause
        pauses = []
        def simulated_controller_exit(run_dir, current_plan, q, cycle):
            event = q.read_json(run / "sencache_recovery/maintenance" / cycle / "event.json", {})
            if event.get("status") not in {"restored", "resumed"}:
                pauses.append(cycle)
                self.paused(run_dir, current_plan, cycle)
            original_wait(run_dir, current_plan, q, cycle)
        outages = []
        def queue(qos):
            state = self.r.get_state(run, self.q)
            if state.get("stage") == "calibration_and_parity" and not outages:
                outages.append("temporary Slurm accounting outage")
                raise self.q.SchedulerUnavailable(outages[-1])
            return []
        def accounting(ids):
            completed = {"191500", "180000", "180002", "180007"}
            return {str(identifier): {"state": "COMPLETED" if str(identifier) in completed else "FAILED",
                                      "exit_code": "0:0" if str(identifier) in completed else "1:0"}
                    for identifier in ids}
        assets = types.SimpleNamespace(prepare_assets=lambda *args, **kwargs: None)
        with self.scheduler() as (_, commands), \
                patch.object(self.r, "import_queue", return_value=self.q), \
                patch.object(self.r, "import_path", return_value=assets), \
                patch.object(self.r, "wait_for_pause", side_effect=simulated_controller_exit), \
                patch.object(self.r.time, "sleep"), \
                patch.object(self.q, "query_queue", side_effect=queue), \
                patch.object(self.q, "query_accounting", side_effect=accounting), \
                contextlib.redirect_stdout(io.StringIO()):
            self.r.run_manager(argparse.Namespace(run_dir=run))
        self.assertEqual(len(outages), 1)
        self.assertEqual(pauses, ["calibration", "unblock"])
        self.assertEqual(len(commands), 3)  # One replacement controller for each cycle.
        self.assertTrue(all("--gres" not in command for command in commands))
        self.assertEqual({path: path.read_bytes() for path in receipts}, receipts)
        self.assertEqual(result_path.read_bytes(), b"existing generation result must remain untouched")
        self.assertEqual(self.r.get_state(run, self.q)["status"], "complete")
        state = self.q.read_json(run / "rolling/state.json")
        self.assertNotIn("parent:sencache", state["failed_prerequisites"])
        self.assertEqual(digest(run / "rolling/manifest.jsonl"), plan["manifest_sha256"])
