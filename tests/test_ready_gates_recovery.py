"""Offline regression tests for stale readiness blocks; no real Slurm/GPU/network."""
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
import shlex
import shutil
import subprocess
import sys
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


class ReadyGateRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sen_tests = load_module("ready_existing_sen_tests", REPO / "tests/test_sencache_recovery_controller.py")
        cls.sen_tests.RecoveryControllerTests.setUpClass()
        cls.ready = load_module("ready_gate_recovery_tests", REPO / "scripts/recover_ready_gates.py")

    def setUp(self):
        self.sen = self.sen_tests.RecoveryControllerTests()
        self.sen.setUp()
        self.addCleanup(self.sen.doCleanups)
        # Adapter changes are process-local. Give each test independent original
        # modules so no patched entry point can leak into another test.
        self.q = load_module("ready_fresh_queue", REPO / "scripts/queue_orion.py")
        self.h = load_module("ready_fresh_hardware", REPO / "scripts/orion_hardware.py")
        self.sen.q = self.sen.fixture.q = self.q
        self.sen.h = self.sen.fixture.h = self.h

    def fixture(self):
        run, sen_plan, rolling, state, rows, checkout = self.sen.prepared()
        shutil.copy2(REPO / "scripts/recover_ready_gates.py", checkout / "scripts/recover_ready_gates.py")
        write_json(run / "sencache_recovery/manager-job.json", {"status": "accepted", "job_id": "185297"})
        write_json(run / "sencache_recovery/state.json", {"status": "running", "stage": "calibration_and_parity", "manager_job": "185297"})
        self.sen.fixture.passed_calibration(run, "magcache")
        self.sen.fixture.passed_hardware(run, "smoke")
        self.sen.fixture.passed_hardware(run, "magcache")
        child = self.h.runtime_path(run, "h200p")
        check = self.h.state_path(run, "h200p", "magcache")
        data = self.q.read_json(check)
        data.update(job_id="185244", phase="hardware_magcache", partition="h200p")
        write_json(check, data)
        write_json(run / "rolling/checks/h200p-magcache.json", {
            "status": "accepted", "job_id": "185244",
            "identity": {"partition": "h200p", "stage": "magcache", "runtime_dir": str(child), "state_file": str(check)},
        })
        reason = f"Unreadable state {check}: [Errno 2] No such file or directory: {str(check)!r}"
        for task in state["tasks"]["h200p"].values():
            if task["method"] == "magcache":
                task.update(status="blocked", error=reason)
        other = [identifier for identifier, task in state["tasks"]["h200p"].items()
                 if task["method"] == "baseline"]
        state["tasks"]["h200p"][other[0]].update(status="complete", job_id="185235_0")
        state["tasks"]["h200p"][other[1]].update(status="submitted", job_id="185310_0")
        write_json(run / "rolling/state.json", state)
        output = child / "development_h200p/results" / other[0] / "video.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"completed experiment output must not change")
        return run, sen_plan, rolling, state, rows, checkout, check

    def prepared(self):
        values = self.fixture()
        run, _, _, _, _, checkout, _ = values
        r = load_module("ready_fresh_sen_maintenance", run / "sencache_recovery/code/recover_sencache.py")
        with contextlib.redirect_stdout(io.StringIO()):
            plan = self.ready.prepare(run, checkout, self.q, r)
        return (*values, plan, r)

    def test_missing_or_temporary_unreadable_phase_waits_but_explicit_failure_survives(self):
        path = self.sen.fixture.root / "phase.json"
        self.assertNotEqual(self.ready.safe_phase_state(path).get("status"), "failed")
        path.write_text('{"status":')
        self.assertNotEqual(self.ready.safe_phase_state(path).get("status"), "failed")
        with patch.object(Path, "read_text", side_effect=FileNotFoundError(2, "No such file or directory", str(path))):
            self.assertNotEqual(self.ready.safe_phase_state(path).get("status"), "failed")
        write_json(path, {"status": "failed", "error": "real GPU parity mismatch"})
        self.assertEqual(self.ready.safe_phase_state(path), {"status": "failed", "error": "real GPU parity mismatch"})

    def test_valid_passed_state_is_preserved_without_rewriting_file(self):
        path = self.sen.fixture.root / "phase.json"
        value = {"status": "passed", "job_id": "185244", "input_identity": {"same": "hash"}}
        write_json(path, value)
        before = path.read_bytes()
        self.assertEqual(self.ready.safe_phase_state(path), value)
        self.assertEqual(path.read_bytes(), before)

    def test_exact_576_stale_blocks_recover_without_touching_other_work_or_results(self):
        run, _, rolling, state, _, _, _, _, _ = self.prepared()
        before = copy.deepcopy(state)
        successful_files = {str(p): p.read_bytes() for p in run.parent.rglob("video.mp4")}
        self.assertEqual(sum(t["method"] == "magcache" and t["status"] == "blocked"
                             for t in state["tasks"]["h200p"].values()), 576)
        self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        restored = []
        for partition, tasks in state["tasks"].items():
            for identifier, task in tasks.items():
                previous = before["tasks"][partition][identifier]
                if task != previous:
                    self.assertEqual((partition, previous["method"], previous["status"]),
                                     ("h200p", "magcache", "blocked"))
                    self.assertEqual(task["status"], "unsubmitted")
                    self.assertNotIn("error", task)
                    restored.append(identifier)
        self.assertEqual(len(restored), 576)
        self.assertEqual(state["failed_prerequisites"], before["failed_prerequisites"])
        self.assertEqual(successful_files, {name: Path(name).read_bytes() for name in successful_files})
        again = copy.deepcopy(state)
        self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        self.assertEqual(state, again)

    def test_changed_profile_or_failed_hardware_check_never_unblocks(self):
        run, _, rolling, state, _, _, check, _, _ = self.prepared()
        before = copy.deepcopy(state)
        original = check.read_bytes()
        data = json.loads(original)
        data["status"] = "failed"
        data["error"] = "full-compute differs from baseline by 7/255"
        write_json(check, data)
        self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        self.assertEqual(state, before)
        check.write_bytes(original)
        profile = run / "calibration/magcache/profile.json"
        profile.write_bytes(profile.read_bytes() + b" ")
        self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        self.assertEqual(state, before)

    def test_unrelated_error_or_true_failed_prerequisite_is_not_reset(self):
        run, _, rolling, state, _, _, _, _, _ = self.prepared()
        candidate = next(t for t in state["tasks"]["h200p"].values() if t["method"] == "magcache")
        candidate["error"] = "Real hardware parity failure"
        before = copy.deepcopy(state)
        self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        self.assertEqual(candidate, next(t for t in before["tasks"]["h200p"].values()
                                         if t["method"] == "magcache"))
        state = copy.deepcopy(before)
        state["failed_prerequisites"]["h200p:magcache"] = "Slurm FAILED exit 1:0"
        before = copy.deepcopy(state)
        self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        self.assertEqual(state, before)

    def test_unreadable_error_must_name_the_current_gate_file(self):
        run, _, rolling, state, _, _, check, _, _ = self.prepared()
        for task in state["tasks"]["h200p"].values():
            if task["method"] == "magcache":
                task["error"] = task["error"].replace(str(check), str(check.with_name("other-check.json")))
        before = copy.deepcopy(state)
        self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        self.assertEqual(state, before)

    def test_check_receipt_must_match_the_passed_hardware_job(self):
        run, _, rolling, state, _, _, _, _, _ = self.prepared()
        receipt = run / "rolling/checks/h200p-magcache.json"
        data = self.q.read_json(receipt)
        data["job_id"] = "999999"
        write_json(receipt, data)
        before = copy.deepcopy(state)
        try:
            self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        except (ValueError, RuntimeError):
            pass
        self.assertEqual(state, before)

    def test_candidate_with_accepted_or_uncertain_batch_is_never_resubmitted(self):
        run, _, rolling, state, _, _, _, _, _ = self.prepared()
        identifier = next(identifier for identifier, task in state["tasks"]["h200p"].items()
                          if task["method"] == "magcache")
        path = run / "rolling/batches/existing-mag/receipt.json"
        for status, job_id in (("accepted", "185333"), ("uncertain", None)):
            with self.subTest(status=status):
                write_json(path, {"status": status, "job_id": job_id,
                                 "identity": {"partition": "h200p", "method": "magcache", "run_ids": [identifier]}})
                before = copy.deepcopy(state)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
                self.assertEqual(state, before)

    def test_adapter_keeps_receipt_reading_strict_and_original_helper_identity(self):
        run, _, _, _, _, _, _, plan, _ = self.prepared()
        original_reader = self.q.read_json
        original_helper_hash = digest(run / "rolling/code/orion_hardware.py")
        self.ready.install_adapter(run, plan, self.q)
        self.assertIs(self.q.read_json, original_reader)
        path = run / "rolling/batches/damaged/receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"status": "accepted",')
        with self.assertRaises(json.JSONDecodeError):
            self.q.read_json(path)
        helper = self.q.hardware_module(run)
        self.assertEqual(Path(helper.__file__).resolve(), (run / "rolling/code/orion_hardware.py").resolve())
        self.assertEqual(digest(helper.__file__), original_helper_hash)
        self.assertEqual(helper.group_gate(run, "h200p", "magcache")["status"], "passed")

    def test_adapter_preserves_policy_and_successor_uses_same_wrapper(self):
        run, _, rolling, _, _, _, _, plan, _ = self.prepared()
        original = self.q.controller_command(run, rolling, "afterany:185306")
        self.ready.install_adapter(run, plan, self.q)
        command = self.q.controller_command(run, rolling, "afterany:185306")
        for flag in ("--partition", "--account", "--qos", "--cpus-per-task", "--mem", "--time", "--dependency"):
            self.assertEqual(command[command.index(flag) + 1], original[original.index(flag) + 1])
        self.assertNotIn("--gres", command)
        shell = shlex.split(command[command.index("--wrap") + 1])
        self.assertEqual(shell[:3], ["exec", "bash", "-c"])
        body = shell[3]
        self.assertIn("/usr/bin/python3", body)
        self.assertIn("recover_ready_gates.py", body)
        self.assertIn("_controller", body)
        self.assertIn(shlex.quote(str(run)), body)

    def test_changed_frozen_controller_source_is_rejected(self):
        run, _, _, _, _, _, _, plan, _ = self.prepared()
        source = run / "rolling/code/orion_hardware.py"
        source.write_bytes(source.read_bytes() + b"\n# changed runtime source\n")
        with self.assertRaises((ValueError, RuntimeError)):
            self.ready.validate_plan(run, plan, self.q)

    def test_start_waits_for_sen_with_dependency_and_reentry_never_duplicates(self):
        run, _, _, _, _, checkout, _, _, r = self.prepared()
        manifest = (run / "rolling/manifest.jsonl").read_bytes()
        queue_state = (run / "rolling/state.json").read_bytes()
        active = [{"job_id": "185297", "qos": "highpriority", "state": "RUNNING", "name": "sen", "reason": "None"}]
        with self.sen.scheduler(active) as (_, commands), \
                patch.object(self.ready, "import_original", return_value=(self.q, r)), \
                patch.object(self.ready, "__file__", str(checkout / "scripts/recover_ready_gates.py")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.ready.start(argparse.Namespace(run_dir=run, submit=True))
            self.ready.start(argparse.Namespace(run_dir=run, submit=True))
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("--dependency") + 1], "afterany:185297")
        self.assertEqual(command[command.index("--partition") + 1], "cpup")
        self.assertNotIn("--gres", command)
        self.assertEqual((run / "rolling/manifest.jsonl").read_bytes(), manifest)
        self.assertEqual((run / "rolling/state.json").read_bytes(), queue_state)

    def test_start_omits_completed_predecessor_id_and_respects_actual_quota(self):
        run, _, _, _, _, checkout, _, _, r = self.prepared()
        active = self.sen.jobs(60)
        waits = []
        def free_one_slot(seconds):
            waits.append(seconds)
            self.assertFalse((run / "ready_gates_recovery/manager-job.json").exists())
            active.pop()
        with self.sen.scheduler(active) as (_, commands), \
                patch.object(self.ready, "import_original", return_value=(self.q, r)), \
                patch.object(self.ready, "__file__", str(checkout / "scripts/recover_ready_gates.py")), \
                patch.object(self.ready.time, "sleep", side_effect=free_one_slot), \
                contextlib.redirect_stdout(io.StringIO()):
            self.ready.start(argparse.Namespace(run_dir=run, submit=True))
        self.assertEqual(len(waits), 1)
        self.assertEqual(len(commands), 1)
        self.assertNotIn("--dependency", commands[0])

    def test_uncertain_manager_receipt_is_never_blindly_retried(self):
        run, _, _, _, _, checkout, _, plan, r = self.prepared()
        with self.sen.scheduler() as (_, commands), \
                patch.object(self.ready, "import_original", return_value=(self.q, r)), \
                patch.object(self.ready, "__file__", str(checkout / "scripts/recover_ready_gates.py")), \
                contextlib.redirect_stdout(io.StringIO()):
            command = self.ready.manager_command(run, plan, self.q)
            write_json(run / "ready_gates_recovery/manager-job.json", {
                "status": "uncertain", "job_id": None, "command": command,
                "identity": {"stage": "readiness_manager", "adapter_sha256": plan["adapter_sha256"]},
            })
            with self.assertRaisesRegex(RuntimeError, "reconciliation"):
                self.ready.start(argparse.Namespace(run_dir=run, submit=True))
        self.assertEqual(commands, [])

    def test_manager_does_not_enter_maintenance_before_sen_terminal_and_lock(self):
        run, _, _, _, _, _, _, _, r = self.prepared()
        events = []
        real_flock = fcntl.flock
        def flock(handle, operation):
            if str(handle.name).endswith("sencache_recovery/.manager.lock"):
                events.append("sen_lock")
                self.assertEqual(operation, fcntl.LOCK_EX)
                raise BlockingIOError("Sen still holds its lifetime lock")
            return real_flock(handle, operation)
        phases = iter(["active", "terminal"])
        def predecessor(plan, q):
            phase = next(phases)
            events.append(phase)
            return phase
        with patch.object(self.ready, "import_original", return_value=(self.q, r)), \
                patch.object(self.ready, "predecessor_state", side_effect=predecessor), \
                patch.object(self.ready.time, "sleep", side_effect=lambda _: events.append("wait")), \
                patch.object(self.ready.fcntl, "flock", side_effect=flock), \
                patch.object(r, "wait_for_pause") as maintenance, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BlockingIOError):
                self.ready.run_manager(argparse.Namespace(run_dir=run))
        self.assertEqual(events, ["active", "wait", "terminal", "sen_lock"])
        maintenance.assert_not_called()

    def test_missing_manifest_guard_runs_before_any_readiness_reactivation(self):
        run, _, rolling, state, _, _, _, plan, _ = self.prepared()
        original = copy.deepcopy(state)
        self.ready.install_adapter(run, plan, self.q)
        (run / "rolling/manifest.jsonl").rename(run / "rolling/manifest-maintenance.jsonl")
        with patch.object(self.ready, "reconcile_stale_blocks") as reconcile:
            with self.assertRaises(FileNotFoundError):
                self.q.tick(run, state, rolling)
        reconcile.assert_not_called()
        self.assertEqual(state, original)

    def test_handover_releases_controller_lock_before_new_wrapper_submission(self):
        run, _, _, _, _, _, _, plan, r = self.prepared()
        r.recovery_root = self.ready.root
        self.ready.install_adapter(run, plan, self.q)
        r.begin_pause(run, plan, self.q, self.ready.CYCLE)
        state = self.q.read_json(run / "rolling/state.json")
        state.update(status="paused", controller_job="185225", error="FileNotFoundError(2, 'No such file or directory')")
        write_json(run / "rolling/state.json", state)
        (run / "rolling/controller-185225.log").write_text(
            "Traceback (most recent call last):\nFileNotFoundError: [Errno 2] No such file or directory: " +
            repr(str(run / "rolling/manifest.jsonl")) + "\n")
        submitted = []
        def execute(command, **kwargs):
            with (run / "rolling/.controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                submitted.append(list(map(str, command)))
            return subprocess.CompletedProcess(command, 0, "191900\n", "")
        with self.sen.scheduler(), patch.object(self.q.subprocess, "run", side_effect=execute):
            receipt = r.resume_original(run, plan, self.q, self.ready.CYCLE)
        self.assertEqual(receipt["job_id"], "191900")
        self.assertEqual(len(submitted), 1)
        self.assertIn("recover_ready_gates.py", submitted[0][-1])
        self.assertEqual(self.q.file_hash(run / "rolling/manifest.jsonl"), plan["manifest_sha256"])
        self.assertEqual(self.q.read_json(run / "rolling/state.json")["status"], "prepared")

    def test_real_import_and_first_prepare_validate_original_sen_snapshot(self):
        run, _, _, _, _, checkout, _ = self.fixture()
        q, r = self.ready.import_original(run)
        self.assertEqual(r.recovery_root(run), run / "sencache_recovery")
        before = (run / "sencache_recovery/plan.json").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            plan = self.ready.prepare(run, checkout, q, r)
        self.ready.validate_plan(run, plan, q)
        self.assertEqual(plan["predecessor_job"], "185297")
        self.assertEqual((run / "sencache_recovery/plan.json").read_bytes(), before)
        self.assertTrue((run / "ready_gates_recovery/code/recover_ready_gates.py").is_file())

    def test_reactivation_proof_records_real_check_and_all_recovered_ids(self):
        run, _, rolling, state, _, _, check, _, _ = self.prepared()
        count = self.ready.reconcile_stale_blocks(run, state, rolling, self.q, self.h)
        self.assertEqual(count, 576)
        files = list((run / "ready_gates_recovery/reactivations").glob("*.json"))
        self.assertEqual(len(files), 1)
        proof = self.q.read_json(files[0])
        self.assertEqual((proof["partition"], proof["method"], proof["check_job_id"]),
                         ("h200p", "magcache", "185244"))
        self.assertEqual(proof["check_sha256"], digest(check))
        self.assertEqual(len(set(proof["run_ids"])), 576)
        self.assertEqual(proof["input_identity"], self.h.input_identity(run, "h200p", "magcache"))
        self.assertTrue(all(state["tasks"]["h200p"][identifier]["readiness_recovery_audit"] == str(files[0])
                            for identifier in proof["run_ids"]))

    def test_manager_failure_retries_only_its_existing_guarded_handover(self):
        run, _, _, _, _, _, _, _, r = self.prepared()
        def pause_then_fail(*args):
            write_json(run / "ready_gates_recovery/maintenance" / self.ready.CYCLE / "event.json",
                       {"status": "withdrawn", "cycle": self.ready.CYCLE})
            raise RuntimeError("synthetic interruption after own maintenance intent")
        with patch.object(self.ready, "import_original", return_value=(self.q, r)), \
                patch.object(self.ready, "predecessor_state", return_value="terminal"), \
                patch.object(r, "wait_for_pause", side_effect=pause_then_fail), \
                patch.object(r, "resume_until_accepted", return_value={"status": "accepted", "job_id": "191901"}) as restore, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self.ready.run_manager(argparse.Namespace(run_dir=run))
        restore.assert_called_once()
        self.assertEqual(restore.call_args.args[0], run)
        self.assertEqual(restore.call_args.args[3], self.ready.CYCLE)
        self.assertEqual(restore.call_args.kwargs, {"unblock": False})
        state = self.q.read_json(run / "ready_gates_recovery/state.json")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["resumed_controller_job"], "191901")




