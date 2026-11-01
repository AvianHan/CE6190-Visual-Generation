"""Offline recovery/queue tests. No Slurm jobs, downloads, or GPU imports."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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


def queue_lines(count, *, qos="highpriority", first=100):
    return "".join(f"{first+i}|{qos}|PENDING|other-project|Priority\n" for i in range(count))


class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = load_module("ce6190_queue_tests", REPO / "scripts/queue_orion.py")
        cls.h = load_module("ce6190_hardware_tests", REPO / "scripts/orion_hardware.py")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="orion queue test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def run_fixture(self):
        """The actual partial-submission shape, with fake weights and no jobs."""
        run = self.root / "runtime/runs/existing"
        code = run / "code"
        code.mkdir(parents=True)
        for directory in ("src", "configs"):
            shutil.copytree(REPO / directory, code / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("requirements.txt", "pyproject.toml",
                     "scripts/start_orion.py", "scripts/submit_slurm.py"):
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        members = sorted(str(p.relative_to(code)) for p in code.rglob("*") if p.is_file())
        (code / "SHA256SUMS.txt").write_text("".join(
            hashlib.sha256((code / name).read_bytes()).hexdigest() + "  " + name + "\n"
            for name in members))
        (run / "logs").mkdir()
        (run / "venv/bin").mkdir(parents=True)
        (run / "venv/bin/python").write_text("fake Python; never executed\n")
        (run / "vendor/wan").mkdir(parents=True)
        checkpoint = self.root / "runtime/models/wan"
        checkpoint.mkdir(parents=True)
        plan = {"run_dir": str(run), "work_root": str(self.root / "runtime"),
                "account": "test_account", "qos": "highpriority", "cpu_partition": "l40sp",
                "gpu_partition": "l40sp", "python_module": "python/3.11.9",
                "cuda_module": "cuda/12.4.1", "checkpoint": str(checkpoint),
                "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "model_revision": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
                "requested_time": "3-00:00:00", "jobs": {"env": "185108", "smoke": "185109"}}
        write_json(run / "plan.json", plan)
        write_json(run / "model_integrity.json", {"checkpoint": str(checkpoint), "files": []})
        write_json(run / "env.json", {"status": "running", "job_id": "185108"})
        write_json(run / "model.json", {"status": "passed", "job_id": None})
        for method, shard, finalize in (("magcache", "185110", "185111"),
                                        ("sencache", "185112", "185113")):
            write_json(run / "calibration" / method / "array-0.json",
                       {"status": "submitted", "job_id": shard})
            write_json(run / "calibration" / method / "finalize-job.json",
                       {"status": "submitted", "job_id": finalize})
        manifest_module = load_module("ce6190_queue_manifest_tests", REPO / "src/manifest.py")
        config = json.loads((code / "configs/validation_grid.json").read_text())
        for item in config["methods"]:
            if item["method"] in {"magcache", "sencache"}:
                item.setdefault("parameters", {})["calibration_path"] = str(
                    run / "calibration" / item["method"] / "profile.json")
        rows = manifest_module.build_manifest(
            manifest_module.load_prompts(code / "configs/development_prompts.jsonl"), config)
        development = run / "development_l40sp"
        development.mkdir()
        (development / "all_methods.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(development / "config.json", config)
        write_json(development / "results/slurm_submission.json",
                   {"status": "submission_failed", "submitted_jobs": [], "runs": 1536,
                    "arrays": [{"offset": 0, "count": 1001, "job_id": None, "status": "rejected"},
                               {"offset": 1001, "count": 535, "job_id": None, "status": "planned"}]})
        return run, plan, rows

    def test_real_quota_counts_every_element_and_other_projects(self):
        lines = queue_lines(20) + queue_lines(7, qos="normal", first=500)
        policy = self.q.parse_qos("highpriority|60|\n", "highpriority")
        active = self.q.parse_queue(lines, "highpriority")
        self.assertEqual(policy["max_submit"], 60)
        self.assertEqual(len(active), 20)
        self.assertEqual(self.q.quota_headroom(policy["max_submit"], active), 40)
        lines += "900|highpriority|RUNNING|ce6190-controller|None\n"
        lines += "901|highpriority|PENDING|ce6190-h200-check|Dependency\n"
        self.assertEqual(self.q.quota_headroom(60, self.q.parse_queue(lines, "highpriority")), 38)
        self.assertEqual(self.q.quota_headroom(60, self.q.parse_queue(queue_lines(65), "highpriority")), 0)

    def test_compressed_array_or_malformed_queue_never_underestimates_usage(self):
        for text in ("185110_[0-7]|highpriority|PENDING|calibration|Dependency\n",
                     "185110|highpriority\n"):
            with self.subTest(text=text), self.assertRaises((ValueError, RuntimeError)):
                self.q.parse_queue(text, "highpriority")

    def test_missing_or_malformed_qos_cannot_be_treated_as_free_capacity(self):
        for text in ("normal|60|\n", "highpriority|not-a-number|\n", ""):
            with self.subTest(text=text), self.assertRaises((ValueError, RuntimeError)):
                self.q.parse_qos(text, "highpriority")

    def test_site_walltime_is_parsed_without_a_new_experiment_cap(self):
        self.assertEqual(self.q.parse_duration("3-00:00:00"), 3 * 86400)
        self.assertEqual(self.q.parse_duration("01:30:00"), 5400)
        self.assertIsNone(self.q.parse_duration("UNLIMITED"))

    def test_controller_submission_clears_inherited_sbatch_options(self):
        with patch.dict(os.environ, {"SBATCH_GRES": "gpu:8", "SBATCH_WAIT": "1",
                                     "SBATCH_PARTITION": "h200p", "SBATCH_TIMELIMIT": "00:01:00",
                                     "CE6190_TEST_KEEP": "kept"}):
            environment = self.q.clean_sbatch_env()
        self.assertFalse(any(key.startswith("SBATCH_") for key in environment))
        self.assertEqual(environment["CE6190_TEST_KEEP"], "kept")

    def test_accepted_submission_is_durable_and_never_sent_twice(self):
        receipt = self.root / "receipt.json"
        command = ["sbatch", "--parsable", "--array", "0-39", "--gres", "gpu:1", "job.sh"]
        accepted = subprocess.CompletedProcess(command, 0, "190000;orion\n", "")
        with patch.object(self.q.subprocess, "run", return_value=accepted) as runner:
            first = self.q.submit_once(receipt, command, {"kind": "experiments", "count": 40})
            second = self.q.submit_once(receipt, command, {"kind": "experiments", "count": 40})
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(str(first["job_id"]), "190000")
        self.assertEqual(first, second)
        self.assertEqual(str(json.loads(receipt.read_text())["job_id"]), "190000")
        self.assertNotIn("%", command[command.index("--array") + 1])

    def test_explicit_quota_rejection_can_be_retried_without_losing_receipt(self):
        receipt = self.root / "receipt.json"
        command = ["sbatch", "--parsable", "job.sh"]
        rejected = subprocess.CompletedProcess(command, 1, "", "sbatch: error: QOSMaxSubmitJobPerUserLimit\n")
        accepted = subprocess.CompletedProcess(command, 0, "190001\n", "")
        with patch.object(self.q.subprocess, "run", side_effect=[rejected, accepted]) as runner:
            first = self.q.submit_once(receipt, command, {"batch": "a"})
            self.assertEqual(first["status"], "quota_wait")
            self.assertIsNone(first.get("job_id"))
            second = self.q.submit_once(receipt, command, {"batch": "a"})
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(str(second["job_id"]), "190001")

    def test_unknown_submission_outcome_blocks_automatic_resubmission(self):
        for outcome in (subprocess.TimeoutExpired(["sbatch"], 30),
                        subprocess.CompletedProcess(["sbatch"], 0, "unexpected reply\n", "")):
            receipt = self.root / ("timeout.json" if isinstance(outcome, Exception) else "garbled.json")
            kw = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
            with patch.object(self.q.subprocess, "run", **kw) as runner:
                with self.assertRaises(RuntimeError):
                    self.q.submit_once(receipt, ["sbatch", "job.sh"], {"batch": "a"})
                self.assertEqual(json.loads(receipt.read_text())["status"], "uncertain")
                with self.assertRaises(RuntimeError):
                    self.q.submit_once(receipt, ["sbatch", "job.sh"], {"batch": "a"})
                self.assertEqual(runner.call_count, 1)

    def passed_parent(self, run):
        write_json(run / "env.json", {"status": "passed", "job_id": "185108"})
        write_json(run / "smoke.json", {"status": "passed", "job_id": "185109"})

    def passed_hardware(self, run, stage):
        self.h.prepare(run, "h200p")
        write_json(self.h.state_path(run, "h200p", stage),
                   {"status": "passed", "input_identity": self.h.input_identity(run, "h200p", stage)})

    def passed_calibration(self, run, method):
        write_json(run / "calibration" / method / "profile.json", {"method": method, "test_values": [1, 2]})
        write_json(run / "calibration" / method / "check.json", {"status": "passed"})

    def test_hardware_preparation_shares_inputs_but_never_inherits_smoke(self):
        run, _, _ = self.run_fixture()
        self.passed_parent(run)
        with patch.dict(sys.modules, {"torch": None, "numpy": None}):
            child = self.h.prepare(run, "h200p")
            self.assertEqual(self.h.prepare(run, "h200p"), child)
            self.assertEqual(self.h.runtime_path(run, "l40sp"), run)
            self.assertFalse((child / "smoke.json").exists())
            self.assertFalse((child / "smoke_results").exists())
            self.assertEqual(self.h.group_gate(run, "h200p", "baseline")["status"], "waiting")
        self.assertNotEqual(child, run)
        for name in ("code", "venv", "vendor"):
            self.assertTrue((child / name).is_symlink())
            self.assertEqual((child / name).resolve(), (run / name).resolve())
        self.assertEqual(json.loads((child / "plan.json").read_text())["gpu_partition"], "h200p")
        self.assertEqual(json.loads((run / "plan.json").read_text())["gpu_partition"], "l40sp")

    def test_initial_runtime_and_calibrated_groups_have_distinct_gates(self):
        run, _, _ = self.run_fixture()
        self.assertEqual(self.h.group_gate(run, "l40sp", "baseline")["status"], "waiting")
        self.passed_parent(run)
        for method in ("baseline", "teacache", "seacache"):
            self.assertEqual(self.h.group_gate(run, "l40sp", method)["status"], "passed")
        for method in ("magcache", "sencache"):
            self.assertEqual(self.h.group_gate(run, "l40sp", method)["status"], "waiting")
        self.passed_calibration(run, "magcache")
        self.assertEqual(self.h.group_gate(run, "l40sp", "magcache")["status"], "passed")
        self.assertEqual(self.h.group_gate(run, "l40sp", "sencache")["status"], "waiting")

    def test_h200_checks_wait_for_actual_prerequisites_and_own_gpu_checks(self):
        run, _, _ = self.run_fixture()
        self.h.prepare(run, "h200p")
        self.assertEqual(self.h.pending_checks(run, ["l40sp", "h200p"]), [])
        self.passed_parent(run)
        specs = self.h.pending_checks(run, ["l40sp", "h200p"])
        self.assertEqual([(x["partition"], x["stage"]) for x in specs], [("h200p", "smoke")])
        self.passed_calibration(run, "magcache")
        self.assertEqual(self.h.group_gate(run, "h200p", "magcache")["status"], "waiting")
        self.passed_hardware(run, "smoke")
        self.assertEqual(self.h.group_gate(run, "h200p", "baseline")["status"], "passed")
        self.assertEqual([x["stage"] for x in self.h.pending_checks(run, ["h200p"])], ["magcache"])
        self.assertEqual(self.h.group_gate(run, "h200p", "magcache")["status"], "waiting")
        self.passed_hardware(run, "magcache")
        self.assertEqual(self.h.group_gate(run, "h200p", "magcache")["status"], "passed")
        self.assertEqual(self.h.group_gate(run, "h200p", "sencache")["status"], "waiting")

    def test_shared_frozen_profile_change_invalidates_h200_pass(self):
        run, _, _ = self.run_fixture()
        self.passed_parent(run)
        self.passed_calibration(run, "sencache")
        self.passed_hardware(run, "smoke")
        self.passed_hardware(run, "sencache")
        identity = self.h.input_identity(run, "h200p", "sencache")
        self.assertEqual(identity["profile_path"], str(run / "calibration/sencache/profile.json"))
        self.assertEqual(identity["calibration_source_partition"], "l40sp")
        self.assertEqual(self.h.group_gate(run, "h200p", "sencache")["status"], "passed")
        write_json(run / "calibration/sencache/profile.json", {"changed": True})
        self.assertEqual(self.h.group_gate(run, "h200p", "sencache")["status"], "failed")

    def test_failed_hardware_validation_is_recorded_and_never_silently_retried(self):
        run, _, _ = self.run_fixture()
        self.passed_parent(run)
        fake = types.SimpleNamespace(verified_release=lambda _: None, runtime_environment=lambda _: None,
                                     gpu_smoke=lambda _: (_ for _ in ()).throw(RuntimeError("GPU mismatch")))
        with patch.object(self.h, "_starter", return_value=fake), \
                patch.object(self.h, "verify_allocated_gpu", return_value={"name": "H200"}) as gpu:
            with self.assertRaisesRegex(RuntimeError, "GPU mismatch"):
                self.h.run_check(run, "h200p", "smoke")
            state = json.loads(self.h.state_path(run, "h200p", "smoke").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(self.h.group_gate(run, "h200p", "baseline")["status"], "failed")
            self.assertEqual(self.h.pending_checks(run, ["h200p"]), [])
            with self.assertRaisesRegex(RuntimeError, "Previous"):
                self.h.run_check(run, "h200p", "smoke")
            self.assertEqual(gpu.call_count, 1)

    def rolling_fixture(self, partitions=("l40sp",)):
        run, original, rows = self.run_fixture()
        checkout = self.root / "updated checkout"
        shutil.copytree(run / "code", checkout)
        for name in ("queue_orion.py", "orion_hardware.py"):
            shutil.copy2(REPO / "scripts" / name, checkout / "scripts" / name)
        members = sorted(str(p.relative_to(checkout)) for p in checkout.rglob("*")
                         if p.is_file() and p.name != "SHA256SUMS.txt")
        (checkout / "SHA256SUMS.txt").write_text("".join(
            hashlib.sha256((checkout / name).read_bytes()).hexdigest() + "  " + name + "\n"
            for name in members))
        with patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "partition_time", return_value={"time": "3-00:00:00", "seconds": 259200}), \
                patch.object(self.q.shutil, "disk_usage", return_value=types.SimpleNamespace(free=2 * 1024**4)):
            plan = self.q.prepare_plan(run, list(partitions), checkout)
        rolling = run / "rolling"
        (rolling / "code").mkdir(parents=True)
        for name in ("queue_orion.py", "orion_hardware.py"):
            shutil.copy2(REPO / "scripts" / name, rolling / "code" / name)
        (rolling / "manifest.jsonl").write_bytes(Path(plan["source_manifest"]).read_bytes())
        write_json(rolling / "plan.json", plan)
        state = self.q.fresh_state(plan, rows)
        write_json(rolling / "state.json", state)
        for partition in partitions:
            self.h.prepare(run, partition)
        return run, plan, state, rows, checkout

    def slurm_recorder(self):
        commands = []
        def execute(command, **kwargs):
            command = list(map(str, command))
            if command[0] == "scontrol":
                return subprocess.CompletedProcess(command, 0, "MaxArraySize = 1001\nSchedulerParameters = (null)\n", "")
            if command[0] == "sbatch":
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, str(190000 + len(commands)) + "\n", "")
            if command[0] == "sacct":
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError("Unexpected external command: " + repr(command))
        return commands, execute

    def test_partial_submission_migrates_all_2880_rows_without_recreating_environment(self):
        run, plan, state, rows, _ = self.rolling_fixture(("l40sp", "h200p"))
        self.assertEqual(plan["rows_per_partition"], 2880)
        self.assertEqual(set(state["tasks"]), {"l40sp", "h200p"})
        self.assertEqual(len(state["tasks"]["l40sp"]), 2880)
        self.assertEqual(len(state["tasks"]["h200p"]), 2880)
        baselines = {row["run_id"] for row in rows if row["method"] == "baseline"}
        self.assertEqual(len(baselines), 192)
        self.assertTrue(all(row["reference_run_id"] in baselines for row in rows if row["method"] != "baseline"))
        old = json.loads((run / "development_l40sp/results/slurm_submission.json").read_text())
        self.assertEqual(old["submitted_jobs"], [])
        self.assertEqual(json.loads((run / "plan.json").read_text())["jobs"], {"env": "185108", "smoke": "185109"})

    def test_migration_refuses_any_accepted_or_uncertain_legacy_array(self):
        run, _, _ = self.run_fixture()
        receipt = run / "development_l40sp/results/slurm_submission.json"
        for record in ({"submitted_jobs": ["123"], "arrays": []},
                       {"submitted_jobs": [], "arrays": [{"job_id": "123", "status": "submitted"}]},
                       {"submitted_jobs": [], "arrays": [{"job_id": None, "status": "uncertain"}]}):
            write_json(receipt, record)
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.q.verify_old_receipts(run)

    def test_controller_uses_cpu_only_system_python_and_actual_partition_policy(self):
        run, plan, _, _, _ = self.rolling_fixture()
        plan["partition_policy"]["cpup"] = {"time": "1-00:00:00", "seconds": 86400}
        command = self.q.controller_command(run, plan, "afterany:190000")
        self.assertEqual(command[command.index("--partition") + 1], "cpup")
        self.assertEqual(command[command.index("--time") + 1], "1-00:00:00")
        self.assertEqual(command[command.index("--dependency") + 1], "afterany:190000")
        self.assertNotIn("--gres", command)
        self.assertIn("/usr/bin/python3", command[-1])
        self.assertNotIn("module load", command[-1])
        with patch.object(self.q, "query", return_value="PartitionName=cpup AllowQos=highpriority MaxTime=1-00:00:00"):
            value = self.q.partition_time("cpup", "highpriority", {"max_wall_seconds": 172800})
        self.assertEqual(value["seconds"], 86400)

    def test_tick_fills_real_remaining_quota_and_never_duplicates_pending_work(self):
        run, plan, state, _, _ = self.rolling_fixture()
        self.passed_parent(run)
        active = self.q.parse_queue(queue_lines(20), "highpriority")
        commands, execute = self.slurm_recorder()
        with patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "query_queue", return_value=active), \
                patch.object(self.q.subprocess, "run", side_effect=execute), \
                patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
            state = self.q.tick(run, state, plan, hardware=self.h)
            tasks = state["tasks"]["l40sp"]
            submitted = {key: value for key, value in tasks.items() if value["status"] == "submitted"}
            self.assertEqual(len(submitted), 40)
            self.assertEqual({x["method"] for x in submitted.values()}, {"baseline", "teacache", "seacache"})
            self.assertTrue(all("%" not in command[command.index("--array") + 1] for command in commands))
            self.assertEqual(sum(int(command[command.index("--array") + 1].split("-")[1]) + 1 for command in commands), 40)
            for task in submitted.values():
                active.append({"job_id": task["job_id"], "qos": "highpriority", "state": "PENDING", "name": "experiment", "reason": "Priority"})
            call_count = len(commands)
            state = self.q.tick(run, state, plan, hardware=self.h)
        self.assertEqual(len(commands), call_count)
        self.assertEqual(sum(x["status"] == "submitted" for x in state["tasks"]["l40sp"].values()), 40)

    def test_repeated_start_does_not_submit_another_active_controller(self):
        run, plan, _, _, checkout = self.rolling_fixture()
        commands, execute = self.slurm_recorder()
        active = self.q.parse_queue(queue_lines(20), "highpriority")
        args = argparse.Namespace(run_dir=run, partitions=["l40sp"], submit=True)
        with patch.object(self.q, "__file__", str(checkout / "scripts/queue_orion.py")), \
                patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "query_queue", return_value=active), \
                patch.object(self.q.subprocess, "run", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            self.q.start(args)
            self.assertEqual(len(commands), 1)
            active.append({"job_id": "190001", "qos": "highpriority", "state": "PENDING", "name": "ce6190-queue", "reason": "Priority"})
            self.q.start(args)
        self.assertEqual(len(commands), 1)

    def test_start_does_not_duplicate_controller_missing_from_a_fresh_queue_snapshot(self):
        run, _, _, _, checkout = self.rolling_fixture()
        args = argparse.Namespace(run_dir=run, partitions=["l40sp"], submit=True)
        commands, execute = self.slurm_recorder()
        with patch.object(self.q, "__file__", str(checkout / "scripts/queue_orion.py")), \
                patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "query_queue", return_value=[]), \
                patch.object(self.q.subprocess, "run", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
            self.q.start(args)
            with self.assertRaisesRegex(RuntimeError, "not yet conclusively terminal"):
                self.q.start(args)
        self.assertEqual(len(commands), 1)

    def test_cancelled_calibration_finalizer_blocks_only_affected_method_and_finishes(self):
        run, plan, state, _, _ = self.rolling_fixture()
        self.passed_parent(run)
        self.passed_calibration(run, "sencache")
        for task in state["tasks"]["l40sp"].values():
            if task["method"] != "magcache":
                task["status"] = "complete"
        missing_check = run / "calibration/magcache/check.json"
        self.assertFalse(missing_check.exists())
        with patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "query_queue", return_value=[]), \
                patch.object(self.q, "query_accounting", return_value={"185111": {"state": "CANCELLED", "exit_code": "0:0"}}), \
                patch.object(self.q, "submit_once") as submit:
            state = self.q.tick(run, state, plan, hardware=self.h)
        self.assertEqual(state["status"], "complete_with_failures")
        self.assertEqual(sum(task["status"] == "blocked" for task in state["tasks"]["l40sp"].values()), 576)
        self.assertIn("parent:magcache", state["failed_prerequisites"])
        self.assertFalse(missing_check.exists(), "The controller must not forge original phase state")
        submit.assert_not_called()

    def test_failed_generation_is_reported_without_an_automatic_retry(self):
        run, plan, state, _, _ = self.rolling_fixture()
        self.passed_parent(run)
        identifier, task = next(iter(state["tasks"]["l40sp"].items()))
        task.update(status="submitted", job_id="190200_0")
        write_json(run / "development_l40sp/results" / identifier / "metrics.json",
                   {"status": "failed", "error": "CUDA out of memory"})
        with patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "query_queue", return_value=self.q.parse_queue(queue_lines(60), "highpriority")), \
                patch.object(self.q, "query_accounting", return_value={}), \
                patch.object(self.q, "submit_once") as submit:
            state = self.q.tick(run, state, plan, hardware=self.h)
            state = self.q.tick(run, state, plan, hardware=self.h)
        self.assertEqual(state["tasks"]["l40sp"][identifier]["status"], "failed")
        self.assertIn("CUDA out of memory", state["tasks"]["l40sp"][identifier]["error"])
        submit.assert_not_called()

    def test_controller_successor_and_h200_check_consume_real_submit_slots(self):
        run, plan, state, _, _ = self.rolling_fixture(("l40sp", "h200p"))
        self.passed_parent(run)
        active = self.q.parse_queue(queue_lines(20), "highpriority")
        active.append({"job_id": "190500", "qos": "highpriority", "state": "RUNNING", "name": "ce6190-queue", "reason": "None"})
        commands, execute = self.slurm_recorder()
        with patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "query_queue", return_value=active), \
                patch.object(self.q.subprocess, "run", side_effect=execute), \
                patch.dict(os.environ, {"SLURM_JOB_ID": "190500"}):
            state = self.q.tick(run, state, plan, hardware=self.h)
        experiment_count = sum(task["status"] == "submitted" for tasks in state["tasks"].values() for task in tasks.values())
        self.assertEqual(experiment_count, 37)  # 60 - 21 active - 1 successor - 1 hardware check.
        self.assertEqual(sum("--array" not in command for command in commands), 2)
        self.assertEqual(sum(task["status"] == "submitted" for task in state["tasks"]["h200p"].values()), 0)
        self.assertIn("h200p:smoke", state["checks"])

    def test_killed_h200_check_with_stale_running_state_cannot_wait_forever(self):
        run, plan, state, _, _ = self.rolling_fixture(("l40sp", "h200p"))
        self.passed_parent(run)
        for task in state["tasks"]["l40sp"].values():
            task["status"] = "complete"
        state_file = self.h.state_path(run, "h200p", "smoke")
        write_json(state_file, {"status": "running", "job_id": "190500"})
        write_json(run / "rolling/checks/h200p-smoke.json",
                   {"status": "accepted", "job_id": "190500", "identity": {
                       "partition": "h200p", "stage": "smoke", "state_file": str(state_file)}})
        with patch.object(self.q, "query_qos", return_value={"max_submit": 60, "max_wall_seconds": None}), \
                patch.object(self.q, "query_queue", return_value=[]), \
                patch.object(self.q, "query_accounting", return_value={"190500": {"state": "TIMEOUT", "exit_code": "0:9"}}), \
                patch.object(self.q, "submit_once") as submit:
            state = self.q.tick(run, state, plan, hardware=self.h)
        self.assertEqual(state["status"], "complete_with_failures")
        self.assertTrue(all(x["status"] == "blocked" for x in state["tasks"]["h200p"].values()))
        self.assertTrue(all(x["status"] == "complete" for x in state["tasks"]["l40sp"].values()))
        self.assertEqual(json.loads(state_file.read_text())["status"], "running")
        submit.assert_not_called()

    def test_transient_scheduler_query_failure_retries_without_permanent_pause(self):
        run, _, _, _, _ = self.rolling_fixture()
        calls = []
        def tick(run_dir, state, plan):
            calls.append(state["status"])
            if len(calls) == 1:
                raise self.q.SchedulerUnavailable("temporary scheduler outage")
            state["status"] = "complete"
            return state
        with patch.object(self.q, "tick", side_effect=tick), patch.object(self.q.time, "sleep") as sleep, \
                contextlib.redirect_stdout(io.StringIO()):
            self.q.run_controller(argparse.Namespace(run_dir=run))
        self.assertEqual(calls, ["prepared", "waiting_scheduler"])
        self.assertEqual(sleep.call_count, 1)
        state = json.loads((run / "rolling/state.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertNotIn("scheduler_error", state)


if __name__ == "__main__":
    unittest.main()
