#!/usr/bin/env python3
"""Recover three failed SenCache shards while preserving running experiments.

The shared runtime and five successful shards stay immutable. Maintenance moves
only the controller's top-level manifest aside; each GPU batch already has its
own immutable manifest. The existing controller detects this at its next tick,
records a pause and exits normally through its exception handler. No job is
signalled or cancelled. State edits occur only after all old controllers are
terminal and the controller lock is held.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

TARGETS = (0, 2, 7)
SCRIPTS = ("recover_sencache.py", "sencache_recovery_worker.py", "sencache_recovery_assets.py")
POLL_SECONDS = 10


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_queue(run_dir):
    return import_path("ce6190_recovery_original_queue", Path(run_dir) / "rolling/code/queue_orion.py")


def recovery_root(run_dir):
    return Path(run_dir) / "sencache_recovery"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def get_state(run_dir, q):
    return q.read_json(recovery_root(run_dir) / "state.json", {})


def update_state(run_dir, q, **values):
    state = get_state(run_dir, q)
    state.update(values, updated_at=now())
    q.atomic_json(recovery_root(run_dir) / "state.json", state)
    print(json.dumps({k: state.get(k) for k in ("status", "stage", "error", "updated_at")}), flush=True)
    return state


def validate_frozen(run_dir, plan, q):
    for name, digest in plan["frozen_files"].items():
        if q.file_hash(Path(run_dir) / name) != digest:
            raise ValueError(f"Frozen recovery input changed: {name}")
    for name, digest in plan["successful_shards"].items():
        if q.file_hash(Path(run_dir) / "calibration/sencache" / name) != digest:
            raise ValueError(f"Previously successful shard changed: {name}")
    for name, digest in plan["recovery_scripts"].items():
        if q.file_hash(recovery_root(run_dir) / "code" / name) != digest:
            raise ValueError(f"Recovery worker snapshot changed: {name}")


def assert_no_sen_submissions(run_dir, q):
    state = q.read_json(Path(run_dir) / "rolling/state.json")
    for tasks in state["tasks"].values():
        for task in tasks.values():
            if task["method"] == "sencache" and task["status"] != "blocked":
                raise ValueError("Recovery requires all SenCache experiment tasks still blocked")
    for path in (Path(run_dir) / "rolling/batches").glob("*/receipt.json"):
        record = q.read_json(path)
        if record.get("identity", {}).get("method") == "sencache" and record.get("status") != "quota_wait":
            raise ValueError("A SenCache experiment submission already exists; refusing duplicate recovery")


def prepare_recovery(run_dir, repo, q):
    run_dir = Path(run_dir).resolve(); repo = Path(repo).resolve()
    root = recovery_root(run_dir)
    previous = q.read_json(root / "plan.json")
    if previous:
        validate_frozen(run_dir, previous, q)
        return previous
    rolling = q.read_json(run_dir / "rolling/plan.json")
    state = q.read_json(run_dir / "rolling/state.json")
    if state.get("status") not in {"running", "waiting_storage", "waiting_scheduler", "complete_with_failures"}:
        raise ValueError("Original queue must be running or naturally complete; investigate an unrelated pause first")
    assert_no_sen_submissions(run_dir, q)
    if "parent:sencache" not in state.get("failed_prerequisites", {}):
        raise ValueError("Expected the recorded failed SenCache prerequisite")
    for phase in ("env", "model", "smoke"):
        if q.read_json(run_dir / (phase + ".json"), {}).get("status") != "passed":
            raise ValueError(f"Original prerequisite is not passed: {phase}")
    if q.read_json(run_dir / "calibration/sencache/check.json", {}).get("status") == "passed":
        raise ValueError("SenCache already passed; do not launch another calibration")
    if (run_dir / "calibration/sencache/profile.json").exists():
        raise ValueError("Existing SenCache profile needs review before recovery")
    original = q.read_json(run_dir / "plan.json")
    starter = import_path("ce6190_recovery_starter", run_dir / "code/scripts/start_orion.py")
    starter.verified_release(run_dir / "code")
    starter.compatible_runtime(repo, run_dir)
    if q.file_hash(run_dir / "rolling/manifest.jsonl") != rolling["manifest_sha256"]:
        raise ValueError("Original rolling manifest differs")
    samples = [json.loads(line) for line in (run_dir / "calibration/samples.jsonl").read_text().splitlines() if line.strip()]
    if len(samples) != 8:
        raise ValueError("This recovery expects the frozen eight-video calibration")
    successful = {}
    for index in range(8):
        path = run_dir / "calibration/sencache" / f"shard-{index:04d}.json"
        if index in TARGETS:
            if path.exists(): raise ValueError(f"Target shard already exists before recovery: {path}")
        else:
            data = q.read_json(path)
            if not data or data.get("method") != "sencache" or data.get("selection", {}).get("sample_indices") != [index]:
                raise ValueError(f"Expected successful shard {index}")
            successful[path.name] = q.file_hash(path)
    source_files = ["plan.json", "rolling/plan.json", "calibration/samples.jsonl", "calibration/config.json",
                    "calibration/inputs.json", "code/SHA256SUMS.txt", "code/src/calibrate.py",
                    "code/scripts/start_orion.py"]
    source_files += ["rolling/code/" + name for name in rolling["script_hashes"]]
    for item in q.read_json(run_dir / "calibration/inputs.json")["exclusions"]:
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(run_dir): raise ValueError("Held-out manifest is outside the original runtime")
        source_files.append(str(path.relative_to(run_dir)))
    (root / "code").mkdir(parents=True, exist_ok=True)
    scripts = {}
    for name in SCRIPTS:
        source = repo / "scripts" / name; destination = root / "code" / name
        digest = q.file_hash(source)
        if destination.exists() and q.file_hash(destination) != digest:
            raise ValueError("Partial recovery snapshot differs")
        if not destination.exists(): shutil.copy2(source, destination)
        scripts[name] = digest
    plan = {"schema_version": 1, "run_dir": str(run_dir), "created_at": now(),
            "targets": list(TARGETS), "successful_shards": successful,
            "frozen_files": {name: q.file_hash(run_dir / name) for name in source_files},
            "recovery_scripts": scripts, "rolling_plan": rolling, "original_plan": original,
            "manifest_sha256": rolling["manifest_sha256"]}
    q.atomic_json(root / "plan.json", plan)
    update_state(run_dir, q, status="prepared", stage="initial_maintenance", error=None)
    return plan


@contextmanager
def controller_lock(run_dir):
    with (Path(run_dir) / "rolling/.controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def controllers_terminal(run_dir, qos, q):
    active = {row["job_id"] for row in q.query_queue(qos)}
    identifiers = []
    for path in (Path(run_dir) / "rolling/controllers").glob("*.json"):
        record = q.read_json(path)
        if record.get("status") in {"submitting", "uncertain"} and not record.get("job_id"):
            raise RuntimeError(f"Original controller submission is uncertain: {path}")
        identifier = record.get("job_id")
        if identifier:
            if identifier in active: return False
            identifiers.append(identifier)
    accounting = q.query_accounting(identifiers)
    return all(identifier in accounting and accounting[identifier]["state"] not in q.ACTIVE for identifier in identifiers)


def begin_pause(run_dir, plan, q, cycle):
    """Write an intent before atomically withdrawing the controller manifest."""
    run_dir = Path(run_dir); folder = recovery_root(run_dir) / "maintenance" / cycle
    folder.mkdir(parents=True, exist_ok=True)
    event_path = folder / "event.json"; event = q.read_json(event_path)
    manifest = run_dir / "rolling/manifest.jsonl"; backup = folder / "manifest.jsonl"
    if event and event.get("status") in {"restored", "resumed"}: return event
    with (run_dir / "rolling/.start.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if event is None:
            state = q.read_json(run_dir / "rolling/state.json")
            if state.get("status") == "paused":
                raise RuntimeError("Original queue was already paused for another reason")
            if q.file_hash(manifest) != plan["manifest_sha256"]:
                raise ValueError("Cannot pause a changed rolling manifest")
            event = {"cycle": cycle, "status": "requested", "requested_at": now(),
                     "manifest_sha256": plan["manifest_sha256"], "previous_status": state["status"],
                     "note": "Intentional maintenance: top-level scheduling manifest withdrawn; GPU batch manifests stay intact."}
            q.atomic_json(folder / "state-before-request.json", state)
            q.atomic_json(event_path, event)
        if backup.exists():
            if q.file_hash(backup) != plan["manifest_sha256"]:
                raise ValueError("Maintenance backup changed")
            if manifest.exists():
                raise RuntimeError("Both scheduling manifest and uncompleted maintenance backup exist")
        else:
            if q.file_hash(manifest) != plan["manifest_sha256"]:
                raise ValueError("Scheduling manifest changed before maintenance")
            os.replace(manifest, backup)
        event.update(status="withdrawn", withdrawn_at=event.get("withdrawn_at", now()))
        q.atomic_json(event_path, event)
    return event


def pause_ready(run_dir, q, cycle):
    plan = q.read_json(recovery_root(run_dir) / "plan.json")
    if not controllers_terminal(run_dir, plan["original_plan"]["qos"], q): return False
    try:
        with controller_lock(run_dir): return True
    except BlockingIOError: return False


def wait_for_pause(run_dir, plan, q, cycle):
    event = begin_pause(run_dir, plan, q, cycle)
    if event.get("status") in {"restored", "resumed"}:
        return
    while True:
        try:
            if pause_ready(run_dir, q, cycle): return
            update_state(run_dir, q, status="maintenance_wait", stage=cycle)
        except q.SchedulerUnavailable as error:
            update_state(run_dir, q, status="waiting_scheduler", stage=cycle, error=str(error))
        time.sleep(POLL_SECONDS)


def validate_recovered_profile(run_dir, plan, q):
    validate_frozen(run_dir, plan, q)
    root = recovery_root(run_dir)
    final = q.read_json(root / "jobs/finalize.json", {})
    check = q.read_json(Path(run_dir) / "calibration/sencache/check.json", {})
    if not final.get("job_id") or check.get("status") != "passed" or str(check.get("job_id")) != final["job_id"]:
        raise ValueError("Recovery finalizer has no matching passed calibration/parity artifact")
    profile_path = Path(run_dir) / "calibration/sencache/profile.json"
    profile = q.read_json(profile_path)
    if not profile or profile.get("method") != "sencache" or len(profile.get("sample_ids", [])) != 8 or len(profile.get("shards", [])) != 8:
        raise ValueError("Recovered profile does not cover all eight samples")
    audits = {}
    for index in TARGETS:
        shard = Path(run_dir) / "calibration/sencache" / f"shard-{index:04d}.json"
        data = q.read_json(shard)
        if not data or not data.get("recovery_decoder"):
            raise ValueError(f"Recovered shard lacks decoder audit: {index}")
        audits[str(index)] = {"path": str(shard), "sha256": q.file_hash(shard), "recovery_decoder": data["recovery_decoder"]}
    for record in profile["shards"]:
        if q.file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Merged profile shard changed")
    return {"status": "passed", "finalizer_job": final["job_id"], "profile_sha256": q.file_hash(profile_path),
            "original_successful_shards": plan["successful_shards"], "recovered_shards": audits,
            "recovery_scripts": plan["recovery_scripts"]}


def owned_manifest_pause(run_dir, state):
    """Match the completed controller traceback, since exception repr omits path."""
    if state.get("status") != "paused" or not state.get("error", "").startswith("FileNotFoundError("):
        return False
    identifier = str(state.get("controller_job", ""))
    if not identifier.isdigit(): return False
    manifest = Path(run_dir) / "rolling/manifest.jsonl"
    if manifest.exists(): return False
    log = Path(run_dir) / "rolling" / ("controller-" + identifier + ".log")
    if not log.is_file(): return False
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 65536))
        tail = stream.read().decode("utf-8", errors="replace")
    expected = "FileNotFoundError: [Errno 2] No such file or directory: " + repr(str(manifest))
    return expected in tail


def resume_original(run_dir, plan, q, cycle, unblock=False):
    """Restore under the controller lock; release it before starting its replacement."""
    run_dir = Path(run_dir); root = recovery_root(run_dir); folder = root / "maintenance" / cycle
    event_path = folder / "event.json"; event = q.read_json(event_path)
    if not event: raise ValueError("No maintenance intent exists")
    receipt = run_dir / "rolling/controllers" / ("sen-recovery-" + cycle + ".json")
    old = q.read_json(receipt)
    if old and old.get("job_id"):
        event.update(status="resumed", controller_job=old["job_id"]); q.atomic_json(event_path, event)
        return old
    if not pause_ready(run_dir, q, cycle):
        return {"status": "maintenance_wait"}
    with (run_dir / "rolling/.start.lock").open("a") as start_handle:
        fcntl.flock(start_handle, fcntl.LOCK_EX)
        with controller_lock(run_dir):
            state = q.read_json(run_dir / "rolling/state.json")
            manifest = run_dir / "rolling/manifest.jsonl"; backup = folder / "manifest.jsonl"
            if q.file_hash(backup) != plan["manifest_sha256"]:
                raise ValueError("Maintenance manifest backup changed")
            if event.get("status") != "restored":
                transition = q.read_json(folder / "transition.json")
                if transition is None:
                    owned_pause = owned_manifest_pause(run_dir, state)
                    natural_end = state.get("status") in {"complete", "complete_with_failures"}
                    if not owned_pause and not natural_end:
                        raise RuntimeError("Queue did not pause at the expected manifest guard; refusing unrelated state reset")
                    # All validations precede writes, allowing a rejected unblock
                    # to restore ordinary four-method processing without a trap.
                    audit = validate_recovered_profile(run_dir, plan, q) if unblock else None
                    if unblock: assert_no_sen_submissions(run_dir, q)
                    q.atomic_json(folder / "state-before-resume.json", state)
                    if unblock:
                        state.get("failed_prerequisites", {}).pop("parent:sencache", None)
                        state.get("unknown_prerequisites", {}).pop("parent:sencache", None)
                        for tasks in state["tasks"].values():
                            for task in tasks.values():
                                if task["method"] == "sencache" and task["status"] == "blocked":
                                    task["status"] = "unsubmitted"; task.pop("error", None)
                    state.update(status="prepared", controller_job=None, updated_at=now())
                    state.pop("error", None)
                    transition = {"state": state, "unblocked_sencache": unblock, "audit": audit}
                    q.atomic_json(folder / "transition.json", transition)
                elif transition["unblocked_sencache"]:
                    audit = validate_recovered_profile(run_dir, plan, q)
                    if audit["profile_sha256"] != transition["audit"]["profile_sha256"]:
                        raise ValueError("Recovered profile changed during the state transition")
                if manifest.exists():
                    if q.file_hash(manifest) != plan["manifest_sha256"]:
                        raise ValueError("Scheduling manifest unexpectedly changed")
                else:
                    temporary = folder / "restore.tmp"
                    shutil.copy2(backup, temporary)
                    os.replace(temporary, manifest)
                if transition["audit"] is not None:
                    q.atomic_json(root / "completed-audit.json", transition["audit"])
                q.atomic_json(run_dir / "rolling/state.json", transition["state"])
                event.update(status="restored", restored_at=now(), unblocked_sencache=transition["unblocked_sencache"])
                q.atomic_json(event_path, event)
        # Starting the replacement while holding .controller.lock would make it
        # immediately fail its nonblocking singleton check on an idle CPU node.
        queue = q.query_queue(plan["original_plan"]["qos"])
        quota = q.query_qos(plan["original_plan"]["qos"])
        if q.quota_headroom(quota["max_submit"], queue) < 1:
            return {"status": "quota_wait"}
        result = q.submit_once(receipt, q.controller_command(run_dir, plan["rolling_plan"]),
                               {"kind": "sencache_recovery_resume", "cycle": cycle})
        if result.get("job_id"):
            event.update(status="resumed", controller_job=result["job_id"]); q.atomic_json(event_path, event)
        return result


def resume_until_accepted(run_dir, plan, q, cycle, unblock=False):
    while True:
        try:
            result = resume_original(run_dir, plan, q, cycle, unblock)
            if result.get("job_id"): return result
            update_state(run_dir, q, status=result["status"], stage="resume_" + cycle)
        except q.SchedulerUnavailable as error:
            update_state(run_dir, q, status="waiting_scheduler", stage="resume_" + cycle, error=str(error))
        time.sleep(POLL_SECONDS)


def manager_command(run_dir, plan, q):
    original = plan["original_plan"]
    policy = q.partition_time("cpup", original["qos"], q.query_qos(original["qos"]))
    body = "unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; exec " + shlex.join([
        "/usr/bin/python3", "-I", str(recovery_root(run_dir) / "code/recover_sencache.py"),
        "_run", "--run-dir", str(run_dir)])
    return ["sbatch", "--parsable", "--partition", "cpup", "--account", original["account"], "--qos", original["qos"],
            "--cpus-per-task", "1", "--mem", "1G", "--time", policy["time"], "--job-name", "ce6190-sen-recovery",
            "--chdir", str(run_dir), "--output", str(recovery_root(run_dir) / "manager-%j.log"),
            "--wrap", "exec bash -c " + shlex.quote(body)]


def gpu_command(run_dir, plan, q, index=None, dependency=None):
    original = plan["original_plan"]; partition = original["gpu_partition"]
    policy = q.partition_time(partition, original["qos"], q.query_qos(original["qos"]))
    root = recovery_root(run_dir)
    if index is None:
        name = "finalize"
        argv = [str(Path(run_dir) / "venv/bin/python"), "-I", str(Path(run_dir) / "code/scripts/start_orion.py"),
                "_calibration", "--run-dir", str(run_dir), "--method", "sencache", "--stage", "finalize",
                "--partition", partition, "--offset", "0"]
    else:
        name = "shard-" + str(index)
        argv = [str(Path(run_dir) / "venv/bin/python"), "-I", str(root / "code/sencache_recovery_worker.py"),
                "--run-dir", str(run_dir), "--sample-index", str(index),
                "--output", str(Path(run_dir) / "calibration/sencache" / f"shard-{index:04d}.json"),
                "--audit-output", str(root / "audits" / f"shard-{index:04d}.json")]
    body = ["set -euo pipefail", "unset PYTHONPATH PYTHONHOME"]
    for module in (original["python_module"], original["cuda_module"]): body.append("module load " + shlex.quote(module))
    body += ["unset PYTHONPATH PYTHONHOME", "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1", "exec " + shlex.join(argv)]
    command = ["sbatch", "--parsable", "--partition", partition, "--account", original["account"], "--qos", original["qos"],
               "--gres", "gpu:1", "--cpus-per-task", "8", "--mem", "64G", "--time", policy["time"],
               "--job-name", "ce6190-sen-recover-" + name, "--chdir", str(Path(run_dir) / "code"),
               "--output", str(root / (name + "-%j.log"))]
    if dependency: command += ["--dependency", dependency, "--kill-on-invalid-dep=yes"]
    return command + ["--wrap", "exec bash -lc " + shlex.quote("\n".join(body))]


def job_state(record, queue, accounting, q):
    identifier = record.get("job_id")
    if not identifier: return "unsubmitted"
    if identifier in queue: return "active"
    item = accounting.get(identifier)
    if item and item["state"] not in q.ACTIVE:
        return "complete" if item["state"] == "COMPLETED" and item["exit_code"] == "0:0" else "failed"
    return "unknown"


def submit_recovery_job(receipt, command, identity, q):
    """Only a conclusively rejected quota attempt may update a changed command.

    In particular, finalizer dependencies shrink as completed jobs leave Slurm's
    active controller memory. Accepted or uncertain submissions are never moved.
    """
    receipt = Path(receipt)
    previous = q.read_json(receipt)
    if (previous and previous.get("status") == "quota_wait" and not previous.get("job_id")
            and previous.get("command") != list(map(str, command))):
        if previous.get("identity") != identity:
            raise ValueError("Recovery submission identity changed")
        digest = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()[:16]
        archive = receipt.parent / "quota_rejections" / (receipt.stem + "-" + digest + ".json")
        q.atomic_json(archive, previous)
        receipt.unlink()
    return q.submit_once(receipt, command, identity)


def submit_gpu_tasks_tick(run_dir, plan, q):
    validate_frozen(run_dir, plan, q)
    root = recovery_root(run_dir); original = plan["original_plan"]
    queue = q.query_queue(original["qos"]); active = {item["job_id"] for item in queue}
    quota = q.query_qos(original["qos"])
    headroom = q.quota_headroom(quota["max_submit"], queue)
    records = {index: q.read_json(root / "jobs" / f"shard-{index}.json", {}) for index in TARGETS}
    accounting = q.query_accounting([record["job_id"] for record in records.values() if record.get("job_id") and record["job_id"] not in active])
    for index, record in records.items():
        if job_state(record, active, accounting, q) == "failed":
            raise RuntimeError(f"Recovery shard {index} failed; inspect {root / ('shard-' + str(index) + '-%j.log')}")
    for index in TARGETS:
        receipt = root / "jobs" / f"shard-{index}.json"
        record = records[index]
        if record.get("job_id"): continue
        if headroom < 1: return False
        result = submit_recovery_job(receipt, gpu_command(run_dir, plan, q, index),
                               {"stage": "shard", "index": index, "worker_sha256": plan["recovery_scripts"]["sencache_recovery_worker.py"]}, q)
        if not result.get("job_id"): return False
        records[index] = result; headroom -= 1
    receipt = root / "jobs/finalize.json"
    final = q.read_json(receipt, {})
    if final.get("job_id"): return True
    if headroom < 1: return False
    # Completed historical IDs need no dependency and may already have left
    # Slurm's active controller memory. Unknown acceptance is never inferred.
    active = {item["job_id"] for item in q.query_queue(original["qos"])}
    accounting = q.query_accounting([record["job_id"] for record in records.values() if record["job_id"] not in active])
    phases = {index: job_state(records[index], active, accounting, q) for index in TARGETS}
    if "failed" in phases.values(): raise RuntimeError("A recovery shard failed before finalizer submission")
    if "unknown" in phases.values(): return False
    for index, phase in phases.items():
        if phase == "complete":
            path = Path(run_dir) / "calibration/sencache" / f"shard-{index:04d}.json"
            data = q.read_json(path, {})
            if (data.get("method") != "sencache" or data.get("selection", {}).get("sample_indices") != [index]
                    or not data.get("recovery_decoder")):
                raise RuntimeError(f"Completed recovery job {index} has no valid recovered shard")
    dependencies = [records[index]["job_id"] for index in TARGETS if phases[index] != "complete"]
    dependency = "afterok:" + ":".join(dependencies) if dependencies else None
    result = submit_recovery_job(receipt, gpu_command(run_dir, plan, q, dependency=dependency),
                           {"stage": "finalize", "shard_job_ids": [records[index]["job_id"] for index in TARGETS]}, q)
    return bool(result.get("job_id"))


def start(args):
    run_dir = args.run_dir.expanduser().resolve(); repo = Path(__file__).resolve().parents[1]
    q = import_queue(run_dir); root = recovery_root(run_dir)
    if not args.submit:
        print(json.dumps({"status": "preview", "run_dir": str(run_dir), "target_indices": TARGETS,
                          "action": "brief controller maintenance, serial asset preparation, three shard retries, parity, Sen-only unblock",
                          "gpu_jobs_cancelled": False}, indent=2)); return
    root.mkdir(exist_ok=True)
    with (root / ".start.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        plan = prepare_recovery(run_dir, repo, q)
        state = get_state(run_dir, q)
        if state.get("status") in {"complete", "failed"}:
            print(json.dumps(state, indent=2)); return
        receipt = root / "manager-job.json"; old = q.read_json(receipt, {})
        if old.get("job_id"):
            active = {item["job_id"] for item in q.query_queue(plan["original_plan"]["qos"])}
            if old["job_id"] in active:
                result = {"status": "already_submitted", "manager_job": old["job_id"], "recovery": str(root)}
            else:
                terminal = q.query_accounting([old["job_id"]]).get(old["job_id"])
                result = {"status": "manager_needs_review", "manager_job": old["job_id"],
                          "accounting": terminal, "recovery": str(root),
                          "note": "Accepted manager is not active. Inspect recovery status/logs; no duplicate manager was submitted."}
            print(json.dumps(result, indent=2)); return
        wait_for_pause(run_dir, plan, q, "initial")
        queue = q.query_queue(plan["original_plan"]["qos"]); quota = q.query_qos(plan["original_plan"]["qos"])
        if q.quota_headroom(quota["max_submit"], queue) < 1:
            print(json.dumps({"status": "quota_wait", "note": "Original GPU jobs continue. Rerun the same recovery start to submit its manager."}, indent=2)); return
        result = q.submit_once(receipt, manager_command(run_dir, plan, q), {"stage": "manager", "recovery_scripts": plan["recovery_scripts"]})
        update_state(run_dir, q, status=result["status"], stage="manager_queued", manager_job=result.get("job_id"))
        print(json.dumps({"status": result["status"], "manager_job": result.get("job_id"), "recovery": str(root),
                          "status_command": shlex.join(["python3", str(repo / "scripts/recover_sencache.py"), "status", "--run-dir", str(run_dir)])}, indent=2))


def perform_recovery(run_dir, plan, q):
    root = recovery_root(run_dir)
    validate_frozen(run_dir, plan, q)
    # Release the old queue promptly: HTTP429 backoff must not hold up healthy
    # generation or H200 validation/refill while a CPU-only download waits.
    resume_until_accepted(run_dir, plan, q, "initial", unblock=False)
    assets = import_path("ce6190_sen_recovery_assets", root / "code/sencache_recovery_assets.py")
    update_state(run_dir, q, status="running", stage="assets")
    assets.prepare_assets(run_dir, audit_path=root / "assets.json",
                          progress_callback=lambda event: update_state(run_dir, q, status="running", stage="assets", asset_progress=event))
    update_state(run_dir, q, status="running", stage="calibration_submission_maintenance")
    wait_for_pause(run_dir, plan, q, "calibration")
    while not submit_gpu_tasks_tick(run_dir, plan, q): time.sleep(POLL_SECONDS)
    resume_until_accepted(run_dir, plan, q, "calibration", unblock=False)
    update_state(run_dir, q, status="running", stage="calibration_and_parity")
    final = q.read_json(root / "jobs/finalize.json")
    while True:
        queue = q.query_queue(plan["original_plan"]["qos"])
        active = {item["job_id"] for item in queue}
        accounting = q.query_accounting([] if final["job_id"] in active else [final["job_id"]])
        phase = job_state(final, active, accounting, q)
        if phase == "failed": raise RuntimeError("Recovery finalizer failed or was cancelled; inspect its log")
        if phase == "complete":
            validate_recovered_profile(run_dir, plan, q); break
        time.sleep(POLL_SECONDS)
    update_state(run_dir, q, status="running", stage="unblock_maintenance")
    wait_for_pause(run_dir, plan, q, "unblock")
    resume_until_accepted(run_dir, plan, q, "unblock", unblock=True)
    update_state(run_dir, q, status="complete", stage="sencache_unblocked", error=None)


def run_manager(args):
    run_dir = args.run_dir.expanduser().resolve(); root = recovery_root(run_dir); q = import_queue(run_dir)
    plan = q.read_json(root / "plan.json")
    with (root / ".manager.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                try:
                    perform_recovery(run_dir, plan, q)
                    break
                except q.SchedulerUnavailable as error:
                    update_state(run_dir, q, status="waiting_scheduler", error=str(error))
                    time.sleep(POLL_SECONDS)
        except Exception as error:
            update_state(run_dir, q, status="failed", stage="recovery_failed", error=repr(error), traceback=traceback.format_exc())
            # Return the healthy methods to their original controller whenever
            # our own maintenance still holds its scheduling manifest aside.
            for cycle in ("initial", "calibration", "unblock"):
                event = q.read_json(root / "maintenance" / cycle / "event.json", {})
                if event and event.get("status") not in {"resumed"}:
                    try: resume_until_accepted(run_dir, plan, q, cycle, unblock=False)
                    except Exception as restore_error:
                        update_state(run_dir, q, status="failed", resume_error=repr(restore_error))
            raise


def status(args):
    run_dir = args.run_dir.expanduser().resolve(); q = import_queue(run_dir); root = recovery_root(run_dir)
    state = get_state(run_dir, q)
    result = {"recovery": state or {"status": "not_started"},
              "manager": q.read_json(root / "manager-job.json"),
              "jobs": {path.stem: q.read_json(path) for path in sorted((root / "jobs").glob("*.json"))},
              "queue": q.summary(q.read_json(run_dir / "rolling/plan.json"), q.read_json(run_dir / "rolling/state.json")),
              "logs": str(root)}
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start_parser = sub.add_parser("start"); start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument("--submit", action="store_true")
    for command in ("status", "_run"):
        child = sub.add_parser(command); child.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    {"start": start, "status": status, "_run": run_manager}[args.action](args)


if __name__ == "__main__": main()
