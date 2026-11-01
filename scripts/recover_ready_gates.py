#!/usr/bin/env python3
"""Repair stale missing-state blocks without changing experiment code or inputs.

The adapter retains the original immutable queue and hardware helper. A short
CPU-controller handover occurs only after the existing Sen recovery manager has
terminated. GPU jobs and completed results are never cancelled or overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import time
import traceback

POLL_SECONDS = 10
CYCLE = "ready-gates"


def root(run_dir):
    return Path(run_dir) / "ready_gates_recovery"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_original(run_dir):
    run_dir = Path(run_dir)
    rolling = json.loads((run_dir / "rolling/plan.json").read_text())
    for name, expected in rolling["script_hashes"].items():
        if digest(run_dir / "rolling/code" / name) != expected:
            raise ValueError("Original immutable queue code changed")
    sen = json.loads((run_dir / "sencache_recovery/plan.json").read_text())
    recovery_path = run_dir / "sencache_recovery/code/recover_sencache.py"
    if digest(recovery_path) != sen["recovery_scripts"]["recover_sencache.py"]:
        raise ValueError("Original maintenance implementation changed")
    q = load_module("ready_original_queue", run_dir / "rolling/code/queue_orion.py")
    r = load_module("ready_original_maintenance", recovery_path)
    return q, r


def update(run_dir, q, **values):
    state = q.read_json(root(run_dir) / "state.json", {})
    state.update(values, updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
    q.atomic_json(root(run_dir) / "state.json", state)
    print(json.dumps(state), flush=True)
    return state


def safe_phase_state(path):
    """Unavailable phase files wait; a successfully parsed failure stays failed.

    This function is NEVER installed for submission receipts or queue state.
    """
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("state must be an object")
        return value
    except FileNotFoundError:
        return {"status": "not_started"}
    except (OSError, ValueError) as error:
        return {"status": "read_unavailable", "error": f"Waiting to read state {path}: {error}"}


def validate_plan(run_dir, plan, q):
    for name, expected in plan["frozen_files"].items():
        if q.file_hash(Path(run_dir) / name) != expected:
            raise ValueError(f"Frozen readiness-recovery input changed: {name}")
    if q.file_hash(root(run_dir) / "code/recover_ready_gates.py") != plan["adapter_sha256"]:
        raise ValueError("Readiness adapter snapshot changed")


def prepare(run_dir, repo, q, r):
    run_dir = Path(run_dir).resolve(); repo = Path(repo).resolve()
    previous = q.read_json(root(run_dir) / "plan.json")
    if previous:
        validate_plan(run_dir, previous, q)
        return previous
    sen = q.read_json(run_dir / "sencache_recovery/plan.json")
    predecessor = q.read_json(run_dir / "sencache_recovery/manager-job.json")
    if not predecessor or predecessor.get("status") != "accepted" or not str(predecessor.get("job_id", "")).isdigit():
        raise ValueError("A recorded Sen recovery manager is required before scheduling this handover")
    r.validate_frozen(run_dir, sen, q)
    starter = load_module("ready_original_starter", run_dir / "code/scripts/start_orion.py")
    starter.compatible_runtime(repo, run_dir)
    rolling = q.read_json(run_dir / "rolling/plan.json")
    original = q.read_json(run_dir / "plan.json")
    files = set(sen["frozen_files"])
    files.update({"sencache_recovery/plan.json", "sencache_recovery/manager-job.json",
                  "sencache_recovery/code/recover_sencache.py"})
    files.update("sencache_recovery/code/" + name for name in sen["recovery_scripts"])
    folder = root(run_dir) / "code"; folder.mkdir(parents=True, exist_ok=True)
    source = repo / "scripts/recover_ready_gates.py"
    destination = folder / source.name
    if destination.exists() and digest(destination) != digest(source):
        raise ValueError("Partial readiness adapter snapshot differs")
    if not destination.exists(): shutil.copy2(source, destination)
    plan = {"schema_version": 1, "run_dir": str(run_dir), "adapter_sha256": digest(destination),
            "predecessor_job": str(predecessor["job_id"]), "original_plan": original,
            "rolling_plan": rolling, "manifest_sha256": rolling["manifest_sha256"],
            "frozen_files": {name: q.file_hash(run_dir / name) for name in sorted(files)},
            "policy": "same immutable science; recover only exact stale ENOENT blocks after validated passed checks"}
    q.atomic_json(root(run_dir) / "plan.json", plan)
    update(run_dir, q, status="prepared", stage="waiting_sen_manager")
    return plan


def predecessor_state(plan, q):
    identifier = plan["predecessor_job"]
    if identifier in {row["job_id"] for row in q.query_queue(plan["original_plan"]["qos"])}:
        return "active"
    record = q.query_accounting([identifier]).get(identifier)
    if record and record["state"] not in q.ACTIVE:
        return "terminal"
    return "unknown"


def wrapper_command(run_dir, original_command, action):
    command = list(original_command)
    body = "unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; exec " + shlex.join([
        "/usr/bin/python3", "-I", str(root(run_dir) / "code/recover_ready_gates.py"), action,
        "--run-dir", str(run_dir)])
    command[command.index("--wrap") + 1] = "exec bash -c " + shlex.quote(body)
    return command


def manager_command(run_dir, plan, q, dependency=None):
    original = plan["original_plan"]
    policy = q.partition_time("cpup", original["qos"], q.query_qos(original["qos"]))
    command = ["sbatch", "--parsable", "--partition", "cpup", "--account", original["account"],
               "--qos", original["qos"], "--cpus-per-task", "1", "--mem", "1G", "--time", policy["time"],
               "--job-name", "ce6190-ready-recovery", "--chdir", str(run_dir),
               "--output", str(root(run_dir) / "manager-%j.log")]
    if dependency: command += ["--dependency", dependency]
    return wrapper_command(run_dir, command + ["--wrap", "placeholder"], "_run")


def failure_keys(partition, method, original_partition):
    keys = ["parent:env", partition + ":smoke"]
    if partition == original_partition: keys += ["parent:smoke"]
    if method in {"magcache", "sencache"}: keys += ["parent:" + method, partition + ":" + method]
    return keys


def reconcile_stale_blocks(run_dir, state, rolling_plan, q, helper):
    """Called only inside the original controller lock, before its normal tick."""
    run_dir = Path(run_dir)
    if q.file_hash(run_dir / "rolling/manifest.jsonl") != rolling_plan["manifest_sha256"]:
        raise ValueError("Immutable experiment manifest changed")
    total = 0
    for partition, tasks in state["tasks"].items():
        # Additional-hardware check receipts establish this repair's ownership.
        if partition == rolling_plan["original_plan"]["gpu_partition"]: continue
        for method in sorted({task["method"] for task in tasks.values()}):
            if any(key in state.get("failed_prerequisites", {}) for key in
                   failure_keys(partition, method, rolling_plan["original_plan"]["gpu_partition"])):
                continue
            stage = method if method in {"magcache", "sencache"} else "smoke"
            path = helper.state_path(run_dir, partition, stage)
            reason = f"Unreadable state {path}: [Errno 2] No such file or directory: {str(path)!r}"
            candidates = {identifier: task for identifier, task in tasks.items()
                          if task["method"] == method and task["status"] == "blocked" and task.get("error") == reason}
            if not candidates: continue
            if helper.group_gate(run_dir, partition, method).get("status") != "passed": continue
            try:
                raw = path.read_bytes(); check = json.loads(raw)
                current_identity = helper.input_identity(run_dir, partition, stage)
            except (OSError, ValueError):
                continue  # A second transient read cannot manufacture success.
            if check.get("status") != "passed" or check.get("input_identity") != current_identity:
                continue
            receipt_path = run_dir / "rolling/checks" / f"{partition}-{stage}.json"
            receipt = q.read_json(receipt_path, {})
            expected_identity = {"partition": partition, "stage": stage,
                                 "runtime_dir": str(helper.runtime_path(run_dir, partition)), "state_file": str(path)}
            if (receipt.get("status") != "accepted" or not receipt.get("job_id")
                    or str(check.get("job_id")) != receipt["job_id"] or receipt.get("identity") != expected_identity):
                raise ValueError("Passed hardware check does not match its accepted submission receipt")
            if any(task.get("job_id") or task.get("batch") for task in candidates.values()):
                raise ValueError("Blocked candidate already carries an experiment submission identity")
            for batch in (run_dir / "rolling/batches").glob("*/receipt.json"):
                record = q.read_json(batch)
                if record.get("status") in {"submitting", "uncertain"} and not record.get("job_id"):
                    raise RuntimeError(f"Uncertain experiment submission: {batch}")
                identity = record.get("identity", {})
                if (record.get("job_id") and identity.get("partition") == partition
                        and set(candidates).intersection(identity.get("run_ids", []))):
                    raise ValueError("Stale-block candidate already has an accepted experiment receipt")
            proof = {"partition": partition, "method": method, "run_ids": sorted(candidates),
                     "previous_error": reason, "check_path": str(path),
                     "check_sha256": hashlib.sha256(raw).hexdigest(), "check_job_id": receipt["job_id"],
                     "check_receipt_sha256": q.file_hash(receipt_path), "input_identity": current_identity,
                     "adapter_sha256": q.file_hash(root(run_dir) / "code/recover_ready_gates.py")}
            token = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
            audit_path = root(run_dir) / "reactivations" / (token + ".json")
            old = q.read_json(audit_path)
            if old is not None and old != proof: raise ValueError("Readiness reactivation audit differs")
            if old is None: q.atomic_json(audit_path, proof)
            for task in candidates.values():
                task["status"] = "unsubmitted"; task.pop("error", None)
                task["readiness_recovery_audit"] = str(audit_path)
            total += len(candidates)
    return total


def install_adapter(run_dir, plan, q):
    validate_plan(run_dir, plan, q)
    old_hardware, old_tick, old_command = q.hardware_module, q.tick, q.controller_command
    def hardware(directory):
        helper = old_hardware(directory)
        helper._state = safe_phase_state
        return helper
    def tick(directory, state, rolling_plan, hardware=None):
        # Preserve the old controller's very first guard for future maintenance.
        if q.file_hash(Path(directory) / "rolling/manifest.jsonl") != rolling_plan["manifest_sha256"]:
            raise ValueError("Immutable experiment manifest changed")
        validate_plan(directory, plan, q)
        helper = hardware or q.hardware_module(directory)
        count = reconcile_stale_blocks(directory, state, rolling_plan, q, helper)
        if count:
            q.atomic_json(Path(directory) / "rolling/state.json", state)
            print(json.dumps({"readiness_reactivated": count}), flush=True)
        return old_tick(directory, state, rolling_plan, hardware=helper)
    q.hardware_module = hardware
    q.tick = tick
    q.controller_command = lambda directory, rolling_plan, dependency=None: wrapper_command(
        directory, old_command(directory, rolling_plan, dependency), "_controller")
    return q


def start(args):
    run_dir = args.run_dir.expanduser().resolve(); q, r = import_original(run_dir)
    if not args.submit:
        print(json.dumps({"status": "preview", "action": "wait for Sen recovery, then repair stale readiness blocks",
                          "gpu_jobs_cancelled": False})); return
    root(run_dir).mkdir(exist_ok=True)
    with (root(run_dir) / ".start.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        plan = prepare(run_dir, Path(__file__).resolve().parents[1], q, r)
        receipt = root(run_dir) / "manager-job.json"
        previous = q.read_json(receipt, {})
        if previous.get("job_id"):
            print(json.dumps({"status": "already_submitted", "manager_job": previous["job_id"],
                              "note": "Use status to inspect this accepted manager; no duplicate was submitted."})); return
        while True:
            try:
                phase = predecessor_state(plan, q)
                if phase == "unknown":
                    update(run_dir, q, status="waiting_scheduler", stage="predecessor_unknown")
                elif q.quota_headroom(q.query_qos(plan["original_plan"]["qos"])["max_submit"],
                                      q.query_queue(plan["original_plan"]["qos"])) < 1:
                    update(run_dir, q, status="quota_wait", stage="manager_submission")
                else:
                    dependency = "afterany:" + plan["predecessor_job"] if phase == "active" else None
                    result = r.submit_recovery_job(receipt, manager_command(run_dir, plan, q, dependency),
                                                   {"stage": "readiness_manager", "adapter_sha256": plan["adapter_sha256"]}, q)
                    if result.get("job_id"):
                        update(run_dir, q, status="accepted", stage="manager_queued", manager_job=result["job_id"], error=None)
                        return
            except q.SchedulerUnavailable as error:
                update(run_dir, q, status="waiting_scheduler", error=str(error))
            time.sleep(POLL_SECONDS)


def run_manager(args):
    run_dir = args.run_dir.expanduser().resolve(); q, r = import_original(run_dir)
    plan = q.read_json(root(run_dir) / "plan.json"); validate_plan(run_dir, plan, q)
    r.recovery_root = root  # Process-local journals; never edit Sen's records.
    with (root(run_dir) / ".manager.lock").open("a") as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if predecessor_state(plan, q) == "terminal": break
                update(run_dir, q, status="waiting_sen_manager", stage="predecessor")
            except q.SchedulerUnavailable as error:
                update(run_dir, q, status="waiting_scheduler", error=str(error))
            time.sleep(POLL_SECONDS)
        with (run_dir / "sencache_recovery/.manager.lock").open("a") as sen_lock:
            fcntl.flock(sen_lock, fcntl.LOCK_EX)
            validate_plan(run_dir, plan, q)
            # The manager cannot still be alive merely because its state file
            # says complete: scheduler terminal AND its exclusive lock are used.
            install_adapter(run_dir, plan, q)
            try:
                r.wait_for_pause(run_dir, plan, q, CYCLE)
                result = r.resume_until_accepted(run_dir, plan, q, CYCLE, unblock=False)
                update(run_dir, q, status="installed", stage="adapter_controller", controller_job=result["job_id"], error=None)
            except Exception as error:
                update(run_dir, q, status="failed", error=repr(error), traceback=traceback.format_exc())
                event = q.read_json(root(run_dir) / "maintenance" / CYCLE / "event.json", {})
                if event and event.get("status") != "resumed":
                    # Retry only the SAME guarded handover. Accepted/uncertain
                    # receipts cannot be replaced with another command, and an
                    # unrelated pause still fails the original ownership check.
                    try:
                        restored = r.resume_until_accepted(run_dir, plan, q, CYCLE, unblock=False)
                        update(run_dir, q, status="failed", error=repr(error),
                               resumed_controller_job=restored.get("job_id"))
                    except Exception as restore_error:
                        update(run_dir, q, status="failed", error=repr(error), resume_error=repr(restore_error))
                raise


def run_controller(args):
    run_dir = args.run_dir.expanduser().resolve(); q, _ = import_original(run_dir)
    plan = q.read_json(root(run_dir) / "plan.json")
    install_adapter(run_dir, plan, q)
    q.run_controller(args)


def status(args):
    run_dir = args.run_dir.expanduser().resolve(); q, _ = import_original(run_dir)
    audits = [q.read_json(path) for path in sorted((root(run_dir) / "reactivations").glob("*.json"))]
    state = q.read_json(run_dir / "rolling/state.json")
    restored = sum(bool(task.get("readiness_recovery_audit")) for tasks in state["tasks"].values()
                   for task in tasks.values())
    print(json.dumps({"recovery": q.read_json(root(run_dir) / "state.json", {"status": "not_started"}),
                      "manager": q.read_json(root(run_dir) / "manager-job.json"),
                      "queue": q.summary(q.read_json(run_dir / "rolling/plan.json"), state),
                      "reactivation_audits": len(audits), "reactivated_tasks_recorded": restored,
                      "adapter_sha256": q.read_json(root(run_dir) / "plan.json", {}).get("adapter_sha256"),
                      "logs": str(root(run_dir))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for command in ("start", "status", "_run", "_controller"):
        child = sub.add_parser(command); child.add_argument("--run-dir", type=Path, required=True)
        if command == "start": child.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    {"start": start, "status": status, "_run": run_manager, "_controller": run_controller}[args.action](args)


if __name__ == "__main__": main()
