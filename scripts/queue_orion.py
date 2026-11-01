#!/usr/bin/env python3
"""Quota-aware, durable Slurm submission for an existing CE6190 runtime.

One CPU controller and one waiting successor keep the queue progressing across
SSH disconnects and controller wall-time limits. They use no GPU. GPU jobs keep
the original scientific code and one-GPU/8-CPU/64-GiB resource request. The
controller never changes QoS limits, cancels other jobs, or retries failed runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

METHODS = ("baseline", "teacache", "seacache", "magcache", "sencache")
POLL_SECONDS = 30
TERMINAL = {"complete", "complete_with_failures", "paused"}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "STAGE_OUT"}


class SchedulerUnavailable(RuntimeError):
    """A read-only scheduler query failed; wait without submitting anything."""


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def clean_sbatch_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}


def query(command):
    try:
        return subprocess.run(list(map(str, command)), check=True, text=True,
                              capture_output=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise SchedulerUnavailable(f"Scheduler query failed: {shlex.join(map(str, command))}: {error}") from error


def parse_duration(value):
    """Slurm time syntax; bare integers mean minutes, None means unlimited."""
    value = str(value).strip()
    if value.upper() in {"", "UNLIMITED", "INFINITE", "NONE", "N/A"}: return None
    days, rest = (value.split("-", 1) if "-" in value else ("0", value))
    if not days.isdigit(): raise ValueError(f"Invalid Slurm duration: {value}")
    parts = rest.split(":")
    if not all(p.isdigit() for p in parts) or len(parts) > 3:
        raise ValueError(f"Invalid Slurm duration: {value}")
    numbers = list(map(int, parts))
    if len(numbers) == 1: seconds = numbers[0] * (3600 if "-" in value else 60)
    elif len(numbers) == 2: seconds = numbers[0] * (3600 if "-" in value else 60) + numbers[1] * (60 if "-" in value else 1)
    else: seconds = numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return int(days) * 86400 + seconds


def format_duration(seconds):
    if seconds is None: return "0"
    if seconds < 1: raise ValueError("Partition/QoS allows no positive job duration")
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600); minutes, seconds = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_qos(output, qos):
    matches = [line.strip().split("|") for line in output.splitlines()
               if line.strip().split("|", 1)[0] == qos]
    if len(matches) != 1 or len(matches[0]) < 3:
        raise ValueError("Cannot identify the requested QoS and its submit limit")
    _, maximum, wall, *_ = matches[0]
    if not maximum.isdigit() or int(maximum) < 1:
        raise ValueError("QoS MaxSubmitJobsPU must be an explicitly readable positive limit")
    return {"max_submit": int(maximum), "max_wall_seconds": parse_duration(wall)}


def query_qos(qos):
    return parse_qos(query(["sacctmgr", "-n", "-P", "show", "qos", qos,
                            "format=Name,MaxSubmitJobsPU,MaxWall"]), qos)


def parse_queue(output, qos):
    records = []
    for line in output.splitlines():
        if not line.strip(): continue
        fields = line.strip().split("|", 4)
        if len(fields) != 5: raise ValueError("Unexpected squeue output; refusing to infer free slots")
        identifier, actual_qos, state, name, reason = [field.strip() for field in fields]
        if actual_qos != qos: continue
        if not re.fullmatch(r"\d+(?:_\d+)?", identifier):
            raise ValueError("squeue did not expand array elements; cannot count submit slots")
        records.append({"job_id": identifier, "qos": actual_qos, "state": state,
                        "name": name, "reason": reason})
    if len({row["job_id"] for row in records}) != len(records):
        raise ValueError("Duplicate squeue job IDs")
    return records


def query_queue(qos):
    return parse_queue(query(["squeue", "-r", "-h", "-u", getpass.getuser(),
                              "-o", "%i|%q|%T|%j|%r"]), qos)


def query_accounting(identifiers):
    identifiers = list(dict.fromkeys(map(str, identifiers)))
    if not identifiers: return {}
    output = query(["sacct", "-X", "-n", "-P", "--array", "-j", ",".join(identifiers),
                    "--format=JobID%80,State%40,ExitCode"])
    result = {}
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] in identifiers:
            result[fields[0]] = {"state": fields[1].split()[0], "exit_code": fields[2]}
    return result


def quota_headroom(limit, queue):
    if type(limit) is not int or limit < 1: raise ValueError("Invalid submit quota")
    return max(0, limit - len(queue))


def output_bytes(row):
    """One retained uint8 video, NPY header, plus estimated MP4/trace headroom.

    generate.py saves only the final timing repeat. The 15% allowance is an
    estimate, not a filesystem reservation or a guarantee against other writers.
    """
    raw = 3
    for name in ("num_frames", "height", "width"):
        value = row[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid output geometry: {name}={value!r}")
        raw *= value
    return ((raw + 1024) * 115 + 99) // 100


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def storage_snapshot(run_dir, plan, state, by_id, helper):
    """Budget future writes across hardware grids sharing a filesystem.

    Submitted tasks keep their full allowance until completion, even if some
    output is already on disk. This intentionally errs toward over-reservation.
    Unfinished smoke/parity pairs also retain room, so generation cannot consume
    the space needed by those checks. No output is deleted or recompressed.
    """
    filesystems = {}; partition_devices = {}; check_devices = {}
    def volume_for(path):
        anchor = existing_parent(path); device = str(anchor.stat().st_dev)
        free = shutil.disk_usage(anchor).free
        volume = filesystems.setdefault(device, {"device": device, "paths": [],
            "free_bytes": free, "generation_reserved_bytes": 0, "check_reserved_bytes": 0})
        volume["free_bytes"] = min(volume["free_bytes"], free)
        if str(path) not in volume["paths"]: volume["paths"].append(str(path))
        return device, volume
    geometry = read_json(Path(run_dir) / "code/configs/correctness_smoke.json")["generation"]
    pair_bytes = 2 * output_bytes(geometry)
    for partition in plan["partitions"]:
        runtime = helper.runtime_path(run_dir, partition)
        destination = runtime / ("development_" + partition) / "results"
        device, volume = volume_for(destination)
        partition_devices[partition] = device
        volume["generation_reserved_bytes"] += sum(output_bytes(by_id[identifier])
            for identifier, task in state["tasks"][partition].items() if task["status"] == "submitted")
        check_device, check_volume = volume_for(runtime)
        check_devices[partition] = check_device
        # These three existing starter routines each save a baseline/method pair.
        for stage in ("smoke", "magcache", "sencache"):
            path = runtime / ("smoke.json" if stage == "smoke" else f"calibration/{stage}/check.json")
            try: phase = read_json(path, {}).get("status")
            except json.JSONDecodeError:
                # Original starter phase JSON is written directly; a transient
                # partial write must retain its allowance, not stop the queue.
                phase = None
            prefix = "parent:" if partition == plan["original_plan"]["gpu_partition"] else partition + ":"
            failure_keys = {"parent:env", prefix + "smoke", prefix + stage}
            if stage != "smoke": failure_keys.update({"parent:smoke", "parent:" + stage})
            blocked = any(key in state.get("failed_prerequisites", {}) for key in failure_keys)
            if phase not in {"passed", "failed", "cancelled", "blocked"} and not blocked:
                check_volume["check_reserved_bytes"] += pair_bytes
    for volume in filesystems.values():
        volume["available_bytes"] = max(0, volume["free_bytes"] - volume["generation_reserved_bytes"] - volume["check_reserved_bytes"])
        volume["free_gib"] = volume["free_bytes"] / 2**30
        volume["reserved_gib"] = (volume["generation_reserved_bytes"] + volume["check_reserved_bytes"]) / 2**30
        volume["available_gib"] = volume["available_bytes"] / 2**30
    return {"scope": "filesystem_free_space_not_user_or_project_quota",
            "whole_grid_remaining_raw_gib_at_start": plan.get("remaining_raw_frames_gib"),
            "filesystems": list(filesystems.values()), "partition_devices": partition_devices,
            "check_devices": check_devices,
            "note": "Estimates retain all raw frames; calibration downloads, environment writes and other users can consume additional space."}


def storage_available(snapshot):
    return {volume["device"]: volume["available_bytes"] for volume in snapshot["filesystems"]}


def partition_time(partition, qos, qos_limits):
    output = query(["scontrol", "-o", "show", "partition", partition])
    fields = dict(token.split("=", 1) for token in output.split() if "=" in token)
    if fields.get("PartitionName") != partition or "MaxTime" not in fields:
        raise ValueError(f"Cannot read partition policy for {partition}")
    allowed = fields.get("AllowQos", "ALL").split(",")
    if "ALL" not in allowed and qos not in allowed:
        raise ValueError(f"Partition {partition} does not allow QoS {qos}")
    limits = [parse_duration(fields["MaxTime"]), qos_limits["max_wall_seconds"]]
    finite = [value for value in limits if value is not None]
    seconds = min(finite) if finite else None
    return {"time": format_duration(seconds), "seconds": seconds,
            "partition_max_time": fields["MaxTime"], "qos_max_wall_seconds": qos_limits["max_wall_seconds"]}


def submit_once(receipt, command, identity):
    """A known rejection can retry; an uncertain acceptance never retries itself."""
    receipt = Path(receipt)
    previous = read_json(receipt)
    if previous:
        if previous.get("identity") != identity or previous.get("command") != list(map(str, command)):
            raise ValueError(f"Submission identity changed: {receipt}")
        if previous.get("job_id"): return previous
        if previous.get("status") in {"submitting", "uncertain"}:
            raise RuntimeError(f"Submission outcome needs reconciliation: {receipt}")
        if previous.get("status") == "rejected":
            raise RuntimeError(f"Previous non-quota rejection needs correction: {receipt}")
    record = {"identity": identity, "command": list(map(str, command)), "status": "submitting", "job_id": None,
              "attempts": (previous or {}).get("attempts", 0) + 1}
    atomic_json(receipt, record)
    try:
        result = subprocess.run(record["command"], text=True, capture_output=True, timeout=60,
                                env=clean_sbatch_env())
        if result.returncode:
            record["stderr"] = result.stderr; record["returncode"] = result.returncode
            if any(code in result.stderr for code in ("QOSMaxSubmitJobPerUserLimit", "AssocMaxSubmitJobLimit",
                                                      "QOSMaxSubmitJobPerAccountLimit", "QOSGrpSubmitJobsLimit")):
                record["status"] = "quota_wait"; atomic_json(receipt, record); return record
            record["status"] = "rejected"; atomic_json(receipt, record)
            raise RuntimeError(f"Slurm rejected submission: {result.stderr.strip()}")
        identifier = result.stdout.strip().split(";", 1)[0]
        if not identifier.isdigit(): raise ValueError(f"Unrecognized sbatch response: {result.stdout!r}")
        record.update(status="accepted", job_id=identifier); atomic_json(receipt, record)
        return record
    except (subprocess.TimeoutExpired, OSError, ValueError) as error:
        record.update(status="uncertain", error=repr(error)); atomic_json(receipt, record)
        raise RuntimeError(f"Submission result is uncertain; inspect {receipt}") from error


def load_plan(run_dir):
    return read_json(Path(run_dir) / "rolling/plan.json")


def verify_old_receipts(run_dir):
    for path in Path(run_dir).glob("development_*/results/slurm*_submission.json"):
        previous = read_json(path)
        if previous.get("submitted_jobs") or any(item.get("job_id") for item in previous.get("arrays", [])):
            raise ValueError(f"An older launcher already submitted experiment jobs: {path}; reconcile before migration")
        if any(item.get("status") in {"submitting", "uncertain"} for item in previous.get("arrays", [])):
            raise ValueError(f"An older submission outcome is uncertain: {path}")


def prepare_plan(run_dir, partitions, repo):
    run_dir = Path(run_dir).expanduser().resolve(); repo = Path(repo).resolve()
    starter = import_path("ce6190_original_starter", run_dir / "code/scripts/start_orion.py")
    starter.verified_release(run_dir / "code")
    starter.compatible_runtime(repo, run_dir)
    original = read_json(run_dir / "plan.json")
    partitions = list(dict.fromkeys(partitions or [original["gpu_partition"]]))
    if not partitions or original["gpu_partition"] not in partitions:
        raise ValueError("Keep the original verified partition in this rollout")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in partitions):
        raise ValueError("Invalid partition name")
    verify_old_receipts(run_dir)
    source = run_dir / ("development_" + original["gpu_partition"]) / "all_methods.jsonl"
    content = source.read_text(); rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    identifiers = [row.get("run_id") for row in rows]
    if not rows or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError("Experiment manifest has invalid or duplicate run IDs")
    if {row.get("method") for row in rows} != set(METHODS):
        raise ValueError("Expected the agreed five-method experiment manifest")
    baselines = {row["run_id"] for row in rows if row["method"] == "baseline"}
    for row in rows:
        if row["method"] != "baseline" and row.get("reference_run_id") not in baselines:
            raise ValueError("Missing uncached pairing in experiment manifest")
        if row["method"] in {"magcache", "sencache"}:
            expected = (run_dir / "calibration" / row["method"] / "profile.json").resolve()
            if Path(row.get("calibration_path", "")).resolve() != expected:
                raise ValueError("Unexpected calibration profile path")
    helper = import_path("ce6190_prepare_hardware", repo / "scripts/orion_hardware.py")
    remaining_bytes = 0
    for partition in partitions:
        destination = helper.runtime_path(run_dir, partition) / ("development_" + partition) / "results"
        for row in rows:
            directory = destination / row["run_id"]
            try: completed = read_json(directory / "metrics.json", {}).get("status") == "complete"
            except (OSError, ValueError): completed = False
            if not completed or not (directory / "frames.npy").is_file():
                remaining_bytes += row["num_frames"] * row["height"] * row["width"] * 3
    free_bytes = shutil.disk_usage(run_dir).free
    limits = query_qos(original["qos"])
    policy = {partition: partition_time(partition, original["qos"], limits)
              for partition in dict.fromkeys(["cpup", *partitions])}
    return {"schema_version": 1, "run_dir": str(run_dir), "source_manifest": str(source),
            "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(), "rows_per_partition": len(rows),
            "partitions": partitions, "account": original["account"], "qos": original["qos"],
            "controller_partition": "cpup", "partition_policy": policy,
            "original_plan": original, "poll_seconds": POLL_SECONDS,
            "remaining_raw_frames_gib": remaining_bytes / 2**30,
            "filesystem_free_gib_at_start": free_bytes / 2**30,
            "storage_policy": "rolling admission; reserve unfinished writes, retain every raw frame; wait and retry if storage is insufficient",
            "control_slots": "one running CPU controller plus one afterany successor; neither requests GPU",
            "script_hashes": {name: file_hash(repo / "scripts" / name)
                              for name in ("queue_orion.py", "orion_hardware.py")}}


def hardware_module(run_dir):
    return import_path("ce6190_hardware", Path(run_dir) / "rolling/code/orion_hardware.py")


def fresh_state(plan, rows):
    return {"status": "prepared", "tasks": {partition: {row["run_id"]: {"method": row["method"], "status": "unsubmitted"}
             for row in rows} for partition in plan["partitions"]}, "cursor": 0, "controller_job": None,
            "last_quota": None, "checks": {}, "batches": [], "updated_at": None}


def controller_command(run_dir, plan, dependency=None):
    command = ["sbatch", "--parsable", "--partition", plan["controller_partition"], "--account", plan["account"],
               "--qos", plan["qos"], "--cpus-per-task", "1", "--mem", "1G", "--job-name", "ce6190-queue",
               "--chdir", str(run_dir), "--output", str(run_dir / "rolling/controller-%j.log"),
               "--time", plan["partition_policy"][plan["controller_partition"]]["time"]]
    if dependency: command += ["--dependency", dependency]
    body = "unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; exec " + shlex.join([
        "/usr/bin/python3", "-I", str(run_dir / "rolling/code/queue_orion.py"), "_run", "--run-dir", str(run_dir)])
    return command + ["--wrap", "exec bash -c " + shlex.quote(body)]


def start(args):
    run_dir = args.run_dir.expanduser().resolve(); repo = Path(__file__).resolve().parents[1]
    if not args.submit:
        plan = prepare_plan(run_dir, args.partitions, repo)
        print(json.dumps({"status": "preview_only", **plan}, indent=2)); return
    rolling = run_dir / "rolling"; rolling.mkdir(exist_ok=True)
    with (rolling / ".start.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        previous = load_plan(run_dir)
        if previous:
            if list(dict.fromkeys(args.partitions or previous["partitions"])) != previous["partitions"]:
                raise ValueError("Partition selection changed; use the existing rollout")
            plan = previous
            for name, digest in plan["script_hashes"].items():
                if file_hash(rolling / "code" / name) != digest:
                    raise ValueError("Immutable rolling launcher changed")
            state = read_json(rolling / "state.json")
            if state["status"] in TERMINAL:
                print(json.dumps({"status": state["status"], "note": "Inspect status/logs; failed work is not silently retried."}, indent=2)); return
        else:
            plan = prepare_plan(run_dir, args.partitions, repo)
            (rolling / "code").mkdir(exist_ok=True)
            for name, digest in plan["script_hashes"].items():
                target = rolling / "code" / name
                if target.exists() and file_hash(target) != digest:
                    raise ValueError("A partial rolling snapshot differs; inspect before replacing it")
                if not target.exists(): shutil.copy2(repo / "scripts" / name, target)
            if any(file_hash(rolling / "code" / name) != digest for name, digest in plan["script_hashes"].items()):
                raise ValueError("Launcher changed during snapshot")
            rows = [json.loads(line) for line in Path(plan["source_manifest"]).read_text().splitlines() if line.strip()]
            (rolling / "manifest.jsonl").write_text(Path(plan["source_manifest"]).read_text())
            helper = hardware_module(run_dir)
            for partition in plan["partitions"]: helper.prepare(run_dir, partition)
            if not (rolling / "state.json").exists(): atomic_json(rolling / "state.json", fresh_state(plan, rows))
            atomic_json(rolling / "plan.json", plan)
        queue = query_queue(plan["qos"]); active = {row["job_id"] for row in queue}
        receipts = sorted((rolling / "controllers").glob("*.json"))
        absent_controllers = []
        for receipt in receipts:
            record = read_json(receipt)
            if record.get("status") in {"submitting", "uncertain"} and not record.get("job_id"):
                raise RuntimeError(f"Prior controller submission is uncertain: {receipt}")
            if record.get("job_id") in active:
                print(json.dumps({"status": "already_submitted", "controller_job": record["job_id"],
                                  "run_dir": str(run_dir)}, indent=2)); return
            if record.get("job_id"): absent_controllers.append(record["job_id"])
        accounting = query_accounting(absent_controllers)
        for identifier in absent_controllers:
            entry = accounting.get(identifier)
            if entry is None or entry["state"] in ACTIVE:
                raise RuntimeError(f"Accepted controller {identifier} is not yet conclusively terminal; do not resubmit")
        quota = query_qos(plan["qos"])
        if quota_headroom(quota["max_submit"], queue) < 1:
            print(json.dumps({"status": "quota_wait", "note": "No free submit slot for the CPU controller; rerun this same start command later."}, indent=2)); return
        receipt = rolling / "controllers" / (f"{len(receipts):06d}.json")
        result = submit_once(receipt, controller_command(run_dir, plan), {"kind": "controller", "sequence": len(receipts)})
        print(json.dumps({"status": result["status"], "controller_job": result.get("job_id"), "run_dir": str(run_dir),
                          "status_command": shlex.join(["python3", str(repo / "scripts/queue_orion.py"), "status", "--run-dir", str(run_dir)])}, indent=2))


def check_command(run_dir, plan, specification):
    original = plan["original_plan"]; partition = specification["partition"]
    runtime = Path(specification["runtime_dir"])
    body = ["set -euo pipefail", "unset PYTHONPATH PYTHONHOME"]
    for module in (original["python_module"], original["cuda_module"]): body.append("module load " + shlex.quote(module))
    body += ["unset PYTHONPATH PYTHONHOME", "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1", "exec " + shlex.join(specification["command"])]
    return ["sbatch", "--parsable", "--partition", partition, "--account", plan["account"], "--qos", plan["qos"],
            "--gres", "gpu:1", "--cpus-per-task", "8", "--mem", "64G", "--time", plan["partition_policy"][partition]["time"],
            "--job-name", "ce6190-" + partition + "-" + specification["stage"], "--chdir", str(runtime / "code"),
            "--output", str(run_dir / "rolling" / (partition + "-" + specification["stage"] + "-%j.log")),
            "--wrap", "exec bash -lc " + shlex.quote("\n".join(body))]


def batch_command(run_dir, plan, partition, method, manifest, count, state_file, receipt_id):
    original = plan["original_plan"]; helper = hardware_module(run_dir); runtime = helper.runtime_path(run_dir, partition)
    launcher = import_path("ce6190_original_arrays", run_dir / "code/scripts/submit_slurm.py")
    args = argparse.Namespace(wan_repo=run_dir / "vendor/wan", python_executable=run_dir / "venv/bin/python",
                             python_module=original["python_module"], cuda_module=original["cuda_module"],
                             chdir=run_dir / "code", cpus=8, runtime_dir=run_dir, require_smoke=state_file,
                             output_dir=runtime / ("development_" + partition) / "results", checkpoint_dir=original["checkpoint"])
    body = launcher.batch_body(args, manifest, 0)
    check = ("import hashlib,sys; from pathlib import Path; "
             "assert len(sys.argv[1:])%2==0; "
             "assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h "
             "for p,h in zip(sys.argv[1::2],sys.argv[2::2])), 'Submitted input changed'")
    verified_inputs = [str(manifest), file_hash(manifest)]
    if method in {"magcache", "sencache"}:
        profile = run_dir / "calibration" / method / "profile.json"
        verified_inputs += [str(profile), file_hash(profile)]
    verify = shlex.join([str(run_dir / "venv/bin/python"), "-I", "-c", check, *verified_inputs])
    # Assert the immutable work list immediately before generate, in the same process wrapper.
    body = body.replace("exec " + shlex.quote(str(run_dir / "venv/bin/python")) + " -m cachebench.generate",
                        verify + "\nexec " + shlex.quote(str(run_dir / "venv/bin/python")) + " -m cachebench.generate")
    return ["sbatch", "--parsable", "--partition", partition, "--account", plan["account"], "--qos", plan["qos"],
            "--array", f"0-{count - 1}", "--gres", "gpu:1", "--cpus-per-task", "8", "--mem", "64G",
            "--time", plan["partition_policy"][partition]["time"], "--job-name", "ce6190-" + method,
            "--chdir", str(run_dir / "code"), "--output", str(run_dir / "rolling/batches" / receipt_id / "%A_%a.log"),
            "--wrap", "exec bash -lc " + shlex.quote(body)]


def reconcile(run_dir, plan, state, queue, helper):
    active = {row["job_id"] for row in queue}; missing = []
    for partition, tasks in state["tasks"].items():
        runtime = helper.runtime_path(run_dir, partition)
        for identifier, task in tasks.items():
            if task["status"] != "submitted": continue
            try:
                metrics = read_json(runtime / ("development_" + partition) / "results" / identifier / "metrics.json")
            except (OSError, ValueError):
                # generate may still be writing the JSON; scheduler state decides
                # whether to wait or record a terminal run without valid output.
                metrics = None
            if metrics and metrics.get("status") in {"complete", "failed"}:
                task["status"] = metrics["status"]
                if task["status"] == "failed": task["error"] = metrics.get("error", "Generation failed; inspect output")
            elif task["job_id"] not in active: missing.append(task)
    if not missing: return
    accounting = query_accounting(t["job_id"] for t in missing)
    for task in missing:
        entry = accounting.get(task["job_id"])
        if entry and entry["state"] not in ACTIVE:
            task.update(status="failed", error=f"Slurm {entry['state']} exit {entry['exit_code']}; no complete output")
        else:
            task.setdefault("missing_since", time.time())
            if time.time() - task["missing_since"] > 300:
                raise RuntimeError(f"Cannot reconcile accepted task {task['job_id']} with queue/accounting; refusing duplicate work")


def reconcile_dependencies(run_dir, plan, state, queue):
    """Track scheduler failures in our ledger; never forge original phase JSON."""
    active = {row["job_id"]: row for row in queue}
    anchors = {}
    original = plan["original_plan"]
    for phase in ("env", "smoke"):
        identifier = original.get("jobs", {}).get(phase)
        if identifier: anchors["parent:" + phase] = (str(identifier), run_dir / (phase + ".json"))
    for method in ("magcache", "sencache"):
        receipt = read_json(run_dir / "calibration" / method / "finalize-job.json", {})
        if receipt.get("job_id"):
            anchors["parent:" + method] = (str(receipt["job_id"]), run_dir / "calibration" / method / "check.json")
    for receipt_path in (run_dir / "rolling/checks").glob("*.json"):
        receipt = read_json(receipt_path)
        if receipt.get("status") in {"submitting", "uncertain"} and not receipt.get("job_id"):
            raise RuntimeError(f"Uncertain hardware-check submission: {receipt_path}")
        if receipt.get("job_id"):
            identity = receipt["identity"]
            anchors[identity["partition"] + ":" + identity["stage"]] = (
                str(receipt["job_id"]), Path(identity["state_file"]))
    failed = state.setdefault("failed_prerequisites", {})
    unknown = state.setdefault("unknown_prerequisites", {})
    absent = {}
    for key, (identifier, path) in anchors.items():
        phase_state = read_json(path, {})
        if phase_state.get("status") == "passed":
            unknown.pop(key, None); continue
        if phase_state.get("status") == "failed":
            failed[key] = phase_state.get("error", "Prerequisite reported failure"); continue
        if identifier in active:
            if "DependencyNeverSatisfied" in active[identifier]["reason"]:
                failed[key] = "Slurm dependency can no longer succeed"
            continue
        absent[key] = identifier
    accounting = query_accounting(absent.values())
    for key, identifier in absent.items():
        entry = accounting.get(identifier)
        if entry and entry["state"] not in ACTIVE:
            failed[key] = f"Slurm {entry['state']} exit {entry['exit_code']}; prerequisite has no passed artifact"
            unknown.pop(key, None)
        else:
            unknown.setdefault(key, time.time())
            if time.time() - unknown[key] > 300:
                raise RuntimeError(f"Cannot determine prerequisite {key} job {identifier}; inspect queue/accounting")
    return failed


def apply_failed_gates(gates, failures, original_partition):
    for (partition, method), gate in gates.items():
        reasons = [failures.get("parent:env"), failures.get(partition + ":smoke")]
        if partition == original_partition: reasons.append(failures.get("parent:smoke"))
        if method in {"magcache", "sencache"}:
            reasons += [failures.get("parent:" + method), failures.get(partition + ":" + method)]
        if any(reasons):
            gate.update(status="failed", reason="; ".join(str(reason) for reason in reasons if reason))
    return gates


def ensure_successor(run_dir, plan, queue, headroom):
    current = os.environ.get("SLURM_JOB_ID")
    if not current or not current.isdigit(): return headroom
    folder = run_dir / "rolling/controllers"; folder.mkdir(exist_ok=True)
    receipt = folder / ("after-" + current + ".json")
    old = read_json(receipt)
    if old and old.get("job_id"): return headroom
    if headroom < 1: return headroom
    result = submit_once(receipt, controller_command(run_dir, plan, "afterany:" + current),
                         {"kind": "successor", "predecessor": current})
    return headroom - (1 if result.get("job_id") else 0)


def tick(run_dir, state, plan, hardware=None):
    run_dir = Path(run_dir); helper = hardware or hardware_module(run_dir)
    manifest = run_dir / "rolling/manifest.jsonl"
    if file_hash(manifest) != plan["manifest_sha256"]: raise ValueError("Immutable experiment manifest changed")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    by_id = {row["run_id"]: row for row in rows}
    limits = query_qos(plan["qos"]); queue = query_queue(plan["qos"])
    headroom = quota_headroom(limits["max_submit"], queue)
    state["last_quota"] = {"maximum": limits["max_submit"], "active_user_qos_tasks": len(queue), "headroom": headroom}
    reconcile(run_dir, plan, state, queue, helper)
    prerequisite_failures = reconcile_dependencies(run_dir, plan, state, queue)
    # Always record an accepted intent in task state before computing another batch.
    for receipt in sorted((run_dir / "rolling/batches").glob("*/receipt.json")):
        record = read_json(receipt)
        if record["status"] in {"submitting", "uncertain"} and not record.get("job_id"):
            raise RuntimeError(f"Uncertain experiment submission: {receipt}")
        if record.get("job_id"):
            identity = record["identity"]
            for index, identifier in enumerate(identity["run_ids"]):
                task = state["tasks"][identity["partition"]][identifier]
                if task["status"] == "unsubmitted":
                    task.update(status="submitted", job_id=record["job_id"] + "_" + str(index), batch=str(receipt))
    gates = {(partition, method): helper.group_gate(run_dir, partition, method)
             for partition in plan["partitions"] for method in METHODS}
    gates = apply_failed_gates(gates, prerequisite_failures, plan["original_plan"]["gpu_partition"])
    for partition, tasks in state["tasks"].items():
        for task in tasks.values():
            if task["status"] == "unsubmitted" and gates[(partition, task["method"])]["status"] == "failed":
                task.update(status="blocked", error=gates[(partition, task["method"])].get("reason", "Prerequisite failed"))
    unfinished = [task for tasks in state["tasks"].values() for task in tasks.values()
                  if task["status"] in {"unsubmitted", "submitted"}]
    if not unfinished:
        failed = any(task["status"] != "complete" for tasks in state["tasks"].values() for task in tasks.values())
        state["status"] = "complete_with_failures" if failed else "complete"; return state
    # Receipt recovery above must precede budgeting: accepted but unrecorded
    # jobs still own their future output allowance.
    state["storage"] = storage_snapshot(run_dir, plan, state, by_id, helper)
    storage_limited = False
    headroom = ensure_successor(run_dir, plan, queue, headroom)
    for specification in helper.pending_checks(run_dir, plan["partitions"]):
        key = specification["partition"] + ":" + specification["stage"]
        stage_method = "baseline" if specification["stage"] == "smoke" else specification["stage"]
        if gates[(specification["partition"], stage_method)]["status"] == "failed": continue
        receipt = run_dir / "rolling/checks" / (key.replace(":", "-") + ".json")
        old = read_json(receipt)
        if old and old.get("job_id"):
            state["checks"][key] = old; continue
        if headroom < 1: break
        state["storage"] = storage_snapshot(run_dir, plan, state, by_id, helper)
        check_device = state["storage"]["check_devices"][specification["partition"]]
        volume = next(item for item in state["storage"]["filesystems"] if item["device"] == check_device)
        if volume["free_bytes"] < volume["generation_reserved_bytes"] + volume["check_reserved_bytes"]:
            storage_limited = True; continue
        identity = {k: specification[k] for k in ("partition", "stage", "runtime_dir", "state_file")}
        result = submit_once(receipt, check_command(run_dir, plan, specification), identity)
        state["checks"][key] = result
        if result["status"] == "quota_wait": headroom = 0; break
        headroom -= 1
    if headroom:
        array = import_path("ce6190_original_arrays", run_dir / "code/scripts/submit_slurm.py")
        try:
            maximum = array.array_limits(argparse.Namespace(max_array_size=None, max_array_tasks=None))["chunk_size"]
        except RuntimeError as error:
            if "Cannot query Slurm" in str(error): raise SchedulerUnavailable(str(error)) from error
            raise
        available = {}
        for partition in plan["partitions"]:
            for method in METHODS:
                if gates[(partition, method)]["status"] == "passed":
                    pending = [identifier for identifier, task in state["tasks"][partition].items()
                               if task["method"] == method and task["status"] == "unsubmitted"]
                    if pending: available[(partition, method)] = pending
        keys = list(available); chosen = defaultdict(list)
        budget = storage_available(state["storage"])
        if keys:
            cursor = state["cursor"] % len(keys)
            while headroom and available:
                key = keys[cursor % len(keys)]; cursor += 1
                if key not in available: continue
                device = state["storage"]["partition_devices"][key[0]]
                needed = output_bytes(by_id[available[key][0]])
                if needed > budget[device]:
                    storage_limited = True; del available[key]; continue
                budget[device] -= needed
                chosen[key].append(available[key].pop(0)); headroom -= 1
                if not available[key]: del available[key]
            state["cursor"] = cursor
        for (partition, method), identifiers in chosen.items():
            for offset in range(0, len(identifiers), maximum):
                # Re-read free space before each sbatch, including reservations
                # for batches accepted earlier in this same tick.
                state["storage"] = storage_snapshot(run_dir, plan, state, by_id, helper)
                device = state["storage"]["partition_devices"][partition]
                capacity = storage_available(state["storage"])[device]
                selected = []
                for identifier in identifiers[offset:offset + maximum]:
                    needed = output_bytes(by_id[identifier])
                    if needed > capacity:
                        storage_limited = True; break
                    selected.append(identifier); capacity -= needed
                if not selected: continue
                token = uuid.uuid4().hex
                directory = run_dir / "rolling/batches" / token; directory.mkdir(parents=True)
                batch = directory / "manifest.jsonl"
                batch.write_text("".join(json.dumps(by_id[x], sort_keys=True) + "\n" for x in selected))
                identity = {"partition": partition, "method": method, "run_ids": selected, "manifest_sha256": file_hash(batch)}
                command = batch_command(run_dir, plan, partition, method, batch, len(selected),
                                        gates[(partition, method)]["state_file"], token)
                result = submit_once(directory / "receipt.json", command, identity)
                if result["status"] == "quota_wait":
                    state["quota_race"] = result.get("stderr"); state["status"] = "running"; return state
                for index, identifier in enumerate(selected):
                    state["tasks"][partition][identifier].update(status="submitted", job_id=result["job_id"] + "_" + str(index), batch=str(directory / "receipt.json"))
                state["batches"].append(str(directory / "receipt.json"))
                atomic_json(run_dir / "rolling/state.json", state)
    state["storage"] = storage_snapshot(run_dir, plan, state, by_id, helper)
    state["status"] = "waiting_storage" if storage_limited else "running"; return state


def run_controller(args):
    run_dir = args.run_dir.expanduser().resolve(); rolling = run_dir / "rolling"; plan = load_plan(run_dir)
    with (rolling / ".controller.lock").open("a") as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("Another controller is already running")
        state = read_json(rolling / "state.json")
        if state["status"] in TERMINAL: return
        for name, digest in plan["script_hashes"].items():
            if file_hash(rolling / "code" / name) != digest:
                raise ValueError("Immutable rolling launcher changed")
        state["controller_job"] = os.environ.get("SLURM_JOB_ID")
        try:
            while state["status"] not in TERMINAL:
                try:
                    state = tick(run_dir, state, plan)
                    state.pop("scheduler_error", None)
                except SchedulerUnavailable as error:
                    state.update(status="waiting_scheduler", scheduler_error=str(error))
                state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                atomic_json(rolling / "state.json", state)
                print(json.dumps(summary(plan, state)), flush=True)
                if state["status"] not in TERMINAL: time.sleep(POLL_SECONDS)
        except Exception as error:
            state.update(status="paused", error=repr(error), updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
            atomic_json(rolling / "state.json", state)
            raise


def summary(plan, state):
    return {"status": state["status"], "controller_job": state.get("controller_job"),
            "quota": state.get("last_quota"), "partitions": {partition: dict(Counter(task["status"] for task in tasks.values()))
              for partition, tasks in state["tasks"].items()}, "error": state.get("error"),
            "updated_at": state.get("updated_at"), "scheduler_error": state.get("scheduler_error"),
            "failed_prerequisites": state.get("failed_prerequisites", {}),
            "storage": state.get("storage"),
            "no_extra_gpu_concurrency_cap": True}


def status(args):
    run_dir = args.run_dir.expanduser().resolve(); plan = load_plan(run_dir)
    if not plan:
        print(json.dumps({"status": "not_prepared", "run_dir": str(run_dir),
              "filesystem_free_gib": shutil.disk_usage(existing_parent(run_dir)).free / 2**30,
              "note": "No rolling plan exists: start stopped before preparation completed. Existing environment, smoke and calibration jobs are unaffected; rerun start after correcting its reported error."}, indent=2)); return
    state = read_json(run_dir / "rolling/state.json")
    result = summary(plan, state)
    result["controllers"] = [read_json(path) for path in sorted((run_dir / "rolling/controllers").glob("*.json"))]
    result["checks"] = state.get("checks", {})
    result["logs"] = str(run_dir / "rolling")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="action", required=True)
    start_parser = commands.add_parser("start"); start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument("--partitions", nargs="+"); start_parser.add_argument("--submit", action="store_true")
    for name in ("status", "_run"):
        sub = commands.add_parser(name); sub.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    {"start": start, "status": status, "_run": run_controller}[args.action](args)


if __name__ == "__main__": main()
