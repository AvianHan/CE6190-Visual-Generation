#!/usr/bin/env python3
"""Reuse one immutable runtime and validate each additional GPU partition.

This module never submits jobs, installs dependencies, downloads weights or
calibrates a new profile. The rolling controller owns Slurm submissions. A new
partition gets its own correctness outputs and measurements while sharing the
original code, environment, vendor repositories and frozen calibration inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback

STAGES = ("smoke", "magcache", "sencache")


def _json(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _state(path):
    path = Path(path)
    if not path.exists():
        return {"status": "not_started"}
    try:
        result = _json(path)
        if not isinstance(result, dict):
            raise ValueError("state must be an object")
        return result
    except (OSError, ValueError) as error:
        return {"status": "failed", "error": f"Unreadable state {path}: {error}"}


def _parent(parent_run):
    parent = Path(parent_run).expanduser().resolve()
    plan = _json(parent / "plan.json")
    if not isinstance(plan.get("gpu_partition"), str):
        raise ValueError("Original runtime plan has no GPU partition")
    return parent, plan


def runtime_path(parent_run, partition):
    parent, plan = _parent(parent_run)
    if not isinstance(partition, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", partition):
        raise ValueError("Invalid GPU partition name")
    if partition == plan["gpu_partition"]:
        return parent
    return parent.parent / (parent.name + "-" + partition)


def _shared_identity(parent, plan, partition):
    return {
        "parent_run": str(parent), "gpu_partition": partition,
        "checkpoint": str(Path(plan["checkpoint"]).resolve()),
        "model_id": plan["model_id"], "model_revision": plan["model_revision"],
        "python_module": plan["python_module"], "cuda_module": plan["cuda_module"],
        "account": plan["account"], "qos": plan["qos"],
        "source_release_sha256": _hash(parent / "code/SHA256SUMS.txt"),
    }


def prepare(parent_run, partition):
    """Create a separate hardware runtime; never inherit a successful smoke.

    Links to an environment/vendor directory that is still being prepared are
    allowed. Its real success is required by pending_checks and run_check.
    """
    parent, plan = _parent(parent_run)
    child = runtime_path(parent, partition)
    if child == parent:
        return parent
    if child.is_symlink():
        raise ValueError(f"Hardware runtime must not be a symlink: {child}")
    identity = _shared_identity(parent, plan, partition)
    (parent / "rolling").mkdir(exist_ok=True)
    with (parent / "rolling" / (".hardware-" + partition + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        child.mkdir(exist_ok=True)
        marker = child / "hardware_parent.json"
        if marker.exists():
            if _json(marker) != identity:
                raise ValueError(f"Existing hardware runtime has different source inputs: {child}")
        else:
            if any(child.iterdir()):
                raise ValueError(f"Refusing to adopt an unrelated nonempty runtime: {child}")
            _atomic(marker, identity)
        for name in ("code", "venv", "vendor"):
            path, target = child / name, parent / name
            if os.path.lexists(path):
                if not path.is_symlink() or path.resolve() != target.resolve():
                    raise ValueError(f"Hardware runtime link differs: {path}")
            else:
                path.symlink_to(target, target_is_directory=True)
        child_plan = {
            **{key: value for key, value in plan.items()
               if key not in {"jobs", "model_reuse", "stages", "run_dir", "requested_time", "time_policy"}},
            "run_dir": str(child), "gpu_partition": partition, "jobs": {},
            "shared_parent_run": str(parent), "shared_inputs": identity,
            "requested_time": None,
            "time_policy": "rolling controller uses this partition's actual limits",
            "calibration_policy": "same frozen parent profiles; separate GPU parity checks",
        }
        plan_path = child / "plan.json"
        if plan_path.exists():
            if _json(plan_path) != child_plan:
                raise ValueError(f"Existing hardware plan differs: {plan_path}")
        else:
            _atomic(plan_path, child_plan)
        for directory in ("logs", "tmp", "calibration/magcache", "calibration/sencache"):
            (child / directory).mkdir(parents=True, exist_ok=True)
    return child


def state_path(parent_run, partition, stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown hardware check stage: {stage}")
    runtime = runtime_path(parent_run, partition)
    return runtime / ("smoke.json" if stage == "smoke" else f"calibration/{stage}/check.json")


def input_identity(parent_run, partition, stage):
    """Inputs that must remain unchanged after a hardware check passes."""
    parent, plan = _parent(parent_run)
    if stage not in STAGES:
        raise ValueError("Unknown hardware check stage")
    identity = {
        **_shared_identity(parent, plan, partition), "stage": stage,
        "hardware_helper_sha256": _hash(__file__),
        "model_integrity_sha256": _hash(parent / "model_integrity.json"),
    }
    if stage != "smoke":
        profile = parent / "calibration" / stage / "profile.json"
        identity.update(profile_path=str(profile), profile_sha256=_hash(profile),
                        calibration_source_partition=plan["gpu_partition"])
    return identity


def _prerequisite_paths(parent, partition, stage):
    paths = [parent / "env.json", parent / "model.json"]
    if stage != "smoke":
        paths += [state_path(parent, partition, "smoke"), parent / "smoke.json",
                  parent / "calibration" / stage / "check.json"]
    return list(dict.fromkeys(paths))


def _prerequisites(parent, partition, stage):
    waiting = []
    for path in _prerequisite_paths(parent, partition, stage):
        state = _state(path)
        if state.get("status") in {"failed", "cancelled", "blocked"}:
            return {"status": "failed", "state_file": str(path),
                    "reason": f"Prerequisite failed: {path}", "detail": state}
        if state.get("status") != "passed":
            waiting.append(str(path))
    if waiting:
        return {"status": "waiting", "state_file": waiting[0],
                "reason": "Waiting for prerequisites", "waiting_for": waiting}
    return {"status": "passed"}


def _own_stage(parent, partition, stage):
    path = state_path(parent, partition, stage)
    state = _state(path)
    status = state.get("status")
    if status == "passed" and runtime_path(parent, partition) != parent:
        try:
            if state.get("input_identity") != input_identity(parent, partition, stage):
                raise ValueError("Inputs changed after the hardware check passed")
        except (OSError, ValueError, KeyError) as error:
            return {"status": "failed", "state_file": str(path), "reason": str(error)}
    mapped = "passed" if status == "passed" else (
        "failed" if status in {"failed", "cancelled", "blocked"} else "waiting")
    return {"status": mapped, "state_file": str(path),
            "reason": state.get("error", status or "not_started"), "detail": state}


def group_gate(parent_run, partition, method):
    """Read-only readiness for a method, including separate hardware checks."""
    parent, _ = _parent(parent_run)
    stage = method if method in {"magcache", "sencache"} else "smoke"
    if method not in {"baseline", "uncached", "teacache", "seacache", "magcache", "sencache"}:
        raise ValueError(f"Unknown cache method: {method}")
    gate = _prerequisites(parent, partition, stage)
    if gate["status"] != "passed":
        return gate
    gate = _own_stage(parent, partition, stage)
    if gate["status"] == "passed" and stage != "smoke":
        if not (parent / "calibration" / stage / "profile.json").is_file():
            return {"status": "failed", "state_file": gate["state_file"],
                    "reason": "Passed calibration has no profile file"}
    return gate


def pending_checks(parent_run, partitions):
    """Return extra-hardware checks whose prerequisites have passed.

    Accepted Slurm jobs are deduplicated by the caller's durable receipts. No
    failed or running state is silently retried, and this function is read-only.
    """
    parent, plan = _parent(parent_run)
    pending = []
    for partition in dict.fromkeys(partitions):
        if partition == plan["gpu_partition"]:
            continue
        for stage in STAGES:
            if _prerequisites(parent, partition, stage)["status"] != "passed":
                continue
            path = state_path(parent, partition, stage)
            if _state(path).get("status") != "not_started":
                continue
            if stage != "smoke" and not (parent / "calibration" / stage / "profile.json").is_file():
                continue
            pending.append({
                "partition": partition, "stage": stage,
                "runtime_dir": str(runtime_path(parent, partition)), "state_file": str(path),
                "command": [str(parent / "venv/bin/python"), "-I", str(Path(__file__).resolve()),
                            "_check", "--parent-run", str(parent), "--partition", partition,
                            "--stage", stage],
            })
    return pending


def _starter(parent):
    path = parent / "code/scripts/start_orion.py"
    spec = importlib.util.spec_from_file_location("ce6190_original_starter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_allocated_gpu(partition):
    """Check the actual allocation; a finished old probe is not sufficient."""
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != partition:
        raise RuntimeError("Hardware check must run in the requested Slurm partition")
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Hardware correctness check requires exactly one visible GPU")
    name = torch.cuda.get_device_name(0)
    expected = "H200" if partition.startswith("h200") else "L40S" if partition.startswith("l40s") else None
    if expected and expected.casefold() not in name.casefold():
        raise RuntimeError(f"Partition {partition} allocated unexpected GPU: {name}")
    return {"name": name, "capability": list(torch.cuda.get_device_capability(0)),
            "partition": partition}


def run_check(parent_run, partition, stage):
    parent, plan = _parent(parent_run)
    if partition == plan["gpu_partition"]:
        raise ValueError("Original partition checks are already owned by the original launcher")
    child = prepare(parent, partition)
    path = state_path(parent, partition, stage)
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is running this hardware check") from error
        prerequisite = _prerequisites(parent, partition, stage)
        if prerequisite["status"] != "passed":
            raise RuntimeError(prerequisite["reason"])
        identity = input_identity(parent, partition, stage)
        previous = _state(path)
        if previous.get("status") == "passed":
            if previous.get("input_identity") != identity:
                raise RuntimeError("Passed hardware check inputs changed; use a new runtime")
            return previous
        if previous.get("status") != "not_started":
            raise RuntimeError(f"Previous {stage} check is {previous.get('status')}; inspect its log before recovery")
        state = {"phase": "hardware_" + stage, "status": "running",
                 "job_id": os.environ.get("SLURM_JOB_ID"), "partition": partition,
                 "input_identity": identity, "started_at_unix": time.time(),
                 "calibration_policy": "parent profiles reused; no additional calibration"}
        _atomic(path, state)
        try:
            state["gpu"] = verify_allocated_gpu(partition)
            starter = _starter(parent)
            starter.verified_release(parent / "code")
            starter.runtime_environment(child)
            for prerequisite_name in ("env", "model"):
                source = parent / (prerequisite_name + ".json")
                inherited = _json(source)
                if inherited.get("status") != "passed":
                    raise RuntimeError(f"Shared {prerequisite_name} no longer passed")
                _atomic(child / (prerequisite_name + ".json"), {
                    "phase": prerequisite_name, "status": "passed", "job_id": None,
                    "inherited_from": str(source), "source_sha256": _hash(source),
                    "note": "shared existing artifact; no new installation/download job",
                })
            if stage == "smoke":
                starter.gpu_smoke(child)
            else:
                starter.calibration_parity(child, stage, Path(identity["profile_path"]))
            if input_identity(parent, partition, stage) != identity:
                raise RuntimeError("Check inputs changed during GPU validation")
            state["status"] = "passed"
        except BaseException as error:
            state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            state["ended_at_unix"] = time.time()
            _atomic(path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("_check")
    check.add_argument("--parent-run", type=Path, required=True)
    check.add_argument("--partition", required=True)
    check.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    print(json.dumps(run_check(args.parent_run, args.partition, args.stage), indent=2))


if __name__ == "__main__":
    main()
