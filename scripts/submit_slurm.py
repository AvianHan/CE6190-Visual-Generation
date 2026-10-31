#!/usr/bin/env python3
"""Submit independent GPU tasks within actual Slurm array limits.

No default concurrency or wall-time cap. This is a finite submission, not a
rolling queue manager. Account, QoS and site limits still apply.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)  # Complete, atomic, no-clobber reservation.
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def array_limits(args):
    if args.max_array_size is not None:
        maximum = args.max_array_size
        task_limit = args.max_array_tasks
        source = "explicit --max-array-size; verify against the target cluster"
    else:
        try:
            result = subprocess.run(["scontrol", "show", "config"], check=True,
                                    text=True, capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("Cannot query Slurm MaxArraySize; offline previews need "
                               "the verified --max-array-size explicitly") from error
        match = re.search(r"^\s*MaxArraySize\s*=\s*(\d+)\b", result.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError("scontrol show config did not report MaxArraySize")
        maximum = int(match.group(1))
        configured = re.search(r"\bmax_array_tasks\s*=\s*(\d+)\b", result.stdout)
        limits = [int(configured.group(1))] if configured else []
        if args.max_array_tasks is not None:
            limits.append(args.max_array_tasks)
        task_limit = min(limits) if limits else None
        source = "scontrol show config"
    if maximum < 1 or (task_limit is not None and task_limit < 1):
        raise ValueError("Slurm array size/task limits must be positive")
    return {"MaxArraySize": maximum, "max_array_tasks": task_limit,
            "chunk_size": min(maximum, task_limit) if task_limit else maximum,
            "source": source}


def batch_body(args, manifest, offset):
    vendor = Path(args.wan_repo).resolve().parent
    # Preserve a venv's bin/python symlink: resolving it would select base Python.
    python = os.path.abspath(os.path.expanduser(args.python_executable))
    lines = ["set -euo pipefail", "unset PYTHONPATH PYTHONHOME"]
    for module in (args.python_module, args.cuda_module):
        if module:
            lines.append("module load " + shlex.quote(module))
    lines.append("unset PYTHONPATH PYTHONHOME")
    lines.append("cd -- " + shlex.quote(str(Path(args.chdir).resolve())))
    exports = {"PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
               "OMP_NUM_THREADS": str(args.cpus), "CACHEBENCH_VENDOR": str(vendor),
               "CACHEBENCH_SEACACHE_SOURCE": str(vendor / "seacache/Wan2.1/seacache_generate.py")}
    if args.runtime_dir:
        runtime = Path(args.runtime_dir).resolve()
        cache = runtime.parent.parent / "caches"
        temporary = runtime / "tmp"
        lines.append("mkdir -p " + shlex.quote(str(temporary)))
        exports.update(PIP_CACHE_DIR=str(cache / "pip"), HF_HOME=str(cache / "huggingface"),
                       TORCH_HOME=str(cache / "torch"), TMPDIR=str(temporary))
    lines.extend("export " + key + "=" + shlex.quote(value) for key, value in exports.items())
    lines.append(': "$' + '{SLURM_ARRAY_TASK_ID:?This command requires a Slurm array task}"')
    if args.require_smoke:
        check = ("import json,sys; from pathlib import Path; "
                 "state=json.loads(Path(sys.argv[1]).read_text()); "
                 "sys.exit(0 if state.get('status') == 'passed' else "
                 "'Required smoke check has not passed')")
        lines.append(shlex.join([python, "-c", check, str(Path(args.require_smoke).resolve())]))
    command = [python, "-m", "cachebench.generate", "--manifest", str(manifest),
               "--output-dir", str(Path(args.output_dir).resolve()),
               "--wan-repo", str(Path(args.wan_repo).resolve()), "--vendor-dir", str(vendor),
               "--checkpoint-dir", str(Path(args.checkpoint_dir).resolve())]
    lines.append("exec " + shlex.join(command) + ' --index "$((SLURM_ARRAY_TASK_ID + ' + str(offset) + '))"')
    return "\n".join(lines)


def submission_summary(result, *, status=None):
    summary = {key: result[key] for key in ("status", "runs", "submitted_jobs", "receipt",
               "concurrency_policy", "time_policy")}
    if status is not None:
        summary["status"] = status
    summary["array_chunks"] = [{key: item[key] for key in ("offset", "count", "job_id", "status")}
                               for item in result["arrays"]]
    for key in ("error", "stderr"):
        if key in result:
            summary[key] = result[key]
    return summary


def submit_arrays(result, content):
    """Resume only jobs known not to have been accepted; serialize submitters."""
    output = Path(result["output_dir"])
    receipt_path = Path(result["receipt"])
    snapshot = Path(result["manifest_snapshot"])
    output.mkdir(parents=True, exist_ok=True)
    with (receipt_path.parent / ("." + receipt_path.stem + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is submitting this output directory") from error
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            for key in ("manifest_sha256", "commands", "manifest_snapshot", "output_dir"):
                if previous.get(key) != result[key]:
                    raise ValueError(f"Existing submission differs in {key}; use a new output directory")
            result = previous
            if any(item.get("job_id") is None and item["status"] in {"submitting", "uncertain"}
                   for item in result["arrays"]):
                raise RuntimeError("A prior sbatch result is uncertain; inspect squeue and the receipt before retrying")
            if all(item.get("job_id") for item in result["arrays"]):
                print(json.dumps(submission_summary(result, status="already_submitted"), indent=2))
                return
        else:
            result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(receipt_path, result, exclusive=True)
        current = None
        try:
            if snapshot.exists():
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() != result["manifest_sha256"]:
                    raise ValueError("Submitted manifest snapshot changed; refusing to resume")
            else:
                if result["submitted_jobs"]:
                    raise FileNotFoundError("Submitted manifest snapshot is missing")
                with snapshot.open("xb") as handle:
                    handle.write(content)
            (output / "slurm_logs").mkdir(exist_ok=True)
            for current in result["arrays"]:
                if current.get("job_id"):
                    continue
                current["status"] = "submitting"
                result["status"] = "submitting"
                atomic_json(receipt_path, result)
                try:
                    response = subprocess.run(current["command"], check=True, text=True, capture_output=True)
                except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
                    current["status"] = "rejected"
                    raise
                job_id = response.stdout.strip().split(";", 1)[0]
                if not job_id.isdigit():
                    current["status"] = "uncertain"
                    raise RuntimeError(f"Unexpected sbatch response: {response.stdout!r}; inspect squeue before retrying")
                current.update(status="submitted", job_id=job_id)
                result["submitted_jobs"].append(job_id)
                atomic_json(receipt_path, result)
            result["status"] = "submitted"
            result.pop("error", None)
            result.pop("stderr", None)
            atomic_json(receipt_path, result)
        except Exception as error:
            if current and current["status"] == "submitting" and not current.get("job_id"):
                current["status"] = "uncertain"
            result["status"] = "partial_submission" if result["submitted_jobs"] else "submission_failed"
            result["error"] = repr(error)
            if isinstance(error, subprocess.CalledProcessError):
                result["stderr"] = error.stderr
            atomic_json(receipt_path, result)
            print(json.dumps(submission_summary(result), indent=2))
            raise
        print(json.dumps(submission_summary(result), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "output-dir", "wan-repo", "checkpoint-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--partition"); p.add_argument("--account"); p.add_argument("--qos")
    p.add_argument("--dependency", help="Slurm dependency, e.g. afterok:12345")
    p.add_argument("--submission-name", default="",
                   help="Optional lowercase name for separate manifests/receipts in a shared result directory")
    p.add_argument("--concurrency", type=int, help="Optional cap per array; no default %%N cap")
    p.add_argument("--time", help="Optional wall time; omitted uses Slurm policy defaults")
    p.add_argument("--cpus", type=int, default=8); p.add_argument("--memory", default="64G")
    p.add_argument("--python-executable", default=sys.executable,
                   help="Prepared environment's Python; may be created by a dependency")
    p.add_argument("--python-module"); p.add_argument("--cuda-module")
    p.add_argument("--chdir", default=str(Path.cwd()))
    p.add_argument("--runtime-dir", help="Existing start_orion runtime for shared caches and temp files")
    p.add_argument("--require-smoke", help="Smoke state JSON that must say passed before each task")
    p.add_argument("--max-array-size", type=int, help="Explicit site MaxArraySize; otherwise query scontrol")
    p.add_argument("--max-array-tasks", type=int, help="Optional additional site max_array_tasks limit")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    if a.submission_name and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", a.submission_name):
        p.error("--submission-name must be a lowercase letter/digit slug, at most 64 characters")
    if a.time is None and os.environ.get("SBATCH_TIMELIMIT"):
        a.time = os.environ["SBATCH_TIMELIMIT"]
    if a.cpus < 1 or (a.concurrency is not None and a.concurrency < 1):
        p.error("CPU count and an explicit concurrency cap must be positive")
    if not Path(a.chdir).is_dir():
        p.error("--chdir must be an existing directory")
    source = Path(a.manifest).resolve()
    content = source.read_bytes()
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        p.error("Manifest must contain nonempty JSON object rows")
    run_ids = [row["run_id"] for row in rows if "run_id" in row]
    if len(set(run_ids)) != len(run_ids):
        p.error("Manifest contains duplicate run_ids")
    n = len(rows)
    output = Path(a.output_dir).resolve()
    prefix = "slurm_" + (a.submission_name + "_" if a.submission_name else "")
    receipt_path = output / (prefix + "submission.json")
    limits = array_limits(a)
    snapshot = output / (prefix + "manifest.jsonl")
    arrays = []
    for offset in range(0, n, limits["chunk_size"]):
        count = min(limits["chunk_size"], n - offset)
        array = f"0-{count - 1}" + (f"%{a.concurrency}" if a.concurrency is not None else "")
        cmd = ["sbatch", "--parsable", "--array", array, "--gres", "gpu:1",
               "--cpus-per-task", str(a.cpus), "--mem", a.memory,
               "--chdir", str(Path(a.chdir).resolve()),
               "--output", str(output / "slurm_logs/%A_%a.log")]
        for option, value in (("partition", a.partition), ("account", a.account),
                              ("qos", a.qos), ("time", a.time), ("dependency", a.dependency)):
            if value is not None:
                cmd += ["--" + option, value]
        if a.dependency:
            cmd += ["--kill-on-invalid-dep=yes"]
        cmd += ["--wrap", "exec bash -lc " + shlex.quote(batch_body(a, snapshot, offset))]
        arrays.append({"offset": offset, "count": count, "array": array, "command": cmd,
                       "status": "planned", "job_id": None})
    result = {"schema_version": 1, "status": "planned", "runs": n,
              "manifest_source": str(source), "manifest_snapshot": str(snapshot),
              "manifest_sha256": hashlib.sha256(content).hexdigest(),
              "output_dir": str(output), "receipt": str(receipt_path), "array_limits": limits,
              "max_parallel_gpus": a.concurrency * len(arrays) if a.concurrency else None,
              "concurrency_policy": "explicit per-array --concurrency" if a.concurrency else "Slurm allocation; no additional array cap",
              "time_policy": "explicit --time" if a.time is not None else "Slurm policy default",
              "command": arrays[0]["command"], "commands": [x["command"] for x in arrays],
              "arrays": arrays, "submitted_jobs": []}
    if not a.submit:
        print(json.dumps(result, indent=2))
        return
    submit_arrays(result, content)


if __name__ == "__main__":
    main()
