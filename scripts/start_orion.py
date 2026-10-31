#!/usr/bin/env python3
"""Prepare an isolated Slurm runtime and run one paired Wan correctness check.

No job is submitted without `submit --submit`. All runtime files stay outside
the Git repository. This launcher does not calibrate or start the full benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid
import zipfile

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
MODEL_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
MODEL_FILES = (
    "diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth", "config.json", "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model", "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json", "LICENSE.txt", "README.md",
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(args, *, cwd=None, capture=False):
    print("+ " + shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def verified_release(repo):
    """Validate every release member before creating an immutable job snapshot."""
    result = []
    for line in (repo / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = repo / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or ".git" in Path(relative).parts
                or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(repo.resolve())):
            raise ValueError(f"Unsafe or missing release member: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Release checksum changed: {relative}; use the intact update package")
        result.append(relative)
    required = {"scripts/start_orion.py", "requirements.txt", "pyproject.toml",
                "configs/correctness_smoke.json", "src/wan_adapter.py"}
    if not required.issubset(result):
        raise ValueError("The release checksum list does not include the complete startup package")
    return result


def collect_cluster_resources(run_dir, plan):
    """Record live allocation and policy evidence; never infer a personal GPU quota."""
    partitions = list(dict.fromkeys([plan["cpu_partition"], plan["gpu_partition"], "l40sp"]))
    commands = [
        ["sinfo", "-N", "-h", "-p", ",".join(partitions), "-o", "%N|%P|%t|%G|%C"],
        ["scontrol", "-o", "show", "nodes"],
        ["sacctmgr", "-n", "-P", "show", "qos", "format=Name,Flags,MaxWall"],
        ["sacctmgr", "-n", "-P", "show", "qos",
         "format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,GrpTRES"],
        ["sacctmgr", "-n", "-P", "show", "assoc", "where", "user=" + getpass.getuser(),
         "format=Cluster,Account,User,Partition,QOS,DefaultQOS,MaxWall,MaxJobs,MaxSubmit"],
        ["squeue", "-u", getpass.getuser(), "-o", "%.18i %.12P %.10T %.12M %.12l %.8D %R"],
    ] + [["scontrol", "-o", "show", "partition", p] for p in partitions]
    def query(command):
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=20)
            return {"command": command, "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"command": command, "error": str(error)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(query, commands))
    write_json(run_dir / "cluster_resources.json", {
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "Live snapshot, not a GPU allocation promise. Count unique NodeName values; policy query failures do not mean unlimited.",
        "queries": results,
    })


def submit(args):
    repo = Path(__file__).resolve().parents[1]
    files = verified_release(repo)
    root = args.work_root.expanduser().resolve()
    if root.is_relative_to(repo):
        raise ValueError("--work-root must be outside the Git repository")
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free / 2**30
    if free < args.min_free_gb:
        raise ValueError(f"Only {free:.1f} GiB free; need {args.min_free_gb} GiB on runtime filesystem")
    run_dir = root / "runs" / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    plan = {"run_dir": str(run_dir), "work_root": str(root), "account": args.account,
            "qos": args.qos, "cpu_partition": args.cpu_partition,
            "gpu_partition": args.gpu_partition, "python_module": args.python_module,
            "cuda_module": args.cuda_module, "checkpoint": str(root / "models" / ("Wan2.1-T2V-1.3B-" + MODEL_REVISION[:12])),
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "jobs": {},
            "requested_time": args.time,
            "time_policy": "explicit --time" if args.time is not None else "Slurm policy default",
            "stages": {"env": "8 CPUs / 64G", "model": "4 CPUs / 8G",
                       "smoke": "1 GPU / 8 CPUs / 64G; after env and model"}}
    if not args.submit:
        print(json.dumps({"status": "preview_only", **plan}, indent=2))
        return
    if not shutil.which("sbatch"):
        raise RuntimeError("sbatch not found; run this from the Orion login terminal")
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    code = run_dir / "code"
    for relative in files + ["SHA256SUMS.txt"]:
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    verified_release(code)  # Reject edits made while the release was being copied.
    write_json(run_dir / "plan.json", plan)
    collect_cluster_resources(run_dir, plan)
    for phase in ("env", "model", "smoke"):
        script = run_dir / (phase + ".sh")
        command = ["python3", str(code / "scripts/start_orion.py"), "_phase",
                   "--run-dir", str(run_dir), "--phase", phase]
        script.write_text("#!/bin/bash -l\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n" +
                          "module load " + shlex.quote(args.python_module) + "\n" +
                          "module load " + shlex.quote(args.cuda_module) + "\n" +
                          "export PYTHONUNBUFFERED=1\n" +
                          "exec " + shlex.join(command) + "\n")
        command = ["sbatch", "--parsable", "--account", args.account, "--qos", args.qos,
                   "--partition", args.gpu_partition if phase == "smoke" else args.cpu_partition,
                   "--cpus-per-task", "4" if phase == "model" else "8",
                   "--mem", "8G" if phase == "model" else "64G",
                   "--job-name", "ce6190-" + phase, "--chdir", str(code),
                   "--output", str(run_dir / "logs" / (phase + "-%j.log"))]
        if args.time is not None:
            command += ["--time", args.time]
        if phase == "smoke":
            command += ["--gres", "gpu:1", "--dependency", "afterok:" + ":".join(plan["jobs"].values()),
                        "--kill-on-invalid-dep=yes"]
        command += [str(script)]
        try:
            response = run(command, capture=True).strip()
            job_id = response.split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {response!r}")
            plan["jobs"][phase] = job_id
            write_json(run_dir / "plan.json", plan)
        except Exception:
            print(json.dumps({"status": "partial_submission", "run_dir": str(run_dir),
                              "submitted_jobs": plan["jobs"]}, indent=2))
            raise
    print(json.dumps({"status": "submitted", "run_dir": str(run_dir), "jobs": plan["jobs"],
                      "status_command": shlex.join(["python3", str(repo / "scripts/start_orion.py"),
                                                     "status", "--run-dir", str(run_dir)])}, indent=2))


def runtime_environment(run_dir):
    cache = run_dir.parent.parent / "caches"
    temporary = run_dir / "tmp"
    temporary.mkdir(exist_ok=True)
    os.environ.update(PIP_CACHE_DIR=str(cache / "pip"), HF_HOME=str(cache / "huggingface"),
                      TORCH_HOME=str(cache / "torch"), TMPDIR=str(temporary),
                      PYTHONNOUSERSITE="1", MAX_JOBS="4", OMP_NUM_THREADS="8",
                      CACHEBENCH_VENDOR=str(run_dir / "vendor"),
                      CACHEBENCH_SEACACHE_SOURCE=str(run_dir / "vendor/seacache/Wan2.1/seacache_generate.py"))


def environment_stage(run_dir):
    code = run_dir / "code"
    run([sys.executable, "-m", "venv", run_dir / "venv"])
    python = run_dir / "venv/bin/python"
    pip = [python, "-m", "pip", "install"]
    run(pip + ["--upgrade", "pip", "setuptools", "wheel", "packaging", "ninja", "psutil"])
    run(pip + ["torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124"])
    run(pip + ["-r", code / "requirements.txt", "huggingface_hub==0.29.3"])
    run(pip + ["flash-attn==2.7.4.post1", "--no-build-isolation"])
    run(pip + ["-e", code, "--no-deps"])
    run([python, "-m", "cachebench.sources", "--directory", run_dir / "vendor"], cwd=code)
    run([python, "-m", "pip", "check"])
    run([python, "-c", "import torch, flash_attn, cachebench; print(torch.__version__, flash_attn.__version__)"])
    run([python, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=code)
    (run_dir / "environment_actual.txt").write_text(run([python, "-m", "pip", "freeze"], capture=True))


def model_stage(run_dir):
    run([sys.executable, "-m", "venv", run_dir / "download_venv"])
    python = run_dir / "download_venv/bin/python"
    run([python, "-m", "pip", "install", "huggingface_hub==0.29.3"])
    run([python, __file__, "_download", "--run-dir", run_dir])


def download_model(run_dir):
    import fcntl
    from huggingface_hub import HfApi, snapshot_download
    plan = json.loads((run_dir / "plan.json").read_text())
    checkpoint = Path(plan["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (checkpoint.parent / (checkpoint.name + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
        if info.sha != MODEL_REVISION:
            raise ValueError("Model revision mismatch")
        metadata = {x.rfilename: x for x in info.siblings}
        if set(MODEL_FILES) - metadata.keys():
            raise ValueError("Pinned checkpoint is missing required model/tokenizer files")
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(checkpoint),
                          allow_patterns=list(MODEL_FILES), max_workers=4)
        records = []
        for name in MODEL_FILES:
            item = metadata[name]
            path = checkpoint / name
            if item.size is None or path.stat().st_size != item.size:
                raise ValueError(f"Model download size mismatch: {name}")
            lfs = item.lfs
            expected = lfs.sha256 if lfs else item.blob_id
            digest = hashlib.sha256() if lfs else hashlib.sha1()
            if not lfs:
                digest.update(f"blob {item.size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Model download hash mismatch: {name}; do not run this checkpoint")
            records.append({"path": name, "bytes": item.size, "hash": expected,
                            "algorithm": "sha256" if lfs else "git_blob_sha1"})
        config = json.loads((checkpoint / "config.json").read_text())
        for key, value in {"model_type": "t2v", "dim": 1536, "num_layers": 30, "in_dim": 16, "out_dim": 16}.items():
            if config.get(key) != value:
                raise ValueError(f"Unexpected checkpoint architecture: {key}")
        write_json(run_dir / "model_integrity.json", {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                   "checkpoint": str(checkpoint), "files": records})


def generate_manifest_rows(run_dir, manifest, results, checkpoint):
    """Run the correctness pair directly; no optional launcher file is required."""
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 2 or {x["method"] for x in rows} != {"baseline", "seacache"}:
        raise ValueError("Smoke must contain exactly one uncached/SeaCache all-compute pair")
    for index in range(len(rows)):
        run([sys.executable, "-m", "cachebench.generate", "--manifest", manifest,
             "--index", str(index), "--output-dir", results,
             "--vendor-dir", run_dir / "vendor", "--wan-repo", run_dir / "vendor/wan",
             "--checkpoint-dir", checkpoint], cwd=run_dir / "code")


def gpu_smoke(run_dir):
    import numpy as np
    import torch
    from flash_attn import flash_attn_func
    code = run_dir / "code"
    plan = json.loads((run_dir / "plan.json").read_text())
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke requires exactly one GPU allocated by Slurm")
    run(["nvidia-smi"])
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Allocated GPU does not support the BF16 experiment configuration")
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    output = flash_attn_func(q, q, q)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("FlashAttention2 GPU check produced nonfinite values")
    torch.cuda.synchronize()
    write_json(run_dir / "gpu.json", {"name": torch.cuda.get_device_name(0),
               "capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__,
               "cuda": torch.version.cuda, "bf16": True, "flash_attention_2": "passed"})
    del q, output
    torch.cuda.empty_cache()
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=code)
    manifest = run_dir / "smoke.jsonl"
    results = run_dir / "smoke_results"
    run([sys.executable, "-m", "cachebench.manifest", "--prompts", code / "configs/smoke_prompts.jsonl",
         "--config", code / "configs/correctness_smoke.json", "--output", manifest], cwd=code)
    generate_manifest_rows(run_dir, manifest, results, plan["checkpoint"])
    run([sys.executable, "-m", "cachebench.evaluate", "fidelity", "--manifest", manifest,
         "--root", results, "--output", run_dir / "smoke_fidelity.jsonl", "--metrics", "psnr", "ssim"], cwd=code)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 2 or {x["method"] for x in rows} != {"baseline", "seacache"}:
        raise ValueError("Smoke must contain exactly one uncached/SeaCache all-compute pair")
    arrays = []
    for row in rows:
        directory = results / row["run_id"]
        metadata = json.loads((directory / "metrics.json").read_text())
        trace = json.loads((directory / "trace.json").read_text())["repeats"]
        expected_calls = 2 * row["num_inference_steps"]
        if (metadata["status"] != "complete" or not trace
                or len(trace) != row["timing_repeats"]
                or not all(len(t) == expected_calls and all(x["refresh"] for x in t) for t in trace)):
            raise ValueError("All-compute smoke unexpectedly skipped Transformer calculations")
        frames = np.load(directory / "frames.npy", mmap_mode="r", allow_pickle=False)
        if frames.shape != (81, 480, 832, 3) or frames.dtype != np.uint8:
            raise ValueError("Smoke output geometry/dtype mismatch")
        arrays.append(frames)
    max_error = max(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) for a, b in zip(*arrays))
    write_json(run_dir / "smoke_parity.json", {"max_uint8_difference": max_error,
               "allowed_uint8_difference": 1, "exact_match": max_error == 0,
               "all_forwards_computed": True})
    if max_error > 1:
        raise ValueError(f"All-compute output differs by {max_error}/255; inspect before caching experiments")


def compatible_runtime(repo, run_dir):
    """The existing editable environment must still load the same experiment code."""
    current = verified_release(repo)
    original = verified_release(run_dir / "code")
    fixed = {"requirements.txt", "pyproject.toml", "configs/correctness_smoke.json", "configs/smoke_prompts.jsonl"}
    needed = fixed | {x for x in current + original if x.startswith("src/")}
    for relative in needed:
        before, after = run_dir / "code" / relative, repo / relative
        if not before.is_file() or not after.is_file() or before.read_bytes() != after.read_bytes():
            raise ValueError(f"Runtime source changed: {relative}; cannot reuse this editable environment")
    return current


def resume_smoke(args):
    """Repair only the GPU entrypoint, retaining the original environment/model jobs."""
    if not args.submit:
        return _resume_smoke_locked(args)
    import fcntl
    with (args.run_dir.expanduser().resolve() / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _resume_smoke_locked(args)


def _resume_smoke_locked(args):
    repo = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.expanduser().resolve()
    files = compatible_runtime(repo, run_dir)
    plan = json.loads((run_dir / "plan.json").read_text())
    old_job = str(plan["jobs"].get("smoke", ""))
    state_file = run_dir / "smoke.json"
    if state_file.exists() and json.loads(state_file.read_text()).get("status") == "passed":
        raise ValueError("Smoke already passed; use queue-development instead of submitting it again")
    dependencies = []
    for prerequisite in ("env", "model"):
        path = run_dir / (prerequisite + ".json")
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get("status") == "failed":
            raise ValueError(f"Prerequisite failed: {prerequisite}; inspect its log before resubmission")
        if state.get("status") != "passed":
            dependencies.append(str(plan["jobs"][prerequisite]))
    if not args.submit:
        print(json.dumps({"status": "preview_only", "run_dir": str(run_dir), "replaces_smoke_job": old_job,
                          "reuse_environment_and_model": True, "dependencies": dependencies,
                          "queue_development": args.queue_development}, indent=2))
        return
    active = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"],
                            check=True, capture_output=True, text=True).stdout.split()
    if old_job and old_job in active:
        raise ValueError(f"Previous smoke job {old_job} is still active; cancel only that job first")
    recovery = run_dir / "recovery" / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    code = recovery / "code"
    for relative in files + ["SHA256SUMS.txt"]:
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    verified_release(code)
    # Keep any failed/partial attempt for diagnosis; never silently overwrite it.
    for name in ("smoke.json", "smoke.jsonl", "smoke.jsonl.meta.json", "smoke_results", "smoke_fidelity.jsonl", "smoke_parity.json", "gpu.json"):
        path = run_dir / name
        if path.exists():
            archive = recovery / "previous_attempt"
            archive.mkdir(exist_ok=True)
            path.rename(archive / name)
    script = recovery / "smoke.sh"
    script.write_text("#!/bin/bash -l\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n" +
                      "module load " + shlex.quote(plan["python_module"]) + "\n" +
                      "module load " + shlex.quote(plan["cuda_module"]) + "\nexport PYTHONUNBUFFERED=1\nexec " +
                      shlex.join(["python3", str(code / "scripts/start_orion.py"), "_phase", "--run-dir", str(run_dir), "--phase", "smoke"]) + "\n")
    command = ["sbatch", "--parsable", "--account", plan["account"], "--qos", plan["qos"],
               "--partition", plan["gpu_partition"], "--cpus-per-task", "8", "--mem", "64G", "--gres", "gpu:1",
               "--job-name", "ce6190-smoke", "--chdir", str(run_dir / "code"),
               "--output", str(run_dir / "logs/smoke-%j.log")]
    if dependencies:
        command += ["--dependency", "afterok:" + ":".join(dependencies), "--kill-on-invalid-dep=yes"]
    if args.time is not None:
        command += ["--time", args.time]
    command += [str(script)]
    record = {"previous_smoke_job": old_job, "launcher": str(code / "scripts/start_orion.py"),
              "command": command, "status": "prepared"}
    write_json(recovery / "repair.json", record)
    job_id = run(command, capture=True).strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {job_id!r}")
    record.update(status="submitted", job_id=job_id)
    write_json(recovery / "repair.json", record)
    plan.setdefault("smoke_history", []).append(record)
    plan["jobs"]["smoke"] = job_id
    write_json(run_dir / "plan.json", plan)
    print(json.dumps({"status": "smoke_resubmitted", "job_id": job_id, "run_dir": str(run_dir), "repair": str(recovery)}, indent=2))
    if args.queue_development:
        queue_development(argparse.Namespace(run_dir=run_dir, submit=True, partitions=args.partitions), repo=code)


def queue_development(args, *, repo=None):
    """Queue the full existing Tea/Sea threshold grid after the correctness check."""
    repo = repo or Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.expanduser().resolve()
    compatible_runtime(repo, run_dir)
    plan = json.loads((run_dir / "plan.json").read_text())
    partitions = list(dict.fromkeys(args.partitions or [plan["gpu_partition"]]))
    if any(p not in {"h200p", "l40sp"} for p in partitions):
        raise ValueError("Use separate h200p/l40sp experiment roots; do not mix hardware in one manifest")
    spec = importlib.util.spec_from_file_location("ce6190_manifest", repo / "src/manifest.py")
    manifest_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest_module)
    config = json.loads((repo / "configs/validation_grid.json").read_text())
    config["methods"] = [x for x in config["methods"] if x["method"] in {"baseline", "teacache", "seacache"}]
    prompts = manifest_module.load_prompts(repo / "configs/development_prompts.jsonl")
    manifest_module.assert_disjoint(prompts, manifest_module.load_prompts(repo / "benchmarks/vbench/VBench_full_info.json"))
    rows = manifest_module.build_manifest(prompts, config)
    # RGB arrays are retained for fidelity; include both hardware-specific copies.
    frame_bytes = sum(r["num_frames"] * r["height"] * r["width"] * 3 for r in rows) * len(partitions)
    stored_bytes = sum(path.stat().st_size for partition in partitions
                       for path in (run_dir / ("development_" + partition) / "results").glob("*/frames.npy"))
    remaining_bytes = max(0, frame_bytes - stored_bytes)
    free_bytes = shutil.disk_usage(run_dir).free
    if free_bytes < remaining_bytes * 1.15:
        raise ValueError(f"Development needs about {frame_bytes / 2**30:.1f} GiB of raw frames plus metadata/videos; only {free_bytes / 2**30:.1f} GiB free")
    detail = {"prompts": len(prompts), "rows_per_partition": len(rows), "partitions": partitions,
              "methods": [x["method"] for x in config["methods"]], "raw_frames_gib": frame_bytes / 2**30,
              "concurrency_policy": "no additional cap", "not_the_final_vbench_test": True}
    if not args.submit:
        print(json.dumps({"status": "preview_only", **detail}, indent=2))
        return
    smoke_file = run_dir / "smoke.json"
    smoke_state = json.loads(smoke_file.read_text()) if smoke_file.exists() else {}
    if smoke_state.get("status") == "failed":
        raise ValueError("Smoke failed; do not queue development until repaired")
    dependency = None if smoke_state.get("status") == "passed" else "afterok:" + str(plan["jobs"]["smoke"])
    print(json.dumps({"status": "queueing_development", **detail}, indent=2))
    for partition in partitions:
        directory = run_dir / ("development_" + partition)
        directory.mkdir(parents=True, exist_ok=True)
        config_path, manifest_path = directory / "config.json", directory / "manifest.jsonl"
        previous_plan_path = directory / "plan.json"
        previous_plan = json.loads(previous_plan_path.read_text()) if previous_plan_path.exists() else {}
        if previous_plan and previous_plan["depends_on_smoke"] != plan["jobs"]["smoke"]:
            raise ValueError("Development was queued against a different smoke attempt; inspect its jobs before reuse")
        effective_dependency = previous_plan.get("dependency", dependency)
        content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        if manifest_path.exists() and manifest_path.read_text() != content:
            raise ValueError("Existing development manifest differs; refusing to overwrite")
        write_json(config_path, config)
        manifest_path.write_text(content)
        write_json(directory / "plan.json", {**detail, "partition": partition,
                   "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(),
                   "depends_on_smoke": plan["jobs"]["smoke"], "dependency": effective_dependency})
        command = [sys.executable, str(repo / "scripts/submit_slurm.py"), "--manifest", str(manifest_path),
                   "--output-dir", str(directory / "results"), "--wan-repo", str(run_dir / "vendor/wan"),
                   "--checkpoint-dir", plan["checkpoint"], "--partition", partition, "--account", plan["account"],
                   "--qos", plan["qos"], "--python-executable", str(run_dir / "venv/bin/python"),
                   "--python-module", plan["python_module"], "--cuda-module", plan["cuda_module"],
                   "--runtime-dir", str(run_dir), "--require-smoke", str(run_dir / "smoke.json"),
                   "--chdir", str(run_dir / "code"), "--submit"]
        if effective_dependency:
            command += ["--dependency", effective_dependency]
        run(command)


def phase(args):
    run_dir = args.run_dir.resolve()
    runtime_environment(run_dir)
    state = {"phase": args.phase, "status": "running", "job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(run_dir / (args.phase + ".json"), state)
    try:
        if args.phase == "env":
            environment_stage(run_dir)
        elif args.phase == "model":
            model_stage(run_dir)
        else:
            for prerequisite in ("env", "model"):
                if json.loads((run_dir / (prerequisite + ".json")).read_text()).get("status") != "passed":
                    raise RuntimeError(f"Prerequisite did not pass: {prerequisite}")
            run([run_dir / "venv/bin/python", __file__, "_smoke", "--run-dir", run_dir])
        state["status"] = "passed"
    except Exception as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(run_dir / (args.phase + ".json"), state)


def status(args):
    run_dir = args.run_dir.resolve()
    plan = json.loads((run_dir / "plan.json").read_text())
    phases = {}
    for name in ("env", "model", "smoke"):
        path = run_dir / (name + ".json")
        phases[name] = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
    ready = all(x["status"] == "passed" for x in phases.values())
    development = []
    for directory in sorted(run_dir.glob("development_*")):
        receipt = directory / "results/slurm_submission.json"
        if not receipt.is_file():
            continue
        submission = json.loads(receipt.read_text())
        counts = {"complete": 0, "failed": 0, "running": 0, "unreadable": 0}
        for path in (directory / "results").glob("*/metrics.json"):
            try:
                key = json.loads(path.read_text()).get("status", "unreadable")
            except (OSError, ValueError):
                key = "unreadable"
            counts[key] = counts.get(key, 0) + 1
        development.append({"directory": str(directory), "submission_status": submission["status"],
                            "jobs": submission.get("submitted_jobs", []), "planned_runs": submission["runs"],
                            "outputs": counts, "note": "Output states can be stale after an interrupted job; verify with squeue/sacct"})
    bundle = None
    if args.bundle:
        bundle = run_dir / "startup_report.zip"
        paths = [run_dir / name for name in ("plan.json", "env.json", "model.json", "smoke.json",
                 "model_integrity.json", "gpu.json", "smoke_parity.json", "smoke_fidelity.jsonl", "environment_actual.txt", "cluster_resources.json")]
        paths += list((run_dir / "logs").glob("*.log"))
        paths += list(run_dir.glob("recovery/*/repair.json"))
        for directory in run_dir.glob("development_*"):
            paths += [directory / "config.json", directory / "plan.json", directory / "results/slurm_submission.json"]
        for name in ("metrics.json", "trace.json"):
            paths += list((run_dir / "smoke_results").glob("*/" + name))
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                if path.is_file():
                    # Keep the diagnostic bundle small even if a build printed extensively.
                    with path.open("rb") as handle:
                        size = path.stat().st_size
                        handle.seek(max(0, size - 2 * 1024 * 1024))
                        data = handle.read()
                    relative = str(path.relative_to(run_dir))
                    archive.writestr(relative + (".tail" if size > len(data) else ""), data)
    print(json.dumps({"status": "FIRST_SMOKE_PASSED" if ready else "NOT_READY", "run_dir": str(run_dir),
                      "jobs": plan["jobs"], "phases": phases, "logs": str(run_dir / "logs"),
                      "videos": str(run_dir / "smoke_results"), "checkpoint": plan["checkpoint"],
                      "development": development,
                      "report_bundle": str(bundle) if bundle else None}, indent=2))
    if plan["jobs"] and shutil.which("sacct"):
        subprocess.run(["sacct", "-j", ",".join(plan["jobs"].values()),
                        "--format=JobID,JobName,State,ExitCode,Elapsed", "--noheader"], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit", help="Preview or submit environment, model, and first GPU check")
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--qos", default="highpriority")
    p.add_argument("--cpu-partition", default="cpup")
    p.add_argument("--gpu-partition", default="h200p")
    p.add_argument("--python-module", default="python/3.11.9")
    p.add_argument("--cuda-module", default="cuda/12.4.1")
    p.add_argument("--time", help="Optional per-job wall time; omitted uses Slurm policy defaults, not unlimited time")
    p.add_argument("--min-free-gb", type=float, default=40)
    p.add_argument("--submit", action="store_true")
    p = sub.add_parser("status", help="Read job states and startup reports")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--bundle", action="store_true", help="Write a small ZIP of diagnostic reports/logs; no model or videos")
    for command in ("resume-smoke", "queue-development"):
        p = sub.add_parser(command, help="Reuse the existing environment/model and repair or continue its experiment jobs")
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--partitions", nargs="+", help="Separate paired experiment sets; e.g. h200p l40sp")
        p.add_argument("--submit", action="store_true")
        if command == "resume-smoke":
            p.add_argument("--queue-development", action="store_true")
            p.add_argument("--time", help="Optional wall time for the replacement smoke job")
    for command in ("_phase", "_download", "_smoke"):
        p = sub.add_parser(command, help=argparse.SUPPRESS)
        p.add_argument("--run-dir", type=Path, required=True)
        if command == "_phase":
            p.add_argument("--phase", choices=("env", "model", "smoke"), required=True)
    args = parser.parse_args()
    if args.command == "submit":
        submit(args)
    elif args.command == "status":
        status(args)
    elif args.command == "resume-smoke":
        resume_smoke(args)
    elif args.command == "queue-development":
        queue_development(args)
    elif args.command == "_phase":
        phase(args)
    else:
        runtime_environment(args.run_dir.resolve())
        (download_model if args.command == "_download" else gpu_smoke)(args.run_dir.resolve())


if __name__ == "__main__":
    main()
