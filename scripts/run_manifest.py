#!/usr/bin/env python3
"""Sequential local launcher; use the Slurm wrapper for independent GPUs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def existing_status(output_dir, row):
    """Return a trustworthy completion state; incomplete outputs never count."""
    directory = Path(output_dir) / row["run_id"]
    if not directory.exists():
        return "new"
    path = directory / "metrics.json"
    if not path.is_file():
        return "incomplete"
    metadata = json.loads(path.read_text())
    if any(metadata.get("config", {}).get(key) != value for key, value in row.items()):
        raise ValueError(f"Existing output config differs for {row['run_id']}")
    status = metadata.get("status", "incomplete")
    if status in {"complete", "completed", "ok", "success"}:
        return "complete" if all((directory / x).is_file() for x in ("frames.npy", "video.mp4")) else "incomplete"
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--wan-repo", required=True); p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--start", type=int, default=0); p.add_argument("--stop", type=int)
    p.add_argument("--method"); p.add_argument("--split"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="Record failures, continue other runs, return nonzero at end")
    p.add_argument("--resume", action="store_true", help="Skip only complete outputs with matching configuration and required files")
    p.add_argument("--retry-failed", action="store_true", help="Allow explicit retry of directories marked failed; never overwrite completed/running outputs")
    a = p.parse_args()
    rows = [json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    failed = []
    for i in range(a.start, min(a.stop if a.stop is not None else len(rows), len(rows))):
        row = rows[i]
        if a.method and row["method"] != a.method or a.split and row["split"] != a.split:
            continue
        status = existing_status(a.output_dir, row)
        if a.resume and status == "complete":
            print(json.dumps({"index": i, "run_id": row["run_id"], "action": "skip_verified_complete"}))
            continue
        cmd = [sys.executable, "-m", "cachebench.generate", "--manifest", str(Path(a.manifest).resolve()),
               "--index", str(i), "--output-dir", str(Path(a.output_dir).resolve()),
               "--wan-repo", str(Path(a.wan_repo).resolve()), "--checkpoint-dir", str(Path(a.checkpoint_dir).resolve())]
        cmd += ["--vendor-dir", str(Path(a.wan_repo).resolve().parent)]
        if a.retry_failed and status == "failed":
            cmd += ["--overwrite-failed"]
        if a.dry_run:
            print(json.dumps({"index": i, "run_id": row["run_id"], "command": cmd}))
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode:
            failed.append({"index": i, "run_id": row["run_id"], "returncode": result.returncode})
            if not a.keep_going:
                break
    if failed:
        out = Path(a.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "launcher_failures.json").write_text(json.dumps(failed, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
