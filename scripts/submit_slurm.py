#!/usr/bin/env python3
"""Submit a bounded Slurm array; defaults to print-only until --submit is used."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "output-dir", "wan-repo", "checkpoint-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--partition"); p.add_argument("--account"); p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--time", default="04:00:00"); p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--memory", default="64G"); p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    n = sum(bool(x.strip()) for x in Path(a.manifest).read_text().splitlines())
    if n == 0 or a.concurrency < 1:
        p.error("Nonempty manifest and positive concurrency required")
    output = Path(a.output_dir).resolve()
    (output / "slurm_logs").mkdir(parents=True, exist_ok=True)
    cmd = ["sbatch", "--parsable", "--array", f"0-{n - 1}%{a.concurrency}", "--gres", "gpu:1",
           "--cpus-per-task", str(a.cpus), "--mem", a.memory, "--time", a.time,
           "--output", str(output / "slurm_logs/%A_%a.log")]
    if a.partition:
        cmd += ["--partition", a.partition]
    if a.account:
        cmd += ["--account", a.account]
    cmd += [str(Path(__file__).with_name("slurm_job.sh").resolve()), str(Path(a.manifest).resolve()),
            str(output), str(Path(a.wan_repo).resolve()), str(Path(a.checkpoint_dir).resolve())]
    print(json.dumps({"runs": n, "max_parallel_gpus": a.concurrency, "command": cmd}))
    if a.submit:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
