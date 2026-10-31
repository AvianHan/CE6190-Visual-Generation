#!/usr/bin/env python3
"""Preview or submit a Slurm array without an additional concurrency/time cap.

Slurm account, QoS, partition, and array-size limits still apply. Optional
--concurrency and --time are explicit overrides, not project defaults.
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "output-dir", "wan-repo", "checkpoint-dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--partition"); p.add_argument("--account"); p.add_argument("--qos")
    p.add_argument("--concurrency", type=int, help="Optional maximum parallel array tasks; omitted adds no %%N cap")
    p.add_argument("--time", help="Optional wall time; omitted uses Slurm policy defaults, not unlimited time")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--memory", default="64G"); p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    n = sum(bool(x.strip()) for x in Path(a.manifest).read_text().splitlines())
    if n == 0 or (a.concurrency is not None and a.concurrency < 1):
        p.error("Nonempty manifest and, when specified, positive concurrency required")
    output = Path(a.output_dir).resolve()
    (output / "slurm_logs").mkdir(parents=True, exist_ok=True)
    array = f"0-{n - 1}" + (f"%{a.concurrency}" if a.concurrency is not None else "")
    cmd = ["sbatch", "--parsable", "--array", array, "--gres", "gpu:1",
           "--cpus-per-task", str(a.cpus), "--mem", a.memory,
           "--output", str(output / "slurm_logs/%A_%a.log")]
    if a.partition:
        cmd += ["--partition", a.partition]
    if a.account:
        cmd += ["--account", a.account]
    if a.qos:
        cmd += ["--qos", a.qos]
    if a.time is not None:
        cmd += ["--time", a.time]
    cmd += [str(Path(__file__).with_name("slurm_job.sh").resolve()), str(Path(a.manifest).resolve()),
            str(output), str(Path(a.wan_repo).resolve()), str(Path(a.checkpoint_dir).resolve())]
    print(json.dumps({"runs": n, "max_parallel_gpus": a.concurrency,
                      "concurrency_policy": "explicit --concurrency" if a.concurrency is not None else "Slurm allocation; no additional array cap",
                      "time_policy": "explicit --time" if a.time is not None else "Slurm policy default",
                      "command": cmd}))
    if a.submit:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
