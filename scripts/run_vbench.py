#!/usr/bin/env python3
"""Run the pinned official VBench evaluator against an already staged suite.

Temporal flickering must use the static-filtered directory. Do not evaluate all
25 candidates as the official post-filter score. See CODEBASE_GUIDE.md.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vbench-repo", required=True); p.add_argument("--stage-dir", required=True)
    p.add_argument("--dimension", nargs="+", required=True)
    p.add_argument("--output", required=True); p.add_argument("--videos-override")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    repo = Path(a.vbench_repo).resolve(); stage = Path(a.stage_dir).resolve()
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    expected = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
    if revision != expected:
        raise ValueError(f"VBench revision {revision} differs from pinned {expected}")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("VBench tracked sources modified; use a clean pinned evaluator")
    details = json.loads((stage / "staging.json").read_text())
    if "temporal_flickering" in a.dimension and not a.videos_override:
        p.error("For temporal_flickering first run official static_filter.py, then --videos-override its filtered_videos directory")
    videos = Path(a.videos_override).resolve() if a.videos_override else stage / "videos"
    cmd = [sys.executable, str(repo / "evaluate.py"), "--videos_path", str(videos),
           "--full_json_dir", str(stage / "VBench_info.json"), "--output_path", str(Path(a.output).resolve()),
           "--mode", "vbench_standard", "--dimension", *a.dimension]
    print(json.dumps({"coverage": details["protocol"], "command": cmd}))
    if a.execute:
        out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
        (out / "evaluator_provenance.json").write_text(json.dumps({"vbench_commit": revision, "command": cmd, "staging": details}, indent=2))
        subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()
