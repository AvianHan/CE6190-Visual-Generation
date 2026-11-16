"""Build a code-submission ZIP from explicit source/evidence paths; never delete inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = "deliverables/CE6190_Friend_Handoff_20261106/report_handoff_20261113"
TREES = ("src", "scripts", "tests", "configs", "benchmarks", "assets",
         "results/drawbench_flux", "results/wan_schedules", "analysis", "docs",
         "report/figures/qualitative_inputs", "report/tables")
REPORT_FIGURES = ("caching_framework", "wan_speed_fidelity", "flux_speed_fidelity",
                  "flux_cases_restyled", "wan_cases_restyled", "wan_component_tradeoffs",
                  "flux_modal_schedules")
FILES = ("README.md", "GUIDE.md", "LICENSE", ".gitignore", "pyproject.toml",
         "requirements.txt", "requirements-flux.txt", "report/README.md",
         "report/references.bib", "report/reference_audit.json",
         "report/mathematical_analysis.md", "report/sen_plateau_proof.md",
         "report/report.tex", "report/report.pdf", "report/authors.tex",
         "deliverables/CE6190_Friend_Handoff_20261106/calibration/magcache.json",
         "deliverables/CE6190_Friend_Handoff_20261106/calibration/sencache.json",
         *(f"report/figures/{name}.pdf" for name in REPORT_FIGURES))

ARCHIVED = (
    "analysis/phaseC_summary.csv", "analysis/phaseC_summary.json",
    "analysis/phaseC_denoise_speedup.json", "analysis/phaseC_fidelity.py",
    "analysis/make_tables.py", "analysis/make_figures.py", "analysis/make_qualitative.py",
    "phaseC/frozen_2.0x_denoise_test.json", "phaseC/VBench_subset_info.json",
    "phaseC/manifest_test.jsonl", "phaseD/manifest_ablation.jsonl",
    "tools/batch_worker.py", "tools/worker.py", "tools/gpu_env.sh",
    "tools/check_run.py", "tools/accept_grid.py",
    "tools/eval_incremental.py", "tools/run_vbench.py", "tools/vbench_phaseC.py",
)
REQUIRED = ("src/flux_cache.py", "src/drawbench_provenance.py", "src/wan_adapter.py",
            "results/drawbench_flux/summary_all.csv", "results/drawbench_flux/sen_plateau_records.zip",
            "analysis/sen_plateau_audit.json")
ALLOWED = {".py", ".json", ".jsonl", ".csv", ".md", ".txt", ".toml", ".bib",
           ".png", ".jpg", ".jpeg", ".pdf", ".tex"}


def select_files(root):
    selected = set(FILES) | {f"{HANDOFF}/{p}" for p in ARCHIVED}
    for folder in TREES:
        for path in (root / folder).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Symlinks are not packaged: {path}")
            if not path.is_file() or "__pycache__" in path.parts or path.name == ".DS_Store":
                continue
            rel = path.relative_to(root).as_posix()
            if path.suffix in ALLOWED or path.name == "LICENSE" or rel == "results/drawbench_flux/sen_plateau_records.zip":
                selected.add(rel)
    missing = sorted(p for p in selected | set(REQUIRED) if not (root / p).is_file())
    if missing:
        raise FileNotFoundError("Missing source/evidence files: " + ", ".join(missing))
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = select_files(ROOT)
    manifest = {
        "status": "Report and selected reproducibility materials; use the course group ID in submission filenames",
        "base_main": "7a23544a006eabfeca45648a79f189069d0ef2f5",
        "drawbench_source": "3e27f5c39b318b2ba2e60d75582467b28cbbd615",
        "excluded": ["duplicate handoff code", "vendor archives", "weights", "image/video datasets",
                     "raw frames", "full run_metadata.zip", "temporary build products", "historical validation", "editorial review files",
                     "old result workbook", "unfinished report template"],
        "limits": "Selected evidence only. Full archived metrics and cases remain in the repository/supplement; this ZIP cannot regenerate missing pixels.",
        "files": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
                  for p in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(ROOT / path, path)
        archive.writestr("SUBMISSION_CONTENTS.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
    print(json.dumps({"output": str(args.output.resolve()), "files": len(paths) + 1,
                      "bytes": args.output.stat().st_size, "crc": "pass"}))


if __name__ == "__main__":
    main()
