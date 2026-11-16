# Reproduction guide

## Evidence and entry points

| Task | Code | Results |
| --- | --- | --- |
| Wan generation | `src/wan_adapter.py`, `src/seacache_base.py`, `src/generate.py` | [Executed Wan handoff][wan] |
| FLUX generation | `src/flux_cache.py`, `scripts/run_drawbench_flux.py` | [FLUX results](results/drawbench_flux/README.md) |
| Metrics and aggregation | `src/evaluate.py`, `src/aggregate.py`, `scripts/eval_drawbench.py`, `scripts/report_drawbench.py` | `analysis/report_data/`, `results/` |
| Schedule analysis | `scripts/wan_schedule_audit.py`, `scripts/audit_sen_plateau.py` | `results/wan_schedules/`, `analysis/sen_plateau_audit.json` |
| Report assembly | `scripts/make_report_*.py`, `scripts/make_method_figure.py` | [Build instructions](report/README.md) |

The Wan handoff contains the executed `code/`, `tools/`, manifests, acceptance records and per-sample evaluations. Its final manifests define the measured experiments; root `configs/main.json` and `configs/ablations.json` are earlier reference configurations. Use final analysis files rather than `*_provisional*`.

## Reanalyze retained results

The committed CSV/JSON data and report can be used without models or GPUs. The read-only SenCache audit checks the 200 retained threshold-2 records:

```bash
python scripts/audit_sen_plateau.py results/drawbench_flux/sen_plateau_records.zip --output /path/to/sen-audit.json
```

For additional retained run directories:

```bash
python scripts/report_drawbench.py --runs /path/to/run1 /path/to/run2 --out /path/to/new-analysis
python scripts/wan_schedule_audit.py --source /path/to/wan-run-metadata --out /path/to/new-schedules.csv
```

The FLUX aggregator needs each run's `per_image_metrics.csv` and per-image JSON records. It uses the first run's Uncached reference, so combine only runs with matching reference outputs and settings. Keep hardware groups separate and write new analysis to a new directory.

Wan summaries are in the handoff's `analysis/phaseC_summary.json`, `analysis/phaseC_denoise_speedup.json`, `analysis/grid_summary.json` and `analysis/ablation_table.csv`; the original VBench scores are in `vbench_out/`. Some archived scripts contain server paths that must be adjusted when reused. Recomputing pixel-based metrics requires the original images/frames; only selected visual outputs are archived here.

## Environment

The measured generation stack used PyTorch 2.6.0 + CUDA 12.4 and Diffusers 0.32.2. For generation or metric evaluation on a compatible NVIDIA system:

```bash
python3.11 -m venv /path/to/ce6190-env
source /path/to/ce6190-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Wan additionally needs FlashAttention:

```bash
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

FLUX evaluation/aggregation additionally uses:

```bash
python -m pip install -r requirements-flux.txt
python -m pip check
```

`requirements-flux.txt` includes the recorded ImageReward version and a compatible pandas range; the original pandas version was not recorded. CLIP, ImageReward and VBench require evaluation weights. VBench used a separate environment, documented in the handoff's `PLAN.md` and archived `logs/pip-freeze-vbench.txt`.

The installed package name is `cachebench`; `pyproject.toml` maps it to `src/`. Retrieve and verify pinned upstream sources with:

```bash
python -m cachebench.sources --directory /path/to/vendor
python -m cachebench.sources --directory /path/to/vendor --verify-only
```

## Execution interfaces

Use an allocated GPU, the frozen manifest, matching calibration profiles and a new output directory. The entry points for single-task Wan generation and paired-result evaluation are:

```bash
python -m cachebench.generate --manifest /path/to/frozen-manifest.jsonl --index 0 --output-dir /path/to/new-runs --dry-run
python -m cachebench.generate --manifest /path/to/frozen-manifest.jsonl --index 0 --output-dir /path/to/new-runs --vendor-dir /path/to/vendor --checkpoint-dir /path/to/Wan2.1-T2V-1.3B
python -m cachebench.evaluate fidelity --manifest /path/to/frozen-manifest.jsonl --root /path/to/new-runs --output /path/to/fidelity.jsonl --metrics psnr ssim lpips --device cuda
python -m cachebench.aggregate --input /path/to/fidelity.jsonl --output /path/to/summary.json
```

Existing MagCache/SenCache profiles can be reused when the checkpoint and generation settings match. Wan used eight Commons videos for calibration. The SenCache FLUX port used 64 DOCCI images; its profile is retained in `results/drawbench_flux/`.

FLUX commands are in the [results README](results/drawbench_flux/README.md). Its runner now records immutable invocation metadata and checks configuration-compatible resumes. These safeguards postdate the reported experiments; legacy outputs without the required provenance should remain unchanged and cannot be resumed by this runner.

The executed VBench wrapper is `tools/run_vbench.py` in the Wan handoff. The study used `--allow-subset` and official static filtering for temporal flickering. Retain the 110-prompt subset label and dimension-specific sample counts when reporting those results.

## Historical setup

The Orion Group runtime was retired on 6 November. Slurm startup/recovery scripts and `validation/` retain earlier procedures and implementation checks. Their original instructions remain in the [frozen guide][old-guide]; those checks are separate from generation-quality results. Historical checksum files describe their own snapshots.

[wan]: deliverables/CE6190_Friend_Handoff_20261106/report_handoff_20261113
[old-guide]: https://github.com/AvianHan/CE6190-Visual-Generation/blob/7a23544a006eabfeca45648a79f189069d0ef2f5/GUIDE.md
