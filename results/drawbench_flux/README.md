# DrawBench × FLUX.1-dev: Uncached / TeaCache / MagCache / SeaCache / SenCache

Text-to-image part of the project (owner: Jiang Nan). The write-up for the report is `REPORT_SECTION.md`, and its figure is `speed_vs_lpips.png`.

## Files

| File | Content |
|---|---|
| `REPORT_SECTION.md` | Report section: setup, matched-speed table, findings, limitations |
| `examples_2.45x.jpg` | 4 randomly drawn prompts (seed 0) × Uncached / Tea / Mag / Sea / Sen at about 2.45× (`scripts/qualitative_drawbench.py`) |
| `speed_vs_lpips.png` | Speedup vs. LPIPS for all configurations, including the static-schedule control |
| `summary_all.csv` | One row per configuration: speedup, refresh ratio, PSNR/SSIM/LPIPS, CLIP, ImageReward, schedule statistics |
| `pairwise_lpips.csv` | Paired LPIPS differences, 95% bootstrap CIs and win rates; check both speed columns |
| `schedules.csv` | Per configuration: number of distinct compute/reuse schedules, the most common schedule, and steps computed in the first 6 |
| `per_image_metrics_all.csv` | Every image × configuration (200 prompts × 23 configurations + uncached) |
| `sensitivity_flux_dev_28step_1024.json` | SenCache FLUX calibration: 64 DOCCI images, J_x/J_t per step, raw values |

Images are not stored in Git (about 6 GB). They are on the server under
`/dataDisk/home/jiangnan/ce6190-runtime/runs/drawbench_flux_28step{,_sencache}/<config>/db_XXX.{png,json}`.

## Settings

- **Model:** FLUX.1-dev, revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`, 1024×1024, 28 FlowMatch-Euler steps, guidance 3.5, BF16, T5 max length 512.
- **Run:** seed 0, batch 1, one H200.
- **Software:** diffusers 0.32.2, torch 2.6.0+cu124.
- **Upstream code:** pinned commits from `src/sources.py`.

## Code

| Path | Role |
|---|---|
| `src/flux_cache.py` | Shared FLUX forward and the per-method WHEN rules. Includes our SenCache port (`SenGate`) and the static control (`FixedGate`) |
| `tests/test_flux_cache.py` | CPU/FP32 tiny-model checks against pinned upstream forwards and Wan Sen decisions (8 tests; not full BF16 model equivalence) |
| `scripts/calibrate_sencache_flux.py` | SenCache sensitivity calibration for FLUX (port of `Wan2.1/sensitivity_calculation.py`) |
| `scripts/run_drawbench_flux.py` | Generation and timing; all configurations of a shard run on one GPU, so speedups are paired |
| `scripts/eval_drawbench.py` | PSNR/SSIM/LPIPS against uncached, CLIP score (ViT-L/14), ImageReward |
| `scripts/qualitative_drawbench.py` | Example image grid |
| `scripts/report_drawbench.py` | Merges runs into the tables and the figure in this folder |
| `benchmarks/drawbench/drawbench_prompts.csv` | The 200 DrawBench prompts |

## Reproduce

```bash
PY=python   # env with diffusers==0.32.2, torch 2.6, image-reward, lpips
V=/path/to/vendor   # python -m cachebench.sources --directory $V
CE6190_VENDOR=$V $PY -m pytest -q tests/test_flux_cache.py

# Optional: existing matching profile can be reused. Recalibration requires DOCCI: docci_descriptions.jsonlines + images; 64 train images)
$PY scripts/calibrate_sencache_flux.py --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 --descriptions D/docci_descriptions.jsonlines \
    --images-dir D/images --num-samples 64 --output CAL
$PY scripts/calibrate_sencache_flux.py --output CAL --merge

# Generation: run 1 = Tea/Mag/Sea grid, run 2 = SenCache, fill-in points and static control
$PY scripts/run_drawbench_flux.py --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 --vendor $V --output RUN1 --shard i/n --resume
$PY scripts/run_drawbench_flux.py --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 --vendor $V --output RUN2 --resume \
    --sen-calibration CAL/sensitivity_flux_dev_28step_1024.json \
    --configs uncached sencache:0.5 sencache:0.6 sencache:0.8 sencache:1.0 sencache:2 \
              teacache:0.5 magcache:0.09 seacache:0.45 fixed:16 fixed:14 fixed:12 fixed:10 fixed:9
$PY scripts/eval_drawbench.py --run RUN1 && $PY scripts/eval_drawbench.py --run RUN2
$PY scripts/report_drawbench.py --runs RUN1 RUN2 --out results/drawbench_flux
```

RUN2's uncached images were checked against RUN1's: all 200 are bit-identical, and the median time ratio is 0.9999.

## Record-derived analysis

`sen_plateau_records.zip` is a compact subset of the supplied supplement: 200 unchanged threshold-2 JSONs, declared provenance, original source, and profile. It contains no images. `scripts/audit_sen_plateau.py` validates the decisions; the proof is in `report/sen_plateau_proof.md`. It does not execute a model.

The cleaned runner adds immutable invocation records and safe resume checks. These changes postdate the experiments. Keep old results unchanged; legacy records without the required provenance cannot be resumed by the new runner. New generation should use a new output directory.
