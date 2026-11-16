# Beyond Skip Counts

**Understanding Adaptive Caching for Efficient Generation** · CE6190 Group Project

[Report PDF](report/report.pdf) · [LaTeX source](report/report.tex) · [Reproduction guide](GUIDE.md) · [Learning notes](docs/learning_notes.md)

We benchmark **Uncached, TeaCache, MagCache, SeaCache and SenCache** for text-to-image and text-to-video generation. Within each backbone, the methods share the cached block scope and residual reuse; the comparison focuses on **when to reuse and when to recompute**.

## Experiments

| Task | Backbone / hardware | Evaluation |
| --- | --- | --- |
| Text-to-video | Wan2.1-T2V-1.3B / A100 | 110-prompt VBench subset; 550 paired samples per cached method; 16 VBench dimensions; threshold and component studies |
| Text-to-image | FLUX.1-dev / H200 | All 200 DrawBench prompts, seed 0; 24 configurations / 4,800 outputs; fidelity, CLIP, ImageReward and schedule analysis |

SenCache on FLUX is a project port. The historical H200/L40S development results are kept separately from the A100 comparison. PSNR, SSIM and LPIPS measure deviation from paired Uncached outputs; content-quality metrics answer a separate question.

## Main findings

- **Equal computation counts do not imply equal fidelity.** Compared with TeaCache at matched computation counts and nearly identical runtime, MagCache lowers Wan LPIPS by 40.3% and SenCache lowers FLUX LPIPS by 51.2%.
- **The SeaCache–SenCache comparison changes with the measured budget.** On FLUX, SeaCache has lower LPIPS near 1.7× speedup; SenCache has lower LPIPS near 2.45×.
- **Reuse constraints can set a computation floor.** In the SenCache threshold-2 FLUX setting, all 200 prompts recompute at the same 11 of 28 steps. After the early stages, the consecutive-reuse limit forces every recomputation.

The report connects these findings to component ablations, recorded schedules, mathematical analysis and visual examples.

## Repository

| Path | Contents |
| --- | --- |
| `report/` | Report, bibliography, tables, figures and mathematical analysis |
| `src/`, `scripts/`, `tests/` | Cache implementations, experiment/analysis tools and checks |
| `analysis/`, `results/` | Report inputs, delivered metrics and schedule records |
| `configs/`, `benchmarks/` | Settings, calibration descriptions and benchmark prompts |
| `deliverables/` | Original handoffs, execution snapshots and experiment evidence |
| `docs/`, `assets/` | Learning notes and their original figures |
| `validation/` | Historical implementation-check records |

Start with the [Wan handoff](deliverables/CE6190_Friend_Handoff_20261106/report_handoff_20261113), [FLUX results](results/drawbench_flux/README.md) and [report build instructions](report/README.md). Selected visual outputs are retained; complete generated videos and images are not included.

## Course submission

The assignment requires at least two methods and one dataset, own experimental results, parameter analysis, component ablations, comparisons, visualizations and discussion of improvements. Methods must differ from Project 1; the report must contain at least four pages.

Submit `group_XX_report_project2.pdf` and `group_XX_code_project2.zip`, replacing `XX` with the group ID. The recorded deadline is **17 November 2026**; check NTULearn for the submission time. Submit the files themselves and exclude model weights and datasets from the code ZIP.

```bash
python scripts/package_submission.py --output /path/to/group_XX_code_project2.zip
```
