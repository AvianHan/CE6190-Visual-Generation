# Report

**Beyond Skip Counts: Understanding Adaptive Caching for Efficient Generation**

[PDF](report.pdf) · [LaTeX source](report.tex)

The 14-page single-column report includes 7 figures, 8 tables, 46 references and 3 appendices. The four authors in `authors.tex` are marked as equal contributors.

## Build

Use TeX Live with PDFLaTeX, BibTeX and `latexmk`. From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex
cp build/report.pdf report.pdf
```

The source uses standard LaTeX packages; no conference style file or shell escape is required.

## Rebuild tables and figures

From the repository root, with NumPy, Matplotlib and Pillow installed:

```bash
python scripts/make_report_tables.py
python scripts/make_report_figures.py --formats pdf
python scripts/make_report_cases.py
python scripts/make_method_figure.py
```

Numeric inputs are in `analysis/report_data/`. Qualitative inputs and their checksums/crop provenance are in `figures/qualitative_inputs/`; the scripts reuse those pixels without generating or enhancing images. `tables/method_rules.tex` is maintained manually.

## Supporting analysis

- [Mathematical analysis](mathematical_analysis.md): computation placement and conditional error propagation.
- [SenCache plateau analysis](sen_plateau_proof.md): reuse constraints and the recorded threshold-2 schedules.
- [Reference audit](reference_audit.json): primary-source metadata and supported claims for all 46 references.
- `analysis/sen_plateau_audit.json`: retained trace-validation results.
- `analysis/report_data/vbench_source_verification.json`: checks against the 80 original VBench evaluator JSONs.

## Experiment versions

| Component | Recorded version |
| --- | --- |
| Wan execution snapshot | `7a23544a006eabfeca45648a79f189069d0ef2f5` |
| DrawBench study snapshot | `3e27f5c39b318b2ba2e60d75582467b28cbbd615` |
| FLUX.1-dev model | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| Generation stack | PyTorch 2.6.0+cu124; Diffusers 0.32.2 |
| Upstream cache implementations | Pinned in `src/sources.py` |

See the repository [reproduction guide](../GUIDE.md) for execution and data locations, and the [README](../README.md#course-submission) for submission filenames.
