# Report

**Beyond Skip Counts: Understanding Adaptive Caching for Efficient Generation**

[PDF](report.pdf) · [LaTeX source](report.tex)

The 14-page report uses the two-column CVPR 2027 author kit (`cvpr.sty`, `ieeenat_fullname.bst`, tag `CVPR2027-v1` of https://github.com/cvpr-org/author-kit) in camera-ready mode with page numbers. It includes 7 figures, 8 tables, 46 references and 3 appendices. The four equal-contribution authors are set in `authors.tex`, after a small teal GROUP 7 tag (`\grouptag` in `preamble.tex`): each name has the student ID below it, followed by an envelope icon that links to the member's email. The equal-contribution note is an unnumbered footnote at the foot of the left column on page 1 (placed right after `\maketitle`, as for `\thanks` notes in CVPR papers); the repository link is the last sentence of the abstract; the appendix starts on its own page. Report-specific packages, colours and table macros are in `preamble.tex`.

## Build

Use TeX Live with PDFLaTeX, BibTeX and `latexmk`. From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex
cp build/report.pdf report.pdf
```

The CVPR style and bibliography style files are included in this directory; no shell escape is required.

## Rebuild tables and figures

From the repository root, with NumPy, Matplotlib and Pillow installed:

```bash
python scripts/make_report_tables.py
python scripts/make_report_figures.py --formats pdf
python scripts/make_report_cases.py
python scripts/make_method_figure.py
python scripts/make_overview_figure.py
```

Numeric inputs are in `analysis/report_data/`. The generated tables use the macros in `preamble.tex` (method colour swatches, bold best values, shaded lowest LPIPS, coloured changes); each caption states its emphasis rule. Qualitative inputs and their checksums/crop provenance are in `figures/qualitative_inputs/`; the scripts reuse those pixels without generating or enhancing images. `tables/method_rules.tex` is maintained manually. Wide tables are set in normal size except the FLUX table (small, to fit its relative-change labels); table rows use `\arraystretch` 1.25. These sizes were chosen so that the body fills page 11 while every float stays within one page of its first reference; check both after changing a table or figure size.

Figure 1 is the final overview `figures/caching_overview.pdf` (6.875 × 5.1 in, STIXGeneral), drawn by `scripts/make_overview_figure.py` from the DrawBench thumbnails in `figures/qualitative_inputs/` and the recorded FLUX schedules; `report.tex` falls back to `figures/caching_framework.pdf` only if the overview file is missing. Its caption describes panels (a)–(d) one by one, and the text cites each panel where it is explained. The drawing brief and review notes are in [figure-briefs/](figure-briefs/fig1_overview.md).

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
