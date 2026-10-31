# English report source template

**Status: TEMPLATE ONLY. No GPU measurements, project result figures, speedup claims, or quality scores have been inserted. Do not submit the unfilled template as a completed report.**

This directory contains source files only:

- `report_template.tex`: an English report scaffold with methods background, experimental protocol, empty tables, analysis prompts, contributions, and reproducibility sections.
- `references.bib`: six verified primary references, with version-pinned SeaCache v2 and SenCache v2 entries.
- `README.md`: completion and build instructions.

The main study guide is separate. No paper result figures have been copied into the report as if they were this project's outputs.

## Build locally

Use a TeX distribution containing `latexmk`, `pdflatex`, BibTeX, and the standard packages named in the source. From this `report/` directory:

```bash
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report_template.tex
```

The temporary preview is `build/report_template.pdf`. It will visibly say **TEMPLATE: RESULTS PENDING** on every page. A successful compilation only verifies the document source; it does not validate any GPU implementation or experimental result.

Without `latexmk`, run from this directory:

```bash
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report_template.tex
```

Then run BibTeX from `build/`, making the bibliography file visible:

```bash
cd build
BIBINPUTS=..: bibtex report_template
cd ..
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report_template.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report_template.tex
```

The source has been designed for ordinary PDFLaTeX; it does not need custom fonts, Python, shell escape, network access, or a conference-style package. Build products should stay outside the submission source tree or be removed before packaging.

Compile check: on 2026-10-29, PDFLaTeX/BibTeX through `latexmk` successfully rendered an eight-page temporary preview. The final log contained no undefined citations/references, LaTeX warnings, or overfull boxes; the rendered pages were inspected. That temporary PDF is deliberately not included in this directory or the source ZIP.

## Complete the report

1. Confirm the exact checkpoint, sampler, hardware, and code revisions actually used. The background is not a substitute for auditing the implemented cache boundary and branch rules.
2. Freeze calibration/validation/test separation and threshold selection before final testing. Store the prompt manifest and configuration hashes.
3. Run the experiments. Keep failures and retries; do not silently drop difficult samples or replace missing measurements with published values.
4. Fill the protocol table from actual configurations. Populate each result row from saved measurements and scripts, including sample counts and uncertainty.
5. Add latency-quality curves and synchronized qualitative examples from the group's own videos. Keep fidelity-to-reference separate from video quality, text alignment, and motion.
6. Complete hyperparameter analysis, key-component ablations, pros/cons, failure analysis, and a supported future extension. Remove proposed experiments that were never run, or explicitly label them as future work.
7. Rewrite the abstract, introduction contributions, discussion, and conclusion around the completed evidence. No result claim in this template is intended to survive unchanged without supporting experiments.
8. Add actual group members and their contributions. Re-run the analysis from raw logs and inspect the final PDF page by page.
9. Only after all placeholders have been replaced, remove the draft title/header, instruction boxes, and unused scaffold material. Search for `TBD`, `Fill:`, `pending`, `Template instruction`, and `placeholder` as a final check.

The generic residual equation is a teaching abstraction. Do not present it as proof that every repository uses identical cache boundaries. TeaCache's Wan proxy may depend only on the time branch in the selected official version; document what the version actually reads. SenCache's practical score is an estimate, not a guaranteed bound on final video quality.

## Course submission

Verified against the supplied `0. coursework 2026.pdf`:

- Group Project is **40%**, due **17 November 2026** (PDF p2). The PDF does not state an hour; check NTULearn for the exact deadline time.
- Efficient generation is allowed, and methods must not overlap Project 1 (p11).
- At least **two methods** and **one evaluation dataset** (p12).
- Own results, hyperparameter analysis, key-component ablation, comparisons, strengths/weaknesses, visualization, and future extension (p13).
- Report **at least four pages**, with introduction/methods/experiments and analysis/conclusion (p13). Empty scaffold pages do not satisfy substantive analysis.
- Final files: `group_XX_report_project2.pdf` and `group_XX_code_project2.zip`, with the real group ID (p6).
- Submit actual PDF and ZIP files; no external/cloud links as substitutes. Exclude datasets and model weights from the code ZIP (p6–7).

Do not include a draft preview PDF in the code archive. After the report is complete, copy/rename the final rendered PDF to the required report filename as a separate submission. Keep these `.tex`, `.bib`, and documentation sources in the code archive if useful for reproducibility.

## Bibliography provenance

Primary records checked on 2026-10-29:

| Key | Verified source |
|---|---|
| `chung2026seacache` | https://arxiv.org/abs/2602.18993v2 — v2, 24 March 2026 |
| `haghighi2026sencache` | https://arxiv.org/abs/2602.24208v2 — v2, 20 September 2026 |
| `liu2025teacache` | https://arxiv.org/abs/2411.19108v2 — CVPR 2025 |
| `ma2025magcache` | https://arxiv.org/abs/2506.09045v2 — NeurIPS 2025 |
| `wan2025wan` | https://arxiv.org/abs/2503.20314v2 — v2, 19 April 2025 |
| `huang2024vbench` | https://arxiv.org/abs/2311.17982 and https://github.com/Vchitect/VBench — official CVPR 2024 citation |

SeaCache and SenCache are cited by the exact arXiv versions used for reading, so settings can be traced without assuming all revisions are identical. The Wan collective author is abbreviated as Team Wan et al. The VBench entry refers to the original benchmark, not VBench++ or VBench-2.0. Add references for any additional metric implementation, dataset, or extension actually used.
