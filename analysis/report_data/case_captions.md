# Verified qualitative assets

## Wan shared figure note

Wan2.1-T2V-1.3B, Phase C frozen configuration: TeaCache 0.12, MagCache 0.12 (K=4), SeaCache 0.2, and SenCache 0.5 (K=3), alongside Uncached. All columns show the same prompt and seed; columns are frames 1, 41, and 81. The configuration targeted 2x denoising speed, not 2x end-to-end speed. The saved selection record contains 550 eligible prompt-seed pairs with all four cached results and sample_index < 5. These are metric-selected extremes, not randomly selected or representative cases. Selection timing before visual inspection is claimed by the supplied script, but is not independently timestamp-verified.

### wan_qualitative_success.png — 1026 x 999

**Low-deviation Wan example.** Prompt 0dea713ccf226afa, seed 2: “a banana on the top of an apple, front view.” This pair minimizes the worst-method LPIPS among the 550 eligible pairs in the saved selection. The displayed frames retain the reference arrangement across all methods (LPIPS 0.0051–0.0083). “Success” here means fidelity to the uncached output, not an independent assessment of video quality.

Pair ID: 3bed2fd412cca0c9e73d.

### wan_qualitative_failure.png — 1361 x 999

**High-deviation Wan example.** Prompt fd17d76d4ced9971, seed 2: “A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, zoom out.” This pair maximizes mean LPIPS across the four cached methods in the same 550-pair selection. People and umbrella configurations visibly differ from the reference. SeaCache has the lowest deviation in this example (0.128), followed by MagCache (0.223), SenCache (0.342), and TeaCache (0.347). The three still frames do not establish motion smoothness.

Pair ID: 3594e7177730de4da5e2.

## FLUX shared figure note

FLUX.1-dev, DrawBench, generation seed 0. Configurations are Uncached, TeaCache 0.5, MagCache 0.12, SeaCache 0.6, and the project’s SenCache FLUX port at threshold 2. Their dataset-median paired speedups are approximately 2.45x; individual-image runtimes need not match. The original four prompts (db_053, db_101, db_126, db_167) were drawn without replacement using selection RNG seed 0. The two cases below were subsequently chosen from that existing grid to illustrate specific interpretation issues, not as a new random sample.

### flux_existing_examples_2.45x.jpg — 1770 x 1558

**Four reproducibly sampled DrawBench examples.** Rows are db_053, db_101, db_126, and db_167; columns follow the configuration order above. LPIPS and PSNR compare each cached output with the same prompt/seed uncached output. The examples illustrate observable changes in composition and appearance; they do not establish overall human-perceived quality rankings.

### db_053 originals — five 1024 x 1024 PNGs

**Reference fidelity does not guarantee prompt correctness.** DrawBench db_053, seed 0: “Three cats and three dogs sitting on the grass.” The uncached output visibly contains two cats and two dogs. Closely reproducing this reference therefore preserves a counting error. This error is already present without caching and should not be attributed to the acceleration method.

Files: flux_db_053_uncached.png; flux_db_053_teacache_0.5.png; flux_db_053_magcache_0.12.png; flux_db_053_seacache_0.6.png; flux_db_053_sencache_2.png.

### db_101 originals — five 1024 x 1024 PNGs

**A visible change in object appearance.** DrawBench db_101, seed 0, asks for a donkey and an octopus playing with a rope while a cat jumps over it. Relative to the uncached image, the TeaCache 0.5 output gives the octopus-like character long ears and changes its face and pose. This illustrates departure from the reference rather than proving a general quality ranking.

Exact prompt: “A donkey and an octopus are playing a game. The donkey is holding a rope on one end, the octopus is holding onto the other. The donkey holds the rope in its mouth. A cat is jumping over the rope.”

Files: flux_db_101_uncached.png; flux_db_101_teacache_0.5.png; flux_db_101_magcache_0.12.png; flux_db_101_seacache_0.6.png; flux_db_101_sencache_2.png.

## Provenance and report details still needed

- Wan figures and selection metadata were read from Git commit 7a23544a006eabfeca45648a79f189069d0ef2f5, under deliverables/CE6190_Friend_Handoff_20261106/report_handoff_20261113/. The matching Phase C manifest confirms prompt IDs, seeds, thresholds, and the frozen configuration. No figure was regenerated.
- The FLUX grid was copied unchanged from results/drawbench_flux/examples_2.45x.jpg. The ten originals were extracted unchanged from the supplied CE6190_FLUX_Cases_and_Sen_Records_20261113(1).zip. The original archive and repository were not edited.
- Current report/README.md explicitly leaves group number, names, and contributions unfinished. The inspected case evidence does not provide a verified complete member list or contribution allocation. Do not invent these details or rubric score weights.
