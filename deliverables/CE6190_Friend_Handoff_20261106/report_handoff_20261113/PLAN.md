# Receiver run plan — A100-SXM4-80GB ×8 (2026-11-10, Asia/Shanghai)

Work dir: `/home/ecs-user/ce6190-runtime/a100x8-20261110` (new, created by handoff `tools/prepare.py`).
Hardware group key: `a100-sxm4-80gb-x8/<run_id>`. All timing comparisons are within this group only.

## Phase A — handoff grid (mandated by handoff README)
* 1,411 tasks = `pending.jsonl` (1,409) + `new_hardware_baselines.jsonl` (2); 97 prompt–seed pairs,
  33 development prompts (dev_03_07 … dev_07_07), seeds 0–2, 15 policy settings.
* Unchanged protocol: 832×480, 81 f, 16 fps, UniPC 50, CFG 5, shift 5, BF16, FA2, offload off,
  warmup 1, timing repeats 3, thresholds from the manifest, reused calibration profiles.
* One `python -m cachebench.generate` per task (handoff code snapshot), one process per GPU,
  bound by GPU UUID. Pair-affinity scheduling (uncached + cached of a pair on one GPU, LPT order),
  single-task stealing only at the tail.
* Acceptance per task: `tools/check_run.py` (status, 100 CFG calls ×3 traces, uncached all refresh,
  frames uint8 (81,480,832,3), MP4 decodes, fingerprint b6725fb6…, profile hashes).
* No overwrites; failed/partial → `attention`, never silently rerun.

## Phase B — paired fidelity on the grid
`cachebench.evaluate fidelity` (PSNR/SSIM/LPIPS vs same-seed uncached frames.npy), CPU while GPUs run.
`cachebench.aggregate` → per-policy bootstrap summary, Pareto, freeze thresholds at a measured
speedup target on this tune split (default 2.0×, lowest LPIPS).

## Phase C — frozen test on VBench prompts (subset; documented deviation)
Full VBench × 5 seeds × 5 methods is ~120k generations — infeasible. Stratified subset of the pinned
VBench prompt suite (disjoint from dev/calibration prompts), frozen thresholds, paired seeds.
Fidelity vs uncached + VBench dimensions on the subset (custom subset, not leaderboard protocol).

## Phase D — ablations (ablations.json variants)
SeaCache raw_proxy / no_gain_normalization; SenCache latent_only / time_only, on a subset of grid
pairs that reuses the same-GPU-model uncached references.

## Phase E — analysis + report
Speed–fidelity Pareto, per-threshold hyper-parameter analysis, refresh schedules from traces,
ablations, VBench comparison, qualitative frames, failure cases; fill `report/report_template.tex`.

## Log of deviations / decisions
* Model fetched with a parallel range downloader (`tools/fetch_model_parallel.py`); identical
  per-file size + sha256/git-blob verification as handoff `fetch_model.py`. Reason: ~100 Mbit/s link.
* Conda base channels need ToS acceptance; Python 3.11 taken from conda-forge instead.
* 12:22 First smoke check passed all checks but ran with mixed cuDNN (pip 9.1 front-end + host 9.19.1
  sub-libraries via host LD_LIBRARY_PATH). Results quarantined unchanged in quarantine/; all GPU processes
  now run with the venv's NVIDIA libs only (tools/gpu_env.sh, worker env); check_run requires cuDNN 90100.
  Smoke re-run with the fixed environment before any batch.
* VBench env: separate venv (py3.10, torch 2.7.1+cu128, transformers 4.33.2, VBench fd18b3d).
  detectron2 (needed only for GRiT dims: object_class, multiple_objects, color, spatial_relationship)
  NOT installed: building it from GitHub was denied by the permission classifier -> needs user decision.
* 15:35 User explicitly authorized the detectron2 build. Cloned facebookresearch/detectron2 and pinned
  247691a1c292654867deb35cabb27f18d75a1363 (2026-10-27); building into venv_vbench only (nice 19, MAX_JOBS=16,
  CUDA 12.8, arch 8.0) via tools/build_detectron2.sh -> logs/detectron2_build.log. Generation env untouched.
* 15:30 Grid status: 96/1411 done, 0 failures. Non-timed overhead ~30%/task (67 s model load + 1 warm-up),
  as mandated by the handoff protocol (fresh process per task); not changed.
* 15:34 detectron2 0.6 @247691a1 built OK in venv_vbench (detectron2._C imports; torch unchanged 2.7.1+cu128). GRiT dims unblocked.

## Progress / next actions (hourly trigger picks up from here)
* 2026-11-12 00:02 Grid 1265/1411, 0 failures, ETA ~04:00. Hourly reports: recurring trigger at :01.
* When grid done (1411 in state/manifest/done, workers exited):
  1. Acceptance: `venv/bin/python tools/check_run.py results/*/` over all runs; any problem -> attention list, no reruns.
  2. GPUs free -> start Phase D (phaseD/manifest_ablation.jsonl, 336 tasks; no dependency on frozen thresholds)
     with tools/batch_worker.py via tools/gpu_env.sh, one per GPU.
  3. Meanwhile CPU: let eval_loop finish remaining paired fidelity, then Phase B `cachebench.aggregate`
     -> Pareto + freeze thresholds (2.0x measured speedup, lowest LPIPS) -> record frozen values here.
  4. Build Phase C manifest from phaseC/VBench_subset_info.json with frozen thresholds; run after Phase D.
* 03:10 Grid 1374/1411 (all 97 pairs claimed, tail). Phase D rows validated (336 valid, 24 reference baselines complete,
  no run_id clashes). Started tools/chain_phaseD.sh per GPU: when that GPU's grid worker reports finished_no_work and
  exits, it launches batch_worker on Phase D (results/ + state/manifest_ablation). Halted grid workers do NOT chain.
  Stop flag: state/STOP_CHAIN. Logs: logs/workers/phaseD_gpu*.log.
* 04:01 Phase A DONE: 1411/1411 complete, 0 failed. Acceptance (tools/accept_grid.py, check_run incl. MP4 decode):
  1411 ok, 0 problems, 0 warnings -> acceptance/grid_check.jsonl.
* 04:01 Phase D running on all 8 GPUs (chained automatically as each grid worker finished). Early SeaCache
  raw_proxy / no_gain_normalization @0.1 show refresh_ratio 1.0 (never reuse) -> analysis item, not a fault.
* 04:05 Phase B: fidelity records 1314/1314 (0 errors) -> eval/grid_fidelity.jsonl. Official aggregate refuses
  unequal pair sets: cells finished on the sender's H200 are missing here and cross-hardware mixing is forbidden.
  DEVIATION: aggregate on the 86 pairs common to all 14 cached policies (29 prompts x seeds 0-2; 1204 records)
  -> eval/grid_fidelity_common.jsonl, eval/grid_common_pairs.json, analysis/grid_summary.{json,csv}. Official
  function unchanged.
* 04:08 Freeze: max end-to-end speedup on A100 is ~1.72x (fixed ~99 s FP32 VAE decode + forced-refresh steps).
  Official freeze at 2.0x -> "magcache cannot reach measured 2.0x" => 2.0x REPORTED NOT REACHED on A100.
  DEVIATION: froze at 1.6x = highest target every method reaches (magcache max 1.648x); target chosen from
  speeds only, not LPIPS. analysis/frozen_1.6x.json: magcache 0.24 (1.648x, LPIPS .0795), seacache 0.35
  (1.710x, .1168), sencache 0.5 (1.645x, .1226), teacache 0.16 (1.719x, .1708). User may override before Phase C.
* 04:15 Phase D fidelity: tools/eval_incremental.py gained --ref-manifest (references from grid manifest/state;
  default behaviour unchanged, backup .bak); loop tools/eval_loop_phaseD.sh running. hourly_report shows Phase D.
* NEXT: (1) when Phase D done -> accept (accept_grid.py on phaseD manifest), assemble, aggregate ablations.
  (2) Phase C: frozen_1.6x with seeds [0..4] (+temporal_flickering_25 per GUIDE) on phaseC subset (110 prompts):
  ~110x5x5=2750 gens, est. ~18 h on 8 GPUs with batch_worker -> build manifest, check size, launch after Phase D.
  (3) denoise-only speedup table (records carry denoise_seconds) alongside end-to-end.
* 05:10 Phase D 173/336, 0 failed; fidelity 166 records, 0 errors.
* 05:15 Phase C manifest: phaseC/frozen_1.6x_test.json (= analysis/frozen_1.6x.json + seeds [0..4],
  temporal_flickering_25=true per GUIDE; warmup 0 / timing_repeats 1 for batch_worker) -> phaseC/manifest_test.jsonl:
  110 prompts (100 x 5 seeds + 10 temporal_flickering x 25 seeds) x 5 methods = 3750 runs, 750 pairs, split=test.
  Official --exclude-prompts check passed vs development_prompts.jsonl and calibration_samples.jsonl (the GUIDE's
  data/{mag,sen}_calibration.jsonl do not exist in the handoff code). 3750 rows validate; no run_id clashes.
  Est. ~26 h on 8 GPUs (Phase D walls: seacache 182 s, sencache 155 s; baseline ~285 s).
  tools/chain_phaseC.sh per GPU: starts Phase C when that GPU's Phase D worker finished AND time >= 07:00
  (user window to change target/seeds). Stop: state/STOP_CHAIN_C.
* 06:01 Phase D DONE: 336/336 complete, 0 failed; acceptance 336 ok (24 warnings = seacache no_gain_normalization
  @0.1 never reuses). Fidelity 336/336 -> eval/phaseD_fidelity.jsonl. Aggregated per method vs the official variant
  on common A100 pairs (seacache 21, sencache 24) -> analysis/ablation_*.json, analysis/ablation_table.csv.
  Findings: SeaCache raw_proxy reuses aggressively (refresh .10-.25, up to 2.34x) but LPIPS .60-.66 (quality collapse);
  no_gain_normalization reuses less (never at 0.1) and sits slightly behind official on the speed/LPIPS frontier.
  SenCache latent_only LPIPS ~.41 (bad); time_only == official for thr >= 1 (latent term inactive), official only
  refreshes more at 0.5.
* 06:03 analysis/denoise_speedup.csv: denoise-only vs end-to-end speedup (86 common pairs). Baseline denoise 172.7 s,
  VAE 98.6 s. The e2e-1.6x freeze corresponds to 2.6-2.9x denoise speedup — far more aggressive than the GUIDE's 2x.
* 06:04 DECISION CHANGE (before Phase C started): freeze at 2.0x DENOISE-ONLY speedup, same rule (min LPIPS among
  candidates meeting target) -> analysis/frozen_2.0x_denoise.json: magcache 0.12 (2.37x dn / 1.58x e2e, LPIPS .074),
  seacache 0.2 (2.07x / 1.49x, .044), sencache 0.5 (2.61x / 1.65x, .123), teacache 0.12 (2.37x / 1.58x, .127).
  Reason: preserves the GUIDE 2x budget on the part caching accelerates and comparability with the H200 run; e2e 2.0x
  remains reported as not reached. phaseC/manifest_test.jsonl regenerated (3750 rows, valid, no clashes); the e2e-1.6x
  version kept as phaseC/manifest_test_e2e1.6x.jsonl. chain_phaseC.sh unchanged (starts >= 07:00). Est. ~27 h.
* 07:06 Phase C running on 8 GPUs (chain started 07:00:28; process warm-up done 07:06). Idle grid/phaseD eval loops
  stopped; tools/eval_loop_phaseC.sh running. hourly_report.py now reports Phase C/D via phase().
* 08:02 Phase C 130/3750, 0 failed, ETA ~11-13 08:30; fidelity 96 records, 0 errors.
* 08:04 Report figures (Phase E, CPU only): analysis/make_figures.py -> analysis/figures/{pareto_speed_lpips,
  refresh_schedule,ablation}.{pdf,png} (validated default palette; node unavailable so validator not run — documented
  values used, 4th series carries marker + direct label). Observation: at frozen thresholds TeaCache/MagCache/SenCache
  refresh schedules are (near-)identical across all 86 pairs (prompt-independent); only SeaCache's varies.
* NEXT (CPU while Phase C runs): fill report/report_template.tex sections with grid/ablation results (copy template to
  a working report dir, never overwrite the template); qualitative frame grid from a representative pair.
* 09:02 Phase C 280/3750, 0 failed, ETA ~11-13 08:30; fidelity 200, 0 errors.
* 09:05 Report work dir report/ (report.tex = copy of the template, references.bib, tables/, figures/). tectonic installed
  in texenv/ (conda-forge); template copy compiles (bundle download works). analysis/make_tables.py ->
  report/tables/{grid_tradeoff,efficiency,ablation}.tex from measured files only.
* 09:10 analysis/schedule_stats.json from traces: CFG branches never disagree (0/4300 for all methods); TeaCache and
  MagCache use ONE schedule for all 86 pairs (prompt-independent), SenCache 2/86, SeaCache 9/86 (only materially
  sample-adaptive policy here). Protected leading/trailing full steps and max reuse runs recorded.
* NEXT: write report text (methods with the audited implementation facts, protocol table, grid/ablation results,
  deviations), qualitative frames; Phase C tables/VBench after the test finishes.
* 09:12 Report draft report/report.tex (+ tables/, figures/) compiles with texenv/bin/tectonic -> report/report.pdf
  (9 pages, no overfull boxes). Filled: abstract/intro/methods (audited implementation facts), protocol table, metrics &
  timing (verified: repeat spread median 0.08 s, max 0.5 s <= 0.27%; SSIM/LPIPS protocols from records), matched-budget
  deviation, grid results + Pareto, refresh schedules, ablations, discussion, conclusion, reproducibility.
  Still TBD (orange): group number/member names/contributions (user must supply), Phase C fidelity + VBench, qualitative
  frames, final abstract/conclusion sentences. poppler added to texenv for page previews (report/preview/).
* 10:02 Phase C 424/3750, 0 failed, ETA ~11-13 08:30; fidelity 328, 0 errors.
* 10:05 NOTE fidelity: official evaluate_manifest defaults to samples_per_prompt=5 (extra temporal-flickering seeds are
  for static filtering only); eval_incremental passes None, so records also exist for TF seeds 5-24. For the main Phase C
  fidelity table assemble ONLY sample_index < 5 (official protocol); the extra records may be shown separately.
* 10:08 GUIDE's scripts/run_vbench.py does not exist in the checkout (project_overview.md says so). Wrote
  tools/run_vbench.py: thin wrapper -> official VBench(device, stage/VBench_info.json, out).evaluate(..., local=True,
  mode='vbench_standard') with VBENCH_CACHE_DIR=vbench_cache; records run_info (commit, cache, dims, n_videos).
  CPU smoke test (2 Phase C baseline videos, subject_consistency + object_class/GRiT) running in scratchpad.
* VBench plan after Phase C (per GUIDE): per method stage-vbench --samples 5 --allow-subset -> 15 dimensions;
  temporal_flickering: stage --dimension temporal_flickering --samples 25 -> official static_filter.py -> evaluate the
  filtered videos (keep warnings when < 5 pass). Spread methods x dimension groups over the 8 GPUs.
* 10:15 VBench smoke #1 failed: DINO's hubconf fetches dino_vitbase16_pretrain.pth from dl.fbaipublicfiles.com (expired
  cert for Python SSL) despite local mode. Fix: tools/run_vbench.py sets TORCH_HOME=vbench_cache/torch_home with the
  pre-fetched file hard-linked into hub/checkpoints (no downloads). Smoke #2 (CPU): subject_consistency 0.968,
  object_class 0.4375 (GRiT/detectron2 OK). Smoke #3 (CPU, 10 videos covering all 15 non-flicker dims) running in
  scratchpad vbench_smoke_all; static_filter (RAFT) still to test.
* 10:10 VBench all-15-dimension CPU smoke PASSED (10 Phase C baseline videos, seed 0) after two env fixes:
  (a) dynamic_degree: RAFT models.zip was never extracted and the host has no `unzip` -> extracted with Python zipfile into
      vbench_cache/raft_model/models/ (same as `unzip -d`); a duplicate models.zip.1 re-fetched by VBench was byte-identical
      and removed. (b) background_consistency: openai-clip 1.0.1 imports pkg_resources, removed in setuptools 84 ->
      venv_vbench pinned to setuptools 80.9.0 (VBench env only; logs/pip-freeze-vbench.txt). One download remains by design:
      ViCLIP tokenizer vocab (now cached in vbench_cache/ViCLIP). static_filter.py hard-codes cuda -> test on GPU after
      Phase C (same RAFT weights already exercised by dynamic_degree). With filter_scope=temporal_flickering it iterates the
      official full TF prompt list, so "<5 results" warnings for prompts outside our subset are expected noise.
* 11:02 Phase C 575/3750, 0 failed, ETA ~11-13 08:30; fidelity 439, 0 errors. Disk 295G used (Phase C adds ~100 MB/run).
* 11:08 analysis/make_qualitative.py: pre-declared selection rule (success = lowest worst-method LPIPS, failure = highest
  mean LPIPS, pairs with sample_index<5 and all 4 methods), frames 1/41/81 uncropped -> figures/qualitative_{success,failure}
  + analysis/qualitative_selection.json. Provisional on 104 pairs: failure = "baseball stadium" s0 (Tea .279, Mag .262,
  Sea .168, Sen .257): near-static high-frequency texture whose streak layout drifts; SeaCache keeps layout closest.
  Re-run after Phase C completes (final selection on all 750 pairs).
* 12:01 previous Claude session lost auth (OAuth expired) -> hourly report at 12:01 not done. ~12:14 host rebooted by user:
  all Phase C workers, eval loop and GPU monitor died. State at reboot: 751 done (0 failed) + 8 in flight.
* 12:20 RESTART: the 8 in-flight runs (7 sencache, 1 seacache) had partial output (metrics.json status=running only) ->
  done records written with status 'attention' (partial output untouched, NOT rerun; user decision whether to quarantine
  and rerun). Their pair locks still held by dead workers -> remaining rows of those pairs are picked up by tail stealing.
  Restarted 8 batch_workers directly (no chain; absolute paths so hourly_report counts them), tools/eval_loop_phaseC.sh,
  and the GPU monitor (now appends, noheader). Each worker repeats the ~67 s load + 1 warm-up generation.
* 13:01 Phase C 839/3750 (831 complete + 8 reboot 'attention'), 0 failed, 8 batch_workers, ETA ~11-13 08:45; fidelity 643, 0 errors. Disk 320G.
* 13:10 USER APPROVED: the 8 reboot-interrupted Phase C runs moved (results dir + done + claim) to quarantine/reboot_20261112/
  unchanged; their dead-worker pair locks released too, so live workers re-claim those pairs next and rerun the 8 runs
  from scratch (same run_ids, fresh output dirs). README.json there lists ids.
* 14:02 The 8 reboot reruns all complete (fresh outputs). Phase C 983/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~11-13 08:45; fidelity 768, 0 errors. Disk 334G.
* 19:02 (15:01-18:01 reports queued, not answered; run unaffected) Phase C 1719/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~11-13 08:47; fidelity 1351, 0 errors. Disk 404G.
* 20:02 Phase C 1868/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~11-13 08:46; fidelity 1479, 0 errors. Disk 418G.
* 21:02 Phase C 2013/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~11-13 08:46; fidelity 1590, 0 errors. Disk 432G.
* 22:02 Phase C 2159/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~11-13 08:48; fidelity 1719, 0 errors. Disk 446G.
* 23:02 Phase C 2311/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~11-13 08:46; fidelity 1824, 0 errors. Disk 460G.
* 11-13 00:02 Phase C 2457/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~11-13 08:46; fidelity 1952, 0 errors. Disk 474G.
* 01:02 Phase C 2603/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:47; fidelity 2060, 0 errors. Disk 488G.
* 02:02 Phase C 2754/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:46; fidelity 2189, 0 errors. Disk 502G.
* 03:02 Phase C 2898/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:46; fidelity 2301, 0 errors. Disk 516G.
* 04:02 Phase C 3045/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:47; fidelity 2431, 0 errors. Disk 530G.
* 05:02 Phase C 3195/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:47; fidelity 2534, 0 errors. Disk 544G.
* 05:20 Prepared tools/vbench_phaseC.py (run after Phase C acceptance): official stage-vbench per method (--samples 5
  --allow-subset -> vbench_stage/<m>; temporal_flickering --samples 25 -> vbench_stage/<m>_flicker), official static_filter.py
  on each flicker stage (GPU), then tools/run_vbench.py per (method, dimension) -> vbench_out/<m>/<dim>/. 85 jobs
  (5 x [filter + flicker + 15 dims]) scheduled one per GPU; resume-safe (skips jobs whose *_eval_results.json exists);
  logs/vbench/<job>.log. Compiles; job list builds; staging itself can only run once all 3750 runs exist.
  Run: setsid nohup venv/bin/python tools/vbench_phaseC.py > logs/vbench_phaseC.log 2>&1 &
* 06:02 Phase C 3341/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:47; fidelity 2664, 0 errors. Disk 558G.
* 06:10 tools/after_phaseC.sh running (logs/after_phaseC.log): waits for 3750 done + 0 batch_workers -> accept_grid.py on
  phaseC manifest -> acceptance/phaseC_check.jsonl; only if bad=0 and all done records 'complete' it execs
  tools/vbench_phaseC.py (logs/vbench_phaseC.log, logs/vbench/*). Stop flag: state/STOP_AFTER_C.
* 07:02 Phase C 3488/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:47; fidelity 2776, 0 errors. Disk 572G.
  after_phaseC.sh waiting.
* 07:20 analysis/phaseC_fidelity.py: main table = sample_index<5 only (official), TF seeds 5-24 separate; official
  cachebench.aggregate.summarize unchanged; + denoise-only speedup (ratio of totals). PROVISIONAL (550 pairs/method,
  --provisional -> *_provisional files): denoise/e2e speedup  mag 2.37/1.58, tea 2.37/1.58, sea 2.06/1.49, sen 2.61/1.65
  (= tune-split freeze values, so the 2x denoise budget holds on test); LPIPS sea .048 < mag .087 < sen .142 < tea .146.
  Final run (no --provisional) after Phase C completes.
* 08:02 Phase C 3636/3750, non-complete [], 0 failed, 8 batch_workers, ETA ~08:47; fidelity 2904, 0 errors. Disk 587G. after_phaseC.sh waiting.
* 08:47 PHASE C DONE: 3750/3750 complete, 0 failed. Acceptance (accept_grid.py) 3750 ok, 0 bad, 0 warnings ->
  acceptance/phaseC_check.jsonl. after_phaseC.sh then started tools/vbench_phaseC.py automatically (08:47:51).
* 09:05 VBench: staging OK (5 methods x 110 prompts x 5 seeds; flicker 10 prompts x 25 seeds). Official static_filter:
  baseline & teacache keep 5/5 static videos for all 10 TF prompts. ~43/85 jobs done by 09:05, 0 failures. Note color dim
  scores only videos where GRiT detects the object (36/50 baseline, 35/50 teacache) - official behaviour.
* 09:08 FINAL Phase C fidelity (analysis/phaseC_fidelity.py, no --provisional): 550 pairs/method (110 prompts x 5 seeds;
  the other 200 test pairs are TF seeds 5-24 -> eval/phaseC_fidelity_tfextra.jsonl). Equal pair sets (official summarize
  passed without allow_unpaired). LPIPS [95% CI]: sea .048 [.044,.052], mag .087 [.081,.094], sen .142 [.133,.152],
  tea .146 [.137,.156]; e2e speedup sea 1.49, mag/tea 1.58, sen 1.65; denoise 2.06/2.37/2.37/2.61. Pareto: sea, mag, sen.
  -> analysis/phaseC_summary.{json,csv}, analysis/phaseC_denoise_speedup.json. (Provisional files identical: by 07:20 all
  550 main pairs were already evaluated.)
* 09:10 Final qualitative (550 candidates): success = "a banana on the top of an apple, front view" s2 (all LPIPS <= .008);
  failure = "couple in formal evening wear ... downpour with umbrellas, zoom out" s2 (Tea .347, Sen .342, Mag .223, Sea .128):
  Tea/Sen change the layout (woman's pose, light-blue umbrella turns white/translucent); Mag/Sea stay close to uncached.
* NEXT: when VBench finishes -> per-method x dimension table (analysis), then report Phase C sections (fidelity table,
  VBench table, qualitative figures), final abstract/conclusion. User still to supply group no./names/contributions.
