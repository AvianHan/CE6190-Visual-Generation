# 项目概述 / Project Overview

## 1. 项目简介 / Introduction

**中文：** 本项目是 CE6190 Visual Generation 课程 Assignment 2 的小组研究项目，聚焦于**通过自适应缓存加速文本到视频生成**。项目以 Wan2.1-T2V-1.3B 为统一生成骨干，比较不使用缓存的基线与 TeaCache、MagCache、SeaCache、SenCache 四种缓存方法，研究推理速度、输出保真度和视频质量之间的权衡。项目定位是可复现的实验比较与技术分析，主要复用固定版本的上游模型和缓存实现，并提供实验组织、校准、评测及集群执行代码。

**English:** This group project for Assignment 2 of CE6190 Visual Generation studies **adaptive caching for efficient text-to-video generation**. Using Wan2.1-T2V-1.3B as a shared backbone, it compares an uncached baseline with TeaCache, MagCache, SeaCache, and SenCache to investigate trade-offs among inference speed, output fidelity, and video quality. The project focuses on reproducible experimental comparison and technical analysis. It reuses pinned upstream model and caching implementations and adds calibration, evaluation, experiment management, and cluster execution code.

## 2. 研究问题 / Research Questions

| 中文 | English |
| --- | --- |
| 在相邻去噪步骤中，何时可以复用缓存，何时必须重新计算？ | When can cached computation be reused across denoising steps, and when must it be refreshed? |
| 在相同实际加速预算下，哪种缓存策略更好地保持输出保真度与视频质量？ | Which caching policy best preserves output fidelity and video quality at a matched measured speedup? |
| 阈值、连续复用限制及校准信息如何影响速度与质量？ | How do thresholds, consecutive-reuse limits, and calibration information affect speed and quality? |
| SeaCache 与 SenCache 的关键组件分别贡献了什么？ | What do the key components of SeaCache and SenCache contribute? |

## 3. 方法与共同计算路径 / Methods and Shared Computation

**中文：** 五种方法共用从固定版本 SeaCache 源码提取的核心去噪循环，通过适配层切换缓存 forward。四种缓存方法共同复用整个 Transformer block stack 的残差：完整计算时保存 `r = h_out - h_in`，复用时计算 `h_current + r_cached`。不同方法保留各自的缓存判别信号、阈值及保护规则。适配层检查共同计算结构，以减少非缓存因素对比较的干扰；源码结构检查仍需真实 GPU 数值实验验证。

**English:** All five methods share a core denoising loop extracted from a pinned SeaCache source revision, with an adapter selecting the caching forward implementation. The four caching methods reuse a residual across the full Transformer block stack: a full computation stores `r = h_out - h_in`, and a reuse step computes `h_current + r_cached`. Each method retains its own decision signals, thresholds, and protection rules. The adapter audits the shared computation structure to reduce confounding factors, although source-level checks still require numerical validation on real GPUs.

| 方法 / Method | 项目中的角色 / Role in This Project |
| --- | --- |
| Uncached（配置名 `baseline`） | 每个去噪步骤执行完整计算，提供速度和输出参考。 / Full computation at every denoising step; reference for timing and output fidelity. |
| TeaCache | 自适应缓存比较方法，保留固定上游实现的判别规则。 / Adaptive caching comparator using the decision rules of the pinned upstream implementation. |
| MagCache | 使用幅值相关校准信息进行复用决策；幅值比不用于缩放复用残差。 / Uses magnitude-related calibration for reuse decisions; magnitude ratios do not scale the reused residual. |
| SeaCache | 重点分析方法之一，同时提供共同去噪循环的来源。 / A primary analysis target and the source of the shared denoising loop. |
| SenCache | 使用敏感度校准信息进行决策，并分析 latent/time 分量。 / Uses sensitivity calibration for decisions, with analysis of latent and time components. |

**中文：** 校准不更新模型参数。MagCache 使用独立 prompts；SenCache 使用独立真实视频及其描述。`src/policies.py` 中的标量策略用于教学和校验，实际 GPU forward 由 `src/wan_adapter.py` 对接上游实现。

**English:** Calibration does not update model parameters. MagCache uses separate prompts, while SenCache uses separate real videos and descriptions. Scalar policies in `src/policies.py` support teaching and validation; actual GPU forward implementations are integrated through `src/wan_adapter.py`.

## 4. 实验设置与数据 / Experimental Setup and Data

以下为仓库当前参考配置，不代表已调优参数或已完成实验。

The following settings describe the repository's current reference configuration, not tuned parameters or completed experiments.

| 项目 / Item | 设置 / Setting |
| --- | --- |
| 生成模型 / Backbone | Wan2.1-T2V-1.3B |
| 视频规格 / Video format | 832 × 480，81 帧 / frames，16 FPS（约 / approximately 5.06 s） |
| 采样 / Sampling | 50 steps，UniPC，CFG = 5，flow shift = 5 |
| 精度与注意力 / Precision and attention | BF16，FlashAttention2 |
| 批大小 / Batch size | 1 |
| CPU offload | 当前参考配置关闭 / Disabled in the current reference configuration |
| 计时 / Timing | 预热 1 次，测量 3 次取中位数 / One warmup, median of three timed runs |
| 主测试参考种子 / Reference test seeds | 0, 1, 2, 3, 4 |
| 数据集 / Dataset | 固定版本 VBench prompts 与维度标签 / Pinned VBench prompts and dimension labels |

**中文：** 仓库中的 VBench 清单包含 946 条记录、944 条不同文本 prompts 和 16 个评测维度。项目使用这些 prompts 自行生成视频，而非直接引用论文成绩。另有 64 条 development prompts 用于调参；校准、调参和测试数据应相互隔离。`negative_prompt: null` 表示采用固定 Wan 源码的默认负提示，不表示空文本。当前协议是统一设置下的比较，不应直接视为原论文表格复现或 VBench 排行榜成绩。

**English:** The bundled VBench manifest contains 946 records, 944 distinct prompt texts, and 16 evaluation dimensions. The project generates its own videos from these prompts rather than treating published scores as project results. A separate set of 64 development prompts supports tuning; calibration, tuning, and test data must remain disjoint. `negative_prompt: null` selects the default negative prompt from the pinned Wan source, rather than an empty string. This is a comparison under shared settings, not an exact reproduction of paper tables or a VBench leaderboard result.

## 5. 评测与实验流程 / Evaluation and Workflow

| 评测方面 / Aspect | 内容 / Description |
| --- | --- |
| 效率 / Efficiency | 测量热运行延迟及相对 uncached 的实际加速比；不同 GPU 型号分别汇总。 / Measure warm-run latency and speedup relative to uncached generation; summarize GPU models separately. |
| 输出保真度 / Output fidelity | 使用相同 prompt 和 seed 的 uncached 输出作为参考，计算 PSNR、SSIM、LPIPS。 / Compute PSNR, SSIM, and LPIPS against uncached outputs with the same prompt and seed. |
| 视频质量 / Video quality | 使用 VBench 分析一致性、运动、视觉质量及语义相关维度。 / Use VBench to assess consistency, motion, visual quality, and semantic dimensions. |
| 消融 / Ablations | SeaCache：`official`、`raw_proxy`、`no_gain_normalization`；SenCache：`official`、`latent_only`、`time_only`。 / Compare the listed SeaCache and SenCache variants. |
| 汇总 / Aggregation | 参数一致性检查、prompt 级 bootstrap、速度—质量 Pareto 分析及阈值冻结。 / Configuration consistency checks, prompt-level bootstrap, speed–quality Pareto analysis, and threshold freezing. |

**中文：** 计时包含文本编码、去噪和 VAE 解码，排除模型加载、策略初始化、CPU 传输和磁盘编码。保真度衡量与 uncached 输出的接近程度，不能替代视频质量评估。每个生成任务保存 `video.mp4`、`frames.npy`、`metrics.json` 和 `trace.json`。

**English:** Timing includes text encoding, denoising, and VAE decoding, while excluding model loading, policy initialization, CPU transfers, and disk encoding. Fidelity measures similarity to uncached outputs and does not replace video quality evaluation. Each generation task saves `video.mp4`, `frames.npy`, `metrics.json`, and `trace.json`.

1. **准备与正确性检查 / Preparation and correctness checks:** 固定源码及权重，准备环境，检查 uncached 与 SeaCache 全计算路径。 / Pin sources and weights, prepare the environment, and check uncached and full-computation SeaCache paths.
2. **独立校准 / Independent calibration:** 为 MagCache 和 SenCache 生成与实验配置匹配的校准文件。 / Produce MagCache and SenCache calibration profiles matching the experiment configuration.
3. **调参与冻结 / Tuning and freezing:** 在 development 数据上搜索阈值，在达到目标实际加速比的候选中选择 LPIPS 最低者，冻结参数。 / Search thresholds on development data, select the lowest-LPIPS candidate meeting the target measured speedup, and freeze parameters.
4. **正式测试 / Final testing:** 使用冻结参数生成 VBench 视频，完成配对保真度与 VBench 评测。 / Generate VBench videos with frozen parameters and evaluate paired fidelity and VBench quality.
5. **分析与报告 / Analysis and reporting:** 完成组件消融、速度—质量比较、可视化、失败案例分析和技术报告。 / Complete component ablations, speed–quality comparisons, visualizations, failure analysis, and the technical report.

## 6. 代码结构 / Code Structure

| 路径 / Path | 功能 / Purpose |
| --- | --- |
| `src/sources.py` | 固定并核对上游源码版本。 / Pin and verify upstream source revisions. |
| `src/manifest.py` | 构建 prompts × seeds × 方法的任务清单，检查数据隔离。 / Build prompts × seeds × methods manifests and check data separation. |
| `src/wan_adapter.py`, `src/seacache_base.py` | 模型适配、缓存接入、共同去噪循环及执行跟踪。 / Model adaptation, cache integration, shared denoising, and execution tracing. |
| `src/generate.py`, `src/calibrate.py` | 单任务生成和方法校准。 / Per-task generation and method calibration. |
| `src/evaluate.py`, `src/aggregate.py` | 保真度评测、VBench 数据准备、统计汇总与参数冻结。 / Fidelity evaluation, VBench staging, statistical summaries, and parameter freezing. |
| `configs/` | 参考配置、正确性检查、阈值搜索、development prompts 和消融设置。 / Reference settings, correctness checks, threshold grids, development prompts, and ablations. |
| `benchmarks/vbench/` | VBench prompt 清单、来源记录及许可证。 / VBench prompt manifest, provenance, and license. |
| `scripts/` | Orion/Slurm 启动、任务数组、队列管理、硬件检查及校准恢复工具。 / Orion/Slurm startup, job arrays, queue management, hardware checks, and calibration recovery tools. |
| `tests/` | 集群编排和恢复流程等自动化检查。 / Automated checks for cluster orchestration and recovery workflows. |
| `validation/` | 历史验证记录，不是正式实验成绩。 / Historical validation records, not final experimental results. |
| `report/` | LaTeX 报告模板、参考文献和提交说明。 / LaTeX report template, bibliography, and submission guidance. |

**中文：** Python 包名为 `ce6190-cachebench`，`src/` 安装后映射为 `cachebench`，因此模块入口采用 `python -m cachebench.<module>`。依赖配置见 `requirements.txt` 和 `pyproject.toml`。运行流程见 [GUIDE.md](GUIDE.md)；其中历史文件清单和命令示例应与当前实际文件核对，例如该文档提及的 `scripts/run_manifest.py`、`scripts/run_vbench.py` 和 `scripts/plot_results.py` 在当前检出中未提供。

**English:** The Python distribution is named `ce6190-cachebench`; installation maps `src/` to the `cachebench` package, so module entry points use `python -m cachebench.<module>`. Dependencies are defined in `requirements.txt` and `pyproject.toml`. See [GUIDE.md](GUIDE.md) for workflow details, but check its historical file listings and command examples against the current checkout: for example, its referenced `scripts/run_manifest.py`, `scripts/run_vbench.py`, and `scripts/plot_results.py` are not present here.

## 7. 当前状态与预期交付 / Current Status and Deliverables

**中文：** 当前仓库包含实验框架、参考配置、集群调度与恢复工具、历史验证记录及报告模板。依据仓库内的状态说明，目前不能宣称已完成正式 GPU 性能比较或取得 VBench 成绩；外部运行目录中的最新进展需要结合实际日志另行确认。`configs/main.json` 明确标为未调优参考设置，`configs/ablations.json` 标为计划配置，`report/` 仍是待填入真实结果的模板。历史离线检查不能替代完整模型的 GPU 验证。

**English:** The repository contains the experimental framework, reference configurations, cluster scheduling and recovery tools, historical validation records, and a report template. Repository status notes do not establish a completed GPU performance comparison or measured VBench scores; any newer progress in external runtime directories must be confirmed from actual logs. `configs/main.json` explicitly contains untuned reference settings, `configs/ablations.json` contains planned configurations, and `report/` remains a template awaiting real results. Historical offline checks do not replace GPU validation of the complete model.

**中文：** 预期交付包括可复现的实验代码与配置、自产视频及评测记录、超参数与消融分析、速度—质量可视化，以及满足课程要求的技术报告。未来可探索基于 FLUX 的文本到图像扩展；该扩展及 SenCache 的相关迁移尚未实现。

**English:** Intended deliverables include reproducible experiment code and configurations, generated videos and evaluation records, hyperparameter and ablation analyses, speed–quality visualizations, and a technical report meeting the course requirements. A future direction is FLUX-based text-to-image generation; this extension and the corresponding SenCache adaptation are not yet implemented.

## 8. 项目内参考文档 / Repository References

- [README.md](README.md)：研究背景、方法笔记与课程要求。 / Research background, method notes, and coursework requirements.
- [GUIDE.md](GUIDE.md)：实现说明、实验协议与运行指南。 / Implementation notes, experimental protocol, and execution guide.
- [Assignment_2_Summary.md](Assignment_2_Summary.md)：课程作业要求。 / Assignment requirements.
- [configs/main.json](configs/main.json)：当前统一参考配置。 / Current shared reference configuration.
- [benchmarks/vbench/PROVENANCE.json](benchmarks/vbench/PROVENANCE.json)：数据版本及来源。 / Dataset version and provenance.
- [report/README.md](report/README.md)：报告状态、编译和提交说明。 / Report status, build instructions, and submission guidance.
