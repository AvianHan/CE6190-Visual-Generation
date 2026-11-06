# CE6190 组员交接：待完成开发实验

日期：2026-11-06（新加坡）。任务完成状态来自 2026-11-05 19:19 SGT 导出快照；提交前与当前运行状态去重。

本包提供原始配置、manifest、校准产物及版本锁定信息。**不含项目源码、vendor 源码、Wan 权重、视频或原始帧，也不含已确认的 DrawBench 实验方案。不是一键运行包。**

## 1. 直接使用哪些文件

| 文件 | 用途 |
|---|---|
| `pending/l40sp.jsonl` | 原始 manifest 中筛出的 1,409 条 L40S 待完成任务 |
| `pending/h200p.jsonl` | 原始 manifest 中筛出的 1,473 条 H200 待完成任务 |
| `original/l40sp/development_l40sp/config.json` | 完整开发实验配置 |
| `original/l40sp/rolling/manifest.jsonl` | 每种 GPU 共用的完整 2,880 条任务清单，尚未按完成状态筛选 |
| `original/l40sp/calibration/magcache/profile.json` | 已用的 MagCache 校准产物 |
| `original/l40sp/calibration/sencache/profile.json` | 已用的 SenCache 校准产物 |
| `locks/vendor_sources.json` | Wan / Tea / Mag / Sea / Sen 的官方仓库 URL 与精确 commit |
| `original/l40sp/model_integrity.json` | Wan revision、所需文件、文件大小和校验值 |
| `locks/environment.json` | 已完成实验记录的 Python / CUDA / 关键依赖版本 |
| `locks/model_and_project.json` | 模型 fingerprint、原项目快照清单 hash、源码核查要求 |
| `references/required_uncached.csv` | 后续质量评测需要的 238 个原 uncached 参考及服务器原路径；本包不含帧 |
| `audit.json`、`SHA256SUMS.txt` | 派生清单数量、来源与本包文件校验 |

待完成 JSONL 每行逐字保留原 manifest 内容，不重新生成 run_id。两个硬件的 run_id 会重复，因此结果主键必须包含 GPU 分组。不要把两份 JSONL 混写到同一个结果目录。

**不要误用 `development_l40sp/manifest.jsonl`：那只是旧的三个方法子集，缺少 MagCache 和 SenCache。** 本包使用 `rolling/manifest.jsonl`，已与完整 `all_methods.jsonl` 做字节比较。

## 2. 统一配置和任务范围

64 条自编开发 prompts，seeds=0/1/2；Wan2.1-T2V-1.3B；832×480、81 帧、16 fps；UniPC 50 步、CFG=5、shift=5；BF16、FlashAttention 2、offload=false；warmup=1，计时重复=3。精确方法参数以原始 manifest 为准。

| 方法 | 阈值 | L40S 待完成 | H200 待完成 |
|---|---|---:|---:|
| TeaCache | 0.04 / 0.08 / 0.12 / 0.16 | 440 | 454 |
| MagCache | 0.06 / 0.12 / 0.24 | 247 | 282 |
| SeaCache | 0.1 / 0.2 / 0.35 | 247 | 261 |
| SenCache | 0.5 / 1 / 2 / 4 | 475 | 476 |
| 合计 | | 1,409 | 1,473 |

这些任务只包括开发集视频生成和计时，不包括组件消融、PSNR/SSIM/LPIPS、正式 VBench、DrawBench 或失败案例分析。Uncached 两种硬件各 192 项已完成；如果仍在原硬件和可兼容环境接续，保留既有参考。跨硬件或不一致环境不能直接把新耗时并入原 L40S/H200 表，应另建硬件分组，并建立新环境自己的 Uncached 对照。

## 3. 补齐运行资源

1. 项目代码：使用团队仓库，并核对原运行的 `code/` 科学代码快照。原快照清单 hash 记录在 `locks/model_and_project.json`。包内只有该 hash 和少量源文件 hash，没有完整代码，因此不能声称当前 GitHub HEAD 已与原实验完全一致。必要时从队长原 run 导出 `code/`、实际队列入口及对应校验清单。
2. vendor：按 `locks/vendor_sources.json` 的每个 URL 和 commit 获取，保留 `.git` 以便原校验流程检查 clean commit；不要直接用各仓库最新 main。
3. 权重：原服务器 checkpoint 位于 `original/l40sp/plan.json` 的 `checkpoint` 路径；若组员不能访问，可按 `Wan-AI/Wan2.1-T2V-1.3B` 的 revision `37ec512624d61f7aa208f7ea8140a131f93afc9a` 获取相同原生文件，按 `model_integrity.json` 验证。这里使用的是原生 checkpoint，不可无说明地替换成另一个转换版本。
4. 校准：L40S 与 H200 共用本包的两份 profile；按同一实验设置接续时复用，不重新抽样校准。H200 复用后的既有检查记录也在包内。校准使用项目选定的 8 个 Commons 视频，不是原论文 MixKit 集。
5. 路径：manifest 内 `calibration_path` 是队长服务器的绝对路径。异机执行需在运行副本或加载层映射到本包 profile 的新位置；保留原 manifest、原 run_id 和路径映射记录，不修改 profile 内容或科学参数。具体 CLI 由实际源码确认，本包不猜测运行命令。
6. 先在新环境做单样本五方法检查，记录 GPU、依赖版本和实际 source/profile hash，确认输出与 trace 正常，再按待完成清单批量运行。原环境的 passed JSON 仅是历史证据，不能替新环境作检查。

## 4. 交付与配对

每项保存 `video.mp4`、`frames.npy`、`metrics.json`、`trace.json`，结果按 GPU 和 run_id 分开；保留失败原因。仅 metrics complete 不等于质量评测完成。

质量评测可由队长将新结果与已有 uncached 帧配对；如由组员评测，需要另外获得 `references/required_uncached.csv` 指定的原始帧。不要用有损 MP4 代替原帧计算需要逐像素比较的指标，也不要仅因已导出统计记录而删除原帧。

全部剩余原帧约 260.5 GiB，按既有 15% 余量估算约 300 GiB，模型/环境另外计。当前 Orion 存储问题尚未证实解决；若仍在原存储中运行，须接上评测和空间安排。异机运行使用组员自己的输出目录。

## 5. DrawBench 单独确认

用户已分配五方法在 DrawBench 上的比较，但尚未明确输出为视频还是图像。

- 如果继续 Wan 视频：可以把 DrawBench 文字作为补充 prompt 集；应明确报告为 Wan T2V on DrawBench prompts，建立独立实验 ID、manifest 和 Uncached，不能称为复现论文 DrawBench 图像实验，也不能直接混入原开发集或官方 VBench 主表。固定版本/样本范围/seeds/阈值和评测定义；阈值在开发集确定后冻结。
- 如果要复现 DrawBench 文生图：这是独立 T2I 实验。SeaCache 论文用 FLUX.1-dev；当前项目记录只确认 Wan 视频路径。不能只换 prompts 就认为五方法的图像适配和校准已完成，尤其 SenCache 图像实现需另行核实。

论文依据：SeaCache §5.1 与附录 §10.2，https://arxiv.org/html/2602.18993v2 （DrawBench 用于图像，VBench 用于视频）；SenCache §5，https://arxiv.org/html/2602.24208v2 （其视频评测使用 VBench）。

截图目前只完整显示第一条准备项，第二条内容被裁切，尚未解读。
