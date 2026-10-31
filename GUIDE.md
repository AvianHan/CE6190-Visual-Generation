# Codebase Guide

**当前：Wan2.1-T2V-1.3B；uncached、TeaCache、MagCache、SeaCache、SenCache；VBench。真实 GPU 生成与评测尚未完成。** 本文合并原来的两个 guide。

## 代码从哪里来

| 部分 | 来源 |
|---|---|
| Wan 网络、VAE、文本编码器、sampler | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 共同去噪循环、SeaCache forward | [jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache) |
| TeaCache / MagCache / SenCache forward | [TeaCache](https://github.com/ali-vilab/TeaCache) / [MagCache](https://github.com/Zehong-Ma/MagCache) / [SenCache](https://github.com/vita-epfl/SenCache) |
| 数据与评测 | [Vchitect/VBench](https://github.com/Vchitect/VBench) |
| 配置、校准组织、计时、结果配对与汇总 | 本项目 |

`src/seacache_base.py` 从固定 SeaCache 源码提取原始核心去噪循环；五种方法共用它，分别切换网络的缓存 forward。模型加载和结果保存由本项目组织，未直接照搬整份 SeaCache CLI。`src/sources.py` 固定各源码 commit。

VBench 清单有 **946 条记录、944 条不同文本 prompts、16 个维度**；重复文本的维度标签合并。流程是 prompt → 自己生成视频 → VBench 评测。另用 PSNR/SSIM/LPIPS 比较缓存输出与相同 prompt/seed 的 uncached 输出。64 条 development prompts 用于调参，独立校准数据仍待准备；两者均不属于主测试集。

后续 T2I 走 SeaCache 的 FLUX 路线，尚未实现。SenCache 的 FLUX 迁移也需要另行实现和校准。

## 完整文件结构

只列交付的项目文件；`.git/` 是你现有仓库的版本记录，不包含在更新包中。运行后新增的模型、环境和结果目录也未列入。

```text
CE6190-Visual-Generation/
├── README.md                         你的知识笔记、论文截图；底部附课程要求
├── Assignment_2_Summary.md           保留的课程要求原文
├── CODEBASE_GUIDE.md                 本文件：代码说明与运行入口
├── LICENSE                          你仓库原有许可证
├── THIRD_PARTY_NOTICES.md            上游代码、数据的来源与许可说明
├── requirements.txt                 合并后的 Python 依赖版本
├── pyproject.toml                   Python 包安装配置；把 src 映射为 cachebench
├── SHA256SUMS.txt                    交付时各文件的 SHA256 校验值
├── .gitignore                       你原有的 Git 忽略规则
├── assets/                          README 中使用的原始截图
│   ├── Wan_Fig8.png                  Wan VAE
│   ├── Wan_Fig9.png                  Wan 整体框架
│   ├── Wan_Fig10.png                 Wan DiT
│   ├── SeaCache_Fig3.png             SeaCache 方法图
│   ├── SenCache_Fig2.png             SenCache 方法图
│   └── Sea_Sen_Comparison.png        两篇方法的对照笔记图
├── benchmarks/vbench/
│   ├── VBench_full_info.json         固定版本的官方 prompts 与维度标签
│   ├── PROVENANCE.json               数据来源、commit、hash 和数量
│   └── LICENSE                       VBench 数据来源仓库的许可证
├── configs/                         实验设置；不包含实验结果
│   ├── correctness_smoke.json        uncached 与 SeaCache 全计算路径检查
│   ├── smoke_prompts.jsonl           一个 dog prompt
│   ├── main.json                     五方法的参考参数入口，尚未经调优
│   ├── reference_settings.json       同一组参考参数的独立保留副本
│   ├── validation_grid.json          独立调参的阈值搜索范围
│   ├── development_prompts.jsonl     64 条独立调参 prompts
│   └── ablations.json                SeaCache / SenCache 组件消融配置
├── src/                             实际实验程序；已去掉内层 cachebench 目录
│   ├── __init__.py                   包说明与版本号
│   ├── sources.py                    获取、核对五个上游仓库的固定版本
│   ├── manifest.py                   prompts × seeds × 方法 → 任务清单
│   ├── generate.py                   执行一个任务，计时并保存视频、帧和日志
│   ├── wan_adapter.py                加载 Wan，绑定缓存策略，调用生成流程
│   ├── seacache_base.py              提取、调用 SeaCache 核心去噪循环
│   ├── calibrate.py                  生成 MagCache / SenCache 校准文件
│   ├── policies.py                   配置校验及教学用标量策略；非 GPU forward
│   ├── evaluate.py                   PSNR/SSIM/LPIPS 配对评测与 VBench 文件准备
│   └── aggregate.py                  汇总、参数一致性检查、选阈值及 Pareto 分析
├── scripts/                         调用 src 的命令行辅助工具
│   ├── run_manifest.py               依次执行任务；支持续跑和显式失败重试
│   ├── submit_slurm.py               生成或提交 Slurm array；集群参数待适配
│   ├── slurm_job.sh                  Slurm 单任务入口
│   ├── run_vbench.py                 调用固定版本的官方 VBench 评测器
│   └── plot_results.py               根据真实结果画速度—质量图
├── tests/                           可重新执行的自动检查程序
│   ├── test_policies.py              标量缓存规则检查
│   ├── test_experiment.py            任务配对、评测和结果汇总检查
│   ├── test_review.py                配置、校准与执行边界检查
│   ├── test_official_tensors.py       上游缓存 forward 的小型张量检查
│   └── test_seacache_base.py          SeaCache 循环、CFG、sampler 和 observer 检查
├── validation/                      已执行检查的记录；不是正式实验成绩
│   ├── cpu_test_log.txt              早期 CPU 检查摘要
│   ├── unittest_log.txt              早期自动检查完整输出
│   ├── environment.json              早期验证环境信息
│   ├── artifact_qa.json              早期交付文件检查摘要
│   ├── synthetic_metric_validation.json  合成数组的指标检查
│   ├── teaching_notebook_validation.json  教学 notebook 检查记录
│   ├── notion_readme_import.json      原 Notion 导入时的完整性记录
│   ├── seacache_base_test_log.txt     SeaCache 循环迁移的检查输出
│   ├── seacache_base_validation.json  SeaCache 循环迁移的检查摘要
│   ├── reorganization_test_log.txt    本次目录调整后的自动检查输出
│   └── reorganization.json            本次合并、安装和文件完整性检查摘要
├── examples/                        教学演示；不参与正式视频生成
│   ├── cache_intuition.py            CPU 演示：频率、敏感度、Jacobian、residual
│   ├── cache_intuition.ipynb         同一组演示的 notebook 版本
│   └── cpu_intuition_notebook_output/ 已保存的教学输出
│       ├── 01_frequency.png          频率示例
│       ├── 02_sensitivity.png        敏感度示例
│       ├── 03_jacobian.png           Jacobian 示例
│       ├── 04_distance.png           输入距离示例
│       ├── 05_residual.png           residual 复用示例
│       └── summary.json              教学输出的数值摘要
└── report/                          尚待填写真实结果的报告模板
    ├── README.md                    编译和课程提交说明
    ├── report_template.tex          LaTeX 报告源文件
    └── references.bib               论文参考文献
```

**三个容易混淆的目录：** `examples/` 帮助理解；`tests/` 检查程序是否符合预期；`validation/` 保存检查执行后留下的记录。正式实验的视频和数值将写到 `runs/`、`results/`。

**两个配置文件：** `requirements.txt` 列安装哪些第三方 Python 库；`pyproject.toml` 告诉 Python 怎样安装我们自己的代码。虽然文件现在直接放在 `src/`，安装后仍使用 `python -m cachebench.…`，不用改已有命令。

**SHA256SUMS.txt：** 每个文件对应一个内容指纹，用来检查交付文件是否改变或损坏，不参与模型运行。正常修改代码后指纹会变化，提交前可重新生成；它不是错误日志。

## 环境与最小运行

以下是在准备好 Linux GPU 环境后执行的入口说明；当前不启动 Orion 作业。Python 3.10/3.11，独立虚拟环境。CUDA 轮子要与服务器驱动匹配；`cu124` 是当前候选配置，尚未在你的 GPU 验证。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e . --no-deps
python -m cachebench.sources --directory vendor
python -m cachebench.sources --directory vendor --verify-only
python -m pip freeze > environment_actual.txt
python -m unittest discover -s tests -v
```

FlashAttention 单独安装，是因为编译需要先装好 PyTorch。合并的 requirements 包含生成与本地保真度评测依赖；**不包含完整 VBench 的额外依赖和评测权重**。VBench 建议另建环境，按固定版本官方说明安装。

checkpoint 要用原生 `Wan-AI/Wan2.1-T2V-1.3B` 目录。以下 `CHECKPOINT` 改成服务器真实路径；从仓库根目录执行：

```bash
CHECKPOINT=/data/models/Wan2.1-T2V-1.3B
python -m cachebench.manifest --prompts configs/smoke_prompts.jsonl --config configs/correctness_smoke.json --output manifests/smoke.jsonl
python scripts/run_manifest.py --manifest manifests/smoke.jsonl --output-dir runs/smoke --wan-repo vendor/wan --checkpoint-dir "$CHECKPOINT"
python -m cachebench.evaluate fidelity --manifest manifests/smoke.jsonl --root runs/smoke --output results/smoke.jsonl --metrics psnr ssim
```

这两条生成任务是 uncached 和 SeaCache threshold=0。检查 `trace.json` 每步是否全计算，确认输出一致后再开展测速。TeaCache threshold=0 不一定全计算，不作为这个一致性控制。

## 正式实验顺序

1. **校准**：独立 prompts 用于 MagCache；独立真实视频及描述用于 SenCache。每行 JSONL 使用 `prompt_id`、`prompt`、`seed`、`split:"calibration"`；Sen 另需 `video_path`。校准不更新模型参数。
2. **调参**：在 `development_prompts.jsonl` 上运行 `validation_grid.json`，按相同实际速度预算选阈值。
3. **冻结参数后测试**：用 VBench prompts 生成，评测保真度和视频质量。
4. **消融**：使用 `ablations.json`，与原方法分开报告。

校准命令：

```bash
python -m cachebench.calibrate --method magcache --config configs/main.json --samples data/mag_calibration.jsonl --output calibration/magcache_81f.json --vendor-dir vendor --checkpoint-dir "$CHECKPOINT"
python -m cachebench.calibrate --method sencache --config configs/main.json --samples data/sen_calibration.jsonl --output calibration/sencache_81f.json --vendor-dir vendor --checkpoint-dir "$CHECKPOINT"
```

数据文件需先自行准备。校准集不得与调参/测试 prompts 重叠；更改模型、尺寸、帧数、steps 或 solver 后重新检查 profile 是否匹配。Sen 多视频 sensitivity 的算术平均是本项目的转换约定，须在报告说明。

调参流程：

```bash
python -m cachebench.manifest --prompts configs/development_prompts.jsonl --config configs/validation_grid.json --exclude-prompts data/mag_calibration.jsonl --exclude-prompts data/sen_calibration.jsonl --output manifests/validation.jsonl
python scripts/run_manifest.py --manifest manifests/validation.jsonl --output-dir runs/validation --wan-repo vendor/wan --checkpoint-dir "$CHECKPOINT"
python -m cachebench.evaluate fidelity --manifest manifests/validation.jsonl --root runs/validation --output results/validation.jsonl --metrics psnr ssim lpips --device cuda
python -m cachebench.aggregate --input results/validation.jsonl --output results/validation_summary.json --freeze-config configs/validation_grid.json --target-speedup 2.0 --frozen-output configs/frozen_2x.json
```

选择规则：达到目标实际 speedup 的候选中选 LPIPS 最低者。未达到预算会报错；不同方法的 threshold 数值不能直接比较。冻结文件生成后，将 `seeds` 改为 `[0,1,2,3,4]`、`temporal_flickering_25` 设为 `true`；不再用测试结果调阈值。

生成正式清单后，按上面相同的 `run_manifest → evaluate fidelity → aggregate` 顺序执行，只替换 manifest、runs、results 路径：

```bash
python -m cachebench.manifest --prompts benchmarks/vbench/VBench_full_info.json --config configs/frozen_2x.json --exclude-prompts configs/development_prompts.jsonl --exclude-prompts data/mag_calibration.jsonl --exclude-prompts data/sen_calibration.jsonl --output manifests/test_2x.jsonl
```

VBench 评测器固定版本，并按方法准备视频（以 SeaCache 为例）：

```bash
git clone https://github.com/Vchitect/VBench.git vendor/vbench
git -C vendor/vbench checkout --detach fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
python -m cachebench.evaluate stage-vbench --manifest manifests/test_2x.jsonl --root runs/test_2x --output vbench_stage/seacache --official-info benchmarks/vbench/VBench_full_info.json --method seacache --samples 5
python scripts/run_vbench.py --vbench-repo vendor/vbench --stage-dir vbench_stage/seacache --dimension subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality object_class multiple_objects human_action color spatial_relationship scene temporal_style appearance_style overall_consistency --output results/vbench_seacache --execute
```

`temporal_flickering` 单独准备 25 个候选、做官方静态筛选，再评测：

```bash
python -m cachebench.evaluate stage-vbench --manifest manifests/test_2x.jsonl --root runs/test_2x --output vbench_stage/seacache_flicker --official-info benchmarks/vbench/VBench_full_info.json --method seacache --dimension temporal_flickering --samples 25
python vendor/vbench/static_filter.py --videos_path vbench_stage/seacache_flicker/videos --result_path vbench_stage/seacache_flicker/static --filter_scope temporal_flickering
python scripts/run_vbench.py --vbench-repo vendor/vbench --stage-dir vbench_stage/seacache_flicker --dimension temporal_flickering --videos-override vbench_stage/seacache_flicker/static/filtered_videos --output results/vbench_seacache_flicker --execute
```

保留筛选不足 5 个样本的 warning。81帧/16fps 约5.06秒，当前协议应报告为共同设置上的 VBench 指标，不能直接宣称符合排行榜协议。

## 结果与执行边界

每次生成保存 `video.mp4`、`frames.npy`、`metrics.json`、`trace.json`。热运行时间包含文本编码、去噪和 VAE decode；排除模型加载、策略初始化、CPU传输和磁盘编码。默认 warmup 1次、计时3次取中位数。不同 GPU 型号分别报告。

`run_manifest.py --resume` 只跳过文件完整且配置匹配的成功任务；`--retry-failed` 显式重试失败任务。Slurm 脚本尚需适配 Orion partition/account/QoS，先完成单卡 smoke；不应直接按旧参数提交。

当前单卡、batch=1、BF16、FlashAttention2；不启用 prompt expansion。空 `negative_prompt` 是空文本，未提供/None 才使用 Wan 默认负提示，与 Sea 原 CLI 的空字符串回退行为不同。旧生成后端的 calibration profile 不自动复用。

离线检查不是完整 Wan 或视频质量实验。已有 `validation/` 文件保留各自历史执行记录；尚无真实 GPU 加速比或 VBench 成绩。上传 GitHub 仍由你在 VS Code 中检查 diff → Stage → Commit → Push。
