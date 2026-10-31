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

**共同 HOW：** 四种缓存均保存整个 Transformer block stack 的 `r = h_out - h_in`，复用时只做 `h_current + r_cached`；前部特征计算、完整 block 计算及输出 head 与 uncached 保持同一实现。`wan_adapter.py` 在绑定缓存 forward 前核对这些源码结构，拒绝改变精度、缩放缓存或只计算部分 blocks 的实现。各方法的判别信号、阈值、首尾保护与连续复用限制属于 WHEN，保留作者规则；MagCache 的幅值比只用于判断，不乘到被复用的 residual 上。这个源码检查不替代真实 GPU 数值验证。

**共同配置已确定：** Wan2.1-T2V-1.3B；832×480、81帧、16fps（约5秒）；50步、UniPC、CFG5、flow shift5；BF16、FlashAttention2、batch=1。五方法全部使用同一套设置。`negative_prompt: null` 使用固定 Wan 源码的默认负提示，不是空文本。H200 上关闭 CPU offload，避免把 CPU/GPU 搬运耗时引入主要比较；这是本项目选择，作者 README 的示例是开启 offload。Sea 论文主实验65帧，故本项目属于统一设置比较，不是逐项复制其原表。缓存阈值仍是待调优参考值；Mag/Sen 校准必须匹配这套 CFG5 配置。

VBench 清单有 **946 条记录、944 条不同文本 prompts、16 个维度**；重复文本的维度标签合并。流程是 prompt → 自己生成视频 → VBench 评测。另用 PSNR/SSIM/LPIPS 比较缓存输出与相同 prompt/seed 的 uncached 输出。64 条 development prompts 用于调参，独立校准数据仍待准备；两者均不属于主测试集。

后续 T2I 走 SeaCache 的 FLUX 路线，尚未实现。SenCache 的 FLUX 迁移也需要另行实现和校准。

## 完整文件结构

只列交付的项目文件；`.git/` 是你现有仓库的版本记录，不包含在更新包中。运行后新增的模型、环境和结果目录也未列入。

```text
CE6190-Visual-Generation/
├── README.md                         你的知识笔记、论文截图；底部附课程要求
├── Assignment_2_Summary.md           保留的课程要求原文
├── GUIDE.md                          本文件：代码说明与运行入口
├── LICENSE                          你仓库原有许可证
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
│   ├── start_orion.py                Slurm准备环境/下载模型，成功后单卡检查与首次生成
│   ├── run_manifest.py               依次执行任务；支持续跑和显式失败重试
│   ├── submit_slurm.py               生成或提交 Slurm array；集群参数待适配
│   ├── slurm_job.sh                  Slurm 单任务入口
│   ├── run_vbench.py                 调用固定版本的官方 VBench 评测器
│   └── plot_results.py               根据真实结果画速度—质量图
├── tests/                           可重新执行的自动检查程序
│   ├── test_orion_start.py           离线模拟提交、依赖与模型完整性检查
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
│   ├── seacache_base_test_log.txt     SeaCache 循环迁移的检查输出
│   ├── seacache_base_validation.json  SeaCache 循环迁移的检查摘要
│   ├── reorganization_test_log.txt    本次目录调整后的自动检查输出
│   └── reorganization.json            本次合并、安装和文件完整性检查摘要
└── report/                          尚待填写真实结果的报告模板
    ├── README.md                    编译和课程提交说明
    ├── report_template.tex          LaTeX 报告源文件
    └── references.bib               论文参考文献
```

**检查与结果：** `tests/` 是可执行检查，`validation/` 是历史检查记录。首次实验的环境、权重、上游仓库和结果全部写到仓库外的 scratch 运行目录，避免进入 Git。

**两个配置文件：** `requirements.txt` 列安装哪些第三方 Python 库；`pyproject.toml` 告诉 Python 怎样安装我们自己的代码。虽然文件现在直接放在 `src/`，安装后仍使用 `python -m cachebench.…`，不用改已有命令。

**SHA256SUMS.txt：** 每个文件对应一个内容指纹，用来检查交付文件是否改变或损坏；启动脚本提交前会核对它。正常修改代码后指纹会变化，需要重新生成后再提交任务；它不是错误日志。

## Orion 首次启动

先将更新包内 `CE6190-Visual-Generation/` 的内容复制到你现有仓库根目录，保留 `.git/`；在 VS Code Source Control 检查 diff，commit/push 仍由你操作。不要把 ZIP 外层再套进仓库。README、图片及你删除文件的取舍保持不变。

在 **VS Code Remote SSH 的远程终端**、仓库根目录执行；下面两个值换成你已获准使用的 scratch 路径和 Slurm account：

```bash
python3 scripts/start_orion.py submit \
  --work-root /scratch/YOUR_PROJECT/YOUR_USER/ce6190-runtime \
  --account YOUR_ACCOUNT --qos highpriority --submit
```

默认 CPU 分区 `cpup`、GPU 分区 `h200p`；使用模块 `python/3.11.9` 和 `cuda/12.4.1`。不加 `--submit` 仅预览。提交后记录打印的 `run_dir`，不要为查询状态重复执行 submit。

| 作业 | 资源 | 做什么 |
|---|---|---|
| environment | CPU，8核/64GB | 独立 Python 环境；固定依赖和 FlashAttention2；下载核对五个上游源码；自动检查 |
| model | CPU，4核/8GB | 下载固定 revision 的原生 Wan1.3B 权重并核验；与 environment 并行 |
| smoke | 一张 H200，8核/64GB CPU 内存 | 前两项成功后，检查 GPU/FA2，生成 uncached 与 Sea 全计算路径的 dog 视频并比较 |

启动脚本不再写死2/4小时时限；默认不传 `--time`，采用Slurm分区默认时限，并不代表无限时间。可用 `--time` 显式指定经过核对的允许值。GPU作业只在准备成功后启动。脚本保存 `cluster_resources.json`，记录当时节点、GPU分配、分区、QoS、用户配额和本人作业；查询失败不会被当作无限资源。

模型固定 `Wan-AI/Wan2.1-T2V-1.3B`、revision `37ec512624d61f7aa208f7ea8140a131f93afc9a`；原生权重约17.6GB。脚本将环境、下载缓存和结果放在 `--work-root` 下，并复制本次代码快照。当前提交不会修改系统 Python、驱动或你的 Git 历史。

查询（将路径替换为提交时打印的 `run_dir`）：

```bash
python3 scripts/start_orion.py status --run-dir /absolute/path/to/run --bundle
squeue -u "$USER"
```

状态显示 `FIRST_SMOKE_PASSED` 才表示首次检查完成。`--bundle` 生成本次 `startup_report.zip`，只含诊断与检查结果，不含模型或视频；可发回以继续排查或推进。

首次检查使用同一个 dog prompt、seed0、完整81帧/50步。uncached 与 Sea threshold=0 各 warmup1 次、计时3次，共运行8遍，保存两份最终视频。这里确认全计算路径一致；不作为缓存加速成绩。PSNR/SSIM 使用编码前帧；不需要 Mag/Sen 校准或 LPIPS 权重。

现有 CLI 仍可单独运行。启动完成后，环境、vendor、checkpoint 的实际路径见本次作业记录；后续不要在登录节点直接执行生成或校准。VBench评测另需官方依赖及评测模型，这次启动不等于16维评测已准备完成。

## 正式实验顺序

以下保留后续阶段的接口示例，**不属于本次首次启动命令**。校准数据、GPU作业、`CHECKPOINT` 和外置 vendor/结果路径需在进入该阶段时准备好，再执行。

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

`run_manifest.py --resume` 只跳过文件完整且配置匹配的成功任务；`--retry-failed` 显式重试失败任务。`start_orion.py` 的单卡只用于首次正确性检查。`submit_slurm.py` 已支持QoS，默认不附加 `%8` 等并发上限，由Slurm分配资源；每任务1GPU不限制同时运行的任务数。大规模array仍须按实际MaxArraySize/提交配额分块，不能直接提交超长清单。H200/L40S可各自运行完整配对实验，耗时按GPU型号分别汇总。

当前单卡、batch=1、BF16、FlashAttention2；不启用 prompt expansion。当前配置的 `negative_prompt: null` 使用 Wan 默认负提示；如果以后显式写空字符串，则本项目将其视为空文本。旧生成后端的 calibration profile 不自动复用。

离线检查不是完整 Wan 或视频质量实验。已有 `validation/` 文件保留各自历史执行记录；尚无真实 GPU 加速比或 VBench 成绩。上传 GitHub 仍由你在 VS Code 中检查 diff → Stage → Commit → Push。
