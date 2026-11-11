# CE6190 交接包：DrawBench × FLUX.1-dev 五种方法（文生图部分）

交出人：Jiang Nan（NJIANG005@e.ntu.edu.sg）。日期：2026-11-11。

本包对应队长分工的第 2 项："在 DrawBench 上跑 SeaCache / SenCache / TeaCache / MagCache / Uncached"。这部分已经全部完成，交付内容包括：
- 生成、评测和分析全部做完；
- 报告段落已写好；
- 不需要任何人再跑实验。

人和 AI 都从本文件开始读。本包自包含，`SHA256SUMS.txt` 可用来校验每个文件。

## 1. 一分钟了解结论

**实验设置**
- 模型：FLUX.1-dev，revision `3de623f`；1024²，28 步，guidance 3.5，BF16。
- 数据：DrawBench 全部 200 条 prompt，seed 0，单张 H200。
- 规模：共 24 个配置。
- 计时：加速比用"同卡、同 prompt"的配对方式计算。

**几点说明**
- **共同的复用方式**：四种缓存方法都缓存整个 Transformer block 栈的残差 `r = h_out − h_in`，复用时直接加回去；各方法只在"何时重算"上不同，规则都保持作者原版。
- **正确性**：在小模型上验证，TeaCache / MagCache / SeaCache 与作者 FLUX 代码逐位一致。
- **SenCache 是我们的移植**：官方没有 FLUX 版本，我们照作者 Wan2.1 代码移植，并用 64 张 DOCCI 真实图片校准（与 DrawBench 无重叠）。逐步决策与作者 Wan 原版规则的一致性已用测试核对。
- **自建对照**：我们加了一个"前 6 步全算、之后均匀刷新"的固定调度作为对照组。它不是已发表的方法。

**相同加速下的 LPIPS（越低越接近 uncached）**

| 加速 | 从好到差 |
|---|---|
| ~1.7× | SeaCache .063 < SenCache .072 = 固定调度 .072 ≪ TeaCache .232 |
| ~1.83× | SenCache .078 < MagCache .097 |
| ~2.25× | SenCache .126 < 固定调度 .138 ≈ SeaCache .140 < MagCache .176 |
| ~2.45× | SenCache .158 < SeaCache .193 < MagCache .200 ≪ TeaCache .323 |
| ~3× | 固定调度 .248 ≲ MagCache .258；TeaCache 3.3× 时为 .421；SeaCache 和 SenCache 按现有阈值档位没有到 3× 的点 |

**五条结论**（正文见 `results/REPORT_SECTION.md`；第 2、4 条已用 Wan 现有 trace 交叉核对，见第 5 节）

1. **保真度差距大，画质指标几乎不动。**
   - 相同加速下，各方法 LPIPS 最多相差约 5 倍。
   - 所有配置的 CLIP 都在 uncached ±0.4 以内；ImageReward 到 2.5× 仍在约 0.03 以内。
   - 因此 PSNR/SSIM/LPIPS 只能说明"离完整计算有多远"，不能当画质看。
2. **刷新步放在哪里，决定了大部分排名。**
   - 各方法的调度几乎不随 prompt 变化：MagCache 每档只有 1 种调度，SenCache 2 也只有 1 种。
   - 前 6 步里算了几步，顺序与各档排名一致：Sen 6 步 > Sea 4–6 步 > Mag 4–5 步 > Tea 2–4 步。
3. **自适应判断只在特定档位有用。**
   - 有用：约 1.7× 时 SeaCache 胜过固定调度；约 2.25× 时 SenCache 胜过固定调度（93.5% 的 prompt）。
   - 打平或略差：其他档位都是这样。
4. **SenCache 在作者默认规则下最多约 2.45×。**
   - 原因有两条：前 20% 的步数实际上全算，且最多连续跳 3 步。
   - 阈值 ≥1.2 后，所有 prompt 的调度完全相同。
   - Wan 表里 Sen 阈值 1/2/4 都卡在约 2.47×，很可能也是这个原因（Wan 上未核对）。
5. **加速几乎完全由刷新比例决定**，SeaCache 的频域滤波只带来很小的额外开销。

## 2. 目录

| 路径 | 内容 |
|---|---|
| `results/REPORT_SECTION.md` | **可直接放进报告的英文段落**：设置、对比表、配图、五条发现、局限性 |
| `results/speed_vs_lpips.png` | 主图：全部配置的加速–LPIPS 曲线，含固定调度对照 |
| `results/examples_2.45x.jpg` | 示例图：随机抽 4 条 prompt（seed 0，非挑选）× 五种配置，均在约 2.45× |
| `results/summary_all.csv` | 每个配置一行：加速、刷新比例、PSNR/SSIM/LPIPS、CLIP、ImageReward、调度统计 |
| `results/pairwise_lpips.csv` | 相同加速下的配对比较：ΔLPIPS、bootstrap 95% CI、胜率 |
| `results/schedules.csv` | 每个配置的不同调度数、最常见的调度、前 6 步算了几步 |
| `results/per_image_metrics_all.csv` | 逐张图 × 配置的全部指标 |
| `results/sensitivity_flux_dev_28step_1024.json` | SenCache 的 FLUX 校准文件 |
| `results/README.md` | 英文文件说明和完整复现命令 |
| `results/wan_schedules/schedules_wan.csv` | Wan 已完成运行的调度统计（交叉核对用） |
| `code/` | 本次实际运行的代码快照（仓库里 `src/`、`scripts/`、`tests/`、`benchmarks/` 有同样的文件） |
| `evidence/` | 运行环境（GPU、torch、diffusers、命令行参数）、关键包版本、测试输出 |

生成的 PNG 共约 6 GB，**没有放进 Git**，保存在 Jiang Nan 的服务器上：

```
/dataDisk/home/jiangnan/ce6190-runtime/runs/drawbench_flux_28step{,_sencache}/<config>/db_XXX.{png,json}
```

报告需要更多示例图时，找 Jiang Nan 取。

## 3. 怎么用

- **写报告**：
  - 文生图一节直接用 `results/REPORT_SECTION.md`，配 `speed_vs_lpips.png` 和 `examples_2.45x.jpg`。
  - 段落里每个数字都能在 `summary_all.csv` 或 `pairwise_lpips.csv` 里找到。
  - 措辞已按证据强度写好：差距小的写"打平"，不排名次。
- **改图或改表**：运行 `code/scripts/report_drawbench.py` 和 `qualitative_drawbench.py`。需要原始运行目录，在 Jiang Nan 的服务器上。
- **复现**：见 `results/README.md`。

## 4. 报告里必须写明的事

- SenCache 的 FLUX 版是我们的移植，不是作者的结果。需要交代以下几点：
  - 校准数据：DOCCI 64 张真实图片；
  - 阈值缩放：作者代码里写死的 1447.98 = Wan latent 元素数的平方根，FLUX 1024² 对应 512；
  - K=3 和起始阈值 0.045 来自作者 Wan README；
  - 各图片的敏感度取算术平均，与 Wan 那边的约定一致。
- 固定调度是我们自建的对照组，不是方法。
- 每条 prompt 只用了一个 seed。
- 保真度是对照 uncached 输出来衡量的，不是对照真实图片，也不是人工评分。

## 5. 与 Wan 部分的关系：不依赖任何人补实验

这部分**不需要 Wan 那边补任何实验**，报告段落按现有数据就能成立。

**Wan 交叉核对（无需 GPU）**
- 做法：读取 Wan 已完成的 183 项运行的 trace，见 `results/wan_schedules/schedules_wan.csv`，脚本为 `code/scripts/wan_schedule_audit.py`。
- 结果：
  - TeaCache 和 SenCache 在 Wan 上同样每个阈值只有一种调度，不随 prompt 变化。这支持第 2 条。
  - SenCache 阈值 1/2/4 的调度完全相同：50 步算 17 步，刷新比例 0.34。这正好解释了 Wan 速度表里约 2.47× 的平台。这支持第 4 条。
  - 第 4 条的成因在两个模型上不同：
    - FLUX：前 20% 全算，加上最多连续跳 3 步；
    - Wan：前期仍会跳步（前 10 步算 7 步），上限只由"最多连续跳 3 步"决定。

**只在 FLUX 上验证的部分**
- 第 3 条依赖固定调度对照，只在 FLUX 上做过。报告里已在局限性中写明。

**Wan 那边以后会自然产生、可以顺手用上的数据（都不是新实验）**
- Wan 评测：对已生成的视频算保真度后，可以直接比较 SenCache 2 和 TeaCache 0.16。两者刷新比例都是 0.34，调度都是固定的，只是分布不同，正好在 Wan 上检验"刷新步放在哪里更重要"。
- 剩余任务：跑完后，对新的 trace 重跑 `wan_schedule_audit.py`，就能补上 MagCache 和 SeaCache 的调度统计。目前它们本地各只有 1 项。

另外，Jiang Nan 之前在群里发过一份中文结论页，那是加入 SenCache 之前写的，**已经过时**，以本包为准。
