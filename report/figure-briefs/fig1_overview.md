# Figure 1 绘图指导：缓存复用决策总览图（大图）

> 本文件给负责画图的 agent 使用。它提供论文背景、可直接使用的事实与数据、素材路径、宏观方向和硬约束。
> 构图、视觉隐喻、配色细节和面板划分由你自主决定；文中的"可选方向"只是启发，不是规定。
> 唯一的硬性要求集中在第 7 节（约束）和第 10 节（交付）。

---

## 0. 任务一句话

为 CVPR 双栏论文 *Beyond Skip Counts: Understanding Adaptive Caching for Efficient Generation*（CE6190 课程组报告，Group 8）画一张全宽总览图 Figure 1。读者只看这张图和图注，就应理解：

1. 本文比较的五种方法共享同一个"计算或复用"的缓存操作；
2. 它们的区别在于"何时复用"的决策：不同的决策信号加上保护规则（guards）；
3. 本文用成对的 Uncached 参考来衡量每个决策带来的速度与保真度变化，并发现"算了几步"不足以说明问题，"在哪几步算"以及"保护规则是否接管了决策"同样关键。

---

## 1. 论文故事（背景，供理解，不必全部画出）

- **问题**：图像/视频扩散模型（diffusion transformer）每个采样步都要跑一次大网络，生成很慢。
- **缓存加速**：相邻采样步之间中间特征变化不大，可以复用上一次算好的特征，跳过昂贵的 transformer block 计算，无需重新训练。
- **核心决策**：每一步"复用旧缓存"还是"重新计算并刷新缓存"。这就是本文关注的 *temporal scheduling*（时间调度）。
- **研究问题**：在实测的生成时间预算下，每种策略保留了多少对 Uncached 输出的保真度？是哪些决策导致了这些差别？
- **实验设置**：两个并行任务，文生图 FLUX.1-dev + DrawBench（200 prompts，28 步），文生视频 Wan2.1 + VBench 子集（50 步，每步两次 CFG 调用，共 100 次 block 调用）。同一 backbone 内，所有方法共享缓存范围和复用方式（block-stack residual reuse），只在决策上不同。
- **主要发现（论文中的固定措辞，请保持一致）**：
  1. *Equal computation counts do not determine final error.* 计算次数相同、运行时间几乎相同的策略，保真度可以差很多。FLUX 约 2.45× 加速时，TeaCache、MagCache、SenCache 都只算 28 步中的 11 步，LPIPS 却分别为 0.3231、0.1996、0.1576。Wan 上 MagCache 与 TeaCache 都执行 42/100 次 block 调用，LPIPS 为 0.0873 和 0.1464。
  2. *The better policy depends on the budget.* FLUX 上约 1.7× 时 SeaCache 更好，约 2.45× 时 SenCache 更好。
  3. *A maximum-reuse cap gives a tight count bound.* SenCache 阈值 2 时，200 个 prompt 全部使用同一个 11 步计算调度；早期阶段之后，每次刷新都由"最多连续复用 K 次"的上限强制触发，因此再放宽主阈值也无法减少计算。
  4. *Reference fidelity is not prompt correctness.* Uncached 只是全计算参考，不保证满足 prompt。

---

## 2. 为什么要重画旧图

旧图文件：`report/figures/caching_framework.pdf`（生成脚本 `scripts/make_method_figure.py`）。作者认为它画得不好，主要问题：

- 它是"文字流程图"：方框里写公式和 `WHEN / WHICH / HOW` 这类内部术语，信息密度低，视觉上平淡；
- 它只画了单步的 compute/reuse 分支，**没有体现五种方法的差别**（各自的决策信号和保护规则）；
- 它**没有体现本文的核心视角**：决策沿采样轨迹的分布（在哪几步计算），以及成对参考评测；
- 大量留白，字号偏小，缺少视觉焦点。

新图应该是一张有视觉层次、能独立讲清"方法框架"的全宽图，而不是旧图的美化版。

---

## 3. 信息粒度：画什么、不画什么

### 必须清楚传达（核心，按优先级）

| 优先级 | 内容 | 说明 |
| --- | --- | --- |
| P1 | 共享的复用操作 | 第 $k$ 步：要么完整计算 block stack 并刷新残差缓存，要么把缓存的残差加到当前输入上；output head 和 sampler 每步都照常运行。 |
| P1 | 决策门（gate）是唯一的变量 | 五种方法 = 同一管线 + 不同的门。门 = 决策信号 + 保护规则。 |
| P2 | 四种决策信号的直观差别 | 每个方法一个 1–4 词的短标签即可（见第 4 节），不写公式。 |
| P2 | 决策沿时间轴的分布 | 采样轨迹上哪些步计算、哪些步复用；相同计算次数可以有不同位置。 |
| P3 | 评测方式 | 每个输出都与同一 prompt、同一 seed 的 Uncached 输出成对比较：测时间（speedup）与保真度（LPIPS 等）。 |
| P3 | 保护规则会接管决策 | 例如首尾步强制计算、最多连续复用 $K$ 次；可以用很轻的方式点到。 |

### 可选（有空间且不拥挤时再加）

- 真实输出缩略图（同一 prompt 下 Uncached 与某个缓存方法的结果），让图更"图像化"；
- 一个小的计数提示，例如 "11 / 28 computed" 或 "42 / 100 calls"；
- FLUX 与 Wan 两个 backbone 的标识（同一框架用于图像和视频）。

### 不要放（属于正文、表格或附录的细节）

- 各方法的公式、阈值大小比较、标定数据集（DOCCI、Commons videos）、多项式系数、频谱滤波指数；
- CFG 双分支细节、硬件、计时边界、Amdahl 分析、误差传播界、Proposition 的证明；
- VBench 的 16 个维度、bootstrap 区间、具体的成对差值；
- 大段文字说明。图中文字总量建议控制在 60–90 个英文单词以内（不含坐标刻度类数字）。

---

## 4. 可直接使用的事实与数据（全部来自论文与仓库数据，勿自行编造新数字）

### 4.1 共享机制（论文 Section 3.1）

- 第 $k$ 个采样阶段，网络预测 $v_k = f_\theta(z_k, t_k, c)$，sampler 用它更新 latent $z_k$。
- 设 $a_k$ 为 block stack 的输入，残差 $r_k = F_k(a_k) - a_k$，$\tau(k)$ 为最近一次刷新的步：
  - compute and refresh：输出 $a_k + r_k$，并把 $r_k$ 存入缓存；
  - reuse：输出 $a_k + r_{\tau(k)}$，跳过 block stack。
- 复用只跳过昂贵的 block 计算，采样更新照常进行。
- 缓存范围（整个 block stack）与重建方式（直接复用残差）在每个 backbone 内固定，所有方法共享。

如果要在图中出现符号，最多用 $a_k$、$r_\tau$、$k$ 这类最少量的记号；不要把上述公式整段搬进图。

### 4.2 五种方法（名字必须与论文完全一致）

| 方法 | 决策信号（论文 Table 1） | 可用的极短标签（示例，可改） | 方法颜色 |
| --- | --- | --- | --- |
| Uncached | 无决策，每步都完整计算 | "always compute" / "reference" | 中性灰 `#B5B5B5` |
| TeaCache | 时间步相关或调制后输入的变化，经多项式重标定后累积 | "input change" | `#466F87` |
| MagCache | 标定好的残差幅值比，估计自上次刷新以来的累积复用误差 | "magnitude ratio" | `#78B6C8` |
| SeaCache | 频谱重加权（滤波 + 增益归一化）后的特征变化 | "spectral change" | `#F2D98E` |
| SenCache | 用标定的敏感度加权的 latent 变化与时间步变化 | "sensitivity-weighted change" | `#E48578` |
| Fixed（仅 FLUX 的对照） | 预先确定的固定调度，不看输入 | "predetermined" | `#98AF1E` |

说明：FLUX 上的 SenCache 是本项目从 Wan 规则移植的版本，其他数据图里标为 "SenCache (port)"；总览图里写 SenCache 即可。Fixed 是对照组，不是五种方法之一，可以不画。

### 4.3 保护规则（论文 Table 7，$K$ = 最多连续复用次数）

| 方法 | Wan（50 步） | FLUX（28 步） |
| --- | --- | --- |
| TeaCache / SeaCache | 首尾步计算；无显式复用上限 | 首尾步计算；无显式复用上限 |
| MagCache | 前 10 步计算；$K=4$ | 前 3 步和第 12 步计算；$K=5$ |
| SenCache | 前 10 步用 $\epsilon_{\rm early}=0.045$；$K=3$ | 前 6 步用 $\epsilon_{\rm early}=0.045$；$K=3$ |

### 4.4 真实调度（FLUX，约 2.45× 加速，28 步，`1` = compute，`0` = reuse）

来源：`analysis/report_data/flux_all_configs.json` 的 `modal_schedule`（最常见调度）及其覆盖率；也可参考已有的 `report/figures/flux_modal_schedules.pdf`。

| 配置 | 最常见调度（第 1 步在左） | 覆盖 prompt 比例 | 计算步数 | 中位加速 | 平均 LPIPS |
| --- | --- | --- | --- | --- | --- |
| TeaCache 0.5 | `1100100100100100100100100101` | 88.0% | 11 | 2.45× | 0.3231 |
| MagCache 0.12 | `1110100100110000100000100101` | 100% | 11 | 2.47× | 0.1996 |
| SeaCache 0.6 | `1110101001001000010000100011` | 56.5% | 11（全体平均 10.6） | 2.44× | 0.1930 |
| SenCache 2 | `1111110001000100010001000100` | 100% | 11 | 2.46× | 0.1576 |
| Uncached | 全 1 | 100% | 28 | 1.00× | 参考 |

可讲的故事：TeaCache、MagCache、SenCache 计算次数相同（11/28），时间几乎相同，但计算放在不同位置，保真度差异很大。SenCache 2 的调度是"前 6 步计算，之后每 4 步算一次"：第 6 步之后的每次刷新都是 $K=3$ 上限强制的，阈值已不起作用。

### 4.5 其他可引用的数字（如需要）

- Wan：Uncached 平均 271.33 s；TeaCache 和 MagCache 都执行 42/100 次 block 调用，约 171.5 s，LPIPS 0.1464 与 0.0873（MagCache 低 40.3%）。
- FLUX：Uncached 每张图约 6.5 s，约 2.45× 的缓存配置每张约 2.65 s；SenCache 2 的 LPIPS 比 TeaCache 0.5 低 51.2%。
- 计数下界：前 $g$ 步都计算且最多连续复用 $K$ 次时，计算次数至少为 $g + \lfloor (N-g)/(K+1) \rfloor$；FLUX 上 $N=28, K=3, g=6$ 得 11。

---

## 5. 可用素材

### 5.1 版面尺寸

- 论文为 CVPR 2027 双栏模板，`\textwidth` = 6.875 in（496.8 pt）。Figure 1 以 `figure*` 全宽插入，位于第 2 页顶部。
- 建议画布宽 6.875 in，高 2.4–3.3 in。高度可以按内容需要决定；更高的图会把正文往后推，放入后需要重新编译检查分页（见第 10 节）。
- 最终打印尺寸下，图中最小字号不低于 7 pt，主要标签建议 8–9.5 pt（正文 10 pt，图注 9 pt）。

### 5.2 风格参考（与论文其他图保持一致）

- 现有数据图：`report/figures/wan_speed_fidelity.pdf`、`flux_speed_fidelity.pdf`、`flux_modal_schedules.pdf`、`wan_component_tradeoffs.pdf`。它们用 STIXGeneral 字体（matplotlib `font.family: STIXGeneral`, `mathtext.fontset: stix`, `pdf.fonttype: 42`），线宽约 0.6–1.35，浅灰网格。
- 方法配色：见 4.2 节（与 `scripts/make_report_figures.py` 的 `COLORS` 和论文表格中的方法色块一致）。总览图里如果给方法上色，必须使用这套颜色。SeaCache 的浅黄色在白底上偏淡，填色时建议加细的深色描边。
- 论文正文是 Times 字体；图中字体用 STIXGeneral 或 Times 系都可以，但全图统一一种。

### 5.3 真实图像（可选，用于让总览图更"图像化"）

目录：`report/figures/qualitative_inputs/`，清单与 SHA256：`manifest.json`。只能裁剪、缩放，不得生成、增强或修改像素内容。

- FLUX，约 2.45× 配置，1024×1024 PNG：
  - `flux_db_053_{uncached,teacache_0.5,magcache_0.12,seacache_0.6,sencache_2}.png`，prompt："Three cats and three dogs sitting on the grass."（注意：Uncached 本身只画了两猫两狗，适合说明"参考不等于 prompt 正确"，不适合当作"缓存造成的错误"。）
  - `flux_db_101_{uncached,teacache_0.5,magcache_0.12,seacache_0.6,sencache_2}.png`，prompt："A donkey and an octopus are playing a game. ..."（TeaCache 0.5 明显改变了章鱼角色的外观和姿态，适合说明"决策导致偏离参考"。）
- Wan 视频帧（第 1、41、81 帧）：
  - `wan_success_*`：低偏差例子（香蕉在苹果上）；
  - `wan_failure_*`：高偏差例子（雨中撑伞的情侣）。
- 这些图已在 Figure 4、Figure 5 中完整展示。总览图里如使用，只放 1–2 组小缩略图作为"输入/输出"示意，不要重复做成 case 对比网格。
- 没有逐图的 LPIPS 数据，不要在缩略图旁标数值。

---

## 6. 宏观方向（启发性，可自由组合或另辟蹊径）

下面是几种可行的叙事骨架，供你挑选、混合或推翻：

- **A. "一条管线 + 可替换的门"**：主体画一条横向的生成管线（prompt/噪声 → 每个采样步 → 输出），每个采样步内部是 compute / reuse 二选一；上方或侧边是可插拔的"决策门"，五种方法各是一个门（颜色区分，标签只写信号名）；门旁用小锁或护栏类图形表示保护规则。右侧或下方用一条小的成对评测示意（cached 输出 vs Uncached 输出 → speed、LPIPS）收尾。
- **B. "步骤视角 + 轨迹视角"双面板**：(a) 放大单个采样步，讲清 compute 与 reuse 的区别以及门的位置；(b) 把 28 步展开成时间条，用 4.4 节的真实调度对比几种方法：同样 11 次计算，位置不同，LPIPS 不同；可以在 SenCache 的条上标出"前 6 步"和"cap-forced"的区域。
- **C. "问题 → 框架 → 发现"三段式**：左侧一句问题（何时复用），中间是共享框架与五个门，右侧点出两个发现（相同计数不同误差、上限接管决策）。注意右侧只能用图形和极短标签表达，不能变成文字段落。

无论哪种方向，都请做到：一眼能看出"共享部分"和"各方法不同的部分"的区别（例如共享部分用中性色，方法相关部分用方法色）。

---

## 7. 硬约束（来自论文写作规范和作者的审阅习惯）

- 图中名字和术语必须与论文一致：Uncached、TeaCache、MagCache、SeaCache、SenCache、Fixed；compute、reuse、refresh；protection rules（或 guard）；consecutive-reuse limit $K$；sampling stage / step；block stack；residual。不要自造缩写（如 Tea、Sen、Mag）。
- 不用口号、双关、修辞化标题。面板标题用平实说法，例如 "(a) Shared reuse step"、"(b) Compute placement"。
- 不要图例栏。确需图例时只放底部一行，并且只给出现约三次以上的符号；少见元素直接用文字标注。
- 不要旋转 90° 的文字。
- 每个箭头都从一个对象出发、指向一个对象，标签贴在箭头旁边；不要漂浮的文字列。
- 设计为全宽横幅，不要拼成两列窄图。
- 数字只能用第 4 节给出的数值，不得编造或四舍五入出新的值。
- 不在图中写 "ours"、"proposed" 之类字样：本文是基准研究，没有"我们的方法"。
- 矢量输出，字体嵌入（TrueType，`pdf.fonttype` 42 或等价设置），不要用位图字。
- 美观与清晰优先于紧凑：宁可适当加高，也不要把元素挤在一起。

---

## 8. 图注（caption）建议

现在论文中的图注（`report/report.tex`，`\label{fig:framework}`）是为旧图写的，换图后请一并更新。规则：粗体短标题，加一到两句话，说明图展示了什么以及要点；不写实验细节或显示说明。示例（可改写）：

> **Adaptive caching as a reuse decision.** All methods share the same compute-or-reuse step and differ only in the gate that decides when to refresh the cache, through their decision signals and protection rules. Methods with the same number of computations can place them at different stages and reach different fidelity to Uncached.

如果图中使用了 FLUX 的真实调度或真实图像，可在图注末尾加一句来源，例如 "Schedules are the most frequent ones on FLUX.1-dev near $2.45\times$ speedup."

正文中引用此图的位置：Introduction 第三段 "(Figure~\ref{fig:framework})" 和 Section 2.3 末尾 "Figure~\ref{fig:framework} shows their role in generation."。图的内容应与这两处说法相符。

---

## 9. 术语与符号速查

| 概念 | 论文用词 |
| --- | --- |
| 每步完整计算 transformer block stack | compute（并 refresh 缓存） |
| 用缓存的残差代替计算 | reuse |
| 最近一次刷新所在的步 | $\tau(k)$ |
| 决策函数 | gate / policy |
| 首尾步强制计算、复用上限等 | protection rules；guard |
| 最多连续复用次数 | $K$（consecutive-reuse limit） |
| 与 Uncached 输出的差距 | reference fidelity（PSNR、SSIM、LPIPS） |
| 实测速度 | generation speedup |
| 计算次数 | computation count；compute rate |

---

## 10. 交付与自检

1. 输出 `report/figures/caching_overview.pdf`（矢量，主文件），可另存同名 `.png` 预览。建议把生成代码写成可复现脚本（例如 `scripts/make_overview_figure.py`），但不强制。
2. 不要删除或覆盖 `report/figures/caching_framework.pdf`。`report.tex` 已写好自动切换：`figures/caching_overview.pdf` 存在时使用新图，否则回退到旧图。
3. 自检：
   - 只看图本身的 PDF 判断质量，不以编译后论文里的效果为准；
   - 在 300 dpi 下渲染检查：元素无重叠、无裁切、文字最小 7 pt、颜色与 4.2 节一致；
   - 遮住图注只看图，能否说出"共享什么、不同什么、为什么计数不够"；
   - 文字总量是否仍在 60–90 词左右，是否有任何一处变成了文字流程图。
4. 更新 `report/report.tex` 中 Figure 1 的图注（第 8 节）。
5. 在 `report/` 目录编译并检查分页：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex
```

   确认 Figure 1 仍在第 2 页顶部；若新图更高，检查正文末页与参考文献页是否仍合理。
