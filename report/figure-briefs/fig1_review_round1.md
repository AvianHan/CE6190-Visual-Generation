# Figure 1 审阅意见（第 1 轮）

审阅对象：`report/figures/caching_overview.pdf`（由 `scripts/make_overview_figure.py` 生成，495 × 306 pt）。
对照文件：`figure-briefs/fig1_overview.md`。

## 结论

内容与原意一致，表达比旧图 `caching_framework.pdf` 明显更好。保留整体结构 (a)–(d)，按下面的意见修改。

已核对无误、请保持不变的内容：

- (d) 三行调度与 `analysis/report_data/flux_all_configs.json` 的 `modal_schedule` 完全一致；11 / 28、2.45× / 2.47× / 2.46×、LPIPS 0.3231 / 0.1996 / 0.1576 均正确。
- SenCache 行的 cap-forced 圆点位于第 10、14、18、22、26 步，即 6 步前缀之后每 4 步一次（$K=3$），正确。
- (b) 机制正确：compute 路径经 block stack 输出并刷新残差缓存；reuse 路径为当前输入加缓存残差；output head 和 sampler 每步都运行；guards 与信号一起进入 gate。
- 方法名、颜色与论文和其他图一致。
- 底部单行图例符合规范。

## 必改

1. **字体**。目前正文用 Comic Neue（手写漫画体），数学符号用 STIX，同一张图混用两种字体。它与论文正文（Times）和其余 6 张数据图（STIXGeneral）风格冲突，在 CVPR 版面里显得不正式。请全图统一为 STIXGeneral（`font.family: STIXGeneral`, `mathtext.fontset: stix`）。换字体后字宽会变，请重新检查重叠和对齐。
2. **高度**。现在高 306 pt（4.25 in），约占页面高度的 48%，超出指导文件建议的 2.4–3.3 in。它把后面的表和图整体往后推，部分浮动体因此离正文 1–2 页。目标高度约 3.0–3.3 in，宽度不变，例如：
   - 压缩 (a) 图标行的高度和上下留白；
   - 缩小 (b)/(c) 与 (d) 之间以及面板之间的空白；
   - 不要缩小字号（主要标签保持 8–9.5 pt，最小 7 pt）。

## 建议

3. **面板标题位置**。"(b) Shared reuse step" 和 "(c) Paired evaluation" 夹在 guards 连线和面板之间，guards 方框横跨在 (c) 标题上方，容易被读成属于 (c)。建议像 (a)、(d) 那样把标题放进各自面板的左上角，guards 方框靠近 gate。
4. **"one policy"**。这个标签含义不清，可改成 "one per run" 或 "select one"，也可以删掉。
5. **× 符号**。"2.45×" 中的 × 由回退字体渲染，显得很小，像下标。换成 STIX 后应恢复正常，请确认。
6. **外框底色**。整张图有浅灰圆角底板，放在白色页面上会形成一个灰框。可以去掉最外层底板，只保留各面板的浅色底。
7. **(c) 的评测指标**。(c) 只有图片配对，没有说明比较什么。可以在配对括号下方加很短的 "speedup · LPIPS"，让 (c) 与 (d) 的列名呼应。此项可选。

## 改完后

- 覆盖输出 `report/figures/caching_overview.{pdf,svg,png}`，并保持 `scripts/make_overview_figure.py` 可复现。
- 图注会在定稿后由论文作者一侧统一调整，可不改 `report.tex`。
