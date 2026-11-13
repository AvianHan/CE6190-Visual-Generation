# 离开 GPU 服务器后继续写报告

实验已完成：网格 1411 + Phase D 336 + Phase C 3750 = 5497 项，三份验收记录均无 bad。VBench 85 jobs done，failed=[]，五方法各 16 维结果。

- report/report.tex 和 report/report.pdf 是报告草稿，仍有 TBD。
- analysis/ 包含最终 Phase C 汇总、速度、消融、定性图和制图脚本；带 provisional 的文件为历史中间版本。
- vbench_out/ 为原始 VBench 分维度分方法评分，需整理进报告。
- eval/、acceptance/、phaseC/、phaseD/、manifest.jsonl 保存评测和任务依据。
- run_metadata.zip 保存所有运行的 metrics.json、trace.json，以及状态、日志和中断历史；解压后可获得 results/<run_id>/ 元数据。
- code/、tools/ 是实际执行代码，vendor_source.zip 为第三方源码快照（不含 .git、权重、二进制构建文件）。依赖版本见压缩包 logs/pip-freeze*.txt。

## 报告剩余工作

补齐 Phase C 和 VBench 结果、定性图、摘要/结论及组号、成员姓名和贡献。保留 PLAN.md 中记录的协议限制与实现差异。脚本内含原服务器绝对路径，在新机器上需调整路径；已有表格和图无需重新生成实验。

## 原始产物处理决定

用户于 2026-11-13 明确同意放弃完整原始帧和视频，仅尽可能保留写报告资料。qualitative_videos/ 保留最终定性案例的少量视频，analysis/figures/ 保留定性图。其余原始产物不备份，服务器释放后不能重新执行依赖原始帧的质量评测。服务器释放条件为本包成功推送 Git 并验证远端提交。

第三方源码压缩包分卷保存，恢复命令：
```sh
cat vendor_source.zip.part* > vendor_source.zip
unzip vendor_source.zip
```
虚拟环境可重建，无需保留。模型权重和外部数据集未纳入。
