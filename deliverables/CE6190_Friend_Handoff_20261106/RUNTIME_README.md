# 实验运行归档

运行内容位于 `runtime/a100x8-20261110/`，从 `~/ce6190-runtime/a100x8-20261110/` 同文件系统移动，旧路径保留符号链接，以兼容现有脚本和 Python 环境的绝对路径。

生成任务共 5497 项 complete：交接网格 1411、Phase D 336、Phase C 3750。逐项检查 metrics 状态及 video.mp4、frames.npy、trace.json 存在且非空；既有三份验收记录均无 bad。本次未重新逐视频解码。原交接任务无缺失。VBench 日志为 85 jobs done、failed=[]，五方法各 16 个结果 JSON 可解析。

报告 `runtime/a100x8-20261110/report/report.pdf` 仍为草稿，report.tex 有 TBD；尚需将最终 Phase C / VBench 结果写入报告，并填写组号、姓名和贡献。

权重及评测模型缓存保留在原目录的 models/、vbench_cache/；第三方源码附带的 LVIS 数据集标注保留在原目录 external_datasets/。生成视频、原始帧属于实验产物，已纳入本项目。虚拟环境也随运行目录归档，依赖旧路径兼容链接；此目录不是跨机器可直接运行的环境包。

详细迁移记录见 `runtime/a100x8-20261110/IMPORT_RECORD.json`。runtime/ 已加入 Git 忽略，避免提交大体积产物。
