# 给接手 AI：CE6190 剩余实验交接

**从此 README 开始。你负责运行本目录 `pending.jsonl` 的 1,409 项未完成任务，加上 `new_hardware_baselines.jsonl` 的 2 项新环境基线（合计 1,411 次运行）。机器数量、分工和 GPU 调度由你与机器主人决定，本包不预分两台机器。**

本包是 2026-11-06 的明确任务转交，不是让你重新跑全量实验。用户已明确停止全部本机实验：183 项已完成结果和 2 项中断现场原样保留；不再保留任何待跑任务。`pending.jsonl` 与 183 项已完成清单按 run_id 完全不重叠。另列的 2 项基线是为了新硬件对照，只在接收方新目录执行，不改动本机原基线。此前“本机跑到凌晨 3 点”的安排已撤销。

## 0. 强制安全要求：先读再执行

1. **绝不能覆盖、删除、移动或修改交出方已生成的结果。** 不清理其他实验、其他用户的 GPU 进程或任何原始视频/帧。不得为腾空间删除任何既有结果。
2. **本次授权仅为 `pending.jsonl` 加 `new_hardware_baselines.jsonl`。** `local_completed.jsonl` 是交出方已经完成的 183 项，不能把它当作待完成清单。唯一需要新环境重新生成的两项基线已单独列在 `new_hardware_baselines.jsonl`；它们只能写到接收方新目录，严禁写回本机原路径。原始 ZIP 中的 1,409/1,473/2,880 项旧清单仅用于追溯，禁止直接提交。
3. 源码快照、原始 manifest、ZIP、校准 profile 和锁定文件均只读。需要路径转换时使用准备脚本生成新副本；不要编辑原 profile、run_id、seed 或实验参数。
4. 每台机器使用独立的新工作目录和输出目录；结果主键为 `机器/硬件环境/run_id`。不能让多个 worker 同时领取同一任务，也不能让两个节点写入同一结果目录。
5. 发现已有输出：只有确认文件完整、配置和模型/源码/profile 哈希兼容时才能跳过成功任务；**失败、半成品、损坏 JSON、遗留 running 都必须停下检查，不能自动覆盖或静默从头重跑。** 本项目 `generate.py` 默认拒绝覆盖已有目录；不要绕过它，不使用 `--overwrite-failed` 批量掩盖错误。需要重试时先征得机器主人对具体任务的同意，保留原目录，在新的 attempt 输出根中执行并记录关系。
6. GPU 开始前检测物理型号、UUID、显存与已有进程。不得杀其他进程，不使用已知故障卡，不偷偷转 CPU，不改分辨率/帧数/offload 来勉强塞入显存。本机 A100 40GB 实际占用约 32GiB，不能假设 24GB 卡可跑。按实际 CPU RAM 余量控制并发加载模型。
7. **Git 禁止 `push --force`、`reset --hard`、`git clean`、丢弃他人改动。** 拉取前检查状态，协作冲突人工审阅、保留双方内容。不得把权重、环境、视频或原始帧提交 Git。
8. 本包不包含访问交出方服务器的授权；不要尝试写入 manifest 中的 `/scratch/...` 原服务器绝对路径。所有执行路径必须映射到接收方自己的新工作目录。

原交出方服务器的物理 GPU 3 已知损坏；这是该服务器的规则，不代表你机器的 GPU 3 也损坏。请遵守接收方实际机器规则。

## 1. 任务与目录

| 方法 | 本次任务数 |
|---|---:|
| Uncached（`baseline`）| 95 待完成＋2 新环境对照 |
| TeaCache | 387 |
| MagCache | 281 |
| SeaCache | 260 |
| SenCache | 386 |
| **合计** | **1,411** |

范围为 97 个 prompt–seed 对，使用队长的自编开发 prompts。**不包括 DrawBench、正式 VBench 评分、组件消融或重新校准。**

- `pending.jsonl`：1,409 项未完成任务，保留原 run_id 和参数。
- `new_hardware_baselines.jsonl`：2 项接收方新环境基线；准备脚本会合并进实际执行清单。
- `local_completed.jsonl`、`ownership.csv`：183 项既有完整结果及归属依据，禁止覆盖。除单列的 2 项新环境基线外，不重跑已完成项。
- `code/`：本机已经实际运行验证过的源码快照；优先使用这里，不随意换仓库最新源码。
- `calibration/{magcache,sencache}.json`：原样复用队长校准产物。
- `locks/`：官方源码 commit、模型 revision/逐文件 hash、原环境版本。
- `upstream/CE6190_Teammate_Handoff_20261106.zip`：**队长发来的原始交接包，完整原件**。
- `evidence/`：本机环境、源码/profile hash、五方法检查记录；仅为历史证据，接收方仍需自己检查。
- `tools/prepare.py`：校验本包、建立本机路径副本；不会启动 GPU。
- `tools/fetch_model.py`：下载并逐文件核验指定原生 Wan 权重；不会启动 GPU。

## 2. 接手执行顺序

机器主人让你处理这个目录后，可以进行环境准备、下载、检查和清单内实验；无需让他再手工整理配置。机器分工由你决定，但分片的并集必须等于准备脚本生成的 1,411 项清单、交集必须为空。**建议同一 pair_id 的 baseline 与缓存项放同一机器/同一环境，避免跨硬件计算加速比。** 如果分散到不同硬件，必须为对应环境建立基线，并先向机器主人说明新增工作；不能用另一机器的基线耗时冒充本机基线。

以下命令从本 README 所在目录执行。`WORK` 必须是接收方一个全新的绝对路径；用机器/日期命名。源码和本包不写入实验结果。

```bash
export WORK=/absolute/new/path/ce6190-friend-run
python3 tools/prepare.py --work-dir "$WORK"
python3.11 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --upgrade pip
"$WORK/venv/bin/pip" install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$WORK/venv/bin/pip" install -r code/requirements.txt
```

还需安装 `flash-attn==2.7.4.post1`，以及 `einops`（FlashAttention 依赖）。本机使用 Python 3.11、PyTorch 2.6.0+cu124、CXX11 ABI FALSE 的官方预编译 wheel，避免用本机 CUDA 13 编译器构建 CUDA 12.4 扩展：

```bash
"$WORK/venv/bin/pip" install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
"$WORK/venv/bin/pip" install --no-deps -e ./code
"$WORK/venv/bin/pip" check
"$WORK/venv/bin/python" -m cachebench.sources --directory "$WORK/vendor"
python3 tools/fetch_model.py --destination "$WORK/models/Wan2.1-T2V-1.3B"
```

wheel 仅适用于相应 Python/平台/ABI。先检查兼容性；不兼容时由 AI 调整安装方式并记录环境差异，不改变算法配置。权重为 **Wan-AI/Wan2.1-T2V-1.3B**，不是任意 Wan 2.1、不是 14B、不是 diffusers 转换权重。revision 固定为 `37ec512624d61f7aa208f7ea8140a131f93afc9a`，原生文件约 17.6GB。

## 3. 先检查，再批量

检测空闲健康卡，将进程绑定到确认过的物理 GPU UUID。先选择 pending 中同一 pair 的五方法各一项，在完整设置下验证；不要把预热/计时次数改小来当正式结果。

单任务 CLI（`TASK_INDEX` 为 **0 起始**的运行清单行号，`GPU_UUID` 是实际接收方设备）：

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  "$WORK/venv/bin/python" -m cachebench.generate \
  --manifest "$WORK/manifest.jsonl" --index "$TASK_INDEX" \
  --output-dir "$WORK/results" --vendor-dir "$WORK/vendor" \
  --checkpoint-dir "$WORK/models/Wan2.1-T2V-1.3B" --device 0
```

这里 `--device 0` 指 UUID 隔离后进程可见的第一张卡，不是固定物理 GPU 0。本包没有擅自选择你的 GPU、启动守护进程或设置你的并发数；这些由接手 AI 根据现场资源安排。

检查：`metrics.status=complete`；三个 trace 各 100 个 CFG 分支调用；Uncached 全部 refresh；frames 为 uint8 RGB `(81,480,832,3)`；视频可解码；记录实际 GPU、依赖、源码版本和 profile hash；模型 fingerprint 应为 `b6725fb6e8b1b4869a82655cb9e9eec6730f51199ed91f7e7230be01cbd4eba4`。校准必须通过现有后端兼容性检查，不绕过验证。五方法通过后才能放开清单内批量运行。

统一设置：832×480、81帧、16fps、UniPC 50步、CFG 5、shift 5、BF16、FlashAttention2、offload=false、warmup=1、timing_repeats=3。所有阈值及其它字段以 manifest 为准，禁止重新调参或重新抽样校准。

需要跨终端运行时自行设置后台服务，并保存机器可读的 running/complete/failed 状态、任务归属、PID/物理 UUID 和日志。服务异常后先检查旧状态与文件；不要自动重建一个全新实验。**现有 generate.py 没有视频内部断点恢复，也没有通用自动恢复队列；不要假称已实现。**

## 4. 交付

每项交回 `video.mp4`、`frames.npy`、`metrics.json`、`trace.json`，以及实际运行 manifest、路径映射、环境版本、机器信息、执行代码 hash、失败列表和分片/归属记录。保留原始帧；质量评测不能用有损视频冒充帧。基线和缓存的时间必须来自同一硬件/兼容环境，本机 A100 或原 H200/L40S 的时间不能直接拿来作接收方基线。

结果通过机器主人指定的存储渠道交付；**不要 push 视频、原始帧或模型到 Git。** 此批原始帧约 128GiB，含 MP4/余量约 150GiB，另留模型和环境空间。此估算针对整个合并清单，不限定你的两台机器如何分配。

全部任务产生后，检查去重后的 run_id 集合等于 `pending.jsonl` 与 `new_hardware_baselines.jsonl` 的并集；除明确列出的 2 项新环境基线外，不包含 local_completed 项，没有漏项。生成完成不代表质量评测完成。

## 5. 来源限制

队长原始运行的完整项目代码快照未提供。本包提供的是交出方实际运行过的科学代码：已知校准/启动脚本及官方核心去噪哈希与原记录一致；详见 evidence。不要声称与原服务器完整代码环境逐字相同。当前源码中的校准路径校验、模型 fingerprint 校验、官方 git commit 校验必须保留。

本 README 与当前 `split_summary.json` 的任务归属优先于原 ZIP 和 evidence 内的历史快照；原 ZIP 不能扩大本次授权范围。最新停止状态见 `evidence/local_stop_status.json`。2 项本机中断任务已包含在 pending 中，接收方在自己的新目录重新生成；不要试图清理或恢复交出方的半成品。
