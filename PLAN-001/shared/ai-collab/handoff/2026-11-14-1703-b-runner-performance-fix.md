# B runner 性能优化交接

时间：2026-11-14 17:03 CST；GPT。

用户明确授权本次优化。开工时本地干净，先 fetch 并快进至
`reproduction@968129a`，包含执行 agent 新增的 C 性能要求。

## 已实现

- 新增 `tools/b_mechanism/trajectory_io.py`，不依赖模型、不在 import 时加载
  torch。`StateRecorder` 每图缓存标签与真值字符串；每个 forward 的输入、输出
  各整块转 CPU 一次，在 Python 列表上构造 VALUE / 前四字段记录。
- early / late / controls 三入口接入同一记录工具。late 新旧真值共用已转 CPU 的
  输入/输出；保留 `truth_ref`、`input_nodes_new`、`output_nodes_new`。
- 新增 `--num_threads`，默认 **4**。在 torch/模型导入前设置 OMP、MKL、OpenBLAS、
  NumExpr 环境变量，torch intra-op 同值、inter-op=1。
- 新增 `--progress_every`，默认每条记录输出进度；每图结束补报 counts、
  已写记录数、耗时、records/s、按已扫描图数估算的 ETA。
- 每条完整轨迹写入并 flush `<out>.jsonl`，同步 `<out>.meta.json` 进度快照；
  正常结束流式生成原有 `{args, counts, records}` JSON，不再积存全部记录。
  JSON 空白格式改为紧凑格式，字段内容保持；args 新增上述运行参数。
- 新增 `run_serial.py`，显式任务清单顺序执行，一次仅一个试跑进程，失败即停。
  **没有生成/执行真实服务器任务清单，没有启动实验。**

未改变：manifest、选点/资格、模型计算、clean、注错位置/值、轮数/时序、
新旧真值规则、汇总指标。`C.rollout` 只增加壁钟计时，没有改变反馈和 hook。

## 已做与待做的检查

- 本地 `unittest discover -s tools/b_mechanism/tests -p 'test_*.py'`：**19 项通过**，
  包括现有检查与新增 3 项（CPU 字段/新旧真值等价及整块传输、流式最终 JSON、
  中断 JSONL 的完整记录保留）。语法编译和 `git diff --check` 通过。
- 新增等价检查使用 CPU 列表/传输计数替身；**不是实际模型/GPU 等价或测速**。
- 执行 agent 做已有交接要求的小验证即可：相同图/权重/manifest 下 horizon 2/8
  对比旧版与新版记录（忽略 args 新增参数、文件路径与空白格式），包含 late 新真值
  和 controls 输入序列；4 图 horizon 64 测时间及训练资源影响。
- `rollout_s` / `record_build_s` 是壁钟辅助定位，没有新增 CUDA synchronize。
  异步 GPU 等待可能计入 record_build；加速结论以端到端时间为准。
  **5× 是原交接期望，尚无实测，不写成已达成。**

## 执行方式

1. 不更新 seed2026 运行中的工作树。需要新代码时使用 reproduction 此提交的独立
   目录快照，数据/权重沿用绝对路径。B 新试跑串行，不动训练进程。
2. 沿用已有 manifest 和六项原参数，仅添加 `--num_threads 4`（可省略，默认4）
   与全新的 `--out` 路径。原 early 两份文件保留；已有输出/JSONL/meta 不覆盖。
3. 小验证后按原授权重跑六项，写新目录；结果分析及 C 探索次序不变。

可选串行启动器（执行 agent 填入实际六项参数，不使用占位符运行）：

```json
{
  "num_threads": 4,
  "jobs": [
    {
      "argv": ["tools/b_mechanism/b_mechanism_early.py", "其余原参数"],
      "gpu": "1",
      "log": "tmp/analysis/新目录/A_early.log"
    }
  ]
}
```

在仓库根目录运行 `python tools/b_mechanism/run_serial.py --jobs <任务清单.json>`。
不同时启动多个串行队列。线程数限制不保证绝对资源隔离，是否影响训练由执行
agent 的小测报告；不按 GPU 空闲就恢复六项并行。

## 中断与复用边界

- 当前正在计算的一条轨迹可能丢失；此前已 flush 的完整记录保留。不是自动续跑，
  也不保证断电时数据持久化。meta 的 counts 是进度快照，可能落后最近一条记录。
- 如需查看中断产物，可运行：
  `python tools/b_mechanism/trajectory_io.py --journal <out>.jsonl --out <新文件>.json`。
  导出显式 `partial=true`、`counts_are_progress_snapshot=true`，只收完整行，
  **不作为完整正式实验结果/最终分母**。原 JSONL 不改动。
- C 后续复用 StateRecorder / RecordWriter 观察离散读出；连续 hidden 仍走 C 自己的
  nointerm 反馈，绝不调用 B 离散 rollout 替代。没有改动或新增 C runner。
