# B 长轮数试跑的性能问题与修改建议（交 GPT）

时间：2026-11-14 16:45 CST
提出：执行 agent（claude）
状态：**未改性能代码**；请 GPT 实现，执行 agent 之后只做小样本等价检查和重跑。

## 背景

用户已批准 B 长轮数小试跑。执行 agent 在 15:39 同时启动六项：

- A′/B early after_u（clean/u/r，64 轮）
- A′/B late complete（late_u/late_r，64 轮）
- A′/B sustained_r（强制 4 轮 + 释放后 64 轮，总 68 轮）

固定 32 张 heldout N32 图、42 个 primary chains，manifest 在
`tmp/analysis/b-horizon-pilot-20261114/manifest.json`。六项跑了 53 分钟后，
只有 A′/B early 两项完成并各写盘 126 条记录；late/sustained 四项被停止
（用户指示），未写盘。原因不是模型/条件逻辑错误，而是记录构造的 CPU/GPU
同步开销和并行线程失控。

## 现象与证据

- 每个 runner 进程约 **224 个 OS 线程**；六项并行时 cgroup CPU 配额
  **84 核**被占满（抽样约 88 核），GPU1–6 利用率只有 1–5%。
- seed2026 训练（GPU0，未动）的 epoch 用时：
  - epoch 60–72：4:23 / epoch，5.9 it/s；
  - epoch 73：8:33，3.0 it/s；
  - epoch 74：34:07，0.76 it/s；
  - 停掉四项后仍只有约 1.07 it/s（累计均值），GPU0 利用率约 64%。
- 六项各自的纯计算量并不大：early 126 条 × 64 轮，late 126 条 × 64 轮，
  controls 84 条 × 68 轮；模型是 batch=1 的小 forward。

## 根因（代码级）

1. **逐标量 GPU 同步**：
   - `tools/b_mechanism/b_mechanism_common.py:354` 的 `factor_strings` 对
     `SYNTAX/VARIABLE/OPERATION/VALUE` 四个 factor 各做一次
     `int(row[idx])`；`node_state`（:366）又同时对 prediction 和
     `label_b[p]` 各做一次，等于 8 次/位置。
   - early fixture 每轮调用两次 `node_state`（input/output），约 16 次/位置；
     再加上 `values_at`/`token_values` 的 2 次/位置，约 18 次/位置/轮。
   - 32 个位置时约 **580 次 GPU→CPU 标量同步/轮**；126 条 × 64 轮
     ≈ **470 万次/任务**。`label_b` 的真值每轮都重新从 GPU 读，实际是常量。
2. **线程失控**：runner 没有限制 torch/OMP 线程；每进程 224 线程，六项
   同时与 seed2026 共用 84 核 cgroup，调度抖动把训练拖慢近 8 倍。
3. **没有进度和中间落盘**：所有记录先在内存累积，最后一次性 `json.dump`；
   53 分钟没有可观察进度，kill 后全部丢失。

## 建议改动（保持实验语义和 JSON schema）

1. **CPU 侧构造记录**：
   - 每轮只做一次 `inputs[t][0].detach().cpu()`、`outputs[t][0].detach().cpu()`；
     `label_b` 每图一次 `.detach().cpu()`，缓存真值字符串，按 position 复用；
   - `factor_strings`/`node_state` 接受 CPU 行或 Python list，先 `tolist()`，
     再按索引取值；不要再对 GPU tensor 逐 factor 做 `int()`；
   - 保留现有 `node_state` 作为兼容 wrapper，新增 CPU 版本供 early/late/
     controls 的 `rollout_records`/`round_records`/`values_at`/`token_values`
     使用。三处重复的 state→string 逻辑可合并到 common。
2. **线程限制**：runner 增加 `--num_threads`（默认 4 或 8），入口
   `torch.set_num_threads(args.num_threads)`；启动脚本同时设置
   `OMP_NUM_THREADS`/`MKL_NUM_THREADS`。selection 很短，可沿用默认。
3. **进度与增量落盘**：
   - 每 N 张图/记录打印并 flush 一次 `processed/emitted/skipped`、records/s、
     elapsed、ETA；
   - 每条记录追加写 `*.jsonl`（即使最终 JSON 还没生成，kill 后也有部分结果）；
     最终仍写现有 JSON 供 `b_mechanism_horizon_summary.py` 读取。可加一个
     小的 jsonl→json 转换/恢复入口。
4. **并行度**：提供启动脚本，最多同时跑 2 项（或串行），按 GPU 固定分配；
   不要六项并行共用 84 核配额。也建议对长任务用较低 nice，避免与训练争抢。
5. **可选（先不做）**：把同一图的 42 chains × 3 conditions 合成 batch rollout，
   提高 GPU 利用率。语义上可行，但会改动 rollout 组织和记录顺序，建议先做
   1–4 并测速，仍不够再考虑。
6. **测速**：在记录构造里加计时（rollout vs record build），用 4 图 / horizon 64
   报告优化前后 records/s 和总时间，确认瓶颈确实转移。

## 验收要求

- CPU 等价测试：同一 synthetic state/label，旧 `node_state`/`values_at` 与新
  CPU 路径输出完全一致的 `input`/`output`/`input_nodes`/`output_nodes`；
- 2–4 图、horizon 2 和 8 的 GPU 小样本：新 runner 生成的记录字段与旧实现逐值
  一致；`tools/b_mechanism/tests/test_horizon_summary.py` 和
  `test_review_regressions.py` 继续通过；`b_mechanism_horizon_summary.py`
  不改就能消费；
- 4 图、3 条件、horizon 64 的测速，目标至少 5× 加速，且限制线程后不再明显
  拖慢 seed2026；
- 不改 manifest/selection/资格/条件/注入值/轮数/JSON 字段；不在未获用户批准时
  启动全量或大规模重跑；不动 seed2026。

## 当前产物与重跑安排

- `tmp/analysis/b-horizon-pilot-20261114/manifest.json`：选样保留，无需重做；
- `A_early_after_u.json`、`B_early_after_u.json`：126 条/个，是优化前完成的数据，
  先保留作对照；按用户“都重跑”的要求，性能修改后用同一 manifest 重跑六项，
  输出加 `_v2` 或放到新目录，避免与旧文件混淆；
- late/sustained 四项无输出，重跑即可；
- 性能修改完成、通过上述验收后，执行 agent 再按 board 的 B 长轮数任务重跑六项，
  然后做 H 表/曲线/报告；C 能力探索仍排在 B 之后，不提前启动。