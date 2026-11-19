# A/B重新使用工作空间；C状态可用性检验

本目录承接2026-11-18首批结果。模型展示名统一A/B/C，旧机器ID只用于精确追溯。
详细问题、执行队列与验收见[交接](../../ai-collab/handoff/2026-11-18-state-reuse.md)。

**直接执行入口已补齐（2026-11-19）：**见[队列与结果包交接](../../ai-collab/handoff/2026-11-19-execution-bundles.md)。`batch.py`生成并串行运行两位agent各自的smoke/main队列，记录日志、来源与完成回执；`deliver.py`自动整理关键表和完整案例。实验范围不变。原单任务入口仍可用于定位环境问题。

## 入口

| 文件 | 内容 |
|---|---|
| `core.py` | 独立模23 ADD求值、重复赋值计划、死槽位版本、方程位置校验、C分层与末两轮定义 |
| `run_discrete.py` | 从已审阅A/B起跑快照出发；repeat或reuse，6分支，实际输入/输出全记录 |
| `run_c_function.py` | C四训练版本，3时点×4扰动来源×3固定检查延迟，10分支历史状态诊断 |
| `summarize.py` | 两路线分别汇总；逐图等权与计数、共同分母、配对差异、完整资格、稳定段 |

运行器不训练、不调用现有运行目录的写操作、不修改原始数据。
所有输出目录必须不存在；大记录/快照留`tmp/`，小表/来源hash/报告进git。

## A/B示例

`--source-dir`是上一批已完成且meta/records/selection哈希匹配的目录；每图只取第一delta对应的原始起跑状态，不使用干预后的旧结果筛样本。

```bash
python tools/state_reuse/run_discrete.py \
  --source-dir tmp/analysis/priority-pilots-20261118/reassign-main-A \
  --mode repeat --horizons 4 --limit-graphs 2 \
  --out-dir tmp/analysis/state-reuse-20261118/smoke-repeat-A

# 小验证后：同一源目录，不设limit，H8/H16分别从相同旧快照起跑。
python tools/state_reuse/run_discrete.py \
  --source-dir tmp/analysis/priority-pilots-20261118/reassign-main-A \
  --mode repeat --out-dir tmp/analysis/state-reuse-20261118/repeat-A

# B对应source目录与输出名换成B，所有预算相同。
python tools/state_reuse/summarize.py \
  --run-dir tmp/analysis/state-reuse-20261118/repeat-A \
  --run-dir tmp/analysis/state-reuse-20261118/repeat-B \
  --out-dir tmp/analysis/state-reuse-20261118/repeat-summary
```

`--mode reuse`另起目录，先两候选图/H4小验证，再默认32候选/H8、H16。
不因reuse无合格死槽位或某模型失败而补样。

## C示例

```bash
python tools/state_reuse/run_c_function.py \
  --candidate-count 2 --horizon 4 --arms initial --snapshot-graphs 1 \
  --out-dir tmp/analysis/state-reuse-20261118/smoke-C

# 小验证后：默认四版本、固定前32图、H16、前2图保存隐藏状态快照。
python tools/state_reuse/run_c_function.py \
  --out-dir tmp/analysis/state-reuse-20261118/C
python tools/state_reuse/summarize.py \
  --run-dir tmp/analysis/state-reuse-20261118/C \
  --out-dir tmp/analysis/state-reuse-20261118/C-summary
```

可按`--arms initial/hold/noise/semantic`逐版本串行写不同目录，再给summarize多个`--run-dir`；数据、窗口、源码必须一致。共同资格按传入版本集合计算，正式四版本表须全部传入。
当前上下文是s2026原始权重及其三份小试后训练权重，**不是四种子**。

## 输出口径

- `coverage.jsonl`涵盖固定候选，包括未入选；`records.jsonl`逐事件保留实际输入输出、条件与真值。meta complete才可汇总。
- C `joint_states.csv`是同一事件的联合条件，不用两个边际统计相除。`common_recorded`只保证结构相同；`common_input_qualified`还要求四版本共同满足输入资格。自身资格和全体结果同时报告。
- `case_curves.csv`/`summary.csv`给1/2/4/8/16和末两轮；原始记录保留每轮。空区域分母0、比例空值，不记100%。
- `events.csv`的稳定段至少包含2次输出，只描述窗口内稳定；不证明收敛。
- `paired_effects.csv`比较同一案例的干预分支；`paired_models.csv`给共同A/B案例末两轮差异。
- `update_retention.csv`给A/B旧值残留、新值接受，以及原真值未变节点相对同时间clean的额外损伤。clean/sham按原题判对，任务分支按当轮新题判对。
- C历史未完成向量不保证擦除答案，干净匹配轨迹也不等于完整正确hidden。结果只对被实施的状态替换及后续计算有效。

## 本地必要检查

```bash
python -m unittest discover -s tools/state_reuse/tests -v
```

涵盖循环赋值回到原题、死槽位消费新版本、循环图拒绝、模抵消分母、真实方程顺序、稳定段、C输入资格与共同分母。GPU tensor操作、真实tokenizer和服务器路径的验收由执行agent完成，不把CPU通过写成实验结果。
