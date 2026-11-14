# C0 capability smoke check（小样本链路检查）

日期：2026-11-14 19:36 CST。  
状态：**仅 smoke/链路检查，不是能力结果**。  
代码基线：`9df92a8` + 本地未提交的 C 入口 `python -I` 路径修复与测试补充。  
上游设计：`docs/records/2026-11-14/c0-capability-exploration-plan.md`  
GPT 审阅/修复：`handoff/2026-11-14-1857-bc-execution-review.md`、`handoff/2026-11-14-1913-bc-review-fixes.md`

## 1. 通过项

### CPU 测试

```bash
python -I -B tools/c_capability/tests/test_c_common.py   # 12 tests, OK
python -I -B tools/c_capability/tests/test_c_summary.py  # 2 tests, OK
```

覆盖：第三父节点与 sink、生成配置/路径、节点加权 vs 逐图等权、空桶支持数、
`tV`/`t4` 完成后有后续观察/无后续观察、既有其他字段错误 vs 真正退化、
曾恢复/恢复后再错/末轮完成。

### 入口依赖

在 `python -I` 下，三个入口都需要显式加入脚本目录才能找到 `c_common`；
已在 `build_data.py`、`run_capability.py`、`summarize.py` 中补上。验证：

```bash
python -I -B tools/c_capability/build_data.py --help
python -I -B tools/c_capability/run_capability.py --help
python -I -B tools/c_capability/summarize.py --help
```

均通过。

### 数据选择 dry-run

`build_data.py --dry-run` 结果：

- N32_shallow：1024 合格，选 64；
- N32_deep：319 合格，选 64；
- N64_shallow：54 合格，需生成；
- N64_deep：954 合格，选 64；
- N128_shallow：0，需生成；
- N128_deep：755 合格，选 64。

### 极小生成

在 `/tmp/c_capability_gen_test` 生成 2 张 N128_shallow（4 候选）：

- `manifest.json` / `N128_shallow.pt` 均可写可读；
- `meta.generation_config` 是实际 `dag_config`，不含 `Path`；
- `source=None`、`generated=True`、`generation_seed=20261114`；
- JSON 序列化和 `torch.load` 均通过。

### Runner → summary

数据：`tmp/analysis/c0-capability-20261114/smoke-data/N32_shallow.pt`，2 张固定图
（source indices 175/242，depth 5/6）。

权重：

- s42_e143：GPU4，2 图 × 64 轮，约 1.9 s；
- s1234_e143：GPU4，同 2 图 × 64 轮，约 2.8 s。

两张图在两个权重下都在实际深度对应的参考轮次附近完成：

| 权重 | graph | depth | tV | t4 |
|---|---|---:|---:|---:|
| s42 | 0000 | 5 | 6 | 6 |
| s42 | 0001 | 6 | 7 | 7 |
| s1234 | 0000 | 5 | 6 | 6 |
| s1234 | 0001 | 6 | 7 | 7 |

`summarize.py` 已输出：

`curves.csv`、`capability_by_H.csv`、`by_node_type.csv`、`by_depth.csv`、
`completion.csv`、`drift.csv`、`drift_by_graph.csv`、`post_completion_curves.csv`、
`examples.json`、`report.md`。

## 2. 观察到的现象（仅 2 图，不作结论）

- 两个权重的 VALUE 在完成后大部分保持正确；
- 两个权重的 first-4 在完成后都出现变错，s42 的 first-4 从 H=16 的 1.0
  降到 H=64 的 0.109；s1234 降到 0.0625；
- s1234 的 VALUE 在 H=64 降到 0.84375，s42 仍为 1.0；
- 这正是“完全算对后漂移”的待观察现象，但 2 张图不能作为能力/稳定性结论。

## 3. 未做/边界

- 未生成六档正式数据；
- 未跑 s42/s1234 的正式 64 图 × 六档；
- 未跑 s2026；
- 未训练 C+、未做人工注错；
- 未改 `board.md` / `log.md`，未提交本轮 C 修改；
- GPU4 仅使用约 3 秒，seed2026 在 GPU0 的训练未受影响。

## 4. 结论

GPT 审阅后的 C 代码链路可用：数据选择/极小生成、真实 checkpoint runner、
summary 全链路均通过。下一步是在正式六档数据上按已批准计划运行 s42/s1234；
本 smoke 不应写成能力结果。
