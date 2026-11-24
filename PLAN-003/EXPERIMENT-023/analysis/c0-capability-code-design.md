# C0 能力探索：代码实现设计（GPT 已审阅）

日期：2026-11-14 18:00 CST；19:36 更新 smoke 状态。  
状态：**代码已写并经 GPT 审阅；CPU 测试、极小生成和 2 图 smoke 通过；正式六档已运行，报告见 `c0-capability-report.md`**。  
上游设计：`docs/records/2026-11-14/c0-capability-exploration-plan.md`

## 1. 交付内容

新增 `tools/c_capability/`：

| 文件 | 作用 |
|---|---|
| `c_common.py` | torch-free 的常量、深度筛选、节点元数据、`tV`/`t4` 与漂移定义 |
| `build_data.py` | 从现有数据池构建六档固定数据；缺档时用官方生成器补齐 |
| `run_capability.py` | 加载一个 C0 checkpoint，连续 hidden 自由续算 64 轮，逐图保存 npz |
| `summarize.py` | 汇总逐轮曲线、H 切片、节点类型/深度分解、完成与漂移、案例 |
| `analyze_existing.py` | 只读现有 NPZ 的 CPU 复分析：逐字段、错误深度/数量、完成延迟、失败/恢复事件 |
| `tests/test_c_common.py` | 纯 CPU 单元测试；不加载模型 |
| `tests/test_c_analysis.py` | 纯 CPU 事件/表定义测试；不加载模型 |
| `README.md` | 文件、数据来源、命令、记录格式与 review 点 |

## 2. 数据档与来源

| 档 | 来源 | 实际深度合格数 | 处理 |
|---|---|---|---|
| N32_shallow | `tmp/analysis/matched-depth/ADD/testdata_N32.pt` | 1024（3–6） | 现有池抽样 64 |
| N32_deep | `tmp/analysis/heldout-seed42/ADD/testdata_N32.pt` | 319（7–11） | 现有池抽样 64 |
| N64_shallow | `experiments/evaluation/val_datasets/Tr32Test128/ADD/testdata_N64.pt` | 54（≤6） | 不足，生成 64 |
| N64_deep | 同上 | 954（7–12） | 现有池抽样 64 |
| N128_shallow | 无 | 0 | 生成 64 |
| N128_deep | `tmp/analysis/heldout-seed42/ADD/testdata_N128.pt` | 755（7–12） | 现有池抽样 64 |

- 抽样只按固定 seed 洗牌，**不使用模型是否答对**；记录 `source_indices`、源文件 SHA256、实际深度统计。
- 生成使用 `generate_data.generate_dataset`，`n_vars=128`（129 词表），ADD/mod23、6 叶、fan-in 1–3；生成 seed 固定。
- 生成的 N64/N128 浅层是**全新池**，不与现有 54 张 N64 浅图混合，避免来源混杂。

## 3. 固定权重

| 名称 | checkpoint | SHA256 |
|---|---|---|
| `s42_e143` | `tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-14-02:35:10/epoch=143-val_loss=0.0000.ckpt` | `4a00d6a2…` |
| `s1234_e143` | 同目录 `seed-1234 - 2026-11-14-02:38:10/epoch=143-...` | `0bca3a2b…` |

- s2025 不测；s2026 训练完成后通过 `--ckpt`/`--weight-name` 补入，不改数据与指标。
- 固定 e143 是为了与看板已有结果一致；不按本批结果重新挑 checkpoint。

## 4. 运行与记录

- 从 `DAGDataModule` 的输入状态出发（变量 VALUE=EMPTY）。
- 每轮 `hidden -> model(hidden, nointerm=True) -> 新的 hidden`；解码只写入记录，不回灌。
- 每图固定 64 轮；显式设置，不依赖旧配置的 `max_val_loop=36`。
- 每个 numeric node 保存前四因子读出和固定真值；节点集合排除 PAD、运算符、题面常量。
- `tV`：整图 VALUE 首次全对；`t4`：整图前四属性首次全对；分别记录。
- 完成后按相对轮次记录第一次 VALUE 退化、第一次前四退化、是否回正、最终是否正确；`tV` 与 `t4` 的漂移分开，不合并成单一“完成”。

## 5. 汇总口径

- 核心：节点 VALUE 正确率、节点前四属性同时正确率；按 seed/档/轮次。
- 分解：全部节点、leaf/internal/sink、按真实深度的节点；逐图等权为主，节点加权并列。
- 辅助：整图 VALUE 全对率、整图前四全对率；明确不等于节点指标。
- 输出：`curves.csv`、`capability_by_H.csv`、`by_node_type.csv`、`by_depth.csv`、`completion.csv`、`drift.csv`、`drift_by_graph.csv`、`examples.json`、`report.md`。

## 6. GPT review 结果与剩余 open questions

GPT 已在 `handoff/2026-11-14-1857-bc-execution-review.md` 审阅，并在
`5669677` 修复第三父节点、生成 meta、逐图等权/漂移口径；本文件上方原有
C1–C4 问题已不再作为未决项。仍建议后续报告时明确：

1. N64_shallow/N128_shallow 采用新生成池，不与现有 54 张 N64 浅图混合。
2. `tV`/`t4` 漂移分开；`other_only_error`（错误存在）与
   `other_only_ever`（已正确字段后变错）不可混称。
3. 每档 64 图是探索预算；整图全对率会较粗，报告需保留覆盖 n。
4. s1234 e143 是固定可比性选择，不是最佳 checkpoint。
5. 逐图 npz + `index.jsonl` 保留；summary 已输出逐图等权与节点加权两套值。

## 7. 检查状态

- `py_compile` 通过。
- `tests/test_c_common.py` 12 项、`tests/test_c_summary.py` 2 项 CPU-only
  检查通过（含第三父节点、既有错误 vs 漂移、逐图等权、空桶支持数）。
- 三个入口在 `python -I` 下的 `--help` 通过；期间修复了脚本目录未加入
  `sys.path` 导致 `c_common` 找不到的问题。
- `build_data.py --dry-run` 与极小 N128_shallow 生成/加载通过。
- 真实 checkpoint runner→summary 小样本检查通过：s42/s1234 各 2 图 × 64 轮，
  GPU4 约 3 秒；详见 `docs/records/2026-11-14/c0-capability-smoke.md`。
- 正式六档数据已由 `build_data.py --allow-generate` 生成；s42_e143 与
  s1234_e143 各完成 384 图 × 64 轮，无 Traceback/OOM；汇总已生成。
  结果、边界与典型轨迹见 `docs/records/2026-11-14/c0-capability-report.md`。
- 现有 NPZ 的 CPU 复分析已加入 `analyze_existing.py`（14 项测试通过），
  输出逐字段 micro/graph-equal、错误深度/数量/完成队列、完成延迟、
  逐图事件（含首次受影响 positions）/事件汇总/同图事件顺序；不重跑模型。
- **未训练 C+、未扩展人工注错矩阵；seed2026 训练未受影响。**

