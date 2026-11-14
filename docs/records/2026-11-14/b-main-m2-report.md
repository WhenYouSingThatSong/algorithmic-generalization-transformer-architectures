# B 主实验 M2：complete 目标类型阶段数据

日期：2026-11-14
代码基线：`reproduction@7ec0db8`（M2 运行时代码；B 实现未改）
状态：**数据表，不是机制结论；M3/M4 尚未启动。**

## 1. 数据范围与完整性

固定候选沿用 M0：`graph_id 32–287`，256 张候选，无 top-up；256 张
common complete，218 张有链，365 条主链。M2 使用同一起点 clean 与
`late_r,late_u,sink,leaf_value,unrelated`，主 H=16，同时保留 H=8 截取；
固定 long32 子集（32–63）同条件跑 H=64。

| 子集 | 模型 | 记录数 | 图 | 各条件 target 数 |
|---|---:|---:|---:|---|
| complete main256 | A′/B | 3,873 each | 256 | R/U/unrelated 365；sink/leaf 512 |
| complete long32 | A′/B | 501 each | 32 | R/U/unrelated 49；sink/leaf 64 |

所有 4 个输出 `complete=true`，`skipped_not_complete=0`，
`skipped_no_common_complete=0`；各条件 `n_skipped_no_clean=0`，
`n_skipped_short=0`。模型与数据同 M1：A′ s42 e86，B-M6 s42 e92；
heldout `testdata_N32.pt`。clean 分支在 H=16/64 的全节点正确率均为
100%。

## 2. 聚合计数：all_affected / VALUE

### 2.1 complete main256，H=8 与 H=16（graph-equal；括号为 node-weighted）

| 模型 | 条件 | damage H=8 | stable H=8 | damage H=16 | stable H=16 |
|---|---|---:|---:|---:|---:|
| A′ | R | 0.1311 | 0/332 | 0.1311 | 0/332 |
| A′ | U | 0.1738 | 0/365 | 0.1738 | 0/365 |
| A′ | leaf_value | 0.0680 | 0/400 | 0.0680 | 0/400 |
| A′ | sink | 1.0000 | 0/512 | 1.0000 | 0/512 |
| A′ | unrelated | 0.5115 | 0/365 | 0.5115 | 0/365 |
| B | R | 0.8315 | 2539/2604 (0.9834 / 0.9750) | 0.8315 | 2604/2604 (1.0000 / 1.0000) |
| B | U | 0.8121 | 2249/2263 (0.9965 / 0.9938) | 0.8121 | 2262/2263 (0.9994 / 0.9996) |
| B | leaf_value | 0.9022 | 6521/6667 (0.9835 / 0.9781) | 0.9022 | 6667/6667 (1.0000 / 1.0000) |
| B | sink | 0.0000 | 0/0 (NA) | 0.0000 | 0/0 (NA) |
| B | unrelated | 0.4873 | 778/782 (0.9988 / 0.9949) | 0.4873 | 782/782 (1.0000 / 1.0000) |

A′ 的 damage 计数（H=16）：R 332/3042、U 365/2677、leaf 400/7240、
sink 512/512、unrelated 365/1149。
B 的 damage 计数（H=16）：R 2604/3042、U 2263/2677、leaf 6667/7240、
sink 0/512、unrelated 782/1149。

### 2.2 complete long32，H=8/H=16/H=64

| 模型 | 条件 | damage H=8/16/64 | stable H=8 | stable H=16 | stable H=64 |
|---|---|---:|---:|---:|---:|
| A′ | R | 0.1189 | 0/40 | 0/40 | 0/40 |
| A′ | U | 0.1793 | 0/49 | 0/49 | 0/49 |
| A′ | leaf_value | 0.0673 | 0/48 | 0/48 | 0/48 |
| A′ | sink | 1.0000 | 0/64 | 0/64 | 0/64 |
| A′ | unrelated | 0.5949 | 0/49 | 0/49 | 0/49 |
| B | R | 0.8370 | 341/349 (0.9724 / 0.9771) | 349/349 | 349/349 |
| B | U | 0.8082 | 299/300 (0.9984 / 0.9967) | 300/300 | 300/300 |
| B | leaf_value | 0.9082 | 805/817 (0.9873 / 0.9853) | 817/817 | 817/817 |
| B | sink | 0.0000 | 0/0 (NA) | 0/0 (NA) | 0/0 (NA) |
| B | unrelated | 0.4051 | 79/79 (1.0000 / 1.0000) | 79/79 | 79/79 |

A′ 的 damage 计数（H=64）：R 40/404、U 49/355、leaf 48/886、
sink 64/64、unrelated 49/128。
B 的 damage 计数（H=64）：R 349/404、U 300/355、leaf 817/886、
sink 0/64、unrelated 79/128。

## 3. unrelated 的 protected 分支（main256 H=16 / long32 H=64）

| 模型 | 子集 | protected_target damage | protected_shared_descendants damage / stable | protected_exclusive damage / stable | direct damage / stable |
|---|---|---|---:|---:|---:|
| A′ | main256 | 0/365 | 0/165 / 0/0 | 0/2512 / 0/0 | 365/365 / 0/365 |
| B | main256 | 0/365 | 165/165 / 165/165 | 12/2512 / 12/12 | 0/365 / 0/0 |
| A′ | long32 | 0/49 | 0/24 / 0/0 | 0/331 / 0/0 | 49/49 / 0/49 |
| B | long32 | 0/49 | 24/24 / 24/24 | 0/331 / 0/0 | 0/49 / 0/0 |

## 4. 直接位置首轮（actual injection）

main256：

| 模型 | R | U | leaf_value | sink | unrelated |
|---|---:|---:|---:|---:|---:|
| A′ 首轮改回 / 仍错 | 33 / 332 | 0 / 365 | 112 / 400 | 0 / 512 | 0 / 365 |
| B 首轮改回 / 仍错 | 365 / 0 | 365 / 0 | 512 / 0 | 512 / 0 | 365 / 0 |

long32：A′ R 9/40、U 0/49、leaf 16/48、sink 0/64、unrelated 0/49；
B 对应为 49/0、49/0、64/0、64/0、49/0。所有条件 `actual` 与条件
target 数一致，未出现 empty-to-value 混入。

## 5. 我认为

以下是我的看法，单独列出，不混入上面的计数：

1. 我认为 M2 的计数已经足够让 GPT 直接核验，但目前最需要先确认的是
   `complete_sink`：B 的 all_affected damage 是 0/512，A′ 是 512/512；
   这类分母为 0 的条件不适合直接写“100% 恢复”，应先确认该条件是否
   真的产生了输出损伤。
2. 我认为 B 在 R/U/leaf_value/unrelated 上的 stable 计数与 early 的模式
   一致；是否把它写成最终结论，由 GPT/用户决定。U 的 2262/2263 中
   唯一不稳定节点是 graph 142 的 pos 128（非 direct），值得单独留档。
3. 我认为 unrelated 的 protected_target 在两个模型、两个子集上都是
   0 损伤；B 的 shared_descendants 出现损伤且 stable 为满分母，
   exclusive 在 main256 只有 12/2512、long32 为 0/331，这些应作为
   原始计数保留。
4. 我认为 M2 数据本身没有出现需要停止后续阶段的明显异常；但在
   sink 和 unrelated 的分母口径确认前，不应把“stable 100%”写成统一
   结论。

## 6. 产物与复现入口

- 原始轨迹：`tmp/analysis/b-main-20261114/late/`
- 汇总：`tmp/analysis/b-main-20261114/summary/complete_main256/`、
  `summary/complete_long32/`
- 关键计数：`docs/records/2026-11-14/b-main-m2-key-metrics.csv`
- 运行命令：`b_mechanism_late.py`，`--conditions
  late_r,late_u,sink,leaf_value,unrelated`，`--delta 3 --u_delta 1
  --value_mode a_plus_1`，main256 H=16 / long32 H=64，4 threads，GPU3
  串行。

M3 sustained 与 M4 D1/D2/D3 未启动。