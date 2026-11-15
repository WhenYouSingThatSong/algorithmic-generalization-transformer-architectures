# B 多父节点与多错值：已有答案更新（repair study 第二批）

日期：2026-11-15
运行 commit：`8907018`（`tools/repair_study/run_b.py`、`multifanin.py`、`summarize.py`）
权重：A′ s42 e86（`0b66e196...`）；B-M6 s42 e92（`ef58ba26...`）
数据：heldout N32（`08f52382...`），同一 main256 manifest
状态：**数据报告；未重训，未启动 M3/M4/C+，不作机制结论。**

## 1. 样本与运行

- 固定候选图 `graph32–95` 共 64 张，使用新的结构选点：每张图分别枚举
  数值父数 1/2/3，至少有一个子节点、depth≥1 的内部目标，每组按固定
  seed=20261115 的哈希选一个；没有缺组，也没有用其他父数补位。
- 每个模型 192 个 target（父数 1/2/3 各 64 个）；两个时点各 192 条，
  共 384 条主记录；graph142/128 另加 2 条 anchor，合计 386 条。
- A′、B 均 `meta.status=complete`，386 条全部 `eligible`；扫描 64 张
  候选图 + graph142，无缺图、无 ineligible。
- 目标的实际深度：1/2/3/4/5/6 = 52/53/45/27/9/6；实际正确 VALUE
  覆盖 0–22（22 只出现 1 次）。原值不筛选，按实际值报告。
- 两个时点：
  - `just_solved`：原 manifest 预算内该模型第一次实际输出目标正确值，
    要求该状态目标及全部数值父 VALUE 正确；
  - `whole_graph_complete`：原 manifest 两模型首次共同整图 VALUE 完成
    状态，重新检查本模型该状态整图正确。
- 固定 `graph32–39` 的选中目标另做全部 22 种错误值枚举；其余
  `graph40–95` 只做 +1/+3/+7/+11 四种错值。因此每个时点有
  24 个枚举 target（8 图×3 父数）和 168 个采样 target。
- 每个分支只改目标 VALUE 一次，随后自由续算 16 轮，父值不钳制。
  所有 `correct_write` 与 `unchanged` 的 16 轮输出逐位相同；没有
  replay/相同输入控制失败。
- 运行：GPU2，model_a 后 model_b，16 轮；命令与原始
  `meta.json` 保存在下节产物目录。A′ 约 8 分钟，B 约 8 分钟。

## 2. 目标 VALUE

两个时点的计数相同，表中每格是 `首轮正确/末两轮正确`，每分支 n=192。

| 分支 | A′ just_solved | A′ whole_graph | B just_solved | B whole_graph |
|---|---:|---:|---:|---:|
| unchanged | 192/192 | 192/192 | 192/192 | 192/192 |
| correct_write | 192/192 | 192/192 | 192/192 | 192/192 |
| empty | 192/192 | 192/192 | 192/192 | 192/192 |
| wrong_plus_1 | 0/0 | 0/0 | 192/192 | 192/192 |
| wrong_plus_3 | 0/0 | 0/0 | 192/192 | 192/192 |
| wrong_plus_7 | 0/0 | 0/0 | 192/192 | 192/192 |
| wrong_plus_11 | 0/0 | 0/0 | 192/192 | 192/192 |

- `empty` 的初始目标 VALUE 是 24（非数值），首轮即回到正确值；
  上表首轮 192/192 不表示初始正确。
- A′ 的四个错值分支 `ever_output_wrong=192/192`；B 为 0/192。
- 全部错值枚举：A′ 的 22 个错值 ×3 父数 ×2 时点 ×每组 8 个 target
  共 1056 个 target-时点观察，首轮和末两轮均 0；B 全部 1056 个均为正确。
- 四个采样错值：A′ 的 4×168×2=1344 个 target-时点观察均 0；
  B 全部 1344 个均为正确。
- 把枚举和采样合并，A′ 的错值 target 观察为 0/2400，B 为
  2400/2400。这里的 2400 是 target×时点×错值的观察数，不是
  2400 个独立图。
- 这些计数在父数 1/2/3 分层中相同：A′ 每个父数每时点 0/64，
  B 每个父数每时点 64/64；没有看到本批目标 VALUE 结果随父数改变。

## 3. 下游、父值与局部外区域

**分母说明**：每个模型每个时点 192 个 target；`all_nodes=6144`
= 192 次干预 ×32 个节点观察。`downstream=795` 是这些干预下重复
观察到的下游节点数，不是 795 个独立节点/独立图；`parents=384`
= 192×2 个父节点观察；`outside_local_region=4773` 是其他节点观察。

第16轮（末两轮）VALUE 正确数：

| 模型/时点/分支 | target | downstream | parents | outside | all_nodes |
|---|---:|---:|---:|---:|---:|
| A′ just wrong_plus_1/3/7/11 | 0/192 | 0/795 | 384/384 | 4773/4773 | 5157/6144 |
| A′ just empty | 192/192 | 795/795 | 384/384 | 4773/4773 | 6144/6144 |
| A′ whole wrong_plus_1/3/7/11 | 0/192 | 795/795 | 384/384 | 4773/4773 | 5952/6144 |
| A′ whole empty | 192/192 | 795/795 | 384/384 | 4773/4773 | 6144/6144 |
| B just/whole wrong_plus_1/3/7 | 192/192 | 795/795 | 384/384 | 4773/4773 | 6144/6144 |
| B just/whole wrong_plus_11 | 192/192 | 760/795 | 384/384 | 4773/4773 | 6109/6144 |
| B just/whole empty | 192/192 | 795/795 | 384/384 | 4773/4773 | 6144/6144 |

- B 的 `wrong_plus_11` 末两轮有 35/795 个下游节点观察不正确；
  涉及 5 个 target（每个 target 在两个时点各出现一次，共 10 条
  case 记录）：`B:44:26`（fan-in1，16 个下游节点）、`B:41:82`
  （fan-in2，7）、`B:39:54`（fan-in3，7）、`B:54:84`（fan-in1，2）、
  `B:89:106`（fan-in3，3）。两个时点的 target 首轮都已回到正确值，
  但上述下游节点在 16 轮末端仍不正确。
- 在 22 个错值的枚举集中，B 的 `wrong_plus_11` 和 `wrong_plus_17`
  各有 7/116 个下游观察末两轮不正确，且都落在 `B:39:54` 的同一批
  7 个节点；其余 20 个错值在该枚举集中为 0/116。
- A′ 的 22 个错值在 `just_solved` 枚举集中每个错值都是
  116/116 下游受损；在 `whole_graph_complete` 枚举集中每个都是
  0/116。采样集的四个错值同样表现为 just_solved 的 795/795 下游
  受损、whole_graph 的 795/795 下游正确。
- `first4` 与 VALUE 的末两轮端点计数：在 1920 个主表分组中逐组
  相同，没有看到本批四属性端点与 VALUE 端点分离。
- 父节点 VALUE 在所有分支的末两轮均为 384/384；这里没有钳制父值，
  但本批窗口内未观察到父值变化。`outside_local_region` 也全部正确。

逐轮 `downstream` 受损数（VALUE，`node_curves`/round summary 的
1/2/4/8/16 轮切片）：

| 模型/时点/分支 | 第1轮 | 第2轮 | 第4轮 | 第8轮 | 第16轮 |
|---|---:|---:|---:|---:|---:|
| A′ just wrong_plus_1 | 265 | 490 | 737 | 795 | 795 |
| A′ just empty | 265 | 197 | 62 | 0 | 0 |
| A′ whole wrong_plus_1 | 0 | 0 | 0 | 0 | 0 |
| A′ whole empty | 3 | 0 | 0 | 0 | 0 |
| B just wrong_plus_1 | 265 | 192 | 60 | 0 | 0 |
| B just wrong_plus_11 | 265 | 192 | 78 | 35 | 35 |
| B just empty | 265 | 208 | 75 | 0 | 0 |
| B whole wrong_plus_1 | 374 | 296 | 92 | 0 | 0 |
| B whole wrong_plus_11 | 365 | 286 | 110 | 35 | 35 |
| B whole empty | 350 | 295 | 89 | 0 | 0 |

表中“第16轮”列是逐轮值；报告主表的“末两轮”取第15、16轮。
B 的 wrong_plus_1/3/7 和 empty 在第1–7轮有临时下游损伤，第8轮
起为 0；wrong_plus_11 的 35 个节点没有在16轮内恢复。A′ 的
wrong_plus 在 just_solved 从第1轮扩大到 795 后保持不变。

## 4. 同目标跨时点

- A′、B 都有 192 个 target 在两个时点都合格；匹配时点后仍是
  384 条 case（192 target×2 时点），没有缺失。
- `target_cases` 的 VALUE 和 first4 各形成 3552 个同目标跨时点
  配对；所有配对的目标末两轮结果都是“两个时点都正确”或“两个
  时点都不正确”，`just_solved` 单独正确、`whole_graph_complete`
  单独正确的计数都是 0。
- 这只描述 target VALUE：A′ 的 wrong_plus 在 just_solved 对
  downstream 是 0/795，在 whole_graph 是 795/795，因此不能把
  “target 跨时点相同”写成下游也相同。完整逐 target 表见
  `b-multiparent-cross-timing.csv`。

## 5. graph142 anchor

graph142/128 的父节点为 `[92,64,22]`，真值 VALUE 分别为
`10/5/10`。两个起点都记录 `state_round=8`，都再经过 2 个 forward：
`after_original_fault` 先把位置26从 5 改成 6，再续跑 2 轮；
`clean_context` 不注入，只续跑 2 轮。两者都要求父值正确。
完整 token×6 属性起跑状态保存于原始 records；fault/clean 的完整
状态分别有 1 个和 6 个 token 位置不同（A′/B）。

| 模型/起点 | target 初始 | unchanged | correct_write | empty | +1 | +3 | +6 | +11 | +17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A′ fault | 2 | 2 | 2 | 2 | 3 | 5 | 8 | 13 | 19 |
| A′ clean | 2 | 2 | 2 | 2 | 3 | 5 | 8 | 13 | 19 |
| B fault | 8 | 8 | 2 | 2 | 2 | 2 | 8 | 2 | 2 |
| B clean | 2 | 2 | 2 | 2 | 2 | 2 | 8 | 2 | 2 |

表中为第16轮/末两轮 target VALUE；A′ 的 22 个错值全部保持为
写入值，B 的 21/22 个错值回到真值 2，只有 `wrong_plus_6` 保持为 8。
B 的 fault 起点 target 已经是 8，所以 `wrong_plus_6` 是当前值的
无操作分支；clean 起点写 8 后同样保持 8。两个起点所有分支的
父节点末轮仍为 `10/5/10`。anchor 的完整目标/父节点逐轮轨迹见
`b-multiparent-anchor-trajectories.csv`，不把该单图结果并入主表。

## 6. 我认为

以下是我的看法，和上面的计数分开：

1. 我认为本批最直接的行为差异仍是：A′ 的四个错值分支在目标
   首轮和末两轮都为 0/192，B 为 192/192；这一差异在父数 1/2/3
   和两个时点上都出现，但这是行为计数，不能解释为内部机制。
2. 我认为 B 的 `wrong_plus_11`（以及枚举集中的 `wrong_plus_17`）
   值得单独复核：target 已经首轮正确，仍有 35/795 个下游观察在
   16 轮末端不正确，且集中在 5 个 target；不能把 target 恢复
   直接写成下游恢复，也不能把这 35 个观察当作独立节点。
3. 我认为 A′ 的 empty 和 B 的 empty/1/3/7 分支显示“末两轮正确”
   会掩盖前期下游临时损伤，尤其 A′ just_solved empty 到第8轮才
   归零；报告必须同时保留逐轮表。
4. 我认为本批没有看到目标 VALUE 结果随父数改变；但三个父数
   目标来自同一批图，且每图每层只有一个 target，不能据此说
   父数对行为没有影响。
5. 我认为 anchor 的 B `wrong_plus_6` 需要按“写入值等于当前值”
   单独标注；它同时出现在 fault 的无操作分支和 clean 的持久
   分支里，不能当作普通错值分支计数。

## 7. 产物与边界

- 原始记录：`tmp/analysis/b-multiparent-20261115/main_A/records.jsonl`、
  `main_B/records.jsonl`（各约 31 MB，含 16 轮全部节点四属性）、
  `selection.jsonl`、`meta.json`。
- 完整汇总（不全部入 git）：
  `tmp/analysis/b-multiparent-20261115/summary_all/`、
  `summary_matched_timings/`；其中 `node_curves.csv` 12 MB、
  `endpoint_cases.csv` 11 MB、`target_cases.csv` 2.2 MB。
- 仓库内小表：
  - `docs/records/2026-11-15/b-multiparent-coverage.csv`（772 条 case）
  - `b-multiparent-selection-coverage.csv`（64 图×3 父数）
  - `b-multiparent-target-summary.csv`
  - `b-multiparent-endpoint-summary.csv`
  - `b-multiparent-wrong-value-summary.csv`（含每个错值的 target/下游）
  - `b-multiparent-round-summary.csv`（1/2/4/8/16 轮）
  - `b-multiparent-cross-timing.csv` / `-cross-timing-summary.csv`
  - `b-multiparent-anchor-trajectories.csv`（target+父节点，17 轮）
- 汇总修正：`--require_timings` 现在按 model_a/model_b 分别要求
  指定时点，再取交集；新增回归测试，repair-study 12 项 CPU 测试
  通过。该修正只影响匹配汇总，不改变主运行记录或选点/时点。
- 边界：本批只覆盖固定 64 张 N32 探索图、单次 VALUE 干预、16 轮
  自由续算；全错值枚举只覆盖 `graph32–39`；anchor 只有 graph142
  一张图。没有 C+、M3/M4、N256、局部自洽或新训练授权。
  “末两轮正确”是观察窗口内的端点，不是永久稳定或持续恢复。
