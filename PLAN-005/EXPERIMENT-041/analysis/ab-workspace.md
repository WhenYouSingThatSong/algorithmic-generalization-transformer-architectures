# AB 共同任务：重新赋值、固定空间长程序与固定边界诊断

- 日期：2026-11-19。
- 状态：**固定开发任务；不训练、不改结构；AB lane 完成，C lane 由 C 执行，尚未合并。**
- 运行：A `A_s42`、B `B_s42`；同一份 `tasks.json`/`models.json`；GPU7 串行。
- Prepare SHA：`87f7a2a4bcc275ffa196e831eba6a154819877b4`；队列实际记录 HEAD：`3c360ece3e6fe41dcf9f4ba07c977f489efda1a7`。后者只改 C 的文档与 `tools/state_reuse/export_c_report_tables.py`；`tools/abc_workspace` 与 `experiments` 的源码 hash 未变，AB 主批的 plan 也与 smoke 相同。报告保留实际 SHA，不替换为 prepare SHA。

## 1. 清单、覆盖和执行

- `tasks.json`：`b13f87df400c4d2596dd43284b6f906b3ae5eda6a2ae44441404cb05358690d8`；80 个新生成开发任务，5 形状 ×16，seed 2026111903。
- `models.json`：`73e99dcbc603f419962e93bce46021b84ed1d9737936ec325dd9ac372d1bfd19`；A/B 使用 s42 主权重；五个登记权重的 hash 均复核。
- assignment：每形状 16 图，共 80；每模型 1,940 条记录（含固定前两图/形状的 2 个诊断阶段）。4 次更新。
- program：每形状前 8 图，共 40；每模型 2,600 条记录；20 窗口，32 物理变量位置/256 padded tokens，末尾累计 526 逻辑节点、520 执行/结构祖先操作（不是不可简化的最小计算量）。诊断固定在第 1/4/10/20 窗口，且只用前两图/形状。
- smoke 5 形状各 1 图、program 2 窗口，已独立检查；主批没有按成功补图、换权重或改预算。所有程序分支记录均为 `complete`，本次没有自然非数值提交。

| run | 记录 | forwards | 秒 |
|---|---:|---:|---:|
| assignment-A | 1,940 | 33,936 | 141.0 |
| assignment-B | 1,940 | 33,936 | 141.1 |
| program-A | 2,600 | 43,680 | 196.5 |
| program-B | 2,600 | 43,680 | 191.0 |

执行验收、独立真值、起跑继承、跨窗口实际提交、诊断分叉和汇总重建见 [AB main checks](ab-workspace/ab-main-checks.md)。复核脚本对 9,080 条记录报告 0 处不一致。

## 2. assignment：合法更新与回到原题

表中为 **最终轮 macro**（先节点/任务内比例，再形状等权）；`terminal` 随形状为 w=2、4 或6个输出槽，`nodes` 是32个节点。末两轮 macro 与最终轮在本批相同，完整分列见 `ab-summary.csv` 的 `*_final_*` 与 `*_macro` 列。

| model | stage | carry terminal/nodes | reset_results terminal/nodes | scratch terminal/nodes |
|---|---:|---:|---:|---:|
| A | 1 (+1) | 0.220 / 0.434 | 1.000 / 1.000 | 1.000 / 1.000 |
| A | 2 (+3) | 0.134 / 0.264 | 1.000 / 1.000 | 1.000 / 1.000 |
| A | 3 (+7/+11) | 0.138 / 0.266 | 1.000 / 1.000 | 1.000 / 1.000 |
| A | 4 (回到原题) | 1.000 / 0.989 | 1.000 / 1.000 | 1.000 / 1.000 |
| B | 1 (+1) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| B | 2 (+3) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| B | 3 (+7/+11) | 0.988 / 0.994 | 1.000 / 1.000 | 1.000 / 1.000 |
| B | 4 (回到原题) | 0.997 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

- 原题对照：A/B 的 `old_clean` 与同值写入均 1.000；A 的 `old_corrupt` 终轮 terminal 1.000、node 0.969，B 为 1.000/1.000。该污染只是目标结果错误，不等于同剂量更新扰动。
- 表中 `reset_results` 只清空受影响结果行的 VALUE；`scratch` 从当前题面的全 EMPTY 结果开始；两者都不是 `carry` 的等价对照。B 的 `carry` 在 stage 3/4 有少量失败，A 的 `carry` 在 stage 1–3 失败较多，但 stage 4 的 terminal 恢复不代表 32 个节点全对（A node 0.989）。这些是解码计数，不定位内部原因。

## 3. program：20 窗口的实际提交边界

- 每窗口使用上一窗口该分支的**实际数值提交**作为前 w 个输入；其余输入是固定外生常数。没有 gold/reference 回填。实际提交、独立原程序真值和 actual-input conditional 真值分开记录。
- 终窗口（stage 20）macro：

| model | branch | terminal final | node final | conditional node |
|---|---|---:|---:|---:|
| A | carry | 0.090 | 0.077 | 0.084 |
| A | reset_results | 1.000 | 1.000 | 1.000 |
| A | scratch | 1.000 | 1.000 | 1.000 |
| B | carry | 0.988 | 0.978 | 1.000 |
| B | reset_results | 1.000 | 1.000 | 1.000 |
| B | scratch | 1.000 | 1.000 | 1.000 |

- 逐图最终成功（`ab-programs.csv`）：A carry 1/40、reset_results 40/40、scratch 40/40；B carry 39/40、reset_results 40/40、scratch 40/40。A carry 的 39 个失败图首次最终失败在第 2 窗口；B 的 1 个失败图首次在第 11 窗口。没有任何 `invalid_commit`。
- 40 个程序均达到 526 累计逻辑节点；输出祖先算术操作数为 520。这里报告的是参考程序的累计计算量，不表示模型一次处理 526 节点，也不表示 526 个节点均为必要历史。`at_least_512_terminal_success` 在 A carry 为 1/40、B carry 39/40，reset/scratch 均为 40/40；该阈值行只是阶段轨迹读数。
- `carry` 的失败会由实际提交传播到下一窗口；`reset_results`/`scratch` 每窗口重新开始，因此不能用它们的高分直接推断模型具有稳定的跨窗口更新能力。后者还改变了状态清空语义。

## 4. 固定诊断分叉

- 同一 carry 状态、同一额外 D+8 预算，5 个分叉；成功与失败起点都保留。`probe_continue` 从主轨迹末态继续，另外四路分别刷新题面、重置结果、同时刷新/重置、同实际输入从空白重解。
- A：`probe_refresh_task` 的修回/改坏为 0/0；`probe_reset_results`、`probe_refresh_and_reset`、`probe_scratch_same_input` 各为 assignment 188/0、program 228/25（节点-事件求和；两模式合计 60 个固定 probe，不能把跨任务总数当局部修复率）。B：五种分叉均为 0/0，因为其 carry 在这些固定起点大多已经正确。
- 这些是固定起点的配对交集，不是总体修复率；它们只说明该批中被选中状态下“清空/重解”与“继续”的差异，不能单独证明检测、定位或选择性清空机制。完整逐 probe 行见 `ab-probe-pairs.csv`。

## 5. 证据与限制

2026-11-19审阅补充：w4_f1内部节点为单父复制，保留为控制，不等同于多操作数加法难度。520为执行/结构祖先节点数，不代表不可简化的最小计算量。B唯一失败程序在窗口11首次提交错误，窗口12–20给定实际输入的conditional终端均正确，但原程序仍错。A probe总416/25分为assignment188/0、program228/25，不能把跨任务总数直接当局部修复率。详见[审阅与CPU补交](../../../ai-collab/handoff/2026-11-19-ab-workspace-review.md)。

- 小表：`ab-summary.csv`（1,460 组）、`ab-coverage.csv`、`ab-programs.csv`（240 条轨迹）、`ab-probe-pairs.csv`（480 行）、`ab-events.csv`、`ab-examples.jsonl`、`ab-summary-meta.json`。示例按程序成功/失败类别固定选取，不是流行率样本。
- 运行验收：`ab-main-checks.md`；完整 records、snapshot、日志和回执仍在 `tmp/analysis/abc-workspace-20261119/AB-main/`。
- 这是单 seed/模型的固定开发集结果；形状宽度、fan-in 并不是完全正交因果设计。长程序使用显式离散数值提交，C 也是混合接口，不是纯连续潜空间；不声称原模型原生处理 526 节点整图。
- C lane 未在本报告合并；C 的主结果、接口检查和三 seed 需要由 C 执行并在共同汇总中单独列出。不能把各 lane 自己的 common 分母拼成 ABC common。
- 尚未区分：A carry 的早期失败是题面写入、旧答案保留、下游传播还是读出/寻址；reset/scratch 的高分也可能来自清空范围本身。当前只报告可复核的行为边界，不作统一机制结论。

## 6. 独立复核：reference 程序与 actual-input conditional

- 复核代码已纳入 `tools/abc_workspace/checks/verify_ab_records.py`；它只读 manifest/records，不加载模型、不跑 GPU。主批命令：
  `python tools/abc_workspace/checks/verify_ab_records.py --bundle tmp/analysis/abc-workspace-20261119/AB-main --report docs/records/2026-11-19/ab-workspace/ab-verifier.json`
- 结果：`ab-verifier.json`，0 errors；9,080 条记录、290,560 个节点真值；4,800 条带 reference 程序的记录；5,400 条带 conditional metrics 的记录；600 个 probe；1,460 个 summary 组、240 条 program 轨迹、480 行 probe 交并。
- 两类真值在代码里分开：
  1. **reference 程序真值**：对每条 program 记录，`truth4` 的节点值必须等于 `protocol.reference_program(case,20)[stage-1]['truth']`，`reference` 字段必须与独立重算的参考程序一致；原程序指标用这份真值重算。
  2. **actual-input conditional 真值**：把该记录的实际 `boundary_inputs`（模型实际提交 + 固定外生常数）送进 `protocol.evaluate`，得到该窗口的条件真值；`conditional_metrics` 用这份真值单独重算。
- 两者不要求相等：B 的 `w4_f3:007` 在窗口 11 首次提交错误，窗口 12–20 的 conditional 终端可以全对，而原程序终端仍错；A 的 `w2_f2:000` 窗口 2 是旧值残留案例。复核将“给定实际输入后的局部求解”和“原程序累计提交是否正确”分别计数，不把前者替代后者。assignment 的 probe 使用规定输入，因此其 conditional 与 full truth 相同，代码仍走单独字段检查。
- 报告 `ab-main-checks.md` 中的 summary/record 计数与本 JSON 一致；JSON 不含 raw records，只记录小结果和来源 hash。

## 7. 文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `ab-summary.csv` | `7fabfd984af500705c09fe159cff477f086b074f785f53f86df631531c3b55f2` |
| `ab-coverage.csv` | `166f3c653fd48e9ec72856fcf48709641e701669d941b1b18e45df42c09adeb8` |
| `ab-programs.csv` | `811370aafb827bda573914fde93f7c3d2a92a768acec654628906016cae4646b` |
| `ab-probe-pairs.csv` | `1d59d164130037ef334040ca87a2c3dc2316e558ad4197c79b54ca2d6e70b951` |
| `ab-summary-meta.json` | `9967eb0cfa4f5e3d7bbcfb87f6b03d54844cb7888254076daeeae943ecabf3ad` |
| `ab-main-checks.md` | `374f55f1d0925fa4a30aea1c731332ea853656e23cf600d978bca18a0bc3f073` |
| `ab-verifier.json` | `ad3ee721510dbe2eba00360cc0ae443fa9c00879ef617bf35b26de329f925e78` |
| `tools/abc_workspace/checks/verify_ab_records.py` | `29a44655bfd640d11e7aea354722553df522074290df26b48cef9a181729c788` |

`ab-main-checks.md` 中的 summary hash 指 tmp 原始文件；本目录的 CSV 副本已统一为 LF 换行，因此上表 hash 与 tmp 原文件不同，但内容字段一致。
