# C 状态替换功能探针（main）：十路结果、恢复分母与边界

日期：2026-11-19。状态：**C lane main 已按执行包完成；只做已授权 C 评估，未训练、未扩图/种子、未改真实题面、未做层/头干预**。  
代码：`reproduction@6ef63e3`（`plan.json` 的 `base_commit`；运行期间未改执行源码）。GPU：2。  
范围：四个 C 版本各 32 张固定候选图、H16、3 时点×4 扰动来源×3 lag；每版本 1,152 条 records，共 4,608 条。四版本同源 seed2026，**不是四个独立种子，也不是独立最终测试**。

## 0. 先给结论（按本批固定网格）

1. **持续、可归因于扰动的 q VALUE 错误只出现在 `noise2` 条件**；`natural`/`same` 在 early/hold 的末两轮没有 q VALUE 损伤；`semantic` 有首轮少量错误，但末两轮全部恢复。`noise2` 的实际 L2 约为 `semantic` 的 2 倍，二者差异不能直接归因于“噪声 vs 语义”，存在剂量混杂。
2. 在故障接收状态下，**q 自己的子操作 keep/pre/solved 对 q VALUE 没有区别**；`observed_pre` 与 `observed_keep` 的 q VALUE 计数逐版本相同。把接收状态改为 clean 后，q VALUE 在三个子操作下都正确（除 12 条 late clean-path 记录）。恢复由**接收状态**决定，不是由 pre donor 决定。
3. `restore_u` 相对 `observed_pre` 只多修回 **10 条** q VALUE，且全部在 lag=1；在 lag=2/4 为 0。它同时把 **5 条**原本正确的 q VALUE 改错，净效应很小。
4. 所有 q VALUE 损伤中，q 在本轮的**解码输入 VALUE 错误**占绝大多数；`input_qualified` 已保证 u VALUE 正确。按 q 输入读数分层，q 输入正确的损伤为 initial 11、hold 0、noise 1、semantic 0；initial 的 11 条全部是同一图（g16）的 late clean-path 漂移（clean_keep 也错），noise 的 1 条是真实的单条 lag=1 记录。**没有可重复的“u 与 q 输入读数都正确、q VALUE 仍错”扰动证据**。
5. 自动结果包的 `late/semantic/lag2` 表（keep 四属性 3、pre 29、clean_pre 31）**主要不是 VALUE 恢复**：该条件 VALUE 为 keep 29、pre 29、clean_pre 31（all_recorded）；initial 在该条件只有 **12/32** 条 `input_qualified`（clean local 四属性已漂移）。四属性 3→29 主要是 VARIABLE/操作字段恢复；不能当作“q 算术恢复”的主表。

## 1. 执行、覆盖、预算

C smoke 的 6 个任务先完成并通过执行检查（见 `c-smoke-checks.md`）；main 随后串行执行：

```bash
python tools/state_reuse/batch.py prepare --lane C --phase main \
  --smoke-bundle tmp/analysis/state-reuse-exec-20261119/C-smoke \
  --out-dir tmp/analysis/state-reuse-exec-20261119/C-main
python tools/state_reuse/batch.py run \
  --bundle tmp/analysis/state-reuse-exec-20261119/C-main --gpu 2
```

| 项目 | initial | hold | noise | semantic |
|---|---:|---:|---:|---:|
| records | 1,152 | 1,152 | 1,152 | 1,152 |
| coverage selected | 32/32 | 32/32 | 32/32 | 32/32 |
| snapshot records | 72 | 72 | 72 | 72 |
| sample forwards | 188,397 | 188,397 | 188,397 | 188,397 |
| token forwards | 48,229,632 | 48,229,632 | 48,229,632 | 48,229,632 |
| 任务耗时 | 110.7 s | 106.8 s | 103.3 s | 109.5 s |

汇总 `C-summary` 68.4 s，结果包 `C-delivery` 14.4 s。main 6 个任务合计约 513 s。smoke 每版本 H4、72 条 records、3,135 sample forwards / 802,560 token forwards。

- 来源 / 代码 / 权重：
  - data `676aabd2…`，plan `1971a635…`；完整 hash 见 `c-audit.json`。
  - initial `40babb20…`、hold `24f6b071…`、noise `cdf398a7…`、semantic `559fed2b…`。
  - `run_c_function.py` `7dd2de1c…`、`core.py` `ea400f81…`、`summarize.py` `d9830ea0…`、`deliver.py` `e4f1f850…`。
- 每个 condition 固定 32 图；没有候选缺失、没有零记录、没有因结果补样。四个版本每个 condition 都是 32 条 records；`common_recorded=1,152`。

## 2. 读法、口径与分母

十路 = 3 个接收状态 × 3 个子操作 + `sham`：

- 接收状态：`observed`（故障状态）、`restore_u`（故障状态但 u 行换成 clean）、`clean`（整状态换成同一绝对轮次的 clean 状态）。
- 子操作：`keep`（保留 q 行）、`pre`（q 行换成 clean 轨迹中 q 在 `depth(q)` 的 pre 向量，该向量是 EMPTY、身份字段正确）、`solved`（q 行换成 clean 轨迹中下一轮 q 的向量）。
- `sham` 与 `observed_keep` 在本批 4,608/4,608 条 records 上逐元素相同。

所有比例先按图内节点计数，再按图等权；本报告表格若写“记录数”，是图×condition 的记录，不是独立图数。**同一图在 lag=1/2/4 是重复测量**；多处给出 unique graph 数，不能把 3 个 lag 的记录相加当独立样本。

### 2.1 两个输出口径

- **VALUE**：q 输出第 4 字段（实际算术值）。本报告的主口径，和既有 C 报告一致。
- **first4**：q 输出四字段全部正确。它额外要求 VARIABLE/OPERATION 字段正确，因此会检测到长轮次身份字段漂移；不能把它等同于 VALUE 恢复。

`clean_pre_last_two_correct` 按 first4 统计；`c-arm-summary.csv` 同时给 VALUE 与 first4。

### 2.2 资格分母

`input_qualified` = 该记录的 u VALUE 正确、q 的其余父 VALUE 正确、clean local 四属性正确、donor pre 为 EMPTY 且身份正确、donor solved 四属性正确。**它不要求 q 自己本轮输入读数正确**；q 输入读数另按 `context.q_value_correct` 分层。

| 版本 | records | input_qualified | common_input_qualified | clean_local first4 | q 输入 VALUE 正确 | donor 三项 |
|---|---:|---:|---:|---:|---:|---:|
| C（原始） | 1,152 | 881 | 870 | 892 | 923 | 1,152/1,152/1,152 |
| C（保持后训练） | 1,152 | 1,138 | 870 | 1,152 | 965 | 1,152/1,152/1,152 |
| C（噪声后训练） | 1,152 | 1,131 | 870 | 1,152 | 969 | 1,152/1,152/1,152 |
| C（语义后训练） | 1,152 | 1,127 | 870 | 1,152 | 992 | 1,152/1,152/1,152 |

- donor 三项依次为 `donor_pre_empty`、`donor_pre_identity_correct`、`donor_solved_first4_correct`；全部 1,152/1,152。
- `donor_pre_round=depth(q)`、`donor_solved_round=pre+1`；所有记录满足 `max(pre,solved) ≤ absolute_round`（最小差 1），没有未来 donor。
- **initial 的 late 条件资格骤降**：`clean_local_first4_correct` 只有 4–15/32；同批记录的 clean 轨迹在 H16 已漂移（例 C），q 的故障态 VALUE 在 late semantic/noise2 也下降（如 late/semantic/lag1 为 6/32）。因此 late initial 同时有 clean 资格和探针后状态两类问题，不能作为干净的机制表。
- `common_input_qualified` 合计 870/1,152；按条件最低为 4（late/lag4），late/semantic/lag2 为 12。**跨版本 late 比较只有 4–15 图，不能与 early/hold 的 29–32 图混用。**
- 自动结果包的 `key-results.csv` 用 `all_recorded`（每 condition 32 图）和 first4；本报告主表用 `input_qualified` 与 VALUE，两者必须并列阅读。

## 3. 十路总表：接收状态决定 q VALUE

下表为全部 condition 合并后的 q `last_two` 正确数（VALUE / first4），`input_qualified` 队列；每行 n 为该版本资格记录数。合并只是十路总览，**不用于跨 condition 推断**。

| 版本 | 接收状态 | keep | pre | solved | n |
|---|---|---|---:|---:|---:|---:|
| C（原始） | observed | 817 / 468 | 817 / 815 | 817 / 728 | 881 |
| C（原始） | restore_u | 821 / 469 | 821 / 819 | 821 / 733 | 881 |
| C（原始） | clean | 869 / 492 | 869 / 869 | 869 / 777 | 881 |
| C（保持后训练） | observed | 1114 / 1100 | 1114 / 1114 | 1114 / 1114 | 1138 |
| C（保持后训练） | restore_u | 1115 / 1101 | 1115 / 1115 | 1115 / 1115 | 1138 |
| C（保持后训练） | clean | 1138 / 1122 | 1138 / 1138 | 1138 / 1138 | 1138 |
| C（噪声后训练） | observed | 1104 / 1074 | 1104 / 1104 | 1104 / 1104 | 1131 |
| C（噪声后训练） | restore_u | 1104 / 1074 | 1104 / 1104 | 1104 / 1104 | 1131 |
| C（噪声后训练） | clean | 1131 / 1100 | 1131 / 1131 | 1131 / 1131 | 1131 |
| C（语义后训练） | observed | 1106 / 1091 | 1106 / 1106 | 1106 / 1106 | 1127 |
| C（语义后训练） | restore_u | 1106 / 1091 | 1106 / 1106 | 1106 / 1106 | 1127 |
| C（语义后训练） | clean | 1127 / 1112 | 1127 / 1127 | 1127 / 1127 | 1127 |

读法：

- **q VALUE**：observed 下 keep/pre/solved 三列完全相等（每版本）；restore_u 只多 4/3/2/1 条；clean 下三列都等于该版本资格数（initial 869，其余 1,138/1,131/1,127）。`sham` 与 observed/keep 相同。
- **q first4**：observed 下 pre/solved 明显高于 keep（initial 468→815/728；其余版本 1074–1100→1104–1114），说明 q 自己的历史向量可以修正身份字段；但 VALUE 不变。
- clean 下 `pre` 的 first4 最好（initial 869 vs clean_keep 492；其余版本全部 1,138/1,131/1,127），`solved` 反而较低（initial 777）。这是 H16 末两轮的格式/身份现象，不是 VALUE 计算结论。
- `clean_pre` 的 VALUE 与 `clean_keep` 在 4,608/4,608 条记录上相同；clean receiver 无需 pre donor 就已恢复所有可归因的 q VALUE 错误。

## 4. 条件级关键表（q LAST_TWO，VALUE / first4）

下表的“n”是 `input_qualified` 记录数；late/semantic initial 的 all_recorded 32 图与本表的 12 图差异见 §2.2。

### 4.1 early/noise2/lag2

| 版本 | n | keep | pre | restore_u_pre | clean_pre | clean_keep |
|---|---:|---|---|---|---|---|
| C（原始） | 32 | 22 / 19 | 22 / 22 | 22 / 22 | 32 / 32 | 32 / 25 |
| C（保持后训练） | 32 | 26 / 26 | 26 / 26 | 25 / 25 | 32 / 32 | 32 / 32 |
| C（噪声后训练） | 32 | 25 / 25 | 25 / 25 | 24 / 24 | 32 / 32 | 32 / 32 |
| C（语义后训练） | 31 | 25 / 25 | 25 / 25 | 25 / 25 | 31 / 31 | 31 / 31 |

### 4.2 hold/noise2/lag2

| 版本 | n | keep | pre | restore_u_pre | clean_pre | clean_keep |
|---|---:|---|---|---|---|---|
| C（原始） | 32 | 28 / 15 | 28 / 28 | 28 / 28 | 32 / 32 | 32 / 18 |
| C（保持后训练） | 32 | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 |
| C（噪声后训练） | 32 | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 |
| C（语义后训练） | 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 |

### 4.3 hold/semantic/lag2 与 late/semantic/lag2

| 版本 | 条件 | n | keep | pre | restore_u_pre | clean_pre | clean_keep |
|---|---|---:|---|---|---|---|---|
| C（原始） | hold/semantic | 31 | 31 / 18 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 18 |
| C（保持后训练） | hold/semantic | 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 |
| C（噪声后训练） | hold/semantic | 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 |
| C（语义后训练） | hold/semantic | 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 31 |
| C（原始） | late/semantic | **12** | 11 / 2 | 11 / 11 | 11 / 11 | 11 / 11 | 11 / 1 |
| C（保持后训练） | late/semantic | 31 | 31 / 30 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 30 |
| C（噪声后训练） | late/semantic | 31 | 31 / 29 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 28 |
| C（语义后训练） | late/semantic | 31 | 31 / 30 | 31 / 31 | 31 / 31 | 31 / 31 | 31 / 30 |

- `semantic` 在 hold/late 的末两轮没有**扰动可归因的** q VALUE 损伤；只有首轮（post round 1）少量错误，之后恢复。initial 的 `hold/semantic` 在 keep 下 first4 18/31 是身份字段漂移；VALUE 31/31。
- `late/semantic` 的 all_recorded initial 为 keep VALUE 29、pre 29、clean_pre 31、clean_keep 31；first4 为 3/29/31/2。**VALUE 的 29→29 说明 pre 没有恢复算术；first4 的 3→29 是格式字段。** initial 的 12 条 input_qualified 中，唯一 VALUE 错误是 g16 late clean-path（clean_keep 也错），不是扰动可归因。
- 完整每条件十路表见 `c-q-tenway.csv`；每条件资格/恢复见 `c-conditions.csv`。

## 5. 损伤、恢复与分母

### 5.1 版本级

| 版本 | 资格 | 损伤记录 / unique图 | clean-path 记录/图 | 可归因扰动记录/图 | q输入对/q输入错 | pre 修回 | restore_u 修回 | clean_pre 修回 | solved 修回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C（原始） | 881 | 64 / 15 | 12 / 1 | 52 / 15 | 11 / 53 | 0 | 4 | 52 | 0 |
| C（保持后训练） | 1,138 | 24 / 8 | 0 / 0 | 24 / 8 | 0 / 24 | 0 | 3 | 24 | 0 |
| C（噪声后训练） | 1,131 | 27 / 8 | 0 / 0 | 27 / 8 | 1 / 26 | 0 | 2 | 27 | 0 |
| C（语义后训练） | 1,127 | 21 / 7 | 0 / 0 | 21 / 7 | 0 / 21 | 0 | 1 | 21 | 0 |

- “损伤”= `observed_keep` 的 q VALUE 在末两轮错误；资格内统计。
- “clean-path”= 同时 `clean_keep` VALUE 错误；initial 的 12 条全部是 g16 late 的 4 mode×3 lag，**是 1 张图的重复测量，不是 12 张图**。该图的 clean_keep 在 post round 1 时 q VALUE 正确（observed_keep 在 late/semantic/lag1 已错），到 H16 才漂移（见 §7 例 C）。
- “可归因扰动”= keep 错、clean_keep 对；pre 修回 0/124，restore_u 修回 10/124（全部 lag=1），clean_pre 修回 124/124，solved 修回 0/124。
- `q输入对/q输入错` 是 `context.q_value_correct` 分层，不是 q 的 hidden；initial 的 11 条 q输入对**全部**是 g16 late clean-path；noise 的 1 条是真实单条记录（g27 late/noise2/lag1，restore_u 与 clean_pre 均修回）。
- 自动结果包的 `examples.jsonl` 固定取 late/semantic/lag2；本报告的固定案例另见 §7，按 VALUE 类别选取，不用于计算发生率。

### 5.2 时点/来源

`noise2` 是唯一产生持续、可归因 q VALUE 损伤的来源：

| 版本 | early/noise2 末两轮 | early/noise2 首轮→末两轮 | hold/noise2 末两轮 | hold/noise2 首轮→末两轮 | early unique 图 | hold unique 图 |
|---|---:|---:|---:|---:|---:|---:|
| C（原始） | 30/96 | 34→30 | 12/96 | 16→12 | 10 | 4 |
| C（保持后训练） | 18/96 | 24→18 | 0/96 | 1→0 | 6 | 0 |
| C（噪声后训练） | 21/96 | 26→21 | 0/96 | 1→0 | 7 | 0 |
| C（语义后训练） | 18/94 | 22→18 | 0/94 | 1→0 | 6 | 0 |

- 分母是 3 个 lag 的记录合计（通常 96；semantic 的 early/hold 为 94）；同一图在 3 个 lag 重复，**unique 图只有 10/6/7/6（early）和 4/0/0/0（hold）**，不能把 30/18/21/18 当独立图数。
- early/noise2 的 10/6/7/6 张图在三个 lag 的损伤集合完全相同；initial 的集合包含 6、16、27、29 等，hold/noise/semantic 只覆盖其中一部分。没有做跨版本图筛选。
- hold/noise2：initial 4 张图（6、14、17、25）在三个 lag 重复；三个后训练版本为 0/96、0/96、0/94。**这是本批最清楚的版本差异之一，但三个后训练版本同源、同起点，不能分离具体训练配方。**
- `semantic`：early/hold 首轮有 7–15 条/约 90 条资格记录的 q VALUE 错误（跨 3 lag 汇总），到末两轮**全部恢复**；`same` 与 `natural` 在 early/hold 末两轮为 0。实际 L2：same 约 0.4–0.8，semantic 约 5–6.5，noise2 约 10–13。**semantic 与 noise2 的差异和剂量同向，不能直接说“语义扰动不持久、噪声才持久”。**
- late：initial 的资格与 clean 轨迹都在漂移（§2.2、§7C），late/noise2 的 13 条损伤中 3 条是 clean-path；其余 10 条可归因但只有 4–15 图资格，故 late initial 不作为主结论。三个后训练版本 late/noise2 只有 1–2 条/版本，也不据此下机制结论。

### 5.3 restore_u 不是稳定的“额外改善”

- 10 条修回：initial 4、hold 3、noise 2、semantic 1；全部在 lag=1，且全部是 `noise2`。lag=2/4 为 0。
- 5 条反向：把原本正确的 q VALUE 改错；3 个 unique 的 q 节点（hold g14、noise g14、semantic g18），全部在 early/noise2（lag1 或 lag2）。版本内净变化：initial +4、hold +1、noise 0、semantic 0。
- 因此“restore_u 有额外改善”只在**首轮**、**少数图**成立；从第二轮起，单独恢复 u 的向量既不能修回 q VALUE，也不是无害的。**不能把 restore_u 当作通用修复。**

## 6. 正确节点是否受损

按 `input_qualified` 记录、VALUE 末两轮、参考分支正确而目标分支错误的直接交集统计。`c-unique-damage.csv` 给 unique (arm, graph, node)；`c-node-region-counts.csv` 给每个 region 的计数（region 有重叠，不能相加：`other_children` 也在 `descendants_excl_q` 内）。

| 对比 | VALUE unique 节点 / 重复测量 | first4 unique 节点 / 重复测量 | 分布 |
|---|---:|---:|---|
| clean_keep → clean_pre | 1 / 8 | 3 / 4 | VALUE：initial g8/n136，late 4 mode×lag1/2；first4：initial g19、g23、semantic g21 |
| observed_keep → observed_pre | 4 / 11 | 7 / 8 | 全部 initial g8 的 4 个后代节点，late |
| observed_keep → restore_u_pre | 9 / 10 | 16 / 20 | q 3 个节点/5 条；其余在 initial g8、g16 的后代/其他子节点 |
| observed_keep → clean_pre | 4 / 17 | 35 / 49 | 全部 initial g8 的 4 个后代节点，late |

- **q VALUE 没有被 clean_pre 损伤**：clean_keep→clean_pre 的 q 列为 0；但 initial late g8 有 1 个后代节点 n136 在 8 个重复条件中 VALUE 被 clean_pre 改错。这不是 8 个独立图/节点。
- `observed_pre` 在故障接收状态下会改错 initial g8 的 4 个后代 VALUE 节点（11 个重复测量），却修回 q 的 first4；**first4 的“恢复”可能以其他正确节点为代价**。
- `restore_u` 的 q 反向 5 条（3 个 q 节点）是本批唯一直接看到“修复操作损伤 q VALUE”的地方；其余 6 个 unique 下游节点在 initial g8/g16。
- `outside`（既非 u 也非 u 的后代）在 VALUE 上从未被任何对比损伤；first4 上有少量（restore_u 2、clean_pre 4 个测量），见 CSV。
- 所有正确**非 q**节点的 VALUE 损伤都集中在 initial 的 late g8/g16；early/hold 与三个后训练版本的 clean_pre 没有 VALUE 正确节点损伤。late initial 的资格只有 4–15 图，不能把这几个节点推广成一般规律。
- first4 的损伤面更宽（clean_pre from observed 35 个 unique 节点），主要是身份字段；报告主结论不基于 first4 的节点损伤。

## 7. 固定案例（不是独立测试）

固定排序：版本 `initial<hold<noise<semantic`；时点 `early<hold<late`；来源 `natural<same<semantic<noise2`；lag `1<2<4`；再按 graph_id。每个类别取第一条满足条件的 `input_qualified` 记录；一条记录可能同时满足多个类别时，按下列操作定义互斥选取。完整字段与分支值见 `c-examples.csv`。

### A. 故障接收状态：q 两个父，keep/pre/restore_u 都错
initial / early / noise2 / lag1 / graph6：u=146，q=160。  
`x146 = x22(10)+x120(7) = 17`；`x160 = x112(2)+x146(17) = 19`。  
q 的真值 VALUE=19；probe 时 u VALUE 正确、q 输入 VALUE 错。post1：keep q 输入 12→输出 12；pre 输入 24→输出 12；restore_u 输入 24→输出 12；clean_pre 输入 24→输出 19；clean_keep 输入 19→输出 19。post16 同样：keep/pre/rupre 输出 12，clean_pre/clean_keep 19。  
说明：q 的解码输入 24 不等于其输出；clean receiver 在相同 q pre 向量下输出正确，故障 receiver 不输出正确。

### B. lag1 的 restore_u 修回：q 单父，但 q 输入读数仍错
initial / early / noise2 / lag1 / graph3：u=138，q=168。  
`x138 = x28(15)+x6(13)+x112(14) = 19`；`x168 = x138(19) = 19`。  
post1：keep q 输入 24→输出 0；pre 24→0；restore_u 24→**19**；clean_pre 24→19；clean_keep 19→19。post16：keep/pre 输出 0；restore_u/clean_pre 19。  
说明：q 的父 u 在解码 VALUE 上已经是 19，但故障接收状态下 q 仍输出 0；把 u 换成 clean 向量在 lag1 可修回。q 的解码输入仍是 24，不能把这个修回解释成“q 读到了 24 或 19”；只能说明该状态替换改变了后续计算。

### C. late clean-path 漂移：不是探针造成
initial / late / natural / lag1 / graph16：u=56，q=120。  
`x56 = x10(16)+x26(12)+x2(14) = 19`；`x120 = x56(19)+x78(18)+x66(11) = 2`。  
post1：keep 与 clean_keep q 输出都是 2（正确）；post16：keep 与 clean_keep q 输出都变成 8；pre/restore_u/clean_pre 也输出 8。末两轮因此全错。  
说明：这是 initial 的 clean 轨迹在 H16 内的自然漂移，不是扰动或 pre 的失败；也是 late initial 资格低、不能作干净机制表的具体原因。该图在 12 个 late 条件中重复出现，仍只是 1 张图。

### D. first4 漂移，不是 VALUE 恢复
initial / early / natural / lag1 / graph3：same q/u as B，VALUE 全对。  
keep 与 clean_keep 的 q 输出在 post16 为 `[0,123,1,19]`，真值为 `[0,75,1,19]`：VALUE 19 正确，VARIABLE 字段从 75 漂到 123；pre/clean_pre 的 q 输出保持 75/19。  
说明：自动包中 late/semantic/lag2 的 first4 3→29 是这一类身份字段恢复，VALUE 在 keep 与 pre 都是 29（all_recorded），不能写成“pre 修复了 q 的算术”。

## 8. 支持的观察与尚未区分的解释

**可以直接说的（限定在本批 32 图、固定网格、同源 seed2026 开发集）：**

- 在本批扰动剂量下，持续 q VALUE 损伤由 `noise2` 产生；`semantic` 只有首轮短暂错误并在末两轮恢复，`same`/`natural` 在 early/hold 末两轮无损伤。实际 L2 不同，不能把差异归因于“语义 vs 噪声”本身。
- 在故障接收状态中，q 自己的 keep/pre/solved 不改变 q VALUE；把接收状态整体换成 clean 才恢复。恢复的 124/124 条可归因记录由 clean receiver 完成，pre donor 对 VALUE 没有增量。
- `restore_u` 只修回 10/124 条，全部 lag=1；同时改错 5 条原本正确的 q VALUE。它不是稳定修复。
- q VALUE 错误几乎都伴随 q 本轮解码输入 VALUE 错误；`u VALUE 正确 + q 输入正确 + q VALUE 错` 只有 1 条真实扰动记录（noise g27/late/noise2/lag1），其余 initial 11 条是 g16 late clean-path。**没有可重复的“输入读数都对、q 算术仍错”证据。**
- `clean_pre` 在 clean 中的 q VALUE 成功率 4,596/4,608（12 条 g16 late clean-path）；first4 也是 4,596/4,608，并修回 692 条 clean_keep 的 first4 失败，未出现 first4 1→0。按本批定义，pre 操作本身可用；但它对 VALUE 的“成功”并不高于 clean_keep。
- 正确节点损伤集中在 initial late g8/g16：`clean_keep→clean_pre` 的 VALUE 只损伤 1 个 unique 后代节点（g8/n136，8 个重复测量）；`restore_u` 额外损伤 3 个 q 节点和 6 个下游节点；outside VALUE 无损伤。early/hold 与后训练版本的 clean_pre 没有 VALUE 正确节点损伤。

**不能从本批区分的：**

- clean receiver 同时替换了所有位置（其他父、其他后代、注意力可见的全局状态），**不能定位是哪一个父节点、注意力成分还是全局 hidden 承载了错误**。需要逐个父节点替换或只替换 q 的输入 V 向量才能进一步分解。
- 故障接收状态下 `observed_pre` 在 initial late g8 修回 first4 却损伤 downstream VALUE，说明 q 的 hidden 既影响自己的四字段，也影响后代；本批不能区分“q 的 hidden 携带下游信息”与“late clean 轨迹自身漂移”。
- `restore_u` 的反向损伤说明 u 的 clean 向量与故障向量在隐藏层面不同；解码 u VALUE 相同不代表 hidden 相同。
- late initial 的 clean 轨迹在 H16 漂移（例 C），所以 late initial 的 q VALUE 末两轮不能当作纯探针结果；本报告不把 late initial 纳入主机制表。
- 四个版本同源 seed2026；三个后训练版本共享起点，不能分离具体训练配方，也不是四个独立种子。32 张验证图是开发集，不是独立最终测试；所有“正确”都是解码字段，不证明隐藏读取或机制。

## 9. 产物、命令与审计

- 原始 records / 隐藏快照 / 全量 CSV 留在 `tmp/analysis/state-reuse-exec-20261119/C-main/`（四版本各 248 MB，含 states；不提交）。
- 小表、结果包、smoke 检查在本目录：
  - `c-arm-summary.csv`、`c-conditions.csv`、`c-q-tenway.csv`、`c-condition-recovery.csv`、`c-node-region-counts.csv`、`c-unique-damage.csv`、`c-examples.csv`、`c-audit.json`；
  - `c-delivery-{key-results.csv,report.md,examples.jsonl,meta.json}`（自动结果包原样；注意其 all_recorded/first4 口径）；
  - `c-smoke-checks.md`（真实 GPU smoke 检查记录；本批 C main 的前置验收）。
- 复导命令（只读 records，不加载权重）：
  ```bash
  python tools/state_reuse/export_c_report_tables.py \
    --bundle tmp/analysis/state-reuse-exec-20261119/C-main \
    --out-dir docs/records/2026-11-19/c-state-reuse
  ```
- `c-audit.json` 记录：commit/plan/base、四版本 meta/records/coverage/权重/data/plan hash、完整 summary meta、各版本分母、`sham=observed_keep` 4,608/4,608、`clean_pre VALUE=clean_keep` 4,608/4,608、clean-path 12 条/1 图、生成文件 SHA256、固定案例规则。导出脚本无结果通过阈值，只记录实际数值。
- `update_retention.csv` 为 0 行：C 连续协议不使用 A/B 的旧值残留字段，不是数据丢失；本报告不引用该表。
- 本批未训练、未扩图/种子、未做层/头干预、未改真实题面、未做持续噪声训练；未改共享源码；未提交原始 weights/records。

## 附：报告数字与表的关系

- 正文的十路总表来自 `c-q-tenway.csv` 按 `cohort=input_qualified` 对全部 condition 求和；每条件表在 `c-conditions.csv`。
- 正文的版本级损伤/恢复表来自 `c-arm-summary.csv`；其中“损伤记录”跨 36 个 condition 重复计数，unique 图列给出非独立图数。
- 正文的节 5.2 时点表来自 `c-condition-recovery.csv` 按 arm/timing/mode 对 3 个 lag 求和；unique 图数来自记录级集合，不是 CSV 行数。
- 正文的节点损伤表来自 `c-unique-damage.csv`（unique arm/graph/node）与 `c-node-region-counts.csv`（region 计数，region 可重叠）。
- 正文的固定案例来自 `c-examples.csv` 与原始 records；方程按 `parents` 与 `truth4` 的模 23 和重建，节点值均为记录中的实际解码值。
