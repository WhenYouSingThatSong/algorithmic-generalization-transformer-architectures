# B 机制实验完整设计（2026-11-14）

状态：设计稿，未执行。供外部审查。代码在 `tools/b_mechanism/*.py`。

当前边界：指标、分母、聚合方式与恢复观察窗口尚未定稿；本文相关选项是待讨论方案，不是已确认的实验验收标准。代码修复与小样本流水线检查不等于正式机制实验已经完成。

## 0. 目标与非目标

要回答三个机制问题：

1. B 在 **u 还没算 / u 刚算对 / 整图已算对** 三个时点，会怎样处理父值错误、已有值和下游？
2. B 是“每轮重算”“检测到父子不一致再重算”，还是“优先保留已有槽位”？错误沿真实依赖走多远？
3. B 的训练腐蚀实际发生在什么状态：`EMPTY→随机值`，还是 `已填值→错误值`？这决定 `4→5` 是训练内行为还是跨时机泛化。

非目标：

- 不把单个 token 恢复率直接当成“纠错机制”结论；
- 不区分 Transformer 层内先后；离散模型一次 forward 输出整图，只能定义同一轮/下一轮；
- 不用 test 选点；机制实验用固定 heldout/匹配深度样本。

## 1. 权重、数据、命名

- **A′ 主对照**：`A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa`，s42，e86。
- **B 主对照**：`B-M6-Nodes32-ADD-L4H16D384-DeBERTa`，s42，`last.ckpt`（e92）。
- **cyhB 复现**：s42 e84 / s1234 e88 / s2025 e74；第二批再做。
- **主数据**：`tmp/analysis/heldout-seed42/ADD/testdata_N32.pt`（1,024 图，N32，depth 3–11）。
- **训练同分布复核**：`tmp/analysis/matched-depth/ADD/testdata_N32.pt`（1,024 图，N32，depth≤6）。
- **旧 N32 测试**不用于 B 机制主表；它属于旧 C0 的旧口径测试。
- 变量名池 128 + N/A = 129；A′/B 的 tokenizer 都是 129。

## 2. 三个时点与状态

记 teacher 快照：
- `S[d]`：u 尚未算出，输入中 u 为 EMPTY；
- `S[d+1]`：u 已算出，输入中 u 为正确值；
- 整图完成起点使用各自 clean rollout 的 `outs[T]`；T 是模型输出轮次下标，不与 teacher 快照 S 的下标混用。

own-state 单独记录：
- `before_u`：u 的 own clean 输出必须为 EMPTY/N/A；r 在改之前必须正确；
- `after_u`：u 的 own clean 输出必须为正确值；r 在改之前必须正确；
- `complete`：当前实现搜索双方所有数值节点 VALUE **首次共同正确**的轮次 T，并冻结该起点；这不保证四字段全对，也不保证之后持续稳定。稳定窗口是否加入仍待讨论。

主比较使用**共同合格样本**：A′ 和 B 都满足对应条件、同一图同一节点。分别报告每个时点/条件的合格 n。若两边 VARIABLE 等字段不同，记录差异，并用共同 teacher 起点复核差异是否影响结果。

## 3. 样本选择

候选在注错前一次性选定并冻结，不按结果重选。

### 3.1 链

主链 `r → u → q1 → q2`：

- r 是 u 的直接父节点；
- u 是内部数值节点；
- q1 是 u 的数值子节点；
- q2 是 q1 的数值子节点；
- 主样本优先：u 单父 r；q1/q2 不直接读 r；
- 多父 u、q 直接读 r、更远下游单独标记；主表只汇总主样本；
- 每个图选少量 u，按深度分层；固定 graph_id、u_pos、r_pos、q_positions。

### 3.2 叶节点

在 token 布局里，叶节点是 `常数 VALUE → = → 变量`：

- 常数位置 `c_pos`：`SYNTAX=VALUE`、`depth=-1`、`VALUE=a`；
- 叶变量位置 `v_pos`：`SYNTAX=VARIABLE`、`depth=0`、`VALUE=a`，`info` 指向 `c_pos`；
- C 格（已填值）：只改 `v_pos.VALUE a→a+delta`，保留 `c_pos` 的常数 a；
- D 格（输入常数）：只改 `c_pos.VALUE a→a+delta`，保留 `v_pos` 和下游旧值；按修改后的题面重算新真值。

### 3.3 最终结果节点与无关节点

- **最终结果节点**：数值 sink，无下游，有数值父节点；优先 depth 最大的 sink。
- **无关节点**：不在目标节点的祖先/后代路径上，depth、注错幅度与目标匹配；用于检查错误是否主要沿真实依赖传播。

### 3.4 共同合格

- 先对 A′ 和 B 做 clean free rollout（从原始 token 状态开始，aux 强制 FALSE）；
- 找到**同一个轮次 T**，双方所有数值节点的 VALUE 都正确；
- 只保留存在共同 T 的图；
- 冻结：graph_id、T、r/u/q1/q2、sink、leaf 的 c_pos/v_pos、无关节点；
- 报告总图数、共同合格图数、各目标类型可选节点数。

## 4. 第一批实验矩阵

第一批只用 A′ s42 + B-M6 s42。

### 4.1 早期 2×2（两条时点）

从 `before_u`（S[d] / own 前一轮）和 `after_u`（S[d+1] / own 当轮）起跑，四条件：

| 条件 | u | r | 用途 |
|---|---|---|---|
| clean | 原值/EMPTY | 原值 | 同题对照 |
| U only | a→a+1（4→5） | 原值 | 已有值被改坏 |
| R only | 原值/EMPTY | r_true→r_wrong（默认 +3） | 子节点是否跟随错父 |
| U+R | a→a+1 | r_true→r_wrong | 两者冲突时优先谁 |

记录全轮 r/u/q1/q2、全部后代、无关分支；区分 early 的 q `not_yet`、到期 EMPTY、数字错。

### 4.2 晚期同节点 R-only（时点直接对照）

同一图、同一 r/u、同一错父值：

- 早期：after_u，u 在注错瞬间正确，q 可能未算；
- 晚期：complete，u 在注错瞬间仍正确，q1/q2 已经正确；
- r 都改成同一个 r_wrong；
- 比较：
  - u 是否仍被带错；
  - q 是“首次算错”还是“从正确变错”；
  - 回正轮次。

这是时点比较，不是晚期四格的一部分。

### 4.3 晚期同节点 U-only

同一 u，`a→a+1`（优先 4→5），r 正确，q1/q2 已正确：

- 比较早期 U-only 与晚期 U-only；
- 观察 u 自恢复、下游是否被覆盖、覆盖后是否回正；
- 这一格同时就是“中间节点”目标格，不重复跑。

### 4.4 晚期目标类型

| 目标 | 注错 | 真值 | 观察 |
|---|---|---|---|
| sink | 最终结果节点 VALUE +delta | 不变 | 无下游重写/自恢复 |
| internal | u VALUE 4→5 | 不变 | 下游覆盖与恢复（与 U-only 共用） |
| leaf_value (C) | 叶变量 v_pos VALUE a→a+delta，常数不变 | 不变 | 能否从常数恢复 |
| leaf_constant (D1) | 常数 c_pos VALUE a→a+delta，保留旧叶/下游 | 变为新真值 | 能否更新旧答案 |
| leaf_constant (D2) | 从初始状态解新题 | 新真值 | 模型是否本来会做新题 |
| leaf_constant (D3) | 每轮固定新常数，保留旧状态 | 新真值 | 持续新题面是否稳定更新 |
| unrelated | 无关节点 VALUE +delta | 不变 | 错误是否沿真实依赖传播 |

D2 是从原始图/空状态做 clean rollout 解新题，只用于判断“不会从旧答案更新”是否因为“不会做新题”。

### 4.5 第一批记录

每个样本保存：

- pre（注错前）、injection（注错瞬间）、之后每轮；
- 实际输入值 vs 模型输出值（强制注入时分开）；
- r、u、q1、q2、全部后代、无关分支；
- 绝对轮次与相对可解轮次；
- 每节点状态：correct / wrong / EMPTY / not_yet / padded；
- 对 D 格，新真值、旧真值、实际输出。

## 5. 分母与输出指标（候选方案，未定稿）

以下三类集合及汇总方式是候选方案；当前 summary 只实现首轮受损、后续曾经恢复的统计，不能解释为全窗口传播或持续恢复：

1. 从 clean 分支取“本来应该正确”的节点集合 N0；
2. 注错后实际被改坏的 N1；
3. N1 中后来回正的 N2。

报告：

- 覆盖/带错率 `N1/N0`；
- 恢复率 `N2/N1`；
- 到期 EMPTY / 未到期 not_yet 单独计数；
- 传播：记录全部后代，报错误最远边数；只追 q1/q2 时结论限定两级；
- 保持：首次回正后连续正确的轮数；
- 图级聚合：同一图多节点时先按图汇总，避免长图/多节点权重过大。

第一批输出：

1. 父值响应表：R-only 下 u 是否跟随错父、保留、未出值；其中被带错的恢复率。
2. 下游覆盖表：clean 正确的 q 有多少被覆盖；覆盖的恢复率；错误最远边数；保持轮数。
3. 时点对照表：early R-only vs late R-only；early U-only vs late U-only。
4. 晚期目标表：sink / internal / leaf_value / leaf_constant(D1/D2/D3) / unrelated。
5. 合格数表：每个条件、每个 seed 的样本量。

## 6. 第二批控制

### 6.1 持续错父值

- 一组只起跑时改 r；另一组每轮输入前把 r 固定为错值 N 轮，然后放开；
- 记录每轮实际喂入 r 和模型输出 r；
- 看 u 是否持续跟随错父，放开后才回正；
- 若持续错父下 u 仍回正，检查是否有其他可利用信息。

### 6.2 局部自洽错误：DEFERRED / NOT RUN

本控制暂缓，见 `docs/records/2026-11-14/b-local-self-consistent-deferred.md`。
当前构造可能只替换 r/u/q1/q2，留下旁路父值导致实际并不自洽；不得用它的
输出做机制结论。后续需按保留的父值逐个构造受影响子图，或替换完整受影响
子图并明确改名为“子图自洽错误”。第一批、late timing、持续错父值、切换和
腐蚀审计均不依赖它。

### 6.3 正确子节点 + 冲突父值

- 保持 u=4，把 r 改成 5；
- 在 after_u 与 complete 两个时点各做一次；
- 比较“u 刚算完就被覆盖”和“整图完成后是否还会覆盖”。

### 6.4 held-state 父值切换（当前实现）

- 从整图完成状态开始，先把 r 固定为第一个错值 `hold_rounds` 轮；
- `switch_held_r` 要求 `switch_at == hold_rounds + 1 <= horizon`，在固定窗口结束后的下一轮把 r 换成第二个错值一次，然后释放；
- `sustained_r` 是同一时点直接释放的 no-switch 对照；
- 该操作比较的是“释放”与“释放时追加一次干扰”，**不是“稳定错误状态之间切换”**；
  不能沿用旧自洽切换的解释。本阶段不纳入必跑项。

## 7. 训练腐蚀审计

### 7.1 候选导出（关键）

直接导出 `candidates`，即 `last` 策略下所有“本轮新可算”的位置：

- 对每个 candidate 读取腐蚀前输入 VALUE；
- 统计 `EMPTY / N/A / 正确数值 / 其他`；
- 按 depth、factor、是否 leaf/内部节点分组。

注意：`corruption_prob=1` 只保证 batch 进入腐蚀分支；`tokenwise_corruption_prob=0.5` 仍会随机漏掉 candidate。要枚举全部 candidate，必须直接读 mask，不能只靠强制概率。

### 7.2 实际采样

用正常 `corruption_prob=0.05`、`tokenwise=0.5` 跑一批 batch：

- 记录实际选中的位置、腐蚀前值、腐蚀后值、正确 label；
- 统计替换后是错误数字 / 正确数字 / EMPTY / N/A；
- 估计 `EMPTY→5` 与 `4→5` 在训练中的实际覆盖。

### 7.3 代码版本

B/cyh 训练于 2026-11-07/08；比对当前 `randomly_corrupt_values` 与训练时版本（现有指纹记录称与 upstream `fe385b8` 一致），仍以实际张量审计为准。

## 8. 代码组织

现有可复用：
- `tmp/scripts/parent_corruption_2x2.py`：early 2×2、teacher/own、before/after。
- `tmp/scripts/inject_value_discrete_v2.py`：单点注入。

新增：
- `tools/b_mechanism/b_mechanism_common.py`：模型/数据加载、DAG 解析、候选、rollout、记录工具。
- `tools/b_mechanism/b_mechanism_select.py`：A′/B clean rollout 共同合格样本，生成 manifest。
- `tools/b_mechanism/b_mechanism_early.py`：early 2×2 全轮轨迹（teacher/own，before/after）。
- `tools/b_mechanism/b_mechanism_late.py`：late R-only/U-only、sink/internal/leaf value/leaf constant、unrelated、D2/D3。
- `tools/b_mechanism/b_mechanism_controls.py`：sustained wrong parent、switch-from-held-state；局部自洽模式已移除/搁置。
- `tools/b_mechanism/b_corruption_audit.py`：candidate 导出 + 实际采样统计。
- `tools/b_mechanism/b_mechanism_summary.py`：N0/N1/N2、VALUE 与 first-four、EMPTY、最远传播边数、CSV/JSON 汇总。
- `tools/b_mechanism/b_mechanism_README.md`：命令示例。

**代码状态（2026-11-14）**：上述脚本通过 `py_compile`；selection、early own/teacher、late 全条件、sustained/switch、D1/D2/D3 和腐蚀审计已用 4 张 heldout 图/1 个 batch 做 smoke。smoke 只验证流水线，不是实验结果；正式执行前仍需核对 token 位置、manifest 合格数、输出 schema 和 GPU 显存。

### 8.1 2026-11-14 审查修复

- `-I` 入口显式加入脚本目录；
- 腐蚀审计使用 `S[:-1]`，加 seed/trial ID；
- selection 提供 A′/B early 共同资格、单父 primary、强制 q2、扫描/入选分离、
  unrelated 排除可达 q 的节点、数据/权重/代码哈希；
- late 增加同起点 clean、D1 new_truth、D2 完整预算、逐节点 first-four；
- controls 移除局部自洽，switch 从持续错父状态切换并保留 no-switch 对照；
- summary 增加 N0/N1/N2 与 first-four/VALUE 两套统计。

### 8.2 GPT review 修复（2026-11-14，post-v2）

- R1：`early_*` 资格键统一；selection 写 common + `_a`/`_b` + `event_id`，
  runner 缺键或共同键不一致直接报错。
- R3：selection 按单图 `n_loops` 截断 `T` 搜索；runner 校验 `0 <= T < n_loops`；
  batch_size 改变不改变同一图的 T/预算。
- R2：summary 保留 source/run 身份、完整目标键和分组键；A′/B、start/timing、
  不同输入文件不再串行合并；重复 clean 显式检查。
- R7：controls 记录 `condition`/`start`/`timing`，新增同起点 clean，
  切换约束经后续复核修正为 `switch_at == hold_rounds + 1 <= horizon`，防止被 hold 分支覆盖。
- R8：D1/D2/D3 记录 `new_truth` 与 `output_nodes_new`/`input_nodes_new`，
  旧 label 字段标 `truth_ref=old`；后续复核补齐被修改题面常数的新真值，不能只更新图节点。
- summary 输出显式标注 `post_round_1` / `any_later_round` 的统计范围；
  指标定义仍未定稿。

所有脚本支持：
- `--ds_file`、`--n_graphs`、`--batch_size`、`--seed`；
- 输出 JSONL/JSON + 汇总 CSV；
- `OMP_NUM_THREADS=4–8`、`num_workers=0/1`；
- 单卡串行，不并发跑多模型。

## 9. 执行顺序与资源

1. Phase 0（CPU/1 卡）：共同合格样本选择，生成 manifest；统计合格 n。
2. Phase 1（1 卡）：early 2×2 全轮 + late R-only/U-only + late 四格；A′/B 主对照。
3. Phase 2（1 卡）：持续错父、switch-from-held-state、无关节点；局部自洽标记 deferred，不跑。
4. Phase 3（CPU/1 卡）：腐蚀审计；可与 Phase 1 并行。
5. Phase 4：cyhB 三 seed 复现。

单卡串行、线程 4–8；预计第一批每个权重几分钟到十几分钟，具体以 smoke 为准。

## 10. 风险与判读边界

- 同一 forward 内看不到层间先后；只能按轮边界说“同轮/下一轮”；
- 共同 T 可能样本不足；需报告 n，并用 matched-depth 复核；
- 自洽值 `7` 保持无法区分重算与保留；现有 `switch_held_r` 也不是稳定错误状态切换，
  不能仅凭它区分每轮重算与“检测到不一致才重算”；该问题留到后续针对性干预/层内分析；
- `Graph/strict` 不是这里的主指标；机制表以事件级覆盖/恢复为主；
- 不把 `4→5` 的恢复直接解释为训练内纠错，除非腐蚀审计证明训练覆盖了已填值。
