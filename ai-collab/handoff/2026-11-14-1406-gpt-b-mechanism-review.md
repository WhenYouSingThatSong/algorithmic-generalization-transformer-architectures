# B 机制代码 Git 复核

日期：2026-11-14 14:06 CST；审阅者：GPT。

审查基线：`origin/reproduction` 的 `51370142dc4d096f7c32825bc7d8f2bbdeecc979`，B 工具提交为 `ab80594`。本次仅写审查记录，未修改实验实现、配置、共识或服务器任务。

## 结论与验证范围

**上一轮 v2 审查的关键问题仍存在，当前版本尚未通过正式实验验收。** 去除 Windows CRLF/LF 差异后，`tools/b_mechanism/` 中全部 7 个 Python 文件与 `b-mechanism-code-20261114-v2.tar.gz` 逐字一致；README 主要更新了迁移路径。

已执行：远端 fetch、版本确认、7 个脚本的换行归一化内容比较、与本次仓库底层接口的源码核对。未执行：本地模型推理、Python 端到端测试、训练。仓库含 smoke 文字摘要，没有该次 smoke 的原始 JSON/CSV/log，不能把这份复核表述为独立复现了 smoke。

优先级：P1 为正式配对/选样/汇总前需修复的缺陷；P2 为相关实验发布结果前需补齐的实现或口径。下面行号均对应审查基线。

## P1：优先修复

### R1：early 共同合格标记写入与读取不一致

- selection 写 `early_own_before/after`、`early_teacher_before/after`（`tools/b_mechanism/b_mechanism_select.py:156`）。
- runner 拼出 `early_own_before_u/after_u` 等，因为 CLI timing 含 `_u`（`tools/b_mechanism/b_mechanism_early.py:89`）。
- 条件是“键存在且为 False 才跳过”，所以共同不合格标记被忽略，实际仍可能由 A′/B 各自筛选。
- 修复：统一 schema，对缺失标记报错；记录配对事件 ID。验收必须含“A 合格、B 不合格”，断言两边都排除。现有 smoke 中全部合格不能覆盖此缺陷。

### R2：summary 多文件输入会串用 clean，并合并不同模型

- `target_key()` 对 early 仅使用 graph_id、chain_index，没有模型、seed、数据身份、start、timing（`b_mechanism_summary.py:26`）。
- 读取文件时丢弃顶层 args/来源信息；clean 字典同键后写覆盖（`:69`、`:76`）。
- 分组只按 start、timing、condition（`:149`），A′/B 同条件输入会合成一个结果。
- 修复：保留运行身份，使用完整配对键与分组键，冲突显式检查。验收覆盖 A/B 同图、own/teacher、before/after，以及交换输入文件顺序。合法结果应对文件顺序不敏感。

### R3：实际选样预算仍按 batch，保存预算却按单图

- rollout 与 `find_common_T()` 搜索使用 batch 最大深度（`b_mechanism_select.py:134`、`:147`）。
- manifest 的 n_loops 却存单图深度预算（`:204`）；late/controls 据此重跑并要求 T 在范围内。
- 反例：单图预算 8 轮，与深图同批时搜索 16 轮，若首次共同完成在第 11 轮，则被 selection 接纳，却在 runner 被跳过。
- 修复：按图截断 T 的搜索范围，断言 `0 <= T < n_loops`。验收同一图换 batch 邻居及 batch_size 后预算、资格一致。

## P2：指标和实验接口

| 编号 | 性质与影响范围 | 证据及修正要求 |
| --- | --- | --- |
| R4 | 统计范围不足：首轮队列不代表完整错误传播 | summary 的 N1 仅取 post1，遗漏 post2 之后才受损的节点，也不包含首轮已经修好的注错点；max_propagation_edges 也只来自该首轮队列（summary:100、:118、:137）。保留现有指标时明确冠以“首轮”，另报告注错点恢复和窗口内下游受损。 |
| R5 | 指标定义不足：曾经正确不等于持续恢复 | `any(later_correct)` 将“错→对→错”记为恢复（summary:126、:132）。分开曾恢复、最终正确、持续恢复，并约定观察窗口和未完成观察的处理。EMPTY 还需区分到期无值与未到期。 |
| R6 | 图级聚合及核心准确率未实现 | summary:158 直接累加目标/节点，未逐图等权；输出是条件损伤/恢复率，没有逐轮节点 VALUE 准确率、节点四属性全对率。按用户已明确的两核心指标补实现，机制事件指标另外保留。 |
| R7 | controls 缺基线且汇总接口未接通 | controls 仅执行干预续算，记录只有 mode，没有 condition（controls:106、:118）；summary:84 遇 condition=None 直接忽略。补同起点同预算 clean，支持模式识别和配对，空结果须显式解释。 |
| R8 | D1/D2/D3 的正确性标记仍引用旧答案 | `new_truth` 已补齐，但 `rollout_records()` 的 node_state 仍接原 label（late:137、:246、:265、:283）。新答案预测正确也可能被标为 value_correct=False。明确 old/new 两套正确性或传入新标签。summary 主动排除三类，包内未见相应专用汇总。原始轨迹可重新计算，不必因标记问题全部重跑。 |

R4/R5 的数学定义在 summary 注释中写明了，因此它们是与研究目标不匹配的口径不足，不应误报为“代码没有按自己的公式计算”。但这些数不能被汇报成完整下游恢复率或持续恢复率。

## R9：正式能力指标与用户新口径尚未同步

用户已明确，初步核心是同一节点集合上的两项：节点 VALUE 准确率、节点前四属性同时正确率；整图指标可以保留作辅助。这里记录用户在本对话已给出的要求，不修改 consensus。

- B common 的 numeric_mask 使用 depth>=0，排除了题面常量（`tools/b_mechanism/b_mechanism_common.py:170`）。
- 正式评估的 numeric 只有 non-PAD 与最终 VALUE 为数字，因此题面常量也会入分母（`experiments/evaluation/metric_utils.py:460`）。
- `final_strict_graph_acc` 是全部非 PAD token 的四属性整图 conjunction，不是节点四属性正确率（`:573`）。
- `aggregate_formal_metrics()` 使用总体计数加权（`:120`），与逐图等权不是相同统计量。

建议保留历史三指标及其原名，新增定义清楚的两个节点核心指标，共用固定的节点定义位置 mask。不要把已有历史数直接改名为新口径。看板当前“背景与口径”仍是旧三指标描述，需要后续同步实现与说明。

## 已核实修复的部分

- 入口已显式加入脚本目录。
- audit 使用 `S[:-1]`，与当前训练的 `batch_ext[:-1]` 一致；新增随机 seed/trial ID。
- teacher/own 起点索引对应正确，aux 按字段处理，逐轮输入输出分开记录。
- late 已增加 clean 续算、D1 new_truth、D2 完整预算和前四属性记录。
- 主组要求单父 u、q2 存在，未混入多父 fallback；“单父”主要检验传值，不能泛化为多父算术已验证。
- 无关节点排除了能到达所选 u/q1/q2 的节点，其负对照结论限定该受保护路径。
- 数据与权重哈希验证已加入，`late_r` 的 injection.pos 已指向实际被改的 r。
- 人工 self_consistent 构造已删除，不再要求当前阶段修复或启动它。

## 暂缓与切换边界

用户已决定先做其他实验，暂缓局部自洽及依赖它的切换分析。

现在的 switch_held_r 是另一种父值干预：固定错父若干轮，再换第二个错值一次并释放。sustained_r 对照在同一时点直接释放，比较的是“释放”与“释放时追加一次干扰”，不是“继续保持旧错值”与“切成新错值”。不能沿用原自洽切换解释。

参数也缺约束：`switch_at <= hold_rounds` 被前面的 hold 分支覆盖，`switch_at=0` 不切换，间隔过大会先释放再二次注错（controls:97）。本阶段不把切换纳入必跑项；独立的持续错父后释放可在补齐基线/统计后推进。设计文档仍残留“自洽值稳定后切换”等旧描述，应同步标注。

## 建议下一步与验收

1. 先修 R1/R3，再生成共同样本 manifest；早晚直接比较另取对应共同事件交集，保留完整样本的覆盖数。
2. 修 R2，建立按运行身份配对的汇总；补齐 R4–R9 所需指标和基线。
3. 小样本验收必须包含：一边合格一边不合格、混合来源汇总、换 batch、延迟受损、短暂回正再错、修改题面后预测新答案、controls 单独汇总。
4. 不因汇总缺陷自动废弃完整原始轨迹。能够按 manifest 补筛交集、按新真值重算或补 clean 的，先修复派生统计；缺失轨迹或起点资格不符的再补跑。

审查记录不等于启动实验的批准。未改变运行中的 C0 训练或任何 GPU 任务。
