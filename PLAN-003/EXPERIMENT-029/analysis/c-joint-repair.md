# C 自然错误联合修复：target + 直接父节点

日期：2026-11-15 16:08 CST；17:13 CST 按 1636 审阅补分类/身份/轨迹口径，未重跑。
状态：**复用原 57 事件/114 条自然与健康状态，新增联合分支 smoke/main 已完成；不是 C+，没有重新选事件或 donor**。  
代码：`reproduction@572cc10`（运行期间未 pull/改代码）。  
交接：[C stages/joint](../../../ai-collab/handoff/2026-11-15-b-multiparent-c-stages-joint.md)。

## 1. 运行

```bash
CUDA_VISIBLE_DEVICES=2 python tools/repair_study/run_c_joint.py \
  --source_dir tmp/analysis/c-local-repair-20261115/s42 \
  --limit_events 2 --out_dir tmp/analysis/c-joint-20261115/smoke_s42 \
  --device cuda:0
# s1234 同参数 smoke_s1234

CUDA_VISIBLE_DEVICES=2 python tools/repair_study/run_c_joint.py \
  --source_dir tmp/analysis/c-local-repair-20261115/s42 \
  --out_dir tmp/analysis/c-joint-20261115/s42 --device cuda:0
CUDA_VISIBLE_DEVICES=2 python tools/repair_study/run_c_joint.py \
  --source_dir tmp/analysis/c-local-repair-20261115/s1234 \
  --out_dir tmp/analysis/c-joint-20261115/s1234 --device cuda:0

python tools/repair_study/summarize.py \
  --inputs tmp/analysis/c-joint-20261115/s42/records.jsonl \
           tmp/analysis/c-joint-20261115/s1234/records.jsonl \
  --out_dir tmp/analysis/c-joint-20261115/summary
```

- smoke：s42/s1234 各 2 个固定事件（含健康对照），各 4 records，complete。
- main：s42 54 records、s1234 60 records，合计 **114**，全部 eligible。
- 沿用原 `c-local-repair-20261115` 的 57 个固定 case 和保存的 hidden；
  不重新跑 64 轮采集、不换 donor、不增加事件。
- runner 检查：snapshot 的权重/数据/位置/轮次身份一致；保存的 base 重放
  unchanged 与原记录逐轮一致；`sham_joint` 与 unchanged 相同；联合 donor
  的初始读数为原正确局部值。均通过，无 failure。

## 2. 结果（target VALUE）

`first` = 事件后第一轮 forward 正确；`final2` = 事件后第 15/16 轮均正确。
`initial` 是替换后尚未 forward 的读出，联合分支的 initial 正确是操作结果，
不算模型修复。

### 自然错误（57 case）

| branch | first correct | ever wrong | final2 correct |
|---|---:|---:|---:|
| unchanged | 12 / 57 (0.211) | 45 | 25 / 57 (0.439) |
| sham_current_target | 12 / 57 | 45 | 25 / 57 |
| restore_target | 20 / 57 (0.351) | 45 | 25 / 57 |
| restore_parents | 15 / 57 (0.263) | 45 | 25 / 57 |
| sham_joint | 12 / 57 | 45 | 25 / 57 |
| restore_joint | **29 / 57 (0.509)** | **43** | 25 / 57 |

- 联合替换把第一轮 forward 正确数从 12 提到 29（高于 target-only 20、
  parent-only 15），ever-wrong 从 45 降到 43；但 **final2 仍是 25/57，
  与 unchanged 相同**。
- 按原采样分类拆开第一轮 VALUE 正确数（分类是采样定义，**父状态取 t−1**，
  不是“干预瞬间父正确/错误”）：

  | 原分类 | n | unchanged | restore_target | restore_parents | restore_joint |
  |---|---:|---:|---:|---:|---:|
  | `value_parent_wrong` | 22 | 0 | 8 | 3 | **17** |
  | `value_parent_correct` | 23 | 0 | 0 | 0 | 0 |
  | `variable_only` | 12 | 12 | 12 | 12 | 12 |

  12→29 的首轮增量全部来自 `value_parent_wrong`（0→17）；
  `value_parent_correct` 与 `variable_only` 各分支完全不变。
- 保存的 `initial` 字段说明实际起点：在 `value_parent_wrong` 和
  `value_parent_correct` 两组共 45 个自然错误里，unchanged/parents 的
  target VALUE 初始均错（0/45 correct），target/joint 初始均对（45/45）；
  后者的“对”是写入局部正确值的结果，不是模型修复。`variable_only` 的
  12 个 case 各分支初始 VALUE 本来就都对。
- 末两轮按同一分类：

  | 原分类 | n | unchanged | restore_target | restore_parents | restore_joint |
  |---|---:|---:|---:|---:|---:|
  | `value_parent_wrong` | 22 | 6 | 6 | 6 | 6 |
  | `value_parent_correct` | 23 | 7 | 7 | 7 | 7 |
  | `variable_only` | 12 | 12 | 12 | 12 | 12 |

  三类合计 6+7+12=25/57；首轮增益没有转成任何一类的末两轮增益。
- 自然 target first4：unchanged 0/57、restore_target 1/57、restore_parents
  1/57、restore_joint 3/57；final2 都只有 1/57。VALUE 与 first4 都不能
  说成持续修复。

### 健康对照（57 条）

| branch | first correct | final2 correct |
|---|---:|---:|
| unchanged | 46 / 57 | 27 / 57 |
| restore_target | 52 / 57 | 27 / 57 |
| restore_parents | 49 / 57 | 29 / 57 |
| restore_joint | 53 / 57 | 29 / 57 |

健康对照也出现首轮变化；final2 只有小幅差异（+2），不能作为联合修复的
特异性收益。并且 parents 与 joint 的 final2 同为 29/57：联合相对仅父
first 为 +4/0，但 final2 为 +0/0，首轮差异同样没有转成末两轮增量。

### 端点区域（natural、VALUE、final2）

| branch | role | n | baseline final2 | final2 | helped | harmed |
|---|---|---:|---:|---:|---:|---:|
| restore_target | target | 57 | 25 | 25 | 0 | 0 |
| restore_parents | target | 57 | 25 | 25 | 1 | 1 |
| restore_joint | target | 57 | 25 | 25 | 1 | 1 |
| restore_target | all_nodes | 4352 | 3526 | 3526 | 0 | 0 |
| restore_parents | all_nodes | 4352 | 3526 | 3526 | 7 | 7 |
| restore_joint | all_nodes | 4352 | 3526 | 3526 | 7 | 7 |

parents/downstream 区域的 helped/harmed 在该轮汇总中均为 0/0；all_nodes
的 7/7 是 **case×节点的末两轮端点观察**（两轮是联合判定，没有把两轮分别
计数），不是节点轮次，也不是独立图数。57 个 case 共 4352 个 case×节点
端点观察，其中 5 个 case 有变化：3 个 case 出现 helped、3 个出现 harmed、
1 个两者都有；具体位置见 `c-joint-allnode-endpoint-deltas.csv`。
逐节点集合核对后，restore_parents 与 restore_joint 的 helped/harmed
位置完全相同（不是只总数相同），所以这 7/7 不能区分联合与仅父；净变化
为 0，不支持“联合替换带来稳定下游恢复”。

### target 末两轮 1 改善 / 1 恶化的身份

每格为 `首轮正确 / 末两轮正确`（1=correct，0=wrong）：

| case | 原分类 | unchanged | target | parents | joint | joint 相对 parents |
|---|---|---:|---:|---:|---:|---|
| `C:s1234_e143:N32_shallow:14:170:53:value_parent_wrong:natural_error_state` | `value_parent_wrong` | 0/0 | 1/0 | 0/1 | 1/1 | 首轮 +1，末两轮 0 |
| `C:s42_e143:N64_deep:0:306:52:value_parent_wrong:natural_error_state` | `value_parent_wrong` | 0/1 | 1/1 | 0/0 | 1/0 | 首轮 +1，末两轮 0 |

- 1 个改善（N32_shallow:14:170:53）和 1 个恶化（N64_deep:0:306:52）
  是**两个不同的 case**，不是同一 case 被同时记一次；25/57→25/57 是
  这两个相反 case 的数值相抵，不代表没有逐例变化。
- 但相对于仅父分支，两个 case 的末两轮结果相同：改善 case 的 parents
  已为 1，恶化 case 的 parents 已为 0；joint 相对 parents 的末两轮是
  0 改善 / 0 恶化。两者都属于首轮 +1（parents 首轮为 0、joint 首轮为 1），
  首轮增益没有改变末两轮。
- 逐轮轨迹（`c-joint-two-case-rounds.csv`）显示：改善 case 的 parents
  从第 3 轮起保持正确值 0、joint 从替换后的 0 一直保持到第 16 轮，而
  target-only 在第 15/16 轮漂到 19；恶化 case 的 joint 初始与第 1 轮
  forward 都是 16，第 2 轮已回到 0 并停在第 16 轮的 7，parents 从未
  回到 16，而 unchanged 和 target-only 从第 4 轮起回到 16。这里只记录
  轨迹，不据此推断机制。
- 全 57 个 case 中，joint 相对 parents 首轮为 **+14/0**（全部落在
  `value_parent_wrong`，与上表 3→17 一致），末两轮为 **+0/0**。
  因此“联合的首轮增益”与“仅父的末两轮收益”必须分开；不能把 25/57
  的相同总数解释成同一批节点或同一轨迹。

## 3. 解释边界

- 这是同一个错误现场的一次性联合替换（target + 全部直接父节点），随后
  自由 16 轮；没有逐轮钳制、解码字段回灌或训练。
- 联合分支的第一轮正确增加，说明更大的局部状态替换可以在短期内改变
  读出；但 final2 与 unchanged 相同，且健康对照也有首轮变化，不能把
  首轮提升当作错误特异修复。
- 不能从 1 个 target 改善、1 个 target 恶化推导总体机制；自然结局分组
  和各分支共享分母的分层表在 CSV。
- 没有增加祖父节点、整图回退或 C+；B 结果不在此解释。

## 4. 产物

- smoke：`tmp/analysis/c-joint-20261115/smoke_s42/`、`smoke_s1234/`
- main：`.../s42/`（54 records）、`.../s1234/`（60 records）
- summary：`.../summary/`（coverage、target_cases、target_summary、
  endpoint_cases、endpoint_summary、node_curves）
- 提交小表：`c-joint-coverage.csv`、`c-joint-target-summary.csv`、
  `c-joint-endpoint-summary.csv`、`c-joint-class-summary.csv`、
  `c-joint-case-deltas.csv`、`c-joint-two-case-rounds.csv`、
  `c-joint-allnode-endpoint-deltas.csv`
- 完整 records、hidden snapshot 留在 tmp。
