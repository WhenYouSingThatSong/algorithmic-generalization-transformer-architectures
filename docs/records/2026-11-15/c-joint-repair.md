# C 自然错误联合修复：target + 直接父节点

日期：2026-11-15 16:08 CST。  
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
特异性收益。

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
的 7/7 是节点轮次计数，不是独立图数。改善和损伤相抵，不支持“联合替换
带来稳定下游恢复”。

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
  `c-joint-endpoint-summary.csv`
- 完整 records、hidden snapshot 留在 tmp。
