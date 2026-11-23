# 第一批 · C 侧 original 交付（2026-11-23）

- 状态：**仅 C 侧 original 阶段**：C_s42/C_s1234 的 natural/history/fanout，以及 A_s42/B_s42 的 fanout。后训练 C 的 trained 阶段由 yhy-57 运行；四模型 report 待 B 侧齐备后生成。
- 未训练、未改结构；固定权重评估。C 侧只写 `natural/history/fanout` 的模型目录，不写 trained 的 mark/standard/layered/expanded。
- 命令：

```bash
export DATA_ROOT=/workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures
export FIRST_BATCH_OUT=/workspace/lab-xj/users/jinxi/code/yhy/management-20261120/tmp/analysis/first-batch-20261123
CUDA_VISIBLE_DEVICES=0 bash tools/c_hard_graphs/run_first_batch.sh original smoke
CUDA_VISIBLE_DEVICES=0 bash tools/c_hard_graphs/run_first_batch.sh original formal
```

- 权重：C_s42 `4a00d6a21a7e…`、C_s1234 `0bca3a2b91e8…`、A_s42 `0b66e196990a…`、B_s42 `ef58ba261fd0…`；plan `b3c4b1be…`。每个 stage/model 的 `identity.json` 和 `fit-source.json` 随 raw 保留。

## 1. Natural：自然错与同错数人工供体

`natural` 使用 C 标准组 192 图/模型的已交付 `natural` 事件；每图最多四条可回放边，保留原先能救和不能救的两类。

| model | opportunities | node events | edge records | selected edges | matched reproduced | donor fail |
|---|---:|---:|---:|---:|---:|---:|
| C_s42 | 337,614 | 217 | 306 | 76 | 66 | 10 |
| C_s1234 | 315,962 | 465 | 623 | 118 | 87 | 31 |

- `matched reproduced` 指 `paired_replay_agrees`：自然错状态下的首个子节点读数与归档 `observed_child` 一致。未匹配的主要原因是 `no_exact_same_clock_wrong_donor`（10/31）；没有 `archived_transition_not_reproduced`。
- 背景可修复性：observed 全可用（66/87）；parents_restored 58/66（8 缺历史）与 86/87（1 缺）；all_other_due_restored 50/66（16 缺）与 68/87（19 缺）。缺失是“找不到先前正确历史”，不是失败。
- 同值人工供体：相同错值、同绝对轮次、同位置，但来自改动题面常数的 donor。自然路径与人工同错数路径在 observed 背景完全相同的比例：C_s42 44/66，C_s1234 54/87；平均 hidden 距离 2.76/3.32。**相同数值不保证相同后续轨迹。**
- 先确认传播（`previously_confirmed=True`）的 observed 子样本：自然/人工/目标回正的首个子节点 specific 分别为 C_s42 28/28、27/28、0/28；C_s1234 17/17、17/17、0/17。末两轮 target+child 都对在这组都是 0；未先确认的组 C_s42 4/38、C_s1234 12/70（分背景见 `natural-counts.csv`）。
- `target_restored` 只把目标放回先前正确状态；它不等于修复背景。observed 下 C_s42 的 12/38、C_s1234 的 39/70 仍出现子节点 specific（背景其他位置仍错），所以“目标回正”不能单独当纠错。
- 本表只有可回放边，不能用它估计所有自然错误的恢复概率；`natural-coverage.csv` 保留逐图分子/分母。

## 2. History：同读数、不同历史 + 新子节点

- 64 题/模型（gid 400–431 × 深度 4/8）；四臂 stable / first_after_empty / recovered / wrong_control；新子节点只读取前缀目标位置。

| model | depth | common | confirmed read opportunities | stable next | first_after next | recovered next | wrong_control next | all end correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C_s42 | 4 | 31/32 | 7/31 | 7/7 | 7/7 | 6/7 | 0/7 | 7/7 |
| C_s42 | 8 | 27/32 | 3/27 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| C_s1234 | 4 | 13/32 | **0** | — | — | — | — | — |
| C_s1234 | 8 | 5/32 | **0** | — | — | — | — | — |

- C_s42：在三段不同历史（stable、持续置空后首次填对、先读错再恢复）上，新子节点下一轮读数都等于 gold；wrong_control 下一轮等于错值（0/7、0/3 正确），阳性对照有效。向量距离和逐题读数见 `same-value-new-reader.csv`。
- **C_s1234 的 zero coverage 是结构性的**：所有 64 个题例的新子节点首步读数都是 24（空），因此 `clean_reader_ok` 与 `positive_control_used` 都为 false，`confirmed_read_opportunities=0`。这不是“新读者答错”，也不是缺失；报告必须保留零分母。该模型 `recovered` 臂自身合格也低（13/32、5/32）。
- `same-value-counts.csv` 保留 attempts/own_eligible/common/confirmed/correct 四列；空值即零分母。

## 3. Fanout：一、二、四子节点传播

- 64 题/模型 × 3 fanout × 4 phase；C_s1234 在 64/8/complete 有 1/32 不合格，其余 cell 32/32 合格。
- `fanout-base-models.csv` 逐模型/深度/fanout/phase 给出 target_next、target_recovered、child_specific、children_specific、child_at_risk、descendant_wrong_number_rounds、descendant_empty_rounds、output_ever_wrong、output_wrong_end、followup、stable_after_specific。下面是 fanout=1、深度 4 的摘要：

| model | phase | target_next | child_specific | descendant wrong rounds | output ever/end wrong |
|---|---|---:|---:|---:|---:|
| C_s42 | before | 31/32 | 0 | 58 | 6/1 |
| C_s42 | just_correct | 31/32 | 32/32 | 155 | 32/1 |
| C_s42 | children_correct | 31/32 | 18/32 | 223 | 28/3 |
| C_s42 | complete | 31/32 | 11/32 | 141 | 18/2 |
| C_s1234 | before | 13/32 | 0 | 68 | 19/0 |
| C_s1234 | just_correct | 13/32 | 32/32 | 150 | 31/0 |
| C_s1234 | children_correct | 12/32 | 19/32 | 154 | 32/0 |
| C_s1234 | complete | 12/32 | 14/32 | 176 | 32/0 |
| A_s42 | before | 0/32 | 32/32 | 1344 | 32/32 |
| A_s42 | just_correct | 0/32 | 32/32 | 1440 | 32/32 |
| A_s42 | children_correct | 0/32 | 0 | 0 | 0/0 |
| A_s42 | complete | 0/32 | 0 | 0 | 0/0 |
| B_s42 | before | 32/32 | 0 | 0 | 0/0 |
| B_s42 | just_correct | 32/32 | 20/32 | 45 | 32/0 |
| B_s42 | children_correct | 32/32 | 32/32 | 81 | 32/0 |
| B_s42 | complete | 32/32 | 32/32 | 96 | 32/0 |

- 随 fanout 增大，C_s42 的 `descendant_wrong_number_rounds` 在 just_correct 从 155（1）→280（2）→454（4）；B_s42 的 `children_specific` 从 20→52?（见 CSV 的 `children_specific` 列，just_correct 为 20/44/86），C_s42 为 32/64/128。A_s42 在 just_correct 的 descendant wrong 随 fanout 增大（1440→2207→3456），且 output 末轮仍错 32/32。
- A_s42 在 children_correct/complete 的 `child_specific=0`、descendant=0，不是“恢复”，而是子节点槽位已经填好、没有被重写；这与其保留已填值的行为一致，但不能把该列读成下游未受损。
- B_s42 的 `descendant_empty_rounds` 在 just/children 阶段非零（51/60 等），提示恢复过程中有清空/重填；C 两模型该列为 0。报告需把“数值对但被清空”与“保持正确”分开。
- 以上是 C/A/B 基础权重；后训练 C 的 fanout 尚未由 B 交付，最终 `fanout` 四模型表由 B 的 report 生成。

## 4. 边界

- 本批只覆盖 C 侧 original；trained 的 mark/standard/layered/expanded/fanout 由 B 运行，四模型 report 未生成。
- natural 只统计可回放边；同值人工供体的 hidden 与自然错不同，轨迹差异不能归因于“模型不识别数值”。
- history 的 C_s1234 zero coverage 必须显式保留；不要用其他臂替代。
- fanout 的 A_s42 已填子节点不被重写，`child_specific=0` 不代表恢复；`descendant_empty_rounds` 是 B 的独立现象。
- 固定权重、固定题面；不作纠错训练或架构的因果结论。

## 5. 文件

- `raw/natural/{C_s42,C_s1234}/`：各 192 个标准图 natural 结果 + identity/fit-source。
- `raw/history/{C_s42,C_s1234}/`：各 64 个新子节点结果。
- `raw/fanout/{C_s42,C_s1234,A_s42,B_s42}/`：各 64 题 × 12 场景。
- `raw-manifest.sha256`：784 个 raw/identity/fit-source 文件的逐文件 SHA-256；raw 实际 416 KB。
- `tables/natural-coverage.csv`、`natural-coverage-summary.csv`、`natural-pairs.csv`、`natural-counts.csv`
- `tables/same-value-new-reader.csv`、`same-value-counts.csv`
- `tables/fanout-base-models.csv`

### SHA-256

| 文件 | SHA-256 |
|---|---|
| `tables/natural-coverage.csv` | `3725bbf1b42b16062c95a49e560f66e4623a5e2e8c744f39fd9f66b29e4c548e` |
| `tables/natural-coverage-summary.csv` | `debc1aaa6afd5abd1612c5cb28fa4ac0c36d8b3ea522f6af40229af16c17c430` |
| `tables/natural-pairs.csv` | `fa919d923c23d85d0df2abf75d4257e4034c0dac1af3226dc58ec8c476d8cd48` |
| `tables/natural-counts.csv` | `d071b5923ea4fa2b6c18ff1b3b34e4eb41b52b25903d4459e9077357a264dad4` |
| `tables/same-value-new-reader.csv` | `2b406b2a9e0f08d1fa9bbac3b8497e272c42223ad4459b1ea64a691ec5a93c42` |
| `tables/same-value-counts.csv` | `a8cc3ea3ed22b759f334ec737e624a16c54fb32faf4bde9cb6aba58c5a05e007` |
| `tables/fanout-base-models.csv` | `7f11d616e4c762acd5c346f12263afcbaa41e16f543525b686a93755dd3772b5` |
| `raw-manifest.sha256` | `4867caecb0953bb50aa8388887da190182235cfbb892613c0b1c98a46cdf71f5` |
