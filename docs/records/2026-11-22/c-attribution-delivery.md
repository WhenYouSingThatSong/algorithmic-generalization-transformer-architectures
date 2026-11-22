# 同题机制比较、计算空间复用与跨任务原型：交付

日期：2026-11-22。状态：**完整队列 exit 0；六项既有观察 + 四个新阶段 + 架构/跨任务原型 smoke 均完成；描述性结果，不作架构因果结论**。  
代码：`reproduction@4df9824`；plan SHA256 `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`。  
运行根：`tmp/analysis/c-attribution-v1/`；管理数据根为 `management-20261120`，旧 c_update 权重通过只读 symlink `tmp/analysis/c-update-20261119` 指向原训练树。

## 1. 执行与覆盖

- 按最新交接用 `attribution_queue.py --include-prior-mechanisms` 串行执行；**没有启动旧停止小试**，它已被新交接覆盖。
- 八份权重：A_s42、B_s42、C_s42、C_s1234，以及 C_s42/C_s1234 各自的 correction 和 update final。身份与 checkpoint 哈希在 `provenance.json`。
- smoke 与 formal 都完成；四个新阶段 delivery 全部生成，无 traceback。
- 六项既有观察（每 seed）：transitions 544 题、signals 64、identity 64、fanout 96、versions 64、readers 64。
- 四个新阶段（每阶段八份权重）：`four_conditions` 15,360 行、`pathway` 43,008、`sources` 51,200、`workspace` 34,560；都保留逐图 `graph-counts.csv`、配对表和 delivery。

## 2. 同题四条件：目标写入后的首轮修复

四条件主表只含 A_s42、B_s42、C_s42、C_s42-correction；三时点各 32 个共同合格事件。数值为“在目标位置实际写入预定错值后，下一轮目标读回正确”的计数。

| 时点 | A_s42 | B_s42 | C_s42（原始） | C_s42-correction |
|---|---:|---:|---:|---:|
| 结构上刚可算之前 | 0/32 | 32/32 | 28/32 | 32/32 |
| 刚算出之后 | 0/32 | 32/32 | 29/32 | 32/32 |
| 常规轮数后再 4 轮 | 0/32 | 32/32 | 25/32 | 31/32 |

- A_s42 在三个时点都不修回；B_s42 全部 32/32 立即修回；C_s42 原始为 25–29/32，C_s42-correction 为 31–32/32。
- 这是固定权重、固定题面下的行为；四条件改变的是模型和训练历史，不能把差异单独归因于离散/连续反馈。
- C_s1234 及其 correction/update 的逐图结果在 `four_conditions/graph-counts.csv` 与 `paired.csv`；它们不在四条件主表的四个共同臂内。
- 补充：`four-way-common.csv` 的 192 行全部 `common=1`，它本身就是四臂共同合格子集；未合格行在 `graph-counts.csv`，不能反过来把 192 行当全体合格率。

### 2.1 下游 specific / damage / stable（按 B 复核表补列）

目标下一轮修回 **不等于** 下游未受损。下表分母是实际写入事件（每模型每时点 32 次；B 复核确认 target 实际写入均为 32/32），`specific`/`damage`/`stable` 的定义与表 1 一致；B 的完整表在 `attribution-b-review/four-conditions-downstream.csv`。

| 模型 | 时点 | 目标修回 | specific | damage | stable |
|---|---|---:|---:|---:|---:|
| A_s42 | pending | 0 | 32 | 32 | 0 |
| A_s42 | just_solved | 0 | 32 | 32 | 0 |
| A_s42 | later | 0 | 0 | 0 | 0 |
| B_s42 | pending | 32 | 0 | 0 | 0 |
| B_s42 | just_solved | 32 | 10 | 10 | 10 |
| B_s42 | later | 32 | 32 | 32 | 32 |
| C_s42 | pending | 28 | 0 | 0 | 0 |
| C_s42 | just_solved | 29 | 32 | 32 | 32 |
| C_s42 | later | 25 | 12 | 12 | 12 |
| C_s42-correction | pending | 32 | 0 | 0 | 0 |
| C_s42-correction | just_solved | 32 | 32 | 32 | 32 |
| C_s42-correction | later | 31 | 31 | 31 | 31 |
| C_s42-update | pending | 31 | 0 | 0 | 0 |
| C_s42-update | just_solved | 31 | 32 | 32 | 32 |
| C_s42-update | later | 30 | 29 | 29 | 29 |
| C_s1234 | pending | 17 | 0 | 0 | 0 |
| C_s1234 | just_solved | 17 | 32 | 32 | 32 |
| C_s1234 | later | 15 | 7 | 7 | 7 |
| C_s1234-correction | pending | 32 | 0 | 0 | 0 |
| C_s1234-correction | just_solved | 31 | 32 | 32 | 32 |
| C_s1234-correction | later | 22 | 13 | 13 | 13 |
| C_s1234-update | pending | 28 | 0 | 0 | 0 |
| C_s1234-update | just_solved | 28 | 32 | 32 | 32 |
| C_s1234-update | later | 19 | 24 | 24 | 24 |

- 典型的“目标修回但下游已受损”：B_s42 just_solved 目标 32/32，specific/damage 只有 10/32；C_s42 later 目标 25/32，specific/damage 12/32；C_s42-update later 目标 30/32，specific/damage 29/32。
- s1234 的目标修回整体低于 s42：s1234 为 15–17/32，correction 为 22–32/32，update 为 19–28/32。它们与 s42 的题号/事件相同，但仍不能把种子差异单独解释为训练版本效应。

### 2.2 自然错误与回放零覆盖

- `metric=natural` 的 `additional/due_at_input` 每模型 12,800 个节点观察：只有 **C_s1234 14 个 new_errors（25 个 numeric_errors）** 和 **C_s1234-update 15 个（42 个）**；其余六个权重均为 0。`within_standard/before_due` 的 3,168 个观察全部是非数值状态，`due_at_input` 的 4,288 个观察全正确。
- **`metric=natural_replay` 行数为 0。** 本批没有产生“前一轮正确、后一轮数值错误”的可回放事件，因此不能评估自然错值与回放上一轮正确状态的对照。这是零分母/零覆盖，不是“自然错误不存在”；C_s1234/update 的 14/15 个 additional 自然错误是实际记录。
- 该零覆盖已按 B 复核补记；B 的 `natural-errors.csv` 给出逐模型自然计数。没有新增 GPU 或重放。

## 3. 父值屏蔽与恢复来源

`pathway` 的三臂共同合格事件为每模型 192 个。屏蔽父值定义位置的注意力会损伤原本正确的计算，屏蔽同深度无关位置则没有一次损伤：

| 模型 | 共同合格 | 未屏蔽仍正确 | 仅屏蔽父值受损 | 仅屏蔽无关位置受损 |
|---|---:|---:|---:|---:|
| A_s42 | 192 | 96 | 28 | 0 |
| B_s42 | 192 | 192 | 182 | 0 |
| C_s42 | 192 | 178 | 104 | 0 |
| C_s42-correction | 192 | 191 | 120 | 0 |
| C_s1234 | 192 | 145 | 74 | 0 |
| C_s1234-correction | 192 | 181 | 95 | 0 |
| C_s42-update | 192 | 188 | 144 | 0 |
| C_s1234-update | 192 | 171 | 143 | 0 |

- 这是注意力干预的因果读数，但只作用于一个位置；其他位置的值副本、残差和先前状态仍可能被使用，因此不能等同于“父值信息是唯一来源”。
- `sources` 的 target/parent/ancestor/constant/parent_target/unrelated/coherent 条件与 fresh-solution 对照在 `sources/graph-counts.csv`；fresh-solution 八份权重 original/edited 均为 128/128，只说明从空状态重解这组题，不是修复率。

## 4. 计算空间复用

`workspace` 在 extra=0 的节点合并读数（全 17 档 ×16 题 ×八权重）：

| 每寄存器定义次数 | 独立命名正确率 | 复用命名正确率 | 仅独立命名对 | 仅复用对 |
|---:|---:|---:|---:|---:|
| 1 | 0.8070 | 0.8070 | 0 | 0 |
| 2 | 0.8070 | 0.7214 | 6,861 | 812 |
| 4 | 0.8070 | 0.6890 | 10,491 | 2,154 |

- 保存结果读取位置对复用更敏感：每寄存器定义 2/4 次时，saved_reader 的复用正确率为 0.367/0.235（独立命名 0.627）；immediate_reader 为 0.516/0.363（独立命名 0.793）。
- 同时存活结果数从 0 增到 8、读取间隔从 0 增到 2 时，复用正确率略升；完整分档、峰值、实际重绑定距离在 `workspace/graph-counts.csv`。
- 表中“仅复用对”很少，而“仅独立命名对”很多；两组不是对称噪声，不能只用总体正确率差解释。

## 5. 既有观察、架构与跨任务

- 六项既有观察：`transitions`、`signals`、`identity`、`fanout`、`versions`、`readers` 的 summary/contrast/delivery 均已提交；原始轨迹留在 tmp。
- 架构原型：CPU 两步小测四臂均完成（loop 23,299 参数、fixed_compute 61,411、fixed_parameter 26,923、written 23,299；四臂 4 张评估图，values 0/2/3/4，整图全对均 0）。**这只是流程验收，不能当能力或架构比较**；`.pt` 权重留在 tmp。
- 跨任务：clean 和 forced 各 16 组、512 个条件实例、4,480 个答案位置、9,216 个轨迹位置；用参考答案回环时全部正确，验证的是评分器与导出格式，不是原图模型预测。真正模型预测来源另行标注。

## 6. 文件与来源

- 全部阶段表、delivery、架构两步 JSON、跨任务参考检查在 `docs/records/2026-11-22/attribution/`。
- 来源身份、八份 checkpoint、各阶段 cases/rows、delivery 与表哈希、完整命令在 `docs/records/2026-11-22/attribution/provenance.json`。
- 原始逐题 `.json.gz`、smoke/formal `complete.json`、架构 `.pt` 留在 `tmp/analysis/`；提交内容不改变这些原始记录。

## 7. 边界

- 这是两份/多份固定权重的行为与干预基线；不是训练新模型，也不是“纠错训练导致”的完整因果识别。
- 四条件改变模型及训练历史；父值屏蔽和恢复来源是局部干预；复用实验同时改变定义次数、存活值数、间隔和位置，不能单列一个因素。
- 空分母在 CSV 中留空；共同合格集外的效果不补零。
- B 仍需按逐题记录复核共同分母、实际写入和八轮下游；本交付只提供 C 侧产物和覆盖。
