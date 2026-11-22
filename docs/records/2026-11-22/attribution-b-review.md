# B 侧复核：同题四条件、屏蔽父值、来源与工作空间复用

- 日期：2026-11-22。
- 状态：**仅复核 C 已交付记录；未加载权重、未重跑模型、未改 C 产物。**
- 依据：`docs/records/2026-11-22/attribution/c-attribution-delivery.md`、`state-repair-attribution.md`、`docs/records/2026-11-22/attribution/`。
- 代码/工作树：`management-20261120@b35ba8e`；plan SHA `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`。

## 1. 身份和表完整性

- 四个 stage 的 `delivery.json` 均有 8 个权重；`rows` 分别与 provenance 的 15,360（four_conditions）、43,008（pathway）、51,200（sources）、34,560（workspace）一致。
- 逐表 SHA-256 与 `delivery.json` 的 `table_hashes`、以及 provenance 的 delivery hash 全部匹配；没有重新生成或替换 C 的表。
- `four_conditions/four-way-common.csv` 的 192 行全部 `common=1`：该文件本身已经是共同合格子集，未合格行在 `graph-counts.csv`，不能反过来用 192 行推断全体合格率。
- 每个 timing 有 32 个 target 实际写入事件、32 个 clean 配对行；`direct_faults` 对 target 均为 32/32。C 报告的四条件主表可由这些行复算。

## 2. 同题四条件的下游读数

C 报告只列了目标下一轮修回（`direct_repaired_next`）。同表的 `specific`、`damage_with_followup`、`stable_after_damage` 是不同分母；复核汇总如下（完整见 `four-conditions-downstream.csv`）：

| model | timing | 写入/32 | 目标修回 | specific | damage | stable |
|---|---:|---:|---:|---:|---:|---:|
| A_s42 | pending / just / later | 32/32/32 | 0/0/0 | 32/32/0 | 32/32/0 | 0/0/0 |
| B_s42 | pending / just / later | 32/32/32 | 32/32/32 | 0/10/32 | 0/10/32 | 0/10/32 |
| C_s42 | pending / just / later | 32/32/32 | 28/29/25 | 0/32/12 | 0/32/12 | 0/32/12 |
| C_s42-correction | pending / just / later | 32/32/32 | 32/32/31 | 0/32/31 | 0/32/31 | 0/32/31 |
| C_s42-update | pending / just / later | 32/32/32 | 31/31/30 | 0/32/29 | 0/32/29 | 0/32/29 |
| C_s1234 | pending / just / later | 32/32/32 | 17/17/15 | 0/32/7 | 0/32/7 | 0/32/7 |
| C_s1234-correction | pending / just / later | 32/32/32 | 32/31/22 | 0/32/13 | 0/32/13 | 0/32/13 |
| C_s1234-update | pending / just / later | 32/32/32 | 28/28/19 | 0/32/24 | 0/32/24 | 0/32/24 |

- “目标修回”不是“下游未受损”。例如 B just_solved 目标 32/32，但 specific/damage 只有 10/32；C_s42 later 目标 25/32，specific/damage 12/32。C 报告主表没有把这两个读数并列，容易把目标修复误读成整条下游恢复。
- s1234 及其 correction/update 的同一事件计数可复算；它们不在四条件主表的四臂共同集内，不能与 s42 主表直接拼成同种子对照。s1234 本身（17/17/15）明显低于 s42 的 28/29/25；这是本批可见的种子差异，不作机制结论。
- 四条件的四臂共同集每 timing 32 个 target；所有 target 都实际写入，未出现“注错未生效”。C 报告的数字与表一致。

## 3. 自然错误与回放：零覆盖，必须显式报告

- `four_conditions/graph-counts.csv` 的 `metric=natural` 行存在；按 `additional / due_at_input` 汇总，只有 C_s1234 有 14 个 `new_errors`（25 个 numeric_errors），C_s1234-update 有 15 个（42 个）；其余 6 个权重均为 0。`before_due` 全部是非数值状态，不构成本轮自然算错。
- **`metric=natural_replay` 行数为 0**。C 交付和报告没有单列这一零覆盖；`natural_replay` 的代码入口存在，但本批没有可用的“前一轮正确、后一轮错”事件产生回放配对。因此本批不能评估“自然错值 vs 回放上一轮正确状态”的对照。
- 这不是“自然错误不存在”：C_s1234/update 的 additional 自然错误 14/15 是实际记录；但可回放事件为零，必须写成零分母/零覆盖，不能省略或把普通 natural 计数当作回放结果。
- 相关文件：`natural-errors.csv`；C 报告的自然分析如需补交，只能从既有 records 重导，不需要 GPU。

## 4. 屏蔽父值阳性对照

- `pathway/key-comparison.csv` 每模型 192 个 eligible 事件；父值屏蔽在 A/B/C 各臂均出现受损（A 28、B 182、C_s42 104、C_s42-correction 120、C_s1234 74、C_s1234-correction 95、C_s42-update 144、C_s1234-update 143），无关位置屏蔽全部为 0。C 报告的表格与表一致。
- 该干预只作用于一个位置；其他位置的值副本、残差和先前状态仍可能可用，所以“父值屏蔽有损伤”支持该位置参与正常计算，但不能单独证明父值信息是唯一来源。C 报告已保留此边界。

## 5. 来源与工作空间复用

- `sources/fresh_solution` 的 original/edited 在八份权重上均 128/128；这是从空状态的参照，不是修复率。来源条件的完整行在 `sources/graph-counts.csv`，C 报告只给了 fresh-solution 控制，没有给出各来源条件的聚合结论；本复核不据此补写机制判断。
- `workspace` 在 `extra=0` 的节点合并读数可复算，并与 C 报告一致：

| definitions/register | unique | reused | saved_reader reused | immediate_reader reused |
|---:|---:|---:|---:|---:|
| 1 | 0.8070 | 0.8070 | 0.6273 | 0.7931 |
| 2 | 0.8070 | 0.7214 | 0.3674 | 0.5155 |
| 4 | 0.8070 | 0.6890 | 0.2349 | 0.3629 |

- 每个 definitions=2/4 的 `extra=0` 行都有 `rebindings>0` 和非空 `mean_rebind_distance`；definitions=1 的行为 0。定义 2/4 的 `unique_correct` 保持不变，说明配对题的独立命名控制没有被复用条件自身的难度拖低。该读数支持“最近定义/旧值被读取”的行为描述，但不证明内部存储或回收机制。
- `saved_reader` 与 `immediate_reader` 的差距明显大于总体 `all`；这与 C 报告的表格一致。复用条件同时改变定义次数、保留结果、间隔和位置，不能把差异单独归因于一个因素。

## 6. 复核结论与待补

- C 的主表数字（A 0/32、B 32/32、C_s42 28/29/25、C_s42-correction 32/32/31）可从 `four-way-common.csv` 复算；父值屏蔽和 workspace 主表也可复算。
- **需要 C/GPT 处理的两点**：
  1. 自然回放零覆盖（`natural_replay=0`）没有被报告；应按交接要求在结果中显式列为零分母，或从既有 records 重导可回放事件；不能把 natural 计数替代回放对照。
  2. 四条件报告只给目标修回，未并列 `specific/damage/stable`，且未单列 s1234 两个连续种子的四条件计数；本复核补充了 `four-conditions-downstream.csv`，正式报告应采用相同分母。
- 本复核没有读取原始逐题轨迹，只核对了提交的 CSV/JSON 和 hash；未复算 C 模型的实际前向，因此“无机制结论”的边界仍适用。
- 基础拟合/训练曲线：`fit-sources.json` 中有新 C 的验证记录，但 C 报告没有给统一拟合表；本复核不把不同训练目标的 loss 横比。

## 7. 文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `four-conditions-downstream.csv` | `5ef761c0d5c49e2cce38ad696c32bf0b349697230b6084b0458b665506465567` |
| `natural-errors.csv` | `27a0d87f977e20f5bdd2df8ea62fc6227eb7225db54f75c81cd9d823b2300042` |
| `workspace-extra0.csv` | `9f01f571baf95ddbdf82bbc3435aa1b5743147c4920a408e1ca044f79e45864c` |
