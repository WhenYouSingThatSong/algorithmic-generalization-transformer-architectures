# ABC 共同任务（前三步）：重新赋值、固定空间长程序与边界诊断

- 日期：2026-11-19。
- 状态：**固定开发任务；未训练、未扩图/种子、未做第四步、未改模型结构**。A/B 为已验收 s42 主权重；C 为 s2026/s42/s1234 三个原始 e143 权重。
- 代码/交接：`reproduction@d92915a`（AB main 运行于 `3c360ec`，C main 运行于 `d92915a`；`tools/abc_workspace` 源码 hash 相同）；[执行交接](../../../ai-collab/handoff/2026-11-19-abc-workspace.md)。
- 数据：80 个新生成 development 赋值任务（5 形状×16）和每形状前 8 个长程序任务（共 40）；seed 2026111903。不是官方测试集，也不声称做过历史图同构排重。
- 模型：A_s42、B_s42、C_s2026、C_s42、C_s1234。C 有三个独立训练种子，A/B 各一个；数据为同一生成 seed 的单批开发任务。
- 共同初始四属性合格任务：**24/80**；跨模型比较同时给 all-candidates 与这 24 个共同起点。

## 1. 执行与覆盖

| lane | records | 说明 |
|---|---:|---|
| AB assignment | 2×1,940 | A/B 各 80 任务×4 阶段；每阶段 carry/reset_results/scratch + unchanged/same-value + 固定诊断 |
| AB program | 2×2,600 | A/B 各 40 任务×20 窗口×3 分支 + 固定诊断 |
| C assignment | 3×1,940 | 三个 C 种子各 80 任务×4 阶段 |
| C program | 2,575 / 2,585 / 2,540 | C_s2026 / C_s42 / C_s1234；carry 非数值提交后诊断减少，见 §1.2 |

- AB main：GPU7，8/8 任务；C main：GPU0，7/7 任务。C smoke 7/7，660 条 records。所有队列 `exit_code=0`。
- C 记录/来源 hash、独立 verifier 结果和 nonnumeric 计数见 [c-main-checks](abc-workspace/c-main-checks.md) 与 [c-smoke-checks](abc-workspace/c-smoke-checks.md)。
- 同步后的唯一清单：`tasks.json b13f87df…`（80 任务）、`models.json 73e99dcb…`（5 权重）。C/AB 都使用同一份；未生成第二份。
- 每个 run meta 固定 `physical_nodes=32`、`padded_tokens=256`、六字段 tokenizer（mod23、128 变量、EMPTY=24）；无截断。
- AB 的完整说明见 [AB 报告](ab-workspace.md)；本报告合并 ABC，C 部分给三个种子分列。

### 1.1 指标口径

- `terminal_value_final_*`：最后一次实际输出；`terminal_value_*`：末两轮同时正确。二者分列，不把稳定段写成数学收敛。
- `nodes_*` 与 `terminal_*` 分开；`task_first4_macro` 是题面行四属性，不等于结果正确。
- 所有比例按任务等权；失败/无效提交保留在分母，不补样、不以 gold 纠正。
- 赋值 `carry`=保留旧结果状态；`reset_results`=只重置受影响结果槽；`scratch`=按当前实际输入从空白重解。A/B 的 reset 只清 VALUE；C 的 reset 是整行 hidden 重初始化（见 §1.3）。
- 程序 `carry`/`reset_results`/`scratch` 都使用**各自分支上一窗口实际提交的数值**作为下一窗口输入。分支输出一旦不同，后续输入也可能不同，不能称为跨分支同输入对照。`scratch` 只在窗口内从空白重解，并不接收正确答案回填。固定起点的 `probe_*` 才是同一实际输入的配对诊断。
- C 程序是“窗口内连续 hidden + 窗口边界离散数值提交”的混合接口，不是始终连续的潜空间工作空间。

### 1.2 C 程序 nonnumeric 提交

`carry` 分支在少数窗口产生非数值提交；`reset_results` 和 `scratch` 本批没有。非数值提交阻断该分支的后续窗口，后续 `invalid_commit` 记录保留在分母；最后一个窗口自身的非数值提交按 terminal 失败计，不再产生下一行 invalid。

| C 模型 | invalid records | 首次 invalid 的轨迹数/40 | complete carry 含 None | 其中 final stage |
|---|---:|---:|---:|---:|
| C_s2026 | 198 | 18 | 19 | 1 |
| C_s42 | 90 | 12 | 13 | 1 |
| C_s1234 | 278 | 24 | 24 | 0 |

### 1.3 C reset 语义

GPU 接口检查（三个 C 种子 + A_s42 对照，`w2_f2:000`）确认：

- C 局部题面写入只改指定行，其余 hidden 逐元素不变；同值写入也只改这些行。
- C `reset_results` 等于对结果行做整行 `write_rows`，非结果行不变。
- A_s42 `reset_results` 只改结果行第 3 字段（VALUE），其他字段/行不变。
- 因此 C 与 A/B 的 reset 不是等价扰动，不能把 reset 差异直接归因架构。详见 [c-smoke-checks](abc-workspace/c-smoke-checks.md)。

## 2. 基础题：模型是否已经会做

`old_clean` 为从完成状态续算；`baseline first4 qualified` 是 `input_qualified` 的初始四属性合格数。

| 模型 | baseline old_clean terminal final 全对 /80 | baseline first4 qualified /80 | common24 内 old_clean terminal final |
|---|---:|---:|---:|
| A_s42 | 80 | 80 | 24/24 |
| B_s42 | 80 | 80 | 24/24 |
| C_s2026 | 78 | 56 | 24/24 |
| C_s42 | 77 | 61 | 24/24 |
| C_s1234 | 61 | 25 | 23/24 |

- 在共同 24 任务上，五个模型都基本会做；C_s1234 有 1 个共同任务基线未全对。
- C 的 all-80 first4 资格明显低于 A/B（25–61 vs 80），因此 all-candidates 的 C 差异有一部分是起点资格/题面四属性问题；跨模型主比较优先看 common24，同时报告 all80。
- 这是开发集上的行为，不是基础能力上限。

## 3. 赋值更新：保留、重置、从空白重解

### 3.1 common24：每阶段 terminal final 全对任务数

下表只列 s1 与 s4 两个端点，每格为 `正确任务数/24`；阶段 4 是回到原题。

| 模型 | carry | reset_results | scratch |
|---|---|---|---|
| A_s42 | 0 → 24 | 24 → 24 | 24 → 24 |
| B_s42 | 24 → 24 | 24 → 24 | 24 → 24 |
| C_s2026 | 23 → 4 | 6 → 0 | 24 → 24 |
| C_s42 | 13 → 7 | 9 → 0 | 24 → 24 |
| C_s1234 | 19 → 4 | 11 → 0 | 23 → 24 |

### 3.2 all80：阶段 4（回到原题）terminal final 全对任务数

| 模型 | carry | reset_results | scratch |
|---|---:|---:|---:|
| A_s42 | 80/80 | 80/80 | 80/80 |
| B_s42 | 79/80 | 80/80 | 80/80 |
| C_s2026 | 11/80 | 0/80 | 80/80 |
| C_s42 | 11/80 | 1/80 | 80/80 |
| C_s1234 | 9/80 | 0/80 | 76/80 |

- **A**：在 changed 输入阶段 carry 连续失败，但回到原题时旧状态恰好是正确答案，所以 stage4 恢复；不能把这写成通用更新能力。`reset_results`/`scratch` 阶段 1–3 均 80/80。
- **B**：carry 大部分阶段成功（stage1–2 80/80，stage3–4 79/80）；`reset_results`/`scratch` 全 80/80。唯一 carry 失误不是输入继承导致的整批失败。
- **C**：`scratch` 在 common24 上，s2026/s42 四个阶段均 24/24，s1234 为 23/24、24/24、24/24、24/24；all80 stage4 为 76–80/80。但 `carry` 随阶段快速下降，`reset_results` 更差（stage2 起 common24 为 0–2）。这表明当前接口下，从空白重解与沿用旧状态存在明显差距。
- `task_first4_macro` 约 0.67–0.92 不代表题面完整保存，也不能据此把失败定位到结果行；它是题面行四属性同时正确的比例，仍存在不可忽略的错误。

### 3.3 同轮数、不改题的保持对照（2026-11-19 审阅补充）

同一 common24、相同续算预算，表中为终端末轮全对任务数：

| 模型 | 不改题 s1/s2/s3/s4 | 连续改题 carry s1/s2/s3/s4 |
|---|---|---|
| C_s2026 | 24/23/20/15 | 23/18/11/4 |
| C_s42 | 24/23/21/19 | 13/10/7/7 |
| C_s1234 | 23/18/13/9 | 19/11/8/4 |

不改题也会退化；连续改题的终端正确数更少。因此需同时考虑长程保持不足和当前更新接口的影响，不能把所有失败归因于重新赋值。本表为边际任务数，未计算逐任务新增损伤交集；不能把两个数字相减直接称为因果损伤率。

完整每阶段/每形状/每条分支表在 [abc-summary.csv](abc-workspace/abc-summary.csv)。

## 4. 固定空间长程序：526 逻辑节点

所有程序固定 20 窗口；每个窗口 32 物理位置/256 padded tokens，累计 526 逻辑节点、520 个结构/执行祖先操作；逻辑深度按形状/任务为 96–260。**这不是模型原生一次处理 526 节点整图，也不是 526 个同时活跃槽位。** `w4_f1` 的内部表达式只有一个父，是复制链；520 是结构祖先计数，不等于 520 次多操作数加法或等难度算术。终端数随形状宽度为 2/4/6，assignment 各形状的节点终端分母为 32/64/96；本报告主表用任务成功数，不把节点 macro 与任务数混用。

### 4.1 程序最终成功 /40

| 模型 | carry | reset_results | scratch |
|---|---:|---:|---:|
| A_s42 | 1 | 40 | 40 |
| B_s42 | 39 | 40 | 40 |
| C_s2026 | 1 | 17 | 36 |
| C_s42 | 0 | 7 | 33 |
| C_s1234 | 0 | 2 | 29 |

- A 的 carry 在长程序上几乎不推进；`reset_results`/`scratch` 40/40。
- B 的 carry 39/40，`reset_results`/`scratch` 40/40；本批 B 的连续更新优势在长程序也成立。
- C 的 `scratch` 在 128/256/512 节点阈值分别 36–39/29–39/29–36 成功（按种子）；`reset_results` 只有 2–17/40；`carry` 0–1/40 且 18/12/24 条轨迹至少一次 nonnumeric 提交。
- 512 节点阈值对应最终 526 逻辑节点；`at_least_512_terminal_success` 与最终成功基本一致（A/B reset/scratch 40；C scratch 29/33/36，reset 2/7/17，carry 0/0/1）。

### 4.2 失败位置与输入继承

- 长程序各分支可能继承不同输入，不能仅凭主分支成功率定位失败原因。同输入证据来自固定 `probe`：在 C_s2026/s42/s1234 的 35/37/28 个可执行程序诊断起点，继续计算的 conditional 末两轮终端全对数为 14/14/14；同实际输入整窗重解为 35/37/26。它支持“当前窗口重解可改善对实际输入的求解”，不等于整个原程序恢复。
- 这仍不能定位具体 hidden 成分：`scratch` 同时重置了所有结果行/内部状态，`reset_results` 只重置结果行，`carry` 不重置；三者差异只能说明“整窗重解有帮助，单结果行重置不够”，不能说某个节点或某个门控是唯一原因。
- A 的 carry 失败、reset/scratch 成功；B 的 carry 大多成功；ABC 的差异是模型/接口行为，不是同一 hidden 机制的三个剂量。

## 5. 固定边界诊断：什么有帮助、什么有损伤

`probe_pairs` 是同一 carry 起点、同一额外 D+8 预算下的节点交集；只统计实际诊断起点，不能当总体恢复率。每格为 `repaired / damaged` 节点数（行数见 CSV）。

| 模型 | refresh_task | reset_results | refresh_and_reset | scratch_same_input |
|---|---|---|---|---|
| A_s42 | 0 / 0 | 416 / 25 | 416 / 25 | 416 / 25 |
| B_s42 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| C_s2026 | 31 / 321 | 29 / 361 | 169 / 15 | 169 / 15 |
| C_s42 | 54 / 271 | 37 / 189 | 218 / 15 | 218 / 15 |
| C_s1234 | 18 / 388 | 21 / 271 | 168 / 15 | 168 / 15 |

- A：只刷新题面行对 A 没有额外作用（A 的结果行是离散值，旧结果仍在）；重置结果/整窗重解修复 416 个节点、损伤 25 个。
- B：诊断起点本身已基本全对，四种 probe 没有可测的修复/损伤。
- C：只刷新题面或只重置结果行会造成大量节点损伤；同时刷新题面+重置结果、或按同一实际输入从头 scratch，修复 168–218、损伤 15。也就是说，C 的“局部刷新/单结果 reset”不够，整窗重解才有帮助。
- 这些是条件化描述，不是对“选择性检测错误”或“统一修复机制”的证明；成功和失败起点都保留。A 的 416/25 汇总需拆开：assignment 188/0、program 228/25；program 探针使用该 carry 分支的实际输入，25 次改坏不能直接等同于清空操作破坏局部算术能力，需与 conditional 结果并读。

## 6. 分种子看 C

| 指标 | C_s2026 | C_s42 | C_s1234 |
|---|---:|---:|---:|
| baseline old_clean terminal final /80 | 78 | 77 | 61 |
| baseline first4 qualified /80 | 56 | 61 | 25 |
| assignment common24 scratch s1/s4（正确任务数，各/24） | 24 / 24 | 24 / 24 | 23 / 24 |
| assignment common24 carry s1/s4 | 23/4 | 13/7 | 19/4 |
| assignment common24 reset s1/s4 | 6/0 | 9/0 | 11/0 |
| program scratch final /40 | 36 | 33 | 29 |
| program reset_results final /40 | 17 | 7 | 2 |
| program carry final /40 | 1 | 0 | 0 |
| invalid carry trajectories /40 | 18 | 12 | 24 |

- 三个 C 种子在 common24 scratch 上都接近满分；差异还见于 baseline 资格、carry/reset 的退化速度和长程序 scratch 成功率。
- C_s1234 的 baseline 最弱（61/80，资格 25/80），但 common24 scratch 仍 24/24、program scratch 29/40；不能只按 baseline 排名。
- C 有三个训练种子、每个一个权重；当前只报告这三个种子的分列结果，不据此断言所有连续模型的机制。

## 7. 结论与未区分解释

**可以说的：**

- 在共同 24 个基础任务上，A/B/C 都基本会做；C 的 all-80 起点资格低于 A/B，所以跨模型主表同时给 common24 与 all80。
- A/B 的连续更新能力不同：B carry 在赋值和长程序上大多成功；A carry 在 changed 输入阶段失败，回到原题靠旧状态恢复；两者 reset/scratch 大多成功。
- C 的边界很清楚：**按当前实际输入从空白重解（scratch）可以解决大多数赋值和 128–512 累计节点程序；保留旧结果（carry）或只重置结果行（reset_results）不能持续更新**。C 的长期 carry 还会产生 nonnumeric 提交，按失败保留。
- 固定同实际输入的 probe 支持当前窗口重解有帮助；长程序主分支本身不是同输入配对对照。不改题继续计算也会退化，当前证据尚不能分离保持不足、局部写入接口和具体 hidden 成分的贡献。
- 诊断上，C 的整窗刷新+reset 或同输入 scratch 有修复作用，单独 refresh task 或 reset results 往往损伤；这是条件化观察，不是统一机制。

**不能说的：**

- 不能说 A/B/C 的能力差异只由架构决定；A/B 与 C 的接口、reset 语义、训练数据和权重配置不同，C reset 不是 A/B 的 VALUE clear。
- 不能说 526 逻辑节点等于 526 个同时活跃槽位或原生整图能力；物理仍是 32 位置/256 tokens，且跨窗口有离散数值提交。
- 不能把 `carry` 的 nonnumeric 提交解释成“模型知道失败”或“选择性中止”；它只是解码结果不是合法数值后的协议阻断。
- 不能把 probe 的 repaired/damaged 当总体成功率或因果机制；诊断起点是固定子集，成功/失败都保留。
- 本批是开发任务、固定超参、单批；没有训练、没有第四步、没有层/头干预，也没有独立最终测试。

## 8. 产物与复现

- 原始 records、程序轨迹、快照和每任务日志在 `tmp/analysis/abc-workspace-20261119/{AB-main,C-main,C-smoke}/`，不提交。
- 合并小表：
  - `abc-summary.csv`、`abc-coverage.csv`、`abc-programs.csv`、`abc-probe-pairs.csv`、`abc-events.csv`、`abc-examples.jsonl`、`abc-meta.json`、`abc-report.md`；
  - `c-main-checks.md`、`c-smoke-checks.md`。
- 复导命令：
  ```bash
  python tools/abc_workspace/merge_lanes.py \
    --ab-dir tmp/analysis/abc-workspace-20261119/AB-main \
    --c-dir tmp/analysis/abc-workspace-20261119/C-main \
    --out-dir tmp/analysis/abc-workspace-20261119/ABC-summary
  ```
- 审计：`abc-meta.json` 记录 5 个模型、两个 lane 的 records/meta/代码/权重 hash；C 的独立 verifier 0 errors；`c-main-checks.md` 记录 C 的 nonnumeric/invalid 分母和 smoke 接口检查。
- 本轮未训练、未扩图/种子、未做第四步、未改模型结构；共享源码在运行期间未改。
