# C 基础能力与父子两点评估（首批）

日期：2026-11-15 20:05 CST。
状态：**四个现有权重的六档 clean 评估和 128 图父子两点评估已完成；评估 only，没有新训练**。
代码：`reproduction@7f25a42`（运行期间固定，未 pull/改配置）。
数据：六档沿用旧诊断集；128 图沿用 long-pilot 的新验证集（已查看过，不是独立最终测试）。
权重：initial=s2026/e143；hold/noise/semantic=腐蚀 pilot 的三个 epoch3 final checkpoint。

## 1. 执行与覆盖

- 7 项 stdlib/torch 测试通过；clean smoke 4 图/模型/档、pair smoke 4 图/模型、两者汇总均通过；正式运行无 NaN/OOM。
- clean：4 模型 × 6 档 × 64 图，batch4，每图到 D+64；共 1536 个 graph-model。
- pair：4 模型 × 128 图 × 2 时点，固定结构选一个直接子 q；8 分支、16 轮；共 1024 个 graph×timing 记录。
- pair 的 128 图全部有结构 q：没有 no_candidate；`value_qualified` 为 256/256 个 graph×timing（四模型共同）；common strict 为 185/256（initial 的 late 严格资格较低，hold/noise/semantic 均 256/256 个 arm×timing）。
- `sham` 分支与 unchanged 逐轮完全一致；`u_same/q_same/both_same` 的输入 VALUE 不变；u_wrong/q_wrong/both_wrong 的实际替换行和读出见 `c-basic-pair-dose.csv`。
- 六档仍是已查看诊断集；batch4/每图 D+64 与旧 clean 报告的 batch/终点不同，旧百分比只作背景，不混成精确逐节点配对。

## 2. 基本能力与长程保持

### D+64 节点 VALUE / first4（每档 64 图；括号为整图 VALUE/all 数）

| stratum | initial | hold | noise | semantic |
|---|---|---|---|---|
| N32_shallow | 0.925 / 0.033 (36) | 0.990 / 0.346 (60) | 0.995 / 0.283 (62) | 0.993 / 0.360 (61) |
| N32_deep | 0.909 / 0.025 (27) | 0.979 / 0.347 (55) | 0.971 / 0.271 (55) | 0.974 / 0.338 (55) |
| N64_shallow | 0.824 / 0.021 (2) | 0.987 / 0.366 (49) | 0.967 / 0.281 (48) | 0.966 / 0.364 (49) |
| N64_deep | 0.804 / 0.021 (0) | 0.981 / 0.331 (48) | 0.967 / 0.271 (44) | 0.970 / 0.332 (47) |
| N128_shallow | 0.716 / 0.020 (0) | 0.968 / 0.369 (23) | 0.965 / 0.301 (21) | 0.966 / 0.357 (19) |
| N128_deep | 0.599 / 0.012 (0) | 0.957 / 0.309 (23) | 0.946 / 0.252 (16) | 0.940 / 0.307 (14) |

- D 处四模型都接近饱和：initial VALUE 0.996–1.000、first4 0.996–1.000；hold/noise/semantic 的 VALUE 0.9999–1.000。因此这里的主差异在 D 之后。
- 三组后训练都把 D+64 VALUE 从 initial 的 0.599–0.925 提高到约 0.94–0.995；first4 从约 0.01–0.03 提高到 0.25–0.37。first4 的整图 all 在所有模型的 D+64 都是 0。
- 在本次六档上，hold 总体最强（D+64 first4 0.309–0.369）；noise 较低（0.252–0.301），semantic 与 hold 接近但并非所有档更高。单 seed，不作总体排序。
- D+64 角色均值：leaf 四模型均为 1.000；initial 的 internal/sink 为 0.832/0.700，hold 为 0.982/0.965，noise 为 0.976/0.950，semantic 为 0.974/0.951。逐深度/父数见 `c-basic-clean-depth-fanin.csv`。

### 完成与完成后稳定性（VALUE；每档 64 图）

| arm | completed range | failure-after-completion range | final-two graph range |
|---|---:|---:|---:|
| initial | 63–64/64 | 28–64 | 0–36 |
| hold | 64/64 | 4–41 | 23–60 |
| noise | 64/64 | 2–50 | 16–62 |
| semantic | 64/64 | 3–50 | 14–61 |

- 所有模型 first4 的 completed 为 63–64/64（initial）或 64/64（三组后训练），但 first4 的 final-two 整图全对在所有组、所有档均为 **0/64**。VALUE 的完成和保持改善不等于四属性保持。
- postD failure / completion 的分母按各档实际完成数；未完成图保留在曲线和分母中，没有用“没完成所以稳定”替代。
- 这些是条件于完成/全图的固定观察，不是永久稳定性；完整事件表见 `c-basic-clean-stability.csv`。

## 3. 父子两点：保留已有答案 vs 重新获得

### 2×2 第一轮

下表是 common VALUE 256 个 graph×timing 上的 q VALUE 正确数（q_wrong 是只把 q 换成错值；both_wrong 是 u、q 都换）。

| timing | arm | u_wrong q r1 | q_wrong q r1 / r2 / r4 / r16 | both_wrong q r1 / r2 / r4 / r16 |
|---|---|---:|---:|---:|
| hold | initial | 8/128 | 116 / 128 / 128 / 128 | 3 / 74 / 126 / 128 |
| hold | hold | 12/128 | 112 / 128 / 128 / 128 | 0 / 67 / 124 / 128 |
| hold | noise | 14/128 | 107 / 128 / 128 / 128 | 0 / 67 / 123 / 128 |
| hold | semantic | 34/128 | 97 / 128 / 128 / 128 | 0 / 61 / 124 / 128 |
| late | initial | 16/128 | 116 / 128 / 128 / 125 | 3 / 61 / 111 / 112 |
| late | hold | 12/128 | 110 / 128 / 128 / 128 | 1 / 69 / 124 / 128 |
| late | noise | 15/128 | 108 / 128 / 128 / 128 | 0 / 64 / 122 / 127 |
| late | semantic | 40/128 | 97 / 128 / 128 / 128 | 3 / 62 / 124 / 128 |

- `sham` 与 unchanged 完全一致；`u_same`、`q_same`、`both_same` 在目标/子节点上都无第一轮错误。覆盖和剂量分别见 `c-basic-pair-coverage.csv`、`c-basic-pair-dose.csv`。
- `q_wrong` 虽然在输入中把 q 置错，第一轮/第二轮已经大量恢复（q_wrong q r1 97–116，r2 128）；这不是因为 q 解码值被正确回灌，而是下一轮状态/recurrence 的行为结果。`both_wrong` 的 q 第一轮 0–3，r2 61–74；hold/训练组到 r4 约 123–126、r16 128，initial 的 late 到 r4 111、r16 112。正确上游 u 对早期 q 恢复有作用（q_wrong 比 both_wrong 早），但到 r4 多数已接近；不能把它写成“必须由 u 重建”或“q 是独立计算头”。
- `u_wrong` 时 q 第一轮只有 8–40/128，r2 71–84，r4 111–126，r16 112–128；u 本身 r1 91–109，r2 126–128，r4 128。u 错误的传播/恢复和 q 的错误并不同步。
- semantic 组在 u_wrong 的 q 第一轮 34/128（hold 12/128）并不更好；q_wrong 的 q 第一轮 97/128（hold 112/128）也更低。它的下游损伤总量（见事件表）在 u_wrong 下较少，但目标 u 本身更易被注入打错；这两点不能合成“恢复更好”。
- 目标 u 在 u_wrong 和 both_wrong 的 r1 相同（例如 hold 105，semantic 91），说明 q 注入没有反向改变 u；符合结构方向。

### 事件与末两轮

- q_wrong：四模型的 q damaged=128/128；hold/noise/semantic 和 initial 的 hold 时点
  final-two among damaged 都是 128，但 initial 的 late 时点是 125/128；恢复时间差异
  比最终是否恢复更明显。
- both_wrong：q damaged=128/128；hold 两个时点和 semantic late 的 final-two among
  damaged 为 128，noise late 为 127/128，initial late 为 112/128。不能把 r1 的
  0–3/128 当成持续失败，也不能说四个模型最终都一样。
- u_wrong：q damaged 约为 100–123/128（semantic 100–106、initial 116–123、
  hold 122、noise 118–120），final-two among damaged 也落在这个范围；没有全部恢复。
  但 u_wrong 的 descendants 仍有部分残留，且 semantic 的 damaged 数较少、excess
  wrong rounds 略低（见 events CSV）。这是传播范围的差异，不是“错误检测机制”证明。
- 完整按 cohort（all_fixed/common_value/common_strict）、u/q/other_children/descendants/unrelated/all、value/first4、post1/2/4/8/16 的表在 `c-basic-pair-accuracy.csv` 与 `c-basic-pair-events.csv`。正文仅列 common VALUE 主视图。

### 两个完整例子

实际 u/q 立即式（完整展开与局部预期见 `c-basic-pair-example-expressions.json`）：

- graph89：`u=x1 = x60(21)+x64(16) mod23=14`；`q=x99 = x76(12)+x78(10)+x1(14) mod23=13`。delta_u=8，局部 q 期望为 21。
- graph68：`u=x105 = x57(14)+x84(15) mod23=6`；`q=x79 = x105(6)+x71(11) mod23=17`。delta_u=17，局部 q 期望为 11。

来自 `c-basic-pair-example-values.csv` 的 late、semantic 权重实际值：

| graph | branch | u 输入→r1→r2/r4 | q 输入→r1→r2/r4→r16 |
|---|---|---|---|
| graph89 | u_wrong | 22→20→14 | 13→13→13→13 |
| graph89 | q_wrong | 14→14 | 21→7→13→13 |
| graph89 | both_wrong | 22→20→14 | 21→1→13→13 |
| graph68 | u_wrong | 0→6 | 17→17→11→17 |
| graph68 | q_wrong | 6→6 | 11→11→17→17 |
| graph68 | both_wrong | 0→6 | 11→11→11→17 |

- graph89 中 q_wrong 和 both_wrong 都能在 r2 回到 13；u_wrong 时 q 保持 13（不同题/不同注入，不能把这一例当成普遍“q 免疫”）。
- graph68 中 q_wrong 在 r2 回到 17，both_wrong 在 r2/r4 仍为 11、r16 才回到 17；这是逐题可核对的恢复延迟例子，不是总体率。
- 两个例子均保留 clean/unchanged 对照（表中输入/输出均为实际解码数值；没有把解码值回灌）。

## 4. 现有 corruption records 的 CPU 联合分析

`c-basic-existing-joint-summary.csv` 和 `c-basic-existing-paired-models.csv` 由已有 128 图×两时点 corruption records 读取，不重跑模型：

- target 输入/输出改变、child_harmed、expected_q 等在同一题上联合统计，而不是把 hold/semantic 的边际总数拼在一起。
- `paired_models.csv` 保留 hold 与 semantic 的逐题/逐轮配对；`semantic_target_wrong_and_less_child_harm` 等列可直接定位“目标仍错但下游损伤较少”的案例。
- 该表是观察性读出/功能一致性，不证明下一轮实际使用了某个解码值，也不替代新的 u/q 受控替换。

## 5. 解释与边界

- 本批是**评估**，没有新训练、没有新种子、没有真实改题、持续污染、层/头干预或祖先局部自洽矩阵。
- 六档和 128 图都已被查看过，不能称最终独立测试；单 seed，不能做总体模型排名。
- 解码头读出与下一轮 Transformer 使用的 hidden 不是同一事实。`q_matches_input_parent_readouts` 仅是行为一致性检查；q_wrong 后的恢复不等于内部一定读取了正确 q 或 u 的隐藏表示。
- `both_wrong` 与 `q_wrong` 的差异只能说明正确 u 对早期 q 恢复有帮助；不能单独证明内部检测、选择性重算或独立 VALUE 通道。
- 未完成图没有被稳定率掩盖；完成时间中位数和 post-completion 事件分别使用条件分母。
- 继承的 `reduce_batch` factor-PAD 裁剪缺口未修；clean 评估 mask 排除 PAD。

## 6. 产物与审计

- 原始运行：`tmp/analysis/c-evaluation-suite-20261115/{clean,pair,existing-joint}/`
- 提交小表：
  - `c-basic-clean-accuracy.csv`、`c-basic-clean-curves.csv`、
    `c-basic-clean-depth-fanin.csv`、`c-basic-clean-stability.csv`
  - `c-basic-pair-accuracy.csv`、`c-basic-pair-events.csv`、
    `c-basic-pair-coverage.csv`、`c-basic-pair-dose.csv`
  - `c-basic-pair-example-expressions.json`、`c-basic-pair-example-values.csv`
  - `c-basic-existing-joint-summary.csv`、`c-basic-existing-paired-models.csv`
  - `c-basic-evaluation-audit.json`
- `c-basic-evaluation-audit.json` 记录四个模型与 source/data/code 哈希、clean/pair meta、每个提交表
  的 source SHA256/export SHA256 和过滤条件；未把源大表哈希冒充为过滤后的提交表哈希。
- 完整 records/NPZ/权重/13MB 级 joint_trajectories 留在 tmp，不提交。GPU2 已释放。
