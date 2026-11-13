# 递归潜空间推理论文复现：阶段汇报（2026-11-11）

> 时间：2026-11-11 12:20 CST
> 仓库：`/workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures`（branch `repro`）
> 论文：*Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning*（arXiv 2510.14095v1）
>
> **文档关系**：
> - 本报告 = 汇报版：按“问题 → 设计 → 证据 → 分析 → 决策”组织，覆盖 A′/B 纠错、旧 C0、连续向量纠错尝试、新 C0、B′ 方案。
> - `tmp/records/2026-11-11/c0-mixedn-full-context.md` = 数据附录：所有原始表格、全条件注错表、旧 C0 完整诊断、新 C0 全轨迹。
> - 原始专题记录在文末“9. 资料索引”。
>
> **更正（2026-11-13）**：本报告中依赖 teacher-state v2 注错和 parent 2×2 的数字
> （尤其“B u 恢复 97.8–100%”“R-only 91.3%”“88.6/99.6/98.4%”）已被两个脚本 bug
> 修复取代：v2 teacher rollout 用绝对轮次当下标；parent 2×2 把 `st[Rg]/st[Md]`
> 当列索引用、整行覆盖 token 4/5。修复与重跑见
> `tmp/records/2026-11-13/code-review-fixes-20261113.md`；新结果在
> `tmp/analysis/inject-v2-fixed/` 与 `tmp/analysis/parent-2x2-fixed/`。
> own-state v2 与旧 47 候选注入不受影响。

---

## 0. 摘要（一页）

### 0.1 我们在做什么

复现论文的两条路线：

1. **离散线（A′ vs B）**：验证 recorrection（纠错训练）是否让模型在推理时真正“发现并改回”错误，而不是靠 raw/clean 指标或 aux 回灌的代理。已完成 16 档/deep_v2、单点 VALUE 注入、parent-corruption 2×2、aux/margin 分析。
2. **连续线（C0 / M4）**：复现论文的连续 hidden 基线，诊断它的失败模式，再决定 C+（连续 + 纠错）怎么做。旧 C0（N32/depth6）已完整训练+诊断；新 C0（mixed N/depth12）正在训练。

### 0.2 当前最重要的结论（附置信度）

| # | 结论 | 证据 | 置信度 |
|---|---|---|---|
| 1 | **raw 是评估 bug，不是鲁棒性/纠错指标** | 训练强制 aux=FALSE，raw 回灌预测的 aux；A-L4fast 无纠错但 aux 翻转 38%/46%、raw 仍 100%；margin 尾部解释 | 高 |
| 2 | **B 的“恢复”是真实行为，但不能证明“检测+改回”** | A′ 在所有 256 图条件 u 恢复 0%；B 97.8–100%；但 parent 2×2 的 R-only 下 B 91.3% 跟随错父值 → 更像父节点重算 | 行为高；机制中 |
| 3 | **4→5 注入与训练腐蚀时机不同** | 训练 `last` 掩码腐蚀 depth==i 的新位置（输入 EMPTY）；4→5 是已算好旧位置 → 跨时机泛化 | 高（代码可证） |
| 4 | **旧 C0 三个 seed 是三种失败** | s42 变量名漂移（VALUE 254/256 全对）；s1234 真 VALUE 错并传播；s2025 坏 seed（5701/8192 首次错再没对） | 高 |
| 5 | **旧 C0 的泛化差不能全归因于架构** | 训练只有 N32/depth6；old N32 是 depth10、new N32 是 depth11、N128 更深、deep_v2 是宽/深压力 | 高（数据分布事实） |
| 6 | **C4 自然 hidden 替换能短暂影响下游** | post1 q→q_u5：s42 77.4%、s1234 73.6%；随机置换/噪声 0–3%；post2/4 回原值 | 高（行为）；机制待 4→4 对照 |
| 7 | **新 C0 在好/坏盆地之间反复跳，但 3/4 seed 已明显收敛** | 四 seed 都到 val 12.7–15.5 又反弹 40–78；12:20 s42 e39 **2.84**、s1234 e43 14.4、s2025 e43 12.7；s7 仍 ~20 | 高（现象）；原因待定 |
| 8 | **新 C0 的震荡不是全新故障** | 旧 C0 也有早期尖峰和 s2025 长期高 loss；depth12/13 loops 放大 | 中高 |

### 0.3 已做出的关键决策

1. **raw 不再作为任何主口径**（2026-11-11 11:10，用户拍板）：主结果只用 clean；raw 只作为“论文原版评估为何不稳定”的 bug 案例。
2. **B 的机制表述降级**：不写“B 学会了纠错/检测错误”，改写“行为上会恢复 / 从父值重算 / 条件重算待定”。
3. **旧 C0 的评估必须用 matched-depth 分离“没训好”与“训好了但 OOD”**。
4. **新 C0 用 controlled mixed-N 数据**：第一版 bucket 生成器深度偏斜，弃用；最终 `ControlledDepthDAG` 精确 N×depth 配额。
5. **新 C0 先跑 50 epochs、4 seeds**，用 top-k best checkpoint 评估；是否延长到 100 epochs 看 e50 状态。
6. **C+ 暂不启动**：先等 C4 的 4→4 对照和自然错误采集，确认有“可重复、影响下游、且 C0 还有恢复空间”的错误现场。
7. **B′ 方案（看板新增）**：见 §6；共享看板/records/mailbox/handoff 没有 B′ 字面条目（另一 session 独立确认）。当前按设计空间处理，**B′-1/B′-2 均为 proposed / unstarted**；其中 B′-1 = 只把 `corruption_strategy` 从 `last` 改为 `random`，B′-2 = 显式 old-only/mixed mask。待用户确认后收敛。

### 0.4 当前状态（15:36，训练完成）

- **新 C0 四个 seed 全部跑完 50 epochs**（s1234/s2025 14:23；s7 ~15:25；s42 15:36）；无 OOM/Traceback。
- **最终结果**：
  - s42 e49：train 2.41 / **val 2.01**；best **e48 1.96**；
  - s1234 e49：train 7.82 / **val 5.89**；best e49 5.89；
  - s2025 e49：train 15.8 / **val 10.30**；best e49 10.30；
  - s7 e49：train 18.7 / **val 15.60**；best 仍是 **e26 12.7**（e46 35.8、e47 16.2、e48 21.4、e49 15.6，晚段仍震荡）。
- best checkpoint：
  - s42 **e48 1.96**（top-k e47 2.07、e48 1.96、e49 2.01）；另有 e0 14.5；
  - s7 **e26 12.7**（top-k e23 12.8、e26 12.7、e45 12.9）；
  - s1234 **e49 5.89**（top-k e47/e48/e49）；
  - s2025 **e49 10.30**（top-k e41/e44/e49）。
- 待办：新 C0 的 best-checkpoint 评估（mixed held-out / matched-depth / old N32/N128/deep_v2）、EMPTY→5 vs 4→5 时机对照、B-42 服务器版本核对、parent 2×2 叶父/扫描强化、C4 4→4 donor、B′ 设计确认。

---

## 1. 研究问题、任务与术语

### 1.1 论文主张与我们的拆解

论文的核心主张可以拆成三层：

1. **离散潜空间 + 中间监督 → 规模和深度 OOD 泛化**（M5/A 线）；
2. **recorrection（M6/B 线）→ 对中间状态错误的自纠错/鲁棒性**；
3. **连续 hidden（M4/C0）是否也能获得同样的泛化**。

我们的实验分别对应：

- A′ vs B-M6：同一架构/数据，只差 recorrection 开关 → 隔离主张 2；
- A′ 的 16 档/deep_v2：检验主张 1；
- C0 的 16 档/deep_v2/matched-depth：检验主张 3，并诊断失败模式；
- 单点 VALUE 注入、parent 2×2、C4：直接测“错误发生后是否恢复/传播”。

### 1.2 任务与 token 表示

- 随机算术 DAG：6 个叶子，N 个节点（内部节点 N−6），fan-in 1–3，运算只有 ADD mod 23。
- 每个 token 6 个因子：
  - SYNTAX(7)、VARIABLE(129)、OPERATION(2)、VALUE(25)、REGRET(2)、MODIFICATION(2)。
- 每轮（loop）对整张图的 token 序列做一次 forward；第 d 轮可以算出 DAG 中 depth=d 的节点。
- 严格指标 `final_label_full_acc`：所有非 PAD 位置、前 4 个主因子在最后一轮全部正确；一个 VARIABLE token 错，整张图就算错。

### 1.3 模型清单

| 名称 | 架构 | 反馈/训练路径 | 纠错训练 | 训练种子 / checkpoint |
|---|---|---|---|---|
| A-demo | L2H16D256 | 慢路径 | 无 | 作者权重（seed 1584791061） |
| A-M5 | L2H16D256 | 慢路径 | 无 | 2025 / 1234 / 42 |
| A-L4fast | L4H16D256 | fast | 无 | 42，e99 |
| A′ | L4H16D384 | fast，离散 token 反馈 | 无 | 42 e86、1234 e90、2025 e94 |
| B-M6 | L4H16D384 | fast，离散 token + teacher predictions | **有**（VALUE 腐蚀） | 42 e92 |
| cyhB | L4H16D384 | 同 B-M6 | 有（迁移包，未训满） | 42 e84、1234 e88、2025 e74 |
| C0 / M4 | L4H16D384 | nointerm，连续 hidden | 无 | 旧：42/1234/2025 e94/e96/e99；新：4 seed 训练中 |
| C+ | 连续 + 纠错 | 未启动 | 计划中 | — |
| B′ | B 的改进版 | 看板新增，待确认 | 待定 | — |

### 1.4 指标与协议

- **clean**：自由续算回灌前把 REGRET/MODIFICATION 强制 FALSE（与训练一致）——**唯一主口径**。
- **raw**：回灌模型预测的所有 6 因子——已知 bug 口径，仅用于解释论文原版评估的不稳定。
- **tf 单步**：每轮喂正确中间状态，测“一步转移算子”。
- **自由续算（fr）**：每轮喂模型自己的输出，测真实推理。
- **matched-depth**：N=32、max_depth=6、1024 图、seed42，与旧 C0 训练分布一致，用于分离“没训好”与“OOD”。
- **K4 / Kall**：每图按深度分层 4 个 u / 全部 eligible u。
- **a→a+1 / 4→5**：注入 VALUE 的两种方式；见 §2.5 的时机区别。

### 1.5 命名规范

离散：`D_{state}_{model}_{seed}_{data}_{K}_{value}`，例如 `D_tchr_Aprime_s42_N32_K4_ap1`。  
连续：`C_{kind}_C0_{seed}_{data}_{detail}`，例如 `C_traj_C0_s42_N32old_256`。  
完整对应表见 `tmp/records/2026-11-10/experiment-registry.md`。

---

## 2. A′ vs B：纠错训练的设计、泛化与机制

### 2.1 训练设计

#### B-M6 的实现（代码事实）

- 架构：L4H16D384，fast 路径（teacher predictions 并行算所有 loop）。
- 训练数据：N=32、max_depth=6、max_length=256、50,000 图；ADD mod 23；fan-in 1–3。
- **recorrection**：训练时以 5% 概率对 batch 做 VALUE 腐蚀：
  - `corruption_strategy='last'`；
  - 掩码 `~value_avail_pos[i-1] & value_avail_pos[i]` = **depth==i 的新可算位置**（输入 EMPTY，target 是本轮应算出的正确值）；
  - tokenwise 腐蚀概率 0.5，随机 VALUE；
  - aux 因子（REGRET/MODIFICATION）损失权重 0，回灌前强制 FALSE。
- B-M6 的 checkpoint：seed42 e92（best val 3.25e-6）。

#### A′ 的对照设计

- 同架构、同数据、同 fast 路径、同训练步数，**只把 recorrection 关闭**。
- 三个 seed（42/1234/2025），最高 top-k e86/e90/e94。
- 这是“纠错开关”的干净消融；A-L4fast（D256）只用于分离架构/宽度，不作为 D384 的最终对照。
- cyhB 三份权重（同配置、未训满）作为外部复现。

#### 为什么这样设计

- 如果直接用 A-M5（L2D256）vs B-M6（L4D384），会把“层数/宽度”与“纠错”混在一起；A′ 与 B-M6 同架构，隔离更干净。
- B-M6 的腐蚀分布（新可算位置）是它训练时真正见过的错误；我们的注错实验做的是“已算好位置”，这两者必须分开（见 §2.5）。

**决策依据**：用 A′ vs B-M6 作为主对照；cyhB 作为跨 seed/权重复现；A-L4fast 作为架构对照。  
**代价/风险**：B-M6 只有单 seed；cyhB 的 checkpoint 比 B-M6 早，报告需注明 epoch。

### 2.2 泛化：raw/clean 与压力池

#### raw/clean 的 bug（已定论）

训练时 REGRET/MODIFICATION 被强制 FALSE；raw eval 却把模型预测的这两个未训练通道原样回灌。A 线三个 seed 在同架构/同数据/同训练步数下，raw N128 从 0% 到 95.1%——差异主要在 aux 通道噪声是否碰巧有害，不在算术能力。

决定性对照：同一 checkpoint，raw vs clean，N=128：

| checkpoint | raw | clean |
|---|---:|---:|
| A-s1234 | 10.6% | **100.0%** |
| A-s42 (e95) | 0% | **100.0%** |
| A-s2025 | 89.8% | 99.6% |
| A-demo | 95.7% | 98.8% |

#### 16 档结果（clean 为主）

- A-demo：clean 99.2–100%，raw 95.1–100%。
- A-M5：clean 99.8–100%，raw 0–99.8%（seed 方差极大）。
- A-L4fast（无纠错）：clean 100%，raw 100%。
- B-M6：clean/raw 全 100%。
- A′：clean 0.9995–1.0，raw 0.9954–0.9998。
- cyhB：clean 99.8–100%；raw 随 checkpoint/规模下降（s2025 N128 21.6%）。

#### 压力池

| 权重 | wide8 | mid60 | chain110 | chain122 | wide8 tf 单步 |
|---|---:|---:|---:|---:|---:|
| A-demo (L2) | **82.7** | 98.9 | 98.7 | 98.3 | **0.592** |
| A-s2025 (L2) | 99.7 | 99.6 | 99.7 | 99.7 | 0.990 |
| A-L4fast (无纠错) | 100 | 100 | 100 | 100 | 1.000 |
| B-M6 | 100 | 100 | 100 | 100 | 1.000 |
| cyhB s2025 (raw) | 36.0 | 8.7 | 5.7 | 4.8 | 1.000 |

#### 分析

1. **raw 不是鲁棒性/纠错指标**：A-L4fast 无纠错、aux 翻转 38%/46%，raw 仍 100%；B-M6 的 aux 翻转 27%/66%，raw 也 100%。raw 满分的机制是 **argmax 饱和**：aux 噪声让 logits 动，但 VALUE margin 均值 28–35、最小 4.9–7.5，翻不动 argmax；A-M5 s42 平均 margin 43.5 但尾部 0.20 → 3008 次翻转 → raw 0%。
2. **clean 下所有强模型都接近满分**：论文图4 的 M5→M6 差距，大半是评估噪声 + 容量混淆（A-demo L2 在 wide8 tf 只有 0.592，L4 架构修到 1.000）。
3. **宽度轴是 A 的真实短板**：论文从未测宽度；B 的优势在 L4 架构的检索容量，不一定是纠错训练。
4. **raw 与 checkpoint 尾部相关**：B e91 raw 86.2 vs e92 raw 100，clean 都 100——raw 随 margin 尾部波动，不是训练后期“稳定性”学习。

**决策依据**：2026-11-11 用户拍板 raw 是 bug，主结果只用 clean；raw 只作为 bug 案例保留。  
**影响**：不能再拿 raw 数字证明“纠错”；纠错证据只来自注错实验（§2.3/2.4）。

### 2.3 单点 VALUE 注入：A′ vs B-M6 vs cyhB

#### 设计

- 在已算对的节点 u（定义位置 p、深度 d、真值 a 或 4）注入 VALUE：
  - 主协议 a→(a+1) mod 23；
  - 4→5 只选值为 4 的 u。
- 起跑：
  - teacher-state：从 clean teacher 快照 `S[d+1]` 注入；
  - own-state：从模型自己自由续算到 u 算出的状态注入，只保留两边 u 都算对的图。
- 规模：N=32 / N=128；K4 / Kall；256 图/档（新留出 seed42）。
- free-run：无 teacher、无 recorrection、aux 强制 FALSE；记录 u 的 r1/r2/r4 和直接子节点 q 的 r1/r2/r4。
- clean 对照：不注入；A′ 三 seed 和 B-M6 均 47/47 通过（旧 64 样本集）。

#### 主结果（256 图条件，u 恢复率）

| 模型 | teacher-state | own-state | 备注意义 |
|---|---:|---:|---|
| A′ s42/s1234/s2025 | **0%** | **0%** | 所有 N32/N128、K4/Kall、a→a+1 条件 |
| B-M6 s42 | **100%** | **100%** | 所有 N32/N128、K4/Kall、a→a+1 条件 |
| cyhB s42 | 97.75% | 97.75% | 唯一略弱 |
| cyhB s1234 | 99.80–99.90% | 99.80–99.90% | — |
| cyhB s2025 | 100% | 100% | — |

4→5 own-state：A′ s42 0%；B-M6 100%。  
q 的行为：

- teacher-state B-M6：q r1 直接 correct；
- own-state B-M6：q r1 大量 `no_value`（q 还没算）或 `expected_wrong`；r2/r4 基本 correct；
- A′：q r1/r4 全是 `expected_wrong`（或 other_wrong）。

#### 同轮/延迟结构

- B-M6 同轮（dq=d+1）41 例：**推迟一轮再算 23/41**（第 0 轮 q EMPTY，第 1 轮直接算对）；**先算错再改 18/41**。
- q **从未**在注入发生的同一 forward 内直接算对 → 纠正发生在**轮边界**，不是单次 forward 内部。

#### 分析

1. **行为差异极大且可复现**：A′ 0% vs B 97.8–100%，跨 seed、N32/N128、K4/Kall、teacher/own、a→a+1/4→5 都成立。
2. **但这不是“检测+改回”的直接证据**：4→5 是“已算好旧位置”的腐蚀，而 B 的训练腐蚀是“新可算位置（EMPTY）”；B 在 4→5 上的恢复是**跨时机泛化**。
3. **同轮结构**说明 B 的纠正在轮边界发生；这与离散状态只在轮边界传递的架构一致。
4. **A′ 的少量“改回”是读出波动**：A′ s42 u 第一步 1/47、s1234 5/47、s2025 0/47；三 seed 合计 6/141=4.3%，q 最终 1/141=0.7%；集中在 d=0 叶节点，且 5/6 不能传播到下游。

**决策依据**：扩到 256 图/档、teacher+own、N32/N128、K4/Kall、a→a+1+4→5；所有条件用 clean；主对照 A′-42 vs B-42，cyhB 做复现。  
**遗留**：own-state Kall 全量（已停，待单线程重跑）；时间/EMPTY→5 对照；B-42 服务器版本核对。

### 2.4 Parent-corruption 2×2：机制确认

#### 设计

- u 已算对，真值 a；r 是 u 的一个父节点，原值 r0；选 r' 使“用 r' 重算 u”得到可追踪的第三值（a_recomp = a+3）。
- 四格：clean / U only（u→a+1）/ R only（r→r'）/ U+R。
- 关键条件 **R only**：u 本身正确，父节点错。若模型从父节点重算，u 会变成 a_recomp；若保留存储值，u 仍是 a。

#### 结果（teacher-state，N32，256 图，K4）

| 模型 | R only 下 u 的状态（post1） |
|---|---|
| B-M6 s42 | **recomp 935/1024（91.3%）** |
| A′-42 | true 1024/1024 |
| cyhB s42 | recomp 907/1024（88.6%） |
| cyhB s1234 | recomp 1020/1024（99.6%） |
| cyhB s2025 | recomp 1008/1024（98.4%） |

- A′ 保持 u 原值，完全忽略父节点；
- B 在“u 正确、父节点错”时仍把 u 改成按错父值重算的值；
- own-state 复现 teacher-state；
- 父节点 r 没修好时，u 基本一直跟着错（B-M6：51/51、35/37、39/40）。

#### 分析（重要：不要过度概括）

2×2 **不能直接证明“recorrect”**；它测的是 u 的输出对“存储值 vs 当前父值”的敏感性：

- **R=7（recomp）**：u 的输出依赖当前父值 → 这次 forward 按父值重算；不能推出“每轮无条件重算”，但至少不是“保留存储值”。
- **R=4**：u 没有用当前坏父值；但有歧义：
  - (a) 直接保留存储值（copy）；
  - (b) 同一 forward 内先把 r 修回 r0、再用 r0 算 u（repair-first）；L4 层间传播使 (b) 不能忽略。
- U-only 4：只能说明“没有沿用坏值”，不能说明是条件纠正还是无条件重算。
- U+R 的冲突结果：7=父值赢，5=存储值赢，4=原值来源异常（记忆/同值 token/实验错误）。

**当前可写的结论**：B 的旧节点输出依赖当前父值，恢复更像“按父值重算”，不是“检测到错误后改回”。  
**不能写的结论**：B 学会了纠错/检测错误。

**决策依据**：采用 handoff `2026-11-10-1116-claudeA-parent-corruption-2x2-epistemics.md` 的判读边界；后续用**叶父（depth0）R-only**（无父可修，最干净）、扫描多个 r'、非父节点对照、r 逐轮轨迹、L1/L2 vs L4、EMPTY→5 vs 4→5 分开报告。  
**当前状态**：2×2 已完成主结果；强化实验未做。

### 2.5 训练时机的区别：EMPTY→5 vs 4→5（关键 confound）

这是目前 A′/B 线最容易被误读的一点，必须单独写清楚。

#### 训练腐蚀的时机（代码事实）

`randomly_corrupt_values`（`experiments/model.py:420-444`）在 `last` 策略下：

```python
value_avail_pos[i] = (depth <= i)   # loop i 的 target 里，这个位置的值已可用
candidates = concat(
    value_avail_pos[0],
    ~value_avail_pos[:-1] & value_avail_pos[1:]
)
```

对 loop i，`~avail[i-1] & avail[i]` = **depth == i** 的位置：

- 在 loop i 的**输入**里，它们还是 EMPTY（上一轮还不可用）；
- 在 loop i 的 **target** 里，它们才是这一轮要算出的正确值；
- 腐蚀是把输入里的 VALUE 改成随机值，target 仍是正确值。

→ 训练教的是：**“这个新位置马上要算，输入里的 VALUE 不可信，从父节点算。”**

#### 我们的注错时机

u 深度 d，在第 d 轮已经算对；在 `S[d+1]` 里 u 的 VALUE 是 4；我们改成 5。

→ 这是 **“已算好的旧位置被改成错值”**，训练中从未出现过；对 B 来说是**跨时机泛化**。

#### 为什么重要

- 如果 B 的 4→5 恢复只是“每轮从父节点重算”，那它不是“检测到存储值错”。
- 如果 B 在 **EMPTY→5**（训练分布内）恢复率也很高，而 4→5 也高，说明它对两种时机都重算/鲁棒；
- 如果 EMPTY→5 高、4→5 低，说明 B 学到的是“只对 frontier 不信任”，4→5 是 OOD；
- 如果两者都低，说明之前的 100% 另有原因（需要复查）。

#### 尚未做

- 同一节点的 EMPTY→5 vs 4→5 对照；
- B-42 服务器训练版本核对（本地 `last` 掩码是否与服务器一致）；
- 4→4 sham（写回正确值，应等价 clean）；
- 多节点/持续上游污染。

**决策依据**：在结果出来前，所有 4→5 的结论都标注“跨时机泛化”；不要把 4→5 的 100% 写成“训练学到的纠错”。  
**下一步优先级**：EMPTY→5 vs 4→5 > 叶父 R-only > 4→4 sham > 服务器版本核对。

### 2.6 B′（看板新增想法）

> **状态说明**：我在共享看板 `tmp/ai-collab/shared/board.md`、`state.md`、`log.md`、所有 records/handoff/mailbox 中没有找到 “B′/B撇/B-prime” 的原文；看板最新条目是 2026-11-11 11:10 的 raw bug 决策。另一 Claude session（yhy-69）也独立搜索了 board/records/mailbox/handoff，确认没有 B′ 字面条目；它认为最接近的实际内容是 **A′/B-M6** 和 **C+**（计划中，未开始）。因此 B′ 可能是用户/另一个 AI 的命名，或就是 C+。下面不冒充“原文”，而是把 B′ 的**设计空间**列出来，每个方案给出分析/依据；用户确认是哪一条后，我再收敛成正式方案。
>
> 如果看板的 B′ 就是 **C+**（连续 + 纠错），那么 §4.4 已经是它的完整定义（从同一 C0 checkpoint 出发，等图/等轮/等步数，普通续训 vs 纠错续训）；当前状态是**未开始**。

#### 候选 B′-1：把 `corruption_strategy` 从 `last` 改为 `random`（精确提议，未实现）

> 以下定义来自另一 Claude session（yhy-69）的复核；仍属于 **proposed / unstarted**，没有 config、run、checkpoint。

- **定义**：架构、数据、优化器与 B-M6 **完全相同**；只把 `recorrection_args.corruption_strategy` 从 `last` 改为 `random`，保持 `corruption_prob=0.05`、`tokenwise_corruption_prob=0.5`。
- **代码语义**：`randomly_corrupt_values` 在非 `last` 分支把 `candidates` 设为全部 `value_avail_pos`，即每轮 `depth <= i` 的所有已可算位置（含旧位置），再按 tokenwise 概率腐蚀；`last` 分支只取 `~value_avail_pos[i-1] & value_avail_pos[i]`（新可算位置）。
- **与 B-M6 的区别**：B-M6 只腐蚀 frontier（输入 EMPTY 的新位置）；B′-1 腐蚀所有已可用位置（含已算好的旧位置）。训练/推理时机更接近 4→5 注错，但**不只是时机改变**。
- **两个 caveat（必须写进设计）**：
  1. `random` 不只是把 old/new 时机混合，还**改变了每轮被腐蚀位置的数量/分布**（候选集更大：从 frontier 变成 depth<=i 的全部位置）。因此 B′-1 的效应不能全部归因于“旧位置腐蚀”；若要干净分离，需要 **B′-2 = 显式 old-only 或 old/new 混合比例的 mask**。
  2. 训练必须 **from scratch、同 seed / 同 100 epochs / 同数据**，不要 resume B-M6；resume 测的是“适配新 mask”，不是“训练分布里是否包含旧位置”。
- **想验证**：
  1. B′-1 在 U-only（4→5）上是否仍 100% 恢复；
  2. **R-only（u 正确、父错）**：B′-1 是保留 u=4（条件纠正）还是跟随父值 a_recomp（无条件重算）？
  3. U+R 冲突时信存储值还是父值；
  4. EMPTY→5 vs 4→5 的差异是否消失。
- **预期对照**：A′（无纠错）、B-M6（last/new-only）、B′-1（random/all）、B′-2（old-only/mixed）；同 seed/同数据/同评估；clean + EMPTY→5/4→5 分开 + parent 2×2。
- **分析/依据**：当前 B 的 4→5 恢复是跨时机；R-only 91.3% 跟随父值说明“无条件重算”很可疑。如果 B′-1 在 R-only 下保留 u，说明**条件纠错可以被训练出来**；如果仍跟随父值，说明 B 的行为是架构/训练目标的**无条件重算**，不是可学的条件检测。
- **成本/风险**：一次 B′-1 训练（100 epochs N32）约数小时–1 天；`random` 改变候选集大小/分布，存在 confound；建议同时准备 B′-2 做归因。
- **状态**：**proposed / unstarted**；没有 config/run/checkpoint。

#### 候选 B′-2：显式 old-only / old+new 混合 mask（用于干净归因）

- **定义**：B-M6 架构 + recorrection；新增腐蚀 mask：
  - `old`：只选 `depth < i` 的已算好位置；
  - `mix`：old+new 按比例（例如 50/50）混合。
- **与 B-M6/B′-1 的区别**：B-M6 只 new；B′-1 是 all（数量/分布也变）；B′-2 只改时机、可控制候选集比例，能干净回答“旧位置腐蚀是否让模型学会条件纠错”。
- **训练/对照**：同 B-M6 数据/超参/100 epochs/from scratch；对照 A′、B-M6、B′-1、B′-2；评估同上。
- **状态**：proposed / unstarted；需要改 `randomly_corrupt_values` 增加 old/mix 策略。

#### 候选 B′-3：B 在 mixed-N/depth12 数据上重训（新 C0 的离散对照）

- **定义**：B-M6 架构 + recorrection，但训练数据换成新 C0 的 `C0-MixedN8-32-D12-ADD`（100k、N8–32、depth1–12、13 loops）。
- **与 B-M6 的区别**：训练分布从 N32/depth6 变成 broad N/depth12。
- **想验证**：离散 B 是否也能从 broad 训练分布获益；与 new C0 在同数据上比较，分离“离散架构”与“训练分布”。
- **分析/依据**：旧 C0 的失败被训练分布（N32/depth6）严重混淆；new C0 是修正这个 confound 的连续线。若 B′-3 在 same data 上明显优于 new C0，说明离散瓶颈/架构是关键；若两者都提升，说明分布是主因。
- **成本/风险**：需要 13 loops、100k、4 seeds；比 B-M6 贵；且与时机问题混在一起（除非同时固定 B-M6 的 last/random 策略）。
- **状态**：proposed / unstarted；如果看板的 B′ 是“离散线也上新数据”，选这个。

#### 候选 B′-4：把 aux（REGRET/MODIFICATION）训练成显式纠错信号

- **定义**：B-M6 + 监督 aux 通道：当输入 VALUE 与父节点重算值不一致时 REGRET=TRUE；当模型改写 VALUE 时 MODIFICATION=TRUE；损失权重从 0 改为 >0。
- **与 B-M6 的区别**：把“设计遗迹”变成可训练/可读出的纠错信号；raw 不再是 bug（因为 aux 被训练），但需要重新定义评估协议。
- **想验证**：显式纠错信号是否提升恢复率/泛化；REGRET/MODIFICATION 是否能作为“检测到错误”的可观测代理。
- **分析/依据**：当前 aux 权重 0、强制 FALSE，raw 是 bug；如果论文的 M6 本意是用 aux 做 regret/modification，B′-4 是对论文设计的更忠实复现。风险：aux 监督可能只让模型学会“标记”，不一定改变计算；raw 协议要重新设计。
- **成本/风险**：需要改 loss/数据标签（REGRET/MODIFICATION 的 ground truth 定义）；评估协议要新增；变量多。
- **状态**：proposed / unstarted。

#### 候选 B′-5：own-state 训练（不用 teacher predictions）

- **定义**：B-M6 但训练时用模型自己的预测作为下一轮输入（不用 teacher predictions），让训练/推理的 own-state 分布对齐。
- **想验证**：B 的 4→5 恢复在 own-state 下是否仍成立；训练/推理分布对齐是否改善 q 的 r1 行为。
- **分析/依据**：own-state 注入下 B 的 q r1 有大量 no_value/expected_wrong；如果训练时也用 own-state，可能让“纠正在轮边界”更自然。但 B-M6 的 fast 路径本来就是 teacher predictions；改成 own-state 会改变训练稳定性/速度。
- **成本/风险**：训练路径改动大；与 B′-1/B′-2 可组合。
- **状态**：proposed / unstarted。

**建议**：如果看板的 B′ 是“纠错训练设计”的变体，优先 **B′-1/B′-2**（单变量、最直接回答 recompute vs recorrect；B′-2 用于归因）；如果是“离散线上新数据”，选 B′-3；如果是“aux 显式纠错”，选 B′-4。请用户确认。

---

## 3. 原始 C0（M4 连续基线）

### 3.1 训练设计

| 项目 | 值 |
|---|---|
| 方法 | nointerm / M4，连续 hidden 状态 |
| 配置 | `C0-M4-Nodes32-ADD-L4H16D384-DeBERTa` |
| 架构 | L4H16D384，DeBERTa 位置编码 |
| 训练数据 | N=32，max_depth=6，max_length=256，50,000 图 |
| 训练轮数 | 100 epochs，1,563 batches/epoch |
| batch size | 32 |
| optimizer | AdamW，lr=1e-3，weight_decay=0.01 |
| lr_scheduler | none（常数 lr，无 warmup） |
| gradient clipping | 无 |
| recorrection | 关闭 |
| 验证数据 | N=128，max_depth=36，max_length=1024（训练时 OOD） |
| max_train_loop | 8 |
| max_val_loop | 36 |
| seeds | 42 / 1234 / 2025 |

运行历史：

- 2026-11-09 10:50 第一次启动，分别跑到 e3/e2/e5 后停止；
- 15:42/16:10 从各自 `last.ckpt` 续训；
- 最终 `last.ckpt`：s42 e94、s1234 e96、s2025 e99；top-k e73/90/94、e89/95/96、e73/97/99；
- 单份 100 epochs 约 8 小时（优化后速度）。

**训练提速决策**（2026-11-09）：

- 发现 `TF_forward` 绕过 `torch.compile`：eager 759ms/call vs 编译 9.29ms/call（82×）；
- 改 `model.py:875` 为 `self.model(depth_batch, True)`；
- 关 `logger.watch`、末轮指标门控、OMP=1；
- 真实管线从 0.43–0.51 it/s 提升到 6.4–6.5 it/s（~11×）；六条 run 全部 resume；
- **依据**：隔离试验 C（2.55 it/s）、D1–D4、A′ resume 验证；`Resume` 需 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，`max_epochs` 必须 > ckpt epoch；`save_last` 只在本轮 top-k 改善时更新。

**决策**：选 checkpoint 不能盲用 `last.ckpt`；按 run 目录选最高/最优 top-k，并注明 epoch。

### 3.2 训练过程

早期 `total_loss/train_epoch`：

| epoch | s42 | s1234 | s2025 |
|---:|---:|---:|---:|
| 0 | 15.4 | 6.18 | 4.66 |
| 1 | 3.31 | 1.30 | 1.94 |
| 2 | 1.35 | 1.22 | **33.1** |
| 3 | 1.31 | 10.7 | **33.4** |
| 4 | 16.5 | 23.5 | 22.0 |
| 5 | 20.1 | 17.4 | 22.4 |

中后期：

| epoch | s42 | s1234 | s2025 |
|---:|---:|---:|---:|
| 10 | 1.81 | 15.8 | 31.7 |
| 20 | 1.11 | 1.48 | 21.3 |
| 30 | 1.04 | 1.25 | 35.3 |
| 40 | 1.02 | 1.07 | 34.4 |
| 50 | 0.322 | 1.04 | 18.8 |
| 60 | 0.171 | 1.01 | 4.78 |
| 70 | 0.163 | 0.992 | 1.53 |
| 80 | 0.196 | 0.335 | 1.41 |
| 90 | 0.107 | 0.19 | 1.33 |
| 99 | 0.143 | 0.165 | 1.28 |

真实 val loss（`total_loss/val`；文件名 `val_loss=0.0000` 是占位）：

| seed | e70 | e90 | e99 |
|---|---:|---:|---:|
| s42 | 1.27 | 0.57 | 2.22 |
| s1234 | 3.59 | 1.34 | 1.43 |
| s2025 | 6.07 | 8.48 | 4.41 |

**分析**：

- 三个 seed 都有早期尖峰，不是 s2025 独有；
- s2025 在 e2–e3 就冲到 33，长期高 loss（e30 35.3、e40 34.4、e50 18.8），e60 后才降到 4.78；
- 但最终 train loss 都能降下来；严格整图却不一定（见 3.3）；
- 训练 loss 是逐位置/逐轮/逐因子平均，包含大量 PAD/EMPTY/叶子；严格整图是 AND，二者不能混读。

### 3.3 测试表现与数据分布

#### 16 档（old Tr32Test128，每档 1,024 题，严格整图）

| N | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s42 | .999 | .946 | .815 | .553 | .002 | .005 | .006 | 0 |
| s1234 | .993 | .835 | .642 | .380 | .019 | .024 | .010 | 0 |
| s2025 | .014 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

16 档均值：s42 0.2079、s1234 0.1814、s2025 0.0009。

#### 新留出 seed42（N32/N128，每档 1,024 题）

| 数据 | s42 | s1234 | s2025 |
|---|---:|---:|---:|
| N=32 | 0.2715 | 0.2949 | 0 |
| N=128 | 0 | 0 | 0 |

#### deep_v2（四个深/宽 OOD 池，每池 1,024 题）

- s42 / s1234：四个池全 0；
- s2025：未跑（16 档/留出已经全 0）。
- 对照：离散 A′ clean 在 deep_v2 上约 **1.0**。

#### matched-depth N32（训练同分布）

N=32、max_depth=6、1024 图、seed=42；路径 `tmp/analysis/matched-depth/ADD/testdata_N32.pt`，SHA256 `3088f313...`。

| seed | matched-depth full | old N32 full |
|---|---:|---:|
| s42 | **0.9736** | 0.5527 |
| s1234 | **0.5059** | 0.3799 |
| s2025 | **0.0000** | 0.0000 |

#### checkpoint sweep（matched-depth）

| seed | e73/89 | e90/95 | e94/96 | last.ckpt |
|---|---:|---:|---:|---:|
| s42 | .9736 (e73) | .9736 (e90) | .9736 (e94) | .9736 (e94) |
| s1234 | .5059 (e89) | .5059 (e95) | .5059 (e96) | .5059 (e96) |
| s2025 | 0 (e73) | 0 (e97) | 0 (e99) | 0 (e99) |

#### 数据分布：为什么旧 C0 的“泛化差”不能全怪架构

| 数据集 | N | max_depth | mean depth | 说明 |
|---|---:|---:|---:|---|
| 旧 C0 训练 | 32 | 6 | — | 50k，max_length256 |
| 旧 C0 训练时验证 | 128 | 36 | — | 训练时 OOD |
| matched-depth N32 | 32 | 6 | 5.541 | 训练同分布对照；nonpad 179.35 |
| old Tr32Test128 N32 | 32 | **10** | 6.016 | 比训练深 4 轮；nonpad 178.7 |
| new heldout N32 | 32 | **11** | 6.034 | 更深；nonpad 179.0 |
| old 16 档 N≥40 | N | 随 N 增大 | — | “N=32 测试”已是 depth10；N≥40 的崩溃混入深度 OOD |
| new heldout N128 | 128 | 17 | 11.548 | nonpad 755.4 |
| deep_v2 | 9008/9060/9100/9122 | 7.86/64.8/110.6/122.0 | 宽/深压力 | width 40.4/6.8/6.0/6.0 |

#### “没训好 vs 训好了泛化差”判定

| seed | matched-depth（同分布） | old N32（depth10） | new N32（depth11） | N≥40 | deep_v2 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| s42 | **0.9736** | 0.5527 | 0.2715 | 0.002–0.006 (N40–56)，N64+ 0 | 0 | **训好了，深度/规模泛化差** |
| s1234 | **0.5059** | 0.3799 | 0.2949 | 0.019–0.024 (N40–48)，N56 .010，N64+ 0 | 0 | **同分布只一半（没训好）+ 泛化差；VALUE 错** |
| s2025 | **0.0000** | 0 | 0 | 0 | 未跑 | **训练失败（没训好）** |

**分析**：

1. s42 的同分布 97.4% 证明模型能学会任务；old N32 55.3%、new N32 27.2%、N≥40 近 0、deep_v2 0 —— 这是**训练分布→测试分布的深度/规模 OOD**，不是“没学会”。
2. s1234 同分布 50.6%，本身就没训好；它的 VALUE 错误在训练分布内就存在，不能当“训好了但 OOD”的干净例子。
3. s2025 同分布 0，是训练失败；不能用来解释 OOD。
4. **old 16 档的 N 不是纯节点数**：每档 max_depth=N，N=32 测试已是 depth10；所以“N≥40 崩”混入了深度/轮数 OOD，不能简单说“规模泛化差”。
5. 旧 C0 没有见过 N≥40 或 depth>6 的数据；用它去证明“连续模型不如离散模型”会把训练分布与架构混在一起。新 C0 的 mixed-N 训练就是为修正这个 confound。

**决策依据**：
- 所有 C0 评测必须同时报告 matched-depth（同分布）和 OOD 档位；
- 不能跨数据集直接比绝对值（old N32 vs new N32；old C0 val vs new C0 val）；
- 新 C0 的评估要分层：新分布 in-distribution、matched-depth N32 depth6、old N32 depth10、new N32 depth11、N≥40/deep_v2（OOD）。

### 3.4 三个 seed 的失败模式

诊断数据：old N32，256 图，8,192 个节点定义，自由续算 10–14 轮；纯推理、无注错、无 teacher forcing。

#### VALUE 节点轨迹

| seed | 始终正确 | 先错后对 | 首次错再没对 | 算对后变错 | 到最后一轮仍错 | 有子节点且语义一致 |
|---|---:|---:|---:|---:|---:|---:|
| s42 | 8165 | 22 | 0 | 5 | 5 | 2 / 2 |
| s1234 | 7460 | 210 | 376 | 146 | 131 | 57 / 65 |
| s2025 | 2051 | 44 | 5701 | 396 | 82 | 37 / 234 |

#### 图级最终分解

| seed | 严格全对 | 值全对但变量错 | 值错 |
|---|---:|---:|---:|
| s42 | 130 / 256 | **124** | 2 |
| s1234 | 88 / 256 | 35 | **133** |
| s2025 | 0 / 256 | 0 | **256** |

#### 最终 token mismatch

| seed | VALUE mismatch | VARIABLE mismatch |
|---|---:|---:|
| s42 | **5** | **366** |
| s1234 | **507** | 139 |
| s2025 | **5930** | 6 |

#### s2025 按深度“始终正确”比例

| depth | 始终正确 / 总数 |
|---:|---:|
| 0 | 1536 / 1536 |
| 1 | 359 / 1536 |
| 2 | 77 / 1536 |
| 3 | 42 / 1536 |
| 4 | 19 / 1536 |
| 5 | 11 / 1536 |
| 6 | 3 / 1536 |
| 7+ | 很少 |

#### 具体例子

- **s42 graph 16**：第 13 轮三个定义位置 VARIABLE 同时漂移（位置 72 真名 x_118 读成 x_75，值仍 13）。位置 72 的子节点 160 真值 22，若按 x_75 的值 6 算应是 15，实际仍 22；子节点 184 真值 4，按错名算应是 20，实际仍 4。→ **名字读错，计算没跟着错名走**。
- **s1234 graph 18**：位置 78（x_81，值 8）第 7 轮起 VALUE 变 12；子节点 118 表达式 14+2+x_81，真值 1，第 7 轮起变 5；14+2+12=28 mod 23=5，精确匹配。→ **父节点 VALUE 真的错了，子节点按错值重算**。

#### 分析

1. **s42：算术对、变量名读出漂移**。254/256 张图 RHS VALUE 全对；126 张失败图里 124 张只错 VARIABLE；0/30 下游按错变量算。严格整图被一个 VARIABLE token 拖垮。
2. **s1234：真 VALUE 错并传播**。133/256 张图有 VALUE 错；146 个节点算对后变错；57/65 下游语义一致。它适合做 C+ 数值纠错的自然错误来源，但基座本身较弱（同分布 50.6%）。
3. **s2025：训练失败**。0/256 全对；5701/8192 首次错再没对；2051 始终正确，主要是 depth0 叶子。不能当纠错反例，也不能当 OOD 证据。
4. **指标陷阱**：train loss 低（s42 0.143、s1234 0.165）但严格整图只有 130/256、88/256；loss 对大量 PAD/EMPTY/叶子平均，严格指标是 AND。

**决策依据**：
- s42 → 变量身份/读出稳定性；不要用它做数值纠错主目标；
- s1234 → 数值纠错；自然错误样本充足；
- s2025 → 排除，不作为 C+ 基座；
- C+ 不能混训三种目标。

### 3.5 旧 C0 对 C+ 的意义

| seed | 现象 | C+ 方向 | 不建议 |
|---|---|---|---|
| s42 | 值对、变量名读出漂移 | 变量身份/读出稳定性；或改用 VALUE-only 指标看算术能力 | 不要用它做“数值纠错”主目标 |
| s1234 | 真 VALUE 错并向下游传播 | 数值纠错；自然错误样本充足 | 基座较弱（0.18–0.29），不要期待“强模型修小错” |
| s2025 | 大部分节点从头不会算 | 排除 | 不要当纠错反例，也不要用它解释恢复机制 |

**决策**：C+ 首选 s1234 的 VALUE 错误；先做 C4 的 4→4 对照和自然错误采集，确认有足够样本再开训。

---

## 4. 初步连续向量纠错尝试

### 4.1 Logit-steering 注错（失败）

- 做法：在 C0 loop d 结束、进入 d+1 前，对 u 的 hidden 加 `δ = α(w_5 − w_4)`，把 VALUE 头 argmax 从 4 翻到 5。
- 结果（N32，64 样本，47 候选）：
  - 注入前读出 4：47/47；注入后读出 5：46/47；
  - **下一 loop u 的读出回到 4：47/47**；
  - 下游 q 第一次读出**从未等于语义错值 14**（q_first_wrong = 0/47）；
  - q 最终正确 47/47。
- 分析：C0 每轮从父节点重算，hidden 扰动是 OOD 的（需要很大的 δ），下一轮被覆盖；没有形成语义上的 u=5。
- 决策：放弃 logit-steering，改用**自然反事实 hidden 替换**（C4）。

### 4.2 C4 自然 hidden 替换

#### 设计

- 选真值 u=4、有直接子节点 q 的节点 u；
- 找一个叶子祖先，改叶子值 delta，使同一张图的 u 变成 5（结构不变）；
- 分别跑 original（u=4）和 counterfactual（u=5）的 clean 轨迹，取 loop=d 后的 hidden；
- 把 original 的 u hidden 换成 counterfactual 的 5-hidden，继续跑，看 q 是否移动到 `q_u5`。
- 条件：`orig_swap45`、`cf_swap54`、`orig_swap_randperm`、`orig_swap_noise`；clean 对照。

#### 结果（old N32，256 图）

有效 pair：s42 99、s1234 90（很多图没有 u=4 且有子节点/叶子祖先的节点）。

| seed | 条件 | n | moved_to_q_u5 | stayed_orig | other |
|---|---|---:|---:|---:|---:|
| s42 | orig_swap45 | 177 | **137 (77.4%)** | 0 | 40 |
| s42 | randperm | 177 | 6 (3.4%) | 5 | 166 |
| s42 | noise | 177 | 0 (0%) | 25 | 152 |
| s1234 | orig_swap45 | 163 | **120 (73.6%)** | 0 | 43 |
| s1234 | randperm | 163 | 3 (1.8%) | 10 | 150 |
| s1234 | noise | 163 | 1 (0.6%) | 47 | 115 |

按 pair 聚合：s42 87/99（87.9%），s1234 76/90（84.4%）；randperm 5/99、2/90；noise 0/99、1/90。

时间演化：post2 大幅回原值（s42 10/177；s1234 3/163）；post4 基本全部回原值（2/177、0/163）。反向 swap 约 47%。

u 自己的读出：s42 99/99 保持原值（hidden 换了，u 仍从父节点重算）；s1234 57/90（更容易被扰动）。

#### 分析

1. **自然 hidden 替换确实能传播到下游**：post1 q→q_u5 77.4%/73.6%，pair 87.9%/84.4%；
2. **效应短暂**：post2/4 大部分回原值——与 C0 每轮重算一致；
3. **随机置换/噪声很少产生 q_u5**（0–3.4%），说明不是“任何扰动都会让 q 动”；
4. **但仍缺 in-distribution 4→4 donor 对照**：随机扰动是 OOD，不能排除“donor hidden 是 in-distribution”这一替代解释；
5. **这不是训练信号**：C4 只证明“可以制造可传播的下游错误”，还没证明 C0 能学会恢复。

**决策**：
- C4 作为 C+ 的候选注错来源，但先补 **4→4 donor**（同值、in-distribution，预期 q 不动）；
- 扩大到更多 target 值/更多图；
- 暂不启动 C+ 训练。

### 4.3 自然错误采集（C0 自己犯的错）

- 256 张 old N32，自由续算 10–14 轮：
  - s42：8165/8192 始终正确；5 个算对后变错；184 次 VARIABLE 漂移；0/30 下游语义；
  - s1234：146 个算对后变错；131 持续；57/65 下游语义一致；VALUE mismatch 507；
  - s2025：5701 首次错再没对；大多数深节点从头不会。
- 分析：
  - s42 的自然错误主要是 VARIABLE 漂移，算术值对 → 不适合数值纠错；
  - s1234 有足够的 VALUE 错误且会传播 → 最适合 C+；
  - s2025 是训练失败 → 排除；
  - 但 s1234 的基座弱（同分布 0.506），C+ 是“在较弱基座上继续学”，不是“强模型修小错”。
- 决策：
  - 先小规模采集（已做 256 图），确认“错误多不多、持续多久、是否影响下游”；
  - 如果 s1234 的错误片段足够多且 C0 还有恢复空间 → 做等步数的普通续训 vs 纠错续训；
  - 如果错误几乎瞬间消失 → 先做候选 5（拉长依赖链/增加下游数）再测。

### 4.4 C+ 设计选项与当前决策

| 方案 | 做法 | 优点 | 风险 |
|---|---|---|---|
| A 自然反事实腐蚀 | 以概率 p 把已可用位置的 h 替换为反事实 h（值被改成错值），训练恢复 | 语义最准、in-distribution | 需要在线/离线构造反事实，成本高 |
| B logit-steering 腐蚀 | 加 δ 把读出翻到随机错值，训练恢复 | 便宜 | 模型会直接重算覆盖 δ，可能只学到抗 OOD 扰动 |
| C 高斯噪声腐蚀 | 对 h 加噪声，训练恢复 | 便宜 | 非语义 |
| D 状态一致性/承诺损失 | 加辅助损失，让模型在节点已可用时保持上一轮读出，再加腐蚀训练 | 改变归纳偏置，鼓励“存值” | 与论文架构意图可能不一致 |
| E readout 层腐蚀 | 只在 VALUE 头加错值偏置，训练下一轮恢复 | 简单 | 可能只训练输出头，不影响状态传递 |

**当前决策**：先做 C4 的 4→4 对照 + s1234 自然错误采集；若确认有可重复、影响下游、且还有恢复空间的错误现场，再从同一 C0 checkpoint 做 **等步数普通续训 vs 纠错续训**。不先假设人工注错一定有效（logit-steering 已失败）。

---

## 5. 新 C0 设计（mixed N/depth12）

### 5.1 动机与决策

旧 C0 的三个问题：

1. 训练分布只有 N32/depth6；old N32 测试已是 depth10，N128 更深，deep_v2 是宽/深压力 → **泛化差被数据分布严重混淆**；
2. 三个 seed 三种失败（s42 变量漂移、s1234 VALUE 错、s2025 坏 seed）；
3. matched-depth 与 old N32 的差距说明“同分布能力”和“OOD 能力”必须分开测。

**用户要求**：新 C0 必须有**不同节点数**、宽内部数据分布、depth12、四个 N 组、4 seeds；其他超参与旧 C0 一致（架构不变）。

**决策**：训练一个 C0（nointerm，recorrection off）在 N=8/16/24/32、depth1–12、100k 数据上，13 loops、50 epochs、4 seeds；用 matched-depth/新分布/旧 OOD 分层评估。

### 5.2 第一版数据生成器失败（已停）

`tmp/scripts/gen_mixed_n_data.py`（bucket + 拓扑池）生成 97,999 样本，但深度严重偏斜：

- N32 depth12 13,333、depth4 10,649；
- depth9–11 几乎为 0；
- depth1 为 0。

训练在 2026-11-11 01:29 启动 seeds 42/1234（seq512），在 sanity check 阶段被停掉；没有 checkpoint、没有 epoch。  
**决策**：弃用第一版数据，改用 `ControlledDepthDAG` 精确 N×depth 配额。

### 5.3 最终数据（controlled）

`tmp/scripts/gen_mixed_n_controlled.py`：

- `ControlledDepthDAG`：先建一条长度恰为 d 的随机链（节点 6..6+d−1），再把剩余节点挂到 depth < d 的父节点上，保证每张图 max depth 精确等于 d；
- N 权重：8/16/24/32 = 10%/20%/30%/40%；
- 可行深度：N8 d=1–2；N16 d=1–10；N24/N32 d=1–12；
- 每个 N×d 格等配额；
- 100k train / 2k val，seed 20261111/20261188；
- `max_length=256`。

审计：

| 项 | train | val |
|---|---:|---:|
| 样本数 | 100,000 | 2,000 |
| 形状 | [100000,256,6] | [2000,256,6] |
| N=8/16/24/32 | 10k/20k/30k/40k | 200/400/600/800 |
| 深度范围 | 1–12，均值 5.80 | 1–12，均值 5.79 |
| 非 PAD 长度 min/mean/max | 31/131.0/213 | 31/130.7/203 |
| fan-in 1/2/3 | 1,199,698/600,799/599,503 ≈ 2:1:1 | 同比例 |
| 同层宽度 mean/max | 8.1/26 | 8.1/26 |

深度直方图：d1/d2 各 12,834；d3–d10 各 7,833–7,834；d11/d12 各 5,833。  
**分析/局限**：每 N×d 等配额导致 marginal depth 1/2 过采样（12,834 vs d11/12 5,833）；N8 只有 d1–2；100k/2k 的 val 较小。这些是设计选择，不是 bug。

### 5.4 训练配置与 loss 语义

- 配置：`C0-M4-MixedN8-32-D12-ADD-L4H16D384-DeBERTa`；
- 4 layers、16 heads、hidden384、DeBERTa、dropout0.1、compile true；
- batch32、AdamW lr1e-3、weight_decay0.01；
- `lr_scheduler: none`、无 warmup、无 grad clip；
- `max_epochs: 50`、`max_train_loop: 13`、`max_val_loop: 13`、`max_train_depth: 12`；
- `max_length: 256`、`num_workers: 4`；
- `nointerm: true`、`recorrection: false`。

Loss（`compute_batch_loss_nointerm`）：

- 每个 loop 一次 forward；label = `batch_ext[loop+1]`；
- 每个因子 `CrossEntropyLoss()`，**无 ignore_index，PAD 计入**；
- `factor_loss_scales=[1,1,1,1,0,0]`：前 4 主因子权重 1，aux 0；
- `depth_loss = Σ factors`；`total_loss += depth_loss`，**不除以 loop 数**；
- 一次 `manual_backward(total_loss); opt.step()`。

含义：`total_loss` = 13 loops × 4 factors 的和；坏状态 ~50 → 3.8/loop；好状态 ~14 → 1.1/loop；旧 C0 2.22/8 loops → 0.28/loop。绝对数不能跨任务直接比。

**决策依据**：50 epochs（旧 C0 100 epochs，但新数据 100k/13 loops 更贵；先看 50 是否能到平台）；seq256（seq512 OOM 60–79GB；seq256 约 26.7GB）；4 seeds（seed 方差大）；不加载旧 C0 checkpoint（配置/数据不同）。

### 5.5 训练过程、迁移与当前状态

#### 时间线

- 01:29 第一版 mixed-N 启动（bucket 数据，sanity check 阶段停掉）；
- 01:45–01:48 新 C0 4 seed 启动：s42→GPU0、s1234→GPU5、s2025→GPU7、s7→GPU2；
- 02:24 s7 GPU2→GPU4；02:26 s42 GPU0→GPU6（GPU0 只剩 2.3GB，避免 OOM）；两次都用各自 epoch0 `last.ckpt` 断点续训；
- 06:36 4h 错误监视结束：无 OOM/Traceback；
- 当前 GPU：4/5/6/7 四个 seed；0/1/2/3 被其他用户占用。

#### 断点续训的澄清

- s42/s7 续的是**它们自己新 run 的 epoch 0 checkpoint**，不是旧 C0（N32/depth6）或第一版 mixed-N；
- epoch=0、global_step=3125、optimizer_states=True；当前 global_step 109375/84375 = 35/27 × 3125，连续；
- 日志 `Restored all states`；s1234/s2025 从未续训；
- 代价：s42 丢 epoch1 约 353 步、s7 丢约 48 步；DataLoader shuffle 可能不同；其余一致。

#### 最终状态（15:36，全部完成）

| seed | GPU | 状态 | e49 train / val | 完成时间 |
|---|---:|---|---:|---|
| 1234 | 5 | **50 epochs 完成** | 7.82 / **5.89** | 14:23 |
| 2025 | 7 | **50 epochs 完成** | 15.8 / **10.30** | 14:23 |
| 7 | 4 | **50 epochs 完成** | 18.7 / 15.60 | ~15:25 |
| 42 | 7 | **50 epochs 完成** | 2.41 / **2.01** | 15:36（14:33 迁到 GPU7 后完成） |

best checkpoint：

- s42 **e48 1.96**（top-k e47 2.07、e48 1.96、e49 2.01）；另有 e0 14.5；
- s7 **e26 12.7**（top-k e23 12.8、e26 12.7、e45 12.9）；
- s1234 **e49 5.89**（top-k e47/e48/e49）；
- s2025 **e49 10.30**（top-k e41/e44/e49）。

### 5.6 训练异常：loss 在好/坏盆地之间反复跳

#### 现象

四个 seed 都在不同 epoch 到达 val 12.7–15.5，又在别的 epoch 反弹到 40–78：

- s42：e0 14.5 → e2–e15 坏（37–52）→ e16–e24 好（19.5–25.4）→ e26–e28 大坏（63.6/65.2/65.4）→ e31–e32 恢复（22.2/16.0）；
- s7：反复在 12.7–15.4 和 48–78 之间跳（e12 78.0、e29 48.4）；
- s1234：e7 14.7 后 e12–e33 超长坏盆地（41–56），e34–e36 恢复（38.1→29.7→23.8）；
- s2025：e24 14.4、e28 14.6，e30 又跳 48.0，e31–e36 回到 20–21；
- 四个 seed 在 e12–e16 都处于坏状态。

#### 分析（事实 vs 假设）

**事实**：

1. 不是数据 bug 的典型表现：四个 seed 都能到 val 12.7–15.5；数据审计通过；train/val 同步跳。
2. 不是评测 bug：训练/验证同一 `compute_batch_loss_nointerm`；train/val 同步；checkpoint monitor 是 `total_loss/val`，只是文件名占位错了。
3. 不是 resume bug：s1234 从未迁移也在 e12 后坏；s7/s42 迁移后正常。
4. 不是 `max_train_loop` off-by-one：13 loops 覆盖 depth0–12，已验证。
5. 不是 OOM：四日志无 OOM/Traceback。
6. 旧 C0（depth6）也有早期尖峰和 s2025 长期高 loss；新 C0 是同一不稳定性的放大版。

**假设**：

- **H1 连续递归误差累积/双盆地**：C0 无离散 token 反馈，13 轮吃自己的 hidden；一次更新把递归推入错误吸引子后，误差累积，train/val 同爆；好状态说明可学，坏状态是自洽的错误递归。
- **H2 优化不稳定**：常数 lr1e-3、无 warmup/clip；`total_loss` 按 loop 相加，不同 batch 的 max depth 不同，梯度尺度随深度变化；depth12/13 loops 放大。
- **H3 PAD/EMPTY/readout 边界崩**：CE 计入 PAD；某次更新让 PAD/EMPTY 预测不稳，loss 上升但未必算术全错；坏状态 ~50 是“部分错”。
- **H4 数据/课程**：四 seed 在 e12–e16 同时坏，可能是模型开始拟合深节点时的共同相变；数据从第 0 轮混合 depth1–12，没有浅→深课程。

**最新进展（15:36，全部完成）**：振荡不是永久的——

- s42 e33 14.4 → e39 2.84 → e44 2.17 → e45 2.14 → e46 2.11 → e47 2.07 → **e48 1.96 → e49 2.01（完成）**；
- s1234 e36 23.8 → e41 14.5 → e44 21.7（小反弹）→ e45 12.3 → e46 10.5 → e48 8.51 → **e49 5.89（完成）**；
- s2025 e39 15.9 → e41 12.2 → e43 12.7 → e44 11.9 → e47 48.4（反弹）→ e48 14.7 → **e49 10.30（完成）**；
- s7 e38 14.8 → e39 20.5 → e44 12.9 → e45 12.9 → e46 35.8（大反弹）→ e47 16.2 → e48 21.4 → **e49 15.60（完成）**；best 仍是 e26 12.7。

三个 seed 在 e35–e49 找到稳定好盆地；这支持“坏盆地是暂时的”解释，**50 epochs 对 3/4 seed 足够**（s42 1.96、s1234 5.89、s2025 10.30）。s7 是唯一没有明显收敛的 seed：晚段 e46/e48 仍有大反弹，final 15.60，best 仍是早期 e26 的 12.7。

**为什么不能直接验证坏状态**：top-k 只保留 best 3；坏状态权重没落盘。seed1234 现在 top-k 是 e41/42/43（val 14.4–14.5），e33 的 41.2 坏状态没保存；s42 top-k e37/38/39，s7 e23/25/26，s2025 e41/42/43。想拆解坏状态，只能从日志轨迹和好 checkpoint 推断，或重跑一个可控的“坏状态复现”实验。

#### 决策

1. 让 50 epochs 跑完；用 top-k best checkpoint 评估，不用当前 loss 判断模型好坏；
2. 如果 e50 仍震荡 → 考虑延长到 100 epochs；旧 C0 也是 60–100 才稳定；
3. 若要改优化，优先单变量：lr 3e-4 / grad clip 1.0 / loop-loss 归一化 / depth curriculum；先做便宜对照；
4. 不在当前训练中途重启；除非出现 OOM 或明确坏状态无法恢复。

### 5.7 新 C0 的评估计划

- **in-distribution**：新 val（2k）不够，需要新的 held-out mixed N/depth 测试集（按 N×depth 分层，固定 seed/hash）；
- **matched-depth N32 depth6**：与旧 C0 直接对比；
- **old N32 depth10 / new N32 depth11**：深度 OOD；
- **N≥40/deep_v2**：规模/宽度 OOD，单独标注；
- **top-k best vs final**：两者都报，注明 epoch；
- **严格整图 + 分因子/分深度**：不要只看 val loss；VARIABLE 漂移和 VALUE 错要分开；
- **与 A′/B 对照**：同数据/同指标；不要把 C0 的连续 val 与离散 raw/clean 混比。

---

## 6. B′（看板新增想法，待确认）

见 §2.6。由于共享看板没有找到 B′ 原文，本节暂以“设计空间”形式给出（全部 **proposed / unstarted**，没有 config/run/checkpoint）：
- **B′-1**：B-M6 完全相同，只把 `corruption_strategy` 从 `last` 改为 `random`（候选集变成全部 `depth<=i`，含旧位置；caveat：数量/分布也变）；
- **B′-2**：显式 old-only / old+new 混合 mask（干净归因“旧位置腐蚀”）；
- **B′-3**：B 在 mixed-N/depth12 数据上重训（新 C0 的离散对照）；
- **B′-4**：aux/REGRET/MODIFICATION 显式纠错信号；
- **B′-5**：own-state 训练。

**建议**：如果是纠错机制线，优先 B′-1 + B′-2（B′-1 是直接提议，B′-2 用于归因）；如果是新数据线，优先 B′-3。请用户确认看板原文后，我把对应方案写成正式训练设计（数据/超参/对照/指标/成本）。

---

## 7. 决策记录（按时间）

| # | 时间 | 决策 | 依据 | 替代方案/风险 |
|---|---|---|---|---|
| 1 | 11-07/08 | raw/clean 双口径；raw 不作为能力指标 | aux 未训练、raw 回灌污染；clean 下 A 全部 ~100% | 只用 raw（论文原协议）会得出错误结论 |
| 2 | 11-08 | B-M6 与 A-L4fast 对照，分离架构 vs 纠错 | B 在 wide8 tf=1.000，A-demo L2 tf=0.592；A-L4fast raw 100% | 只用 A-M5 vs B 会混入容量 |
| 3 | 11-09 | 训练 A′（L4H16D384 无纠错，3 seed） | 与 B-M6 同架构，干净消融 | 单 seed B；cyhB 复现 |
| 4 | 11-09 | 单点 VALUE 注入作为纠错主证据 | raw/clean 是代理；注错直接测恢复 | 注入时机与训练不同（后续发现） |
| 5 | 11-09 | 旧 C0 训练加速（875 编译修、watch off、日志门控、OMP=1、resume） | 隔离试验：0.43–0.51→6.4–6.5 it/s；resume 验证 | 不改语义，只改监控/编译/线程 |
| 6 | 11-10 | 扩样到 256 图/档、teacher+own、N32/N128、K4/Kall | 64 样本/单批不足；own-state 更接近真实推理 | 仍有选点/深度分层偏差 |
| 7 | 11-10 | 2×2 判读降级：不写“recorrect” | R=4 有 copy/repair-first 歧义；L4 层间可修父 | 需要叶父/扫描/attention 证据 |
| 8 | 11-10 | 4→5 与 EMPTY→5 分开报告 | 训练 `last` 掩码是 depth==i 新位置；4→5 是旧位置 | 服务器版本未核对 |
| 9 | 11-10 | C0 诊断用 matched-depth 分离“没训好”与 OOD | s42 0.9736 vs old N32 0.5527；s1234 0.5059；s2025 0 | 只测一个 matched-depth 样本 |
| 10 | 11-10 | C4 自然 hidden 替换作为 C+ 候选注错 | post1 77.4%/73.6%，随机/噪声 0–3% | 缺 4→4 donor；效应短暂 |
| 11 | 11-10 | C+ 不立即启动；先采自然错误 + 4→4 对照 | logit-steering 失败；s1234 错误多但基座弱 | 可能错失训练窗口 |
| 12 | 11-11 01:29 | 弃用第一版 mixed-N bucket 数据 | 深度严重偏斜（N32 d12 13,333、d9–11~0、d1=0） | 继续训练会得到不可解释的分布 |
| 13 | 11-11 01:45 | 用 ControlledDepthDAG 重新生成 100k/2k | 精确 N×depth 配额；审计通过 | 每格等配额导致 marginal depth 1/2 过采样 |
| 14 | 11-11 01:45–01:48 | 新 C0：4 seeds、50 epochs、13 loops、seq256 | 旧 C0 100 epochs 但 depth6；新数据更贵；seq512 OOM | 50 可能不够，需延长 |
| 15 | 11-11 02:24/02:26 | s7/s42 断点迁移到 GPU4/6 | GPU0 只剩 2.3GB；GPU2 共享；续训自身 epoch0 ckpt | 丢 partial epoch1；shuffle 差异 |
| 16 | 11-11 11:10 | raw 是 bug，主结果只用 clean | 用户拍板；aux 未训练；raw 受 margin 尾部 | raw 数字保留为档案/案例 |
| 17 | 11-11 | B 的表述改为“行为上恢复/从父值重算” | 2×2 R-only 91.3% 跟随父值；R=4 歧义 | 需要新实验确认条件纠正 |
| 18 | 11-11 | 新 C0 不中途重启，先跑完 50 epochs | 四个 seed 都在恢复；best checkpoint 都在 | 若 e50 仍震荡，延长/改优化 |

---

## 8. 当前状态与下一步

### 8.1 当前状态（15:36，训练完成）

- 新 C0：四个 seed **全部完成 50 epochs**；无 OOM/Traceback；
- final val：s42 **2.01**（best e48 **1.96**）、s1234 **5.89**、s2025 **10.30**、s7 **15.60**（best e26 **12.7**）；
- s42 于 14:33 从 GPU6 迁到 GPU7，从 e45 续训完成；s7 在 GPU4 完成；
- best checkpoint：s42 e48 1.96；s7 e26 12.7；s1234 e49 5.89；s2025 e49 10.30。

### 8.2 待做实验（按优先级）

1. **EMPTY→5 vs 4→5 时机对照**（B/A′，同节点；回答“训练分布内 vs 跨时机”）；
2. **B-42 服务器训练版本核对**（确认 `last` 掩码与本地一致）；
3. **parent 2×2 强化**：叶父 R-only、扫描多个 r'、非父节点对照、r 轨迹、L1/L2 vs L4；
4. **C4 4→4 donor**（in-distribution 同值替换）；
5. **新 C0 评估**（mixed held-out、matched-depth、old N32/N128/deep_v2；top-k vs final）；
6. **B′ 设计确认/启动**（看板原文；候选见 §2.6）；
7. **C+ 决策**：等自然错误采集和 4→4 对照，确认有恢复空间后再开训。

### 8.3 想请外部审阅者判断的问题

1. B 的 R-only 91.3% 是否足以把机制表述为“父节点重算”，还是仍必须等叶父/扫描？
2. B′ 应该优先做“旧位置腐蚀训练”还是“mixed-N 离散对照”？哪个更能回答当前的核心问题？
3. 新 C0 的震荡，最可能是 H1（连续递归误差累积）还是 H2（优化不稳定）？最省算力的区分实验是什么？
4. 50 epochs 是否足够？如果 e50 仍在好/坏盆地跳，应该延长到 100 还是先改优化（lr/clip/课程）？
5. loss 的按 loop 相加、PAD 计入 CE、aux 权重 0，是否应该改成 per-loop 平均/非 PAD 平均？
6. 新 C0 的评估如何公平对比旧 C0/A′/B？哪些测试集算 in-distribution，哪些只能算 OOD？
7. C4 的 4→4 donor 应该怎么构造（同图同值、不同图同值、还是写回原 hidden）？
8. 还有没有我们忽略的 setup bug/红旗？

---

## 9. 资料索引

### 9.1 数据附录

- `tmp/records/2026-11-11/c0-mixedn-full-context.md`：所有原始表（AB 全条件注错、旧 C0 诊断、新 C0 全轨迹、AB raw/clean）。

### 9.2 关键记录

- `tmp/records/2026-11-07/aux-channel-finding.md`
- `tmp/records/2026-11-08/experiment-results-20261108.md`
- `tmp/records/2026-11-08/replication-report-20261108.md`
- `tmp/records/2026-11-08/margin-saturation-finding.md`
- `tmp/records/2026-11-09/inject-discrete-finding-20261109.md`
- `tmp/records/2026-11-09/injection-design-for-gpt.md`
- `tmp/records/2026-11-10/all-experiments-synthesis.md`
- `tmp/records/2026-11-10/c-line-status-data.md`
- `tmp/records/2026-11-10/c0-data-report.md`
- `tmp/records/2026-11-10/c0-matched-depth-finding.md`
- `tmp/records/2026-11-10/c0-checkpoint-sweep.md`
- `tmp/records/2026-11-10/c0-natural-swap-finding.md`
- `tmp/records/2026-11-10/datasets-and-protocols.md`
- `tmp/records/2026-11-10/experiment-registry.md`
- `tmp/records/2026-11-10/parent-2x2-finding.md`
- `tmp/records/2026-11-10/b-raw-clean-same-protocol.md`
- `tmp/ai-collab/handoff/2026-11-10-1116-claudeA-parent-corruption-2x2-epistemics.md`
- `tmp/ai-collab/shared/board.md`

### 9.3 代码/数据/权重路径

- 新 C0 配置：`experiments/configs/C0-M4-MixedN8-32-D12-ADD-L4H16D384-DeBERTa/`
- 新 C0 数据：`experiments/data/C0-MixedN8-32-D12-ADD/`
- 新 C0 生成器：`tmp/scripts/gen_mixed_n_controlled.py`
- 新 C0 日志：`tmp/runs/C0-M4-MixedN-controlled/logs/`
- 新 C0 checkpoints：`tmp/runs/C0-M4-MixedN-controlled/checkpoints/`
- 旧 C0 配置：`experiments/configs/C0-M4-Nodes32-ADD-L4H16D384-DeBERTa/`
- 旧 C0 数据：`experiments/data/Tr32Test128-ADD/`
- 旧 C0 checkpoints：`tmp/runs/C0-M4/checkpoints/`
- AB 注错：`tmp/analysis/inject-v2/`
- AB 评测：`tmp/analysis/Aprime-full/`、`B-full/`、`AL4fast-full/`、`Braw-check/`
- C0 诊断：`tmp/analysis/c0-trajectories/`、`c0-allfactor/`、`c0-decomposition/`
- C4：`tmp/analysis/c0-natural-swap/`
- matched-depth：`tmp/analysis/matched-depth/ADD/testdata_N32.pt`

### 9.4 复现命令

新 C0 单 seed：

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/train.py \
  --config_dir experiments/configs/C0-M4-MixedN8-32-D12-ADD-L4H16D384-DeBERTa \
  --seed <seed> \
  --wandb_project repro-C0-M4-MixedN8-32-D12 \
  --checkpoint_dir tmp/runs/C0-M4-MixedN-controlled/checkpoints
```

续训：

```bash
... python experiments/train.py ... \
  --resume_from_checkpoint "<checkpoint dir>/last.ckpt"
```

日志解析：`tr '\r' '\n'` 后再 grep；`total_loss/val` 属于刚完成的 epoch；文件名 `val_loss=0.0000` 是占位。

---

## 10. 一句话总结

> A′/B 线已经证明 B 的“恢复”是真实且跨条件成立的行为，但 parent 2×2 和训练时机分析把它从“检测并改回”降级为“更像按父节点重算”；旧 C0 的失败是三种失败模式 + 数据分布 OOD 的混合，不能简单归因于连续架构；C4 证明连续 hidden 可以制造可传播的下游错误但效应短暂；新 C0 在经历早期好/坏盆地反复后，s42/s1234/s2025 已收敛到 final val 2.01/5.89/10.30（best 1.96/5.89/10.30），s7 final 15.60、best 仍是 e26 12.7。下一步最关键的是新 C0 的 best-checkpoint 评估、EMPTY→5 vs 4→5 的时机对照、以及 B′ 的训练设计确认。