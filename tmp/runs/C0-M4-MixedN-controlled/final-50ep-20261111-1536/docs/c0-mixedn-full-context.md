# C0 MixedN（depth12）训练异常：完整上下文与现状

> 时间：2026-11-11 14:35（本文所有“当前”数据以此为准；§6.2/§7 已更新到 14:35）
> 仓库：`/workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures`，branch `repro`
> 目标：复现论文 *Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning*（arXiv 2510.14095v1），比较离散/连续两条路线。
>
> **本文整合了：** 2026-11-07/08 的 A/B 离散实验、2026-11-09 的旧 C0 训练、2026-11-10 的 C0 诊断/C4/AB 注错全量数据、2026-11-11 第一版 mixed-N 未完成训练和新 C0 的当前状态。此前各专题记录在文末列出。
>
> **更正（2026-11-13）**：§2.5 的 teacher-state v2 表和 §2.6 的 parent 2×2
> 数字已被修复取代（v2 teacher 轮次索引 bug；parent 行/列索引 bug），详见
> `tmp/records/2026-11-13/code-review-fixes-20261113.md`。§2.5 的 own-state 结果不受影响。

---

## 0. 摘要

- 两条线：
  1. **离散线（AB）**：A（无纠错）vs B（B-M6，有 recorrection）。已完成 raw/clean 16 档、deep_v2 压力池、A/B 单点 VALUE 注入（teacher/own、N32/N128、K4/Kall、a→a+1、4→5）、parent-corruption 2×2、aux 通道/margin 分析。
  2. **连续线（C0/M4）**：旧 C0（N32、depth6、8 loops、50k、100 epochs、3 seeds）已完成训练和全部诊断；新 C0（N8–32、depth1–12、100k、13 loops、50 epochs、4 seeds）正在训练。
- **AB 的核心结论（行为层）**：B 在单点 VALUE 注入下 u 恢复率 97.8–100%，A′ 在全部 256 图条件下 0%；但 parent 2×2 显示 B 更像“从父节点重算”，不是独立错误检测。raw 满分是 margin 尾部饱和，不是纠错证据。
- **旧 C0 的核心结论**：三个 seed 是三种失败——s42 变量名读出漂移（算术值对）、s1234 真 VALUE 错并传播、s2025 坏 seed；旧 C0 在 depth6 下最终能收敛（train loss 0.14/0.17/1.28），但严格整图在 e73–e99 已平台化。
- **新 C0 的现状**：训练曲线大震荡——四个 seed 都反复到达 val ≈ 12.7–15.5 的好状态，又反弹到 val ≈ 40–78；**截至 14:35，s1234/s2025 已完成 50 epochs（final val 5.89/10.30），s42 e45 2.14（GPU7 续训 e46–e49），s7 e45 12.9（best e26 12.7）**；best checkpoint 都在。
- **想请 GPT 判断**：这是 C0 连续递归的固有误差累积，还是优化设置（常数 lr、无 clip、loss 按 loop 直接相加、PAD 计入 CE）导致的不稳定？是否延长到 100 epochs / 加 grad clip / 降 lr / 做深度课程？如何用便宜实验区分？

---

## 1. 任务、模型与术语

### 1.1 任务

- 随机算术 DAG：6 个叶子，N 个节点（内部节点 N−6），fan-in 1–3，运算只有 ADD mod 23。
- 每个 token 用 6 个因子表示：
  - SYNTAX（7）、VARIABLE（129）、OPERATION（2）、VALUE（25）、REGRET（2）、MODIFICATION（2）。
- 模型每轮（loop）对整张图的 token 序列做一次 forward；第 d 轮可以算出 DAG 中 depth=d 的节点。
- 严格指标 `final_label_full_acc`：所有非 PAD 位置、前 4 个主因子在最后一轮全部正确；一个 VARIABLE token 错，整张图就算错。

### 1.2 关键术语

| 名称 | 含义 |
|---|---|
| A-demo | 作者公开权重，L2H16D256，无纠错 |
| A-M5 / A-seed* | 我们训练的 L2H16D256，慢路径，无纠错，seed 2025/1234/42 |
| A-L4fast | L4H16D256，fast，无纠错，seed42；用于分离架构/宽度 vs 纠错 |
| A′ | 注入实验用：L4H16D384，fast，无纠错，3 seed（e86/e90/e94） |
| B-M6 | 离散 token 反馈 + teacher predictions + recorrection VALUE corruption；L4H16D384 fast，seed42 e92 |
| cyhB | 同学迁移的 B 权重（同 B-M6 配置，未训满）：s42 e84 / s1234 e88 / s2025 e74 |
| C0 / nointerm | 论文 M4 连续基线：loop0 用 token embedding，后续 loop 用 hidden；无离散反馈；recorrection 关闭 |
| C+ | 计划中的“连续反馈 + recorrection”，尚未开始 |
| raw / clean | raw 推理把 6 个预测因子全部反馈；clean 强制 REGRET/MODIFICATION=FALSE（与训练一致） |
| max_train_loop | 训练 loop 数；`max_train_loop=13` 训练 loop 0..12，覆盖 depth 0..12 |
| K4 / Kall | 每图按深度分层 4 个 u / 全部 eligible u |
| a→a+1 / 4→5 | 注入 VALUE：a 改成 a+1 mod 23 / 4 改成 5 |

### 1.3 raw/clean 的已知 bug

训练时辅助因子 REGRET/MODIFICATION 被强制 FALSE；raw eval 却把模型预测的辅助因子反馈回去。因此 raw 正确率受“训练/推理辅助位不一致”影响，**raw 不是鲁棒性/自纠错指标**。clean 才是与训练一致的推理协议。

### 1.4 命名规范

离散：`D_{state}_{model}_{seed}_{data}_{K}_{value}`，例如 `D_tchr_Aprime_s42_N32_K4_ap1`。  
连续：`C_{kind}_C0_{seed}_{data}_{detail}`，例如 `C_traj_C0_s42_N32old_256`。  
旧 tag 与规范名的对应见 `tmp/records/2026-11-10/experiment-registry.md`。

---

## 2. AB 离散线：完整数据

### 2.1 权重清单

| 名称 | 架构 | 纠错训练 | 训练路径 | 训练种子 / checkpoint |
|---|---|---|---|---|
| A-demo | L2H16D256 | 无 | 作者权重（fast） | 1584791061 |
| A-M5 | L2H16D256 | 无 | 慢路径 | 2025 / 1234 / 42，各 156,300 步 |
| A-L4fast | L4H16D256 | 无 | fast | 42，e99 |
| A′ | L4H16D384 | 无 | fast | 42 e86、1234 e90、2025 e94 |
| B-M6 | L4H16D384 | **有**（corruption_prob=0.05，strategy last，tokenwise 0.5） | fast | 42 e92（best val 3.25e-6） |
| cyhB | L4H16D384 | 有 | 迁移包（未训满） | 42 e84、1234 e88、2025 e74 |

A′ 与 B-M6 架构/数据完全相同，只差纠错开关，是干净的消融对照；B-M6 只有单 seed，A′ 有 3 seed。

### 2.2 标准 16 档 raw/clean（论文协议）

每格 = 该规模 1,024 题的整题全对率（%）。N=32 是训练规模，>32 为 OOD。  
数据：`Tr32Test128/ADD/testdata_N{8..128}.pt`，每档 1,024 题。

| 权重 | 口径 | N8 | N16 | N24 | N32 | N40 | N48 | N56 | N64 | N72 | N80 | N88 | N96 | N104 | N112 | N120 | N128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-demo | raw | 100 | 100 | 99.7 | 99.8 | 100 | 99.4 | 99.6 | 99.2 | 99.5 | 99.4 | 98.8 | 98.1 | 98.8 | 97.5 | 97.4 | 95.1 |
| A-demo | clean | 100 | 100 | 99.9 | 99.9 | 99.9 | 100 | 99.9 | 99.9 | 99.9 | 99.9 | 99.9 | 99.7 | 99.6 | 99.6 | 99.2 | 99.4 |
| A-s2025 | raw | 99.8 | 99.2 | 98.6 | 98.2 | 97.1 | 95.7 | 96.6 | 95.1 | 94.9 | 93.7 | 92.6 | 93.4 | 90.9 | 90.4 | 89.6 | 88.1 |
| A-s2025 | clean | 100 | 99.9 | 100 | 100 | 100 | 100 | 100 | 99.9 | 99.8 | 99.8 | 100 | 100 | 99.9 | 100 | 99.9 | 99.9 |
| A-s1234 | raw | 99.3 | 88.0 | 77.1 | 67.9 | 58.5 | 51.8 | 45.1 | 39.3 | 31.8 | 28.5 | 23.6 | 21.1 | 17.5 | 15.0 | 11.8 | 9.8 |
| A-s1234 | clean | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| A-s42 | raw | 88.4 | 47.4 | 22.5 | 10.0 | 3.7 | 1.2 | 0.4 | 0.2 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| A-s42 | clean | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99.9 | 100 | 99.9 |
| B-M6 | raw | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| B-M6 | clean | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99.9 | 100 |

关键读法：

- raw 的四行 A 同架构/同数据/同训练步数，**只差 seed**，N128 从 0% 到 95.1%；
- clean 下四份 A 权重全部 ~100%；论文图4 的 M5→M6 差距大半是评估协议噪声；
- B-M6 raw/clean 全满分，不能单独证明“纠错”，因为 raw 是 margin 尾部问题（见 2.4）。

#### A′ per-N 16 档（`tmp/analysis/Aprime-full/logs/`）

| seed | 口径 | N8 | N16 | N24 | N32 | N40 | N48 | N56 | N64 | N72 | N80 | N88 | N96 | N104 | N112 | N120 | N128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s42 | clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 1.0000 | 1.0000 | 0.9980 | 0.9990 | 1.0000 | 0.9980 | 0.9980 |
| s42 | raw | 1.0000 | 1.0000 | 1.0000 | 0.9990 | 0.9971 | 0.9990 | 1.0000 | 0.9971 | 0.9951 | 0.9990 | 0.9932 | 0.9951 | 0.9912 | 0.9863 | 0.9883 | 0.9854 |
| s1234 | clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| s1234 | raw | 1.0000 | 0.9990 | 0.9990 | 0.9980 | 0.9980 | 1.0000 | 0.9980 | 0.9961 | 0.9961 | 0.9932 | 0.9941 | 0.9971 | 0.9951 | 0.9941 | 0.9922 | 0.9902 |
| s2025 | clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| s2025 | raw | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 1.0000 | 1.0000 |

A′ 与 B-M6 同架构/同数据，只差纠错开关；A′ 在 clean 下 16 档几乎全满分，raw 也 0.985+。这进一步说明 raw/clean 的差别不是算术能力。

#### B-M6 / cyhB 同协议复核（fig4 N32/N128，1,024 样本）

| 权重 | N | raw | clean |
|---|---:|---:|---:|
| BM6 s42 | 32 | 1.0000 | 1.0000 |
| BM6 s42 | 128 | 1.0000 | 1.0000 |
| cyhB s42 | 32 | 0.9609 | 1.0000 |
| cyhB s42 | 128 | 0.7910 | 1.0000 |
| cyhB s1234 | 32 | 0.9941 | 1.0000 |
| cyhB s1234 | 128 | 0.9619 | 0.9980 |
| cyhB s2025 | 32 | 0.6650 | 1.0000 |
| cyhB s2025 | 128 | **0.2158** | **0.9990** |

clean 四份权重几乎满分；raw 与权重强相关（cyhB s2025 N128 崩到 0.2158）。raw 不是鲁棒性/自纠错指标。

### 2.3 A′ / A-L4fast / cyhB 的补充评测

A′ 最终评测（fig4 16 档均值 / deep_v2 四池均值）：

| seed | clean fig4 | raw fig4 | clean deep_v2 | raw deep_v2 |
|---|---:|---:|---:|---:|
| s42 | 0.9995 | 0.9954 | 0.9993 | 0.9866 |
| s1234 | 1.0000 | 0.9963 | 1.0000 | 0.9609 |
| s2025 | 1.0000 | 0.9998 | 1.0000 | 0.9983 |

A-L4fast / cyhB 压力池（整题全对 %；wide8 是宽度压力，mid60/chain110/chain122 是深度压力）：

| 权重 | 口径 | N64 | N128 | wide8 | mid60 | chain110 | chain122 | wide8 tf 单步 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B-M6 (ours, e92) | clean | 100 | 100 | 100 | 100 | 100 | 100 | 1.000 |
| B-M6 (ours) | raw | 100 | 100 | 100 | 100 | 100 | 100 | 1.000 |
| cyhB s42 (e84) | clean | 100 | 100 | 100 | 100 | 99.9 | 99.9 | 1.000 |
| cyhB s42 | raw | 90.3 | 79.1 | 79.4 | 74.8 | 72.6 | 67.1 | 1.000 |
| cyhB s1234 (e88) | clean | 100 | 99.8 | 100 | 99.9 | 99.5 | 100 | 1.000 |
| cyhB s1234 | raw | 98.6 | 96.2 | 97.9 | 85.4 | 83.4 | 80.0 | 1.000 |
| cyhB s2025 (e74) | clean | 100 | 99.9 | 100 | 99.9 | 99.9 | 100 | 1.000 |
| cyhB s2025 | raw | 42.6 | 21.6 | 36.0 | 8.7 | 5.7 | 4.8 | 1.000 |
| A-L4fast (e99) | clean | 100 | 100 | 100 | 100 | 100 | 100 | 1.000 |
| A-L4fast | raw | 100 | 100 | 99.9 | 99.9 | 99.8 | 99.9 | 1.000 |
| A-demo (L2) | clean | 99.9 | 99.4 | **82.7** | 98.9 | 98.7 | 98.3 | **0.592** |
| A-s2025 (L2) | clean | 99.9 | 99.9 | 99.7 | 99.6 | 99.7 | 99.7 | 0.990 |

现象：

- A-L4fast **无纠错、aux 翻转 38%/46%，raw 仍 100%** → 纠错不是 raw 鲁棒性的来源；
- cyhB 有纠错但未训满，raw 随深度/宽度下降（s2025 最差 4.8%）；
- A-demo（L2）在 wide8 的 **tf 单步只有 0.592**，说明宽度检索才是 A 的真实短板；L4 架构把它修到 1.000。
- v1 深度池（64 题/池）：A-demo 98.4/96.9/95.3，A-s2025 100/100/98.4，B 100/100/100（depth 27–48 / 49–76 / 77–97）。

### 2.4 aux 通道与 margin 尾部：raw 满分的真实机制

aux 翻转率（raw 回灌，N128）：

| 模型 | REGRET=TRUE | MODIFICATION=TRUE | raw 成绩 | 纠错训练 |
|---|---:|---:|---:|---|
| B-M6 (ours) | 27% | 66% | 100% | 有 |
| A-L4fast | 38% | 46% | 100% | **无** |
| A-M5 各 seed | 不等 | 不等 | 0–95% | 无 |

logit/margin 分析（N128，32 题，24,290 个非 PAD 位置）：

| 模型 | VALUE margin 最小 / 均值 | argmax 翻转数 |
|---|---|---:|
| A-L4fast | 7.5 / 34.8 | 0 |
| B-M6 s42 | 4.9 / 28.3 | 0 |
| A-M5 s42 | **0.20 / 43.5** | **3008** |
| A-M5 s2025 | 8.1 / 32.6 | 6 |

- aux 噪声让 logits 偏移，但只有偏移 > margin 的位置才翻 argmax；
- 整图全对是数千个位置的 AND，决定成败的是 margin 左尾，不是平均值；
- A-M5 s42 平均 margin 最高，但尾部到 0.20 → 3008 次翻转 → raw 归零；
- **raw 满分 = 被动饱和/不敏感，不是主动纠错。**
- B checkpoint 筛查：e91 clean 100/raw 86.2；e92 clean 100/raw 100 → raw 随 checkpoint 尾部波动。

### 2.5 单点 VALUE 注入：A′ vs B-M6 vs cyhB（全条件）

#### 设计与协议

- 在已算对的节点 u（定义位置 p、深度 d、真值 4 或 a）注入 VALUE：
  - 主协议 a→(a+1) mod 23；
  - 4→5 只选值为 4 的 u。
- teacher-state：从 clean teacher 快照 `S[d+1]` 注入；own-state：从模型自己自由续算到 u 算出的状态注入，只保留两边 u 都算对的图。
- K4 = 每图按深度分层 4 个 u；Kall = 所有 eligible u。
- free-run：无 teacher、无 recorrection、aux 强制 FALSE；记录 u 的 r1/r2/r4 和直接子节点 q 的 r1/r2/r4。
- clean 对照：不注入，确认 u/q 读出正常。A′ 三 seed 和 B-M6 均 47/47 通过（旧 64 样本集）。
- 指标：u recovered（回到注入前正确值）、q correct/no_value/expected_wrong/other_wrong。

#### 全条件表（新留出 seed42，256 图；`tmp/analysis/inject-v2/`）

> 说明：`u recovered` 是 u 回到正确值的比例；`exp_wrong` 表示 q 精确等于“用注入值重算”的错值；`no_value` 表示 q 还没算出来（EMPTY）。  
> `4→5` 行的 K 列标为 4→5，是因为 tag 用 `45`；其余 K4/Kall 均为主协议 a→a+1。

| model | seed | state | data | K | value | n_graphs | n_records | u recovered | q r1 | q r4 |
|---|---|---|---|---|---:|---:|---:|---|---|---|
| Aprime | s1234 | teacher | N128 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1567 | exp_wrong 2274; other_wrong 155 |
| Aprime | s1234 | teacher | N32 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1537 | exp_wrong 1956; other_wrong 197 |
| Aprime | s1234 | teacher | N32 | Kall | a→a+1 | 256 | 11868 | 0.00% | exp_wrong 5767 | exp_wrong 7159; other_wrong 662 |
| Aprime | s1234 | own | N128 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1567 | exp_wrong 2274; other_wrong 155 |
| Aprime | s1234 | own | N32 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1537 | exp_wrong 1956; other_wrong 197 |
| Aprime | s2025 | teacher | N128 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1567 | exp_wrong 2274; other_wrong 155 |
| Aprime | s2025 | teacher | N32 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1537 | exp_wrong 1956; other_wrong 197 |
| Aprime | s2025 | own | N128 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1567 | exp_wrong 2274; other_wrong 155 |
| Aprime | s2025 | own | N32 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1537 | exp_wrong 1956; other_wrong 197 |
| Aprime | s42 | teacher | N128 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1567 | exp_wrong 2274; other_wrong 155 |
| Aprime | s42 | teacher | N32 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1537 | exp_wrong 1956; other_wrong 197 |
| Aprime | s42 | teacher | N32 | Kall | a→a+1 | 256 | 11868 | 0.00% | exp_wrong 5767 | exp_wrong 7159; other_wrong 662 |
| Aprime | s42 | own | N128 | 4→5 | 4→5 | 256 | 159 | 0.00% | exp_wrong 79 | exp_wrong 110; other_wrong 12 |
| Aprime | s42 | own | N128 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1567 | exp_wrong 2274; other_wrong 155 |
| Aprime | s42 | own | N32 | 4→5 | 4→5 | 256 | 123 | 0.00% | exp_wrong 62 | exp_wrong 76; other_wrong 8 |
| Aprime | s42 | own | N32 | K4 | a→a+1 | 256 | 3072 | 0.00% | exp_wrong 1537 | exp_wrong 1956; other_wrong 197 |
| BM6 | s42 | teacher | N128 | K4 | a→a+1 | 256 | 3072 | 100.00% | correct 1567 | correct 2429 |
| BM6 | s42 | teacher | N32 | K4 | a→a+1 | 256 | 3072 | 100.00% | correct 1537 | correct 2153 |
| BM6 | s42 | teacher | N32 | Kall | a→a+1 | 256 | 11868 | 100.00% | correct 5767 | correct 7821 |
| BM6 | s42 | own | N128 | 4→5 | 4→5 | 256 | 159 | 100.00% | no_value 48; exp_wrong 31 | correct 116; no_value 6 |
| BM6 | s42 | own | N128 | K4 | a→a+1 | 256 | 3072 | 100.00% | no_value 539; exp_wrong 1028 | correct 2306; no_value 123 |
| BM6 | s42 | own | N32 | 4→5 | 4→5 | 256 | 123 | 100.00% | no_value 35; exp_wrong 27 | correct 82; no_value 2 |
| BM6 | s42 | own | N32 | K4 | a→a+1 | 256 | 3072 | 100.00% | no_value 610; exp_wrong 927 | correct 2121; no_value 29; exp_wrong 3 |
| cyhB | s1234 | teacher | N128 | K4 | a→a+1 | 256 | 3072 | 99.90% | correct 1564; exp_wrong 3 | correct 2425; exp_wrong 4 |
| cyhB | s1234 | teacher | N32 | K4 | a→a+1 | 256 | 3072 | 99.80% | correct 1531; exp_wrong 6 | correct 2143; exp_wrong 8; other_wrong 2 |
| cyhB | s1234 | teacher | N32 | Kall | a→a+1 | 256 | 11868 | 99.87% | correct 5749; exp_wrong 18 | correct 7792; exp_wrong 26; other_wrong 3 |
| cyhB | s1234 | own | N128 | K4 | a→a+1 | 256 | 3072 | 99.90% | no_value 213; exp_wrong 1354 | correct 2298; no_value 119; exp_wrong 11; other_wrong 1 |
| cyhB | s1234 | own | N32 | K4 | a→a+1 | 256 | 3072 | 99.80% | no_value 295; exp_wrong 1242 | correct 2109; no_value 23; exp_wrong 18; other_wrong 3 |
| cyhB | s2025 | teacher | N128 | K4 | a→a+1 | 256 | 3072 | 100.00% | correct 1567 | correct 2403; no_value 26 |
| cyhB | s2025 | teacher | N32 | K4 | a→a+1 | 256 | 3072 | 100.00% | correct 1537 | correct 2153 |
| cyhB | s2025 | own | N128 | K4 | a→a+1 | 256 | 3072 | 100.00% | no_value 843; exp_wrong 724 | correct 2135; no_value 294 |
| cyhB | s2025 | own | N32 | K4 | a→a+1 | 256 | 3072 | 100.00% | no_value 938; exp_wrong 599 | correct 2113; no_value 40 |
| cyhB | s42 | teacher | N32 | K4 | a→a+1 | 256 | 3072 | 97.75% | correct 1470; exp_wrong 67 | correct 2066; exp_wrong 80; other_wrong 7 |
| cyhB | s42 | own | N32 | K4 | a→a+1 | 256 | 3072 | 97.75% | no_value 87; exp_wrong 1450 | correct 2039; no_value 13; exp_wrong 94; other_wrong 7 |

未完成的条件：own-state Kall（K0）全量（已停，待单线程重跑）；cyhB s42 的 N128 teacher/own；部分 4→5（cyhB/BM6 teacher）。cf. registry。

#### 早期 4→5 teacher-state（旧 64 样本集，47 候选）

| 指标 | A′ s42 | A′ s1234 | A′ s2025 | B-M6 s42 |
|---|---:|---:|---:|---:|
| u 第一步改回正确值 | 2.1% (1/47) | 10.6% (5/47) | 0% | **100%** (47/47) |
| u 最终改回 | 2.1% | 10.6% | 0% | 100% |
| q 第一次出现即正确 | 0% | 2.1% | 0% | **61.7%** (29/47) |
| q 最终正确 | 0% | 2.1% | 0% | **100%** (47/47) |
| q 先错后改回 | 0% | 0% | 0% | 38.3% (18/47) |

A′ 三 seed 合计：u 第一步改回 6/141 = 4.3%；q 最终正确 1/141 = 0.7%。  
B-M6 同轮（dq=d+1）41 例：**推迟一轮再算 23/41**（第 0 轮 q EMPTY，第 1 轮直接算对）；**先算错再改 18/41**。q 从未在注入发生的同一 forward 内直接算对。延迟下游 6/6 第一次出现就正确。clean 对照 47/47。

#### 现象总结

- **A′**：所有 256 图条件 u 恢复 0%；q 永远停在 expected_wrong（或 r4 的 other_wrong）。A′ 的行为是复制/漂移，不是纠错。
- **B-M6**：u 恢复 100%；teacher-state 下 q r1 直接 correct；own-state 下 q r1 是 no_value 或 expected_wrong，r2/r4 基本全 correct。
- **cyhB**：u 恢复 97.75–100%；s42 e84 略弱（97.75%），s1234 99.8–99.9%，s2025 100%。
- **teacher vs own**：u 恢复方向完全一致；差异主要在 q r1：own-state 有大量 no_value（模型还没算 q），说明 q 的可算时机受 own 状态影响。
- **N32 vs N128**：方向一致；N128 的 q 数更多，但比例不变。
- **K4 vs Kall**：方向一致；Kall 的 q 数更大，结论不变。
- **a→a+1 vs 4→5**：方向一致；4→5 的 eligible u 少（s42 N32 41 个 u、N128 53 个）。
- **同轮结构性**：B 的纠错发生在**轮边界**，不是单次 forward 内部；同一 forward 里 u 的纠正无法传给同轮计算的 q，所以 q 要么推迟、要么先错后改。

### 2.6 Parent-corruption 2×2：区分 recompute vs recorrect

协议：teacher-state，N=32，256 图，K=4。选 u 的父节点 r，做 2×2：clean / 只改 u（u 真值 a → a+1）/ 只改父 r（r_true → r_true+3）/ 两者。关键条件是 **R only**：u 本身正确，父节点错；若模型从父节点重算，u 会变成按错父值算出的 `a_recomp = a+3`。

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

解读（未最终定论）：B 更像 **父节点优先重算**，不是“检测到 u 错再改回”。

### 2.7 连续线对照：logit-steering 注错失败

在 C0 loop d 结束、进入 d+1 前，对 u 的 hidden 加 `δ = α(w_5 − w_4)`，把 VALUE 头 argmax 从 4 翻到 5（N32，64 样本，47 候选）：

- 注入前读出 4：47/47；注入后读出 5：46/47；
- **下一 loop u 的读出回到 4：47/47**；
- 下游 q 第一次读出**从未等于语义错值 14**（q_first_wrong = 0/47）；同轮 q 是其他值后收敛；q 最终正确 47/47；
- clean 对照 47/47。

解释：C0 每轮从父节点重算，hidden 扰动是 OOD 的，下一轮被覆盖。这直接推动了 C4“自然反事实 hidden 替换”的设计。

### 2.8 AB 数据与当前 C0 问题的关系

- AB 的 B 会“恢复”被注错的 u，但 parent 2×2 说明它很可能是**从父节点重算**，而不是独立检测错误；A′ 则是复制输入值。
- raw/clean 的差异和 aux 翻转不能作为纠错证据；raw 满分是 margin 尾部饱和。
- C0 没有离散 token 反馈，既不能像 A′ 那样“复制离散 token”，也没有 B 的离散重算路径；它的中间状态是 continuous hidden，误差会在 loop 间累积。C4 显示自然 hidden 替换能短暂传播到下游（post1 77.4%/73.6%），但 post2/4 大多被改回——这与 C0 每轮重算的图景一致。
- 因此当前新 C0 的 loss 震荡，不能直接套用 AB 的“纠错”解释；需要从连续递归/优化稳定性角度分析。

---

## 3. 旧 C0（N32、depth6、100 epochs）：完整训练与诊断

### 3.1 设置与运行历史

| 项目 | 值 |
|---|---|
| 方法 | nointerm / M4，连续 hidden 状态 |
| 配置 | `experiments/configs/C0-M4-Nodes32-ADD-L4H16D384-DeBERTa` |
| 架构 | L4H16D384，DeBERTa 位置编码 |
| 训练数据 | N=32，max_depth=6，max_length=256，50,000 图（`experiments/data/Tr32Test128-ADD/`） |
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

- 2026-11-09 10:50 第一次启动三个 seed，分别跑到 e3 / e2 / e5（不同 seed 不同）后停止；
- 15:42（s42）/ 16:10（s1234、s2025）从各自 `last.ckpt` 续训；
- 最终 `last.ckpt`：s42 e94、s1234 e96、s2025 e99；top-k 保存 e73/90/94、e89/95/96、e73/97/99；
- 单份 100 epochs 约 8 小时（优化后速度）。

注意：验证集 N128/depth36 是训练时 OOD，val loss 绝对值不能和新 C0 的 in-distribution val 直接比。

### 3.2 训练曲线

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

真实 val loss（监控 key 是 `total_loss/val`；文件名 `val_loss=0.0000` 是占位，不是真值）：

| seed | e70 | e90 | e99 |
|---|---:|---:|---:|
| s42 | 1.27 | 0.57 | 2.22 |
| s1234 | 3.59 | 1.34 | 1.43 |
| s2025 | 6.07 | 8.48 | 4.41 |

现象：

- 三个 seed 都有早期尖峰，不是 s2025 独有；
- s2025 在 e2–e3 就冲到 33，长期在高 loss 区（e30 35.3、e40 34.4、e50 18.8），到 e60 后才降到 4.78；
- 但最终 train loss 都能降下来；严格整图却不一定（见 3.3）。

### 3.3 评测

#### 16 档（old Tr32Test128，每档 1,024 题，严格整图）

| N | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s42 | .999 | .946 | .815 | .553 | .002 | .005 | .006 | 0 |
| s1234 | .993 | .835 | .642 | .380 | .019 | .024 | .010 | 0 |
| s2025 | .014 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

16 档均值：s42 0.2079，s1234 0.1814，s2025 0.0009。

注意：old N=32 测试集 max_depth=10；新留出 N32 max_depth=11；训练 max_depth=6。所以“N=32 测试”不是训练同分布，是更深/更多轮的 OOD。

#### 新留出 seed42（N32/N128，每档 1,024 题）

| 数据 | s42 | s1234 | s2025 |
|---|---:|---:|---:|
| N=32 | 0.2715 | 0.2949 | 0 |
| N=128 | 0 | 0 | 0 |

#### deep_v2（四个深/宽 OOD 池，每池 1,024 题）

- s42 / s1234：四个池全 0；
- s2025：未跑（16 档/留出已经全 0）。

对照：离散 A′ clean 在 deep_v2 上约 **1.0**——数据和 label 没问题，问题在 C0 的连续中间状态。

#### matched-depth N32（训练同分布）

新生成 N=32、max_depth=6、1024 图、seed=42；路径 `tmp/analysis/matched-depth/ADD/testdata_N32.pt`，SHA256 `3088f313...`。

| seed | matched-depth full | old N32 full |
|---|---:|---:|
| s42 | **0.9736** | 0.5527 |
| s1234 | **0.5059** | 0.3799 |
| s2025 | **0.0000** | 0.0000 |

- s42：训练同分布 97.4%，到 old N32（depth 10）掉到 55.3% → 主要是深度/轮数 OOD；
- s1234：训练同分布也只有 50.6% → VALUE 错误在训练分布内就存在；
- s2025：训练同分布 0% → 训练本身失败，不是 OOD。

#### checkpoint sweep（matched-depth）

| seed | e73/89 | e90/95 | e94/96 | last.ckpt |
|---|---:|---:|---:|---:|
| s42 | .9736 (e73) | .9736 (e90) | .9736 (e94) | .9736 (e94) |
| s1234 | .5059 (e89) | .5059 (e95) | .5059 (e96) | .5059 (e96) |
| s2025 | 0 (e73) | 0 (e97) | 0 (e99) | 0 (e99) |

**严格整图在 e73–e99 完全平台化**，同时 train loss 仍在变化（s1234 从 e70 0.992 降到 e99 0.165）。  
含义：继续同样的训练没有带来任务收益；要提升需要改变优化轨迹/训练设置，而不只是多做同样的步数。

### 3.4 三个 seed 的失败模式（C0 诊断）

诊断数据：old N32，256 图，8,192 个节点定义，自由续算 10–14 轮；纯推理、无注错、无 teacher forcing。  
三个脚本：`collect_c0_trajectories.py`、`collect_c0_allfactor.py`、`decompose_c0_failures.py`。

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

#### s2025 按深度的“始终正确”比例

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

模式：叶子（depth0）全对，越深越不会算；不是“错后恢复”，而是“从头不会”。

#### 具体例子

- **s42 graph 16**：第 13 轮三个定义位置 VARIABLE 同时漂移（位置 72 真名 x_118 读成 x_75，值仍 13；位置 78、90 类似）。位置 72 的子节点 160 真值 22，若真按 x_75 的值 6 算应是 15，实际仍 22；子节点 184 真值 4，按错名算应是 20，实际仍 4。→ **名字读错，计算没跟着错名走**。
- **s1234 graph 18**：位置 78（x_81，值 8）第 7 轮起 VALUE 变成 12；子节点 118 表达式 14+2+x_81，真值 1，第 7 轮起变成 5；14+2+12=28 mod 23=5，精确匹配。位置 84（值 13）第 4 轮起变 17；子节点 104 真值 21，第 4 轮起变 2；17+8=25 mod 23=2，精确匹配。→ **父节点 VALUE 真的错了，子节点按错值重算**。

#### Caveats

1. 诊断只覆盖 256 张 old N32 图，不是 16 档全量。
2. `first_wrong_loop=0` 是 artifact：第 0 轮很多值还没算，整图必然不全对；要看 `ref_round`。
3. per-loop 曲线是条件准确率（后面轮次只有还活着的图），图级结论用“各自最后一轮”的 130/256。
4. 逐因子准确率包含 PAD/N/A；严格整图才是任务指标。
5. 语义检验用其他父节点的真值；本报告例子都检查过其他父节点正确。
6. 变量漂移是读出层面的推断，没有直接打开 hidden state。
7. `last.ckpt` 不是“最佳 checkpoint”；三个 seed 的文件名 `val_loss=0.0000` 是占位。
8. C0 评测是 nointerm 自由续算，不涉及 raw/clean 辅助位回灌。

### 3.5 C4 自然 hidden 替换（旧 C0）

做法：

- 选真值 u=4、有直接子节点 q 的节点 u；
- 找一个叶子祖先，改叶子值 delta，使同一张图的 u 变成 5（结构不变）；
- 分别跑 original（u=4）和 counterfactual（u=5）的 clean 轨迹，取 loop=d 后的 hidden；
- 把 original 的 u hidden 换成 counterfactual 的 5-hidden，继续跑，看 q 是否移动到 `q_u5`。

条件：`orig_noswap`、`cf_noswap`、`orig_swap45`、`cf_swap54`；对照 `orig_swap_randperm`（随机置换 hidden 维度）和 `orig_swap_noise`（同尺度高斯噪声）。

有效样本：

| seed | 图数 | 有效 pair | 跳过 | 其中无 counterfactual | u 读出不符 |
|---|---:|---:|---:|---:|---:|
| s42 | 256 | **99** | 157 | 157 | 0 |
| s1234 | 256 | **90** | 166 | 157 | 9 |

主要结果（4→5，post1）：

| seed | 条件 | n | moved_to_q_u5 | stayed_orig | other |
|---|---|---:|---:|---:|---:|
| s42 | orig_swap45 | 177 | **137 (77.4%)** | 0 | 40 |
| s42 | randperm | 177 | 6 (3.4%) | 5 | 166 |
| s42 | noise | 177 | 0 (0%) | 25 | 152 |
| s1234 | orig_swap45 | 163 | **120 (73.6%)** | 0 | 43 |
| s1234 | randperm | 163 | 3 (1.8%) | 10 | 150 |
| s1234 | noise | 163 | 1 (0.6%) | 47 | 115 |

按 pair 聚合（任一 q 移动）：

| seed | 条件 | pairs | any_moved |
|---|---|---:|---:|
| s42 | orig_swap45 | 99 | **87 (87.9%)** |
| s42 | randperm | 99 | 5 |
| s42 | noise | 99 | 0 |
| s1234 | orig_swap45 | 90 | **76 (84.4%)** |
| s1234 | randperm | 90 | 2 |
| s1234 | noise | 90 | 1 |

时间演化：

- post2：s42 10/177 moved；s1234 3/163；
- post4：s42 2/177；s1234 0/163；
- 效应是 transient，post2/4 大部分回到原值。

反向 swap（cf_swap54，post1）：s42 84/177 = 47.5%；s1234 77/163 = 47.2%——比正向弱，方向仍一致。

u 自己的读出（orig_swap45）：

| seed | post1 保持原值 | post2 | post4 |
|---|---:|---:|---:|
| s42 | 99/99 | 99/99 | 99/99 |
| s1234 | 57/90 | 82/90 | 88/90 |

- s42：hidden 换了，u 的 VALUE 读出仍保持 4 → u 自己是从父节点重算的，hidden 替换主要影响下游 q；
- s1234：u 读出更容易被 swap 扰动，和它本身 VALUE 不稳定一致。

C4 结论：

1. 自然 hidden 替换确实能传播到下游（post1 77.4%/73.6%，pair 87.9%/84.4%）；
2. 传播是短暂的（post2/4 大部分回原值）；
3. 随机置换/噪声很少产生 q_u5（3.4%/1.8%、0%/0.6%），说明不是“任何扰动都会让 q 动”；
4. 但仍缺 **in-distribution 4→4 donor** 对照；随机置换/噪声都是 OOD，不能完全排除“donor hidden 是 in-distribution”这一替代解释。

### 3.6 旧 C0 对当前新 C0 的含义

- 旧 C0 的训练不稳定是既有的：s2025 e2–e3 就冲到 33，e30 仍 35.3，e60 才降到 4.8；不是新问题。
- 旧 C0 最终在 depth6 上能收敛（train 0.14/0.17/1.28），但严格整图在 e73–e99 平台化；s42 0.97、s1234 0.51、s2025 0。
- 新 C0 把 depth 从 6 拉到 12、loops 从 8 拉到 13、N 从固定 32 变成 8–32、数据从 50k 增到 100k；这会把旧 C0 的早期不稳定放大。
- 旧 C0 已经说明：**train/val loss 下降不等于严格整图提升**；评估必须看 best checkpoint 的严格整图，而不是当前 loss。
- 旧 C0 的三种失败模式也提醒：新 C0 即使 loss 降下来，也可能分别出现 VARIABLE 漂移（s42 型）、VALUE 传播错（s1234 型）或整体没学会（s2025 型）；需要分 seed、分深度诊断。

---

## 4. 新 C0 数据：最终版（controlled）

### 4.1 第一版 mixed-N 训练（已停，未产生 epoch）

- 2026-11-11 01:29 用 `tmp/scripts/gen_mixed_n_data.py`（bucket + 拓扑池）生成的 97,999 样本启动 seeds 42/1234，`max_length=512`；
- 训练在 sanity check 阶段被停掉（日志 `tmp/runs/C0-M4-MixedN/logs/`，无 checkpoint、无 epoch）；
- 原因：第一版数据深度严重偏斜：
  - N32 depth12 13,333、depth4 10,649；
  - depth9–11 几乎为 0；
  - depth1 为 0；
- 这类分布无法支撑“可变 N + depth12”的泛化结论，因此弃用并重新生成。

### 4.2 最终生成器

`tmp/scripts/gen_mixed_n_controlled.py`：

- `ControlledDepthDAG`：先建一条长度恰为 d 的随机链（节点 6..6+d−1），再把剩余节点挂到 depth < d 的父节点上，保证每张图 max depth 精确等于 d；
- N 权重：8/16/24/32 = 10%/20%/30%/40%；
- 可行深度：
  - N=8：d=1–2（N=8 只有 2 个内部节点，最大深度只能是 2）；
  - N=16：d=1–10；
  - N=24 / N=32：d=1–12；
- 每个 N×d 格等配额；
- 100k train / 2k val，seed 20261111 / 20261188；
- `max_length=256`。

### 4.3 数据审计（manifest）

| 项 | train | val |
|---|---:|---:|
| 样本数 | 100,000 | 2,000 |
| 张量形状 | [100000, 256, 6] | [2000, 256, 6] |
| N=8/16/24/32 | 10k/20k/30k/40k | 200/400/600/800 |
| 深度范围 | 1–12，均值 5.80 | 1–12，均值 5.79 |
| 非 PAD 长度 min/mean/max | 31 / 131.0 / 213 | 31 / 130.7 / 203 |
| fan-in 1/2/3 | 1,199,698 / 600,799 / 599,503 ≈ 2:1:1 | 同比例 |
| 同层宽度 mean/max | 8.1 / 26 | 8.1 / 26 |

train 深度直方图：d1/d2 各 12,834；d3–d10 各 7,833–7,834；d11/d12 各 5,833。  
每个 N×d 格都被精确填满，没有第一版的空洞。

数据路径：`experiments/data/C0-MixedN8-32-D12-ADD/{train_data.pt,val_data.pt,manifest.json}`。

---

## 5. 新 C0 训练配置与 loss 语义

配置：`experiments/configs/C0-M4-MixedN8-32-D12-ADD-L4H16D384-DeBERTa`

### 5.1 参数

- 4 layers，16 heads，hidden 384，DeBERTa 位置编码，dropout 0.1，input_size 256，compile true；
- batch 32，AdamW lr 1e-3，weight decay 0.01；
- `lr_scheduler: none`（常数 lr），无 warmup，无 gradient clipping；
- `max_epochs: 50`，`max_train_loop: 13`，`max_val_loop: 13`，`max_train_depth: 12`；
- `max_length: 256`，`num_workers: 4`；
- `nointerm: true`，`recorrection enable: false`（C0/M4）。

### 5.2 Loss 实现（`compute_batch_loss_nointerm`）

- 对每个 loop 调 `self.model(depth_batch, True)`，label 是 `batch_ext[loop+1]`；
- 对 6 个因子逐个算 `CrossEntropyLoss()`；**没有 ignore_index，PAD 位置也计入 CE**；
- `factor_loss_scales = [1, 1, 1, 1, 0, 0]`：前 4 个主因子权重 1，REGRET/MODIFICATION 权重 0；
- `depth_loss = Σ_factor scaled_loss`；
- `total_loss += depth_loss`，**不除以 loop 数**；
- 训练时一次 `manual_backward(total_loss); opt.step()`；
- 因此 `total_loss` 是“13 个 loop × 4 个主因子”的和：
  - 坏状态 ~50 → 50/13 ≈ 3.8 per loop；
  - 好状态 ~14 → 14/13 ≈ 1.1 per loop；
  - 旧 C0 收敛 val 2.22 是 8 loops → 0.28 per loop。绝对数不能直接跨任务比较。

### 5.3 Checkpoint 行为

- `ModelCheckpoint(dirpath=..., filename='{epoch}-{val_loss:.4f}', monitor='total_loss/val', mode='min', save_top_k=3, save_last=True)`；
- `filename` 里的 `val_loss` 没有被 log，文件名永远显示 `val_loss=0.0000`；真实指标是日志里的 `total_loss/val`；
- 这个 Lightning 版本的 `save_last=True` 是“**每次保存 checkpoint 时顺便更新 last.ckpt**”，不是每 epoch 更新；如果连续多个 epoch 没有进入 top-3，`last.ckpt` 会停留在最后一次 top-k；
- `on_train_end` 会保证训练结束时保存最终模型；之后仍可用于延长训练；
- top-k 轮转：新的更好 checkpoint 进入 top-3 时，旧 top-k 文件被删除；坏状态不会保留。

### 5.4 吞吐与内存

- seq512 + batch32 + depth12 会 OOM（约 60–79GB），因此改用 seq256；
- seq256 下单进程实际约 **26.7GB**（之前看到的 46–48GB 含同卡他人任务）；
- 3125 steps/epoch（100k / batch32）；
- 干净 H100 上 3.4–3.7 it/s，约 **14–15 min/epoch**；50 epochs 约 12h。

---

## 6. 新 C0 启动、迁移与当前状态（10:36）

### 6.1 启动与迁移

- 01:40 第一次启动 4 seed，遇到 OOM / GPU 竞争，失败；
- 01:45–01:48 用 seq256 重新启动：s42→GPU0，s1234→GPU5，s2025→GPU7，s7→GPU2；
- **02:24** s7 从 GPU2 迁移到刚空出的 GPU4，从 epoch0 的 `last.ckpt` 断点续训；
- **02:26** s42 从 GPU0 迁移到 GPU6：GPU0 总占用达到 79.3/81.5GB（同卡他人任务从 21GB 涨到 52GB，s42 只剩 2.3GB 余量），为避免 OOM 主动迁移；
- 两次迁移日志均打印 `Restored all states from the checkpoint`，config/seed 不变；
- 06:36 四小时错误监视结束：四个日志均无 OOM/Traceback；
- 当前 GPU：4/5/6/7 我们的四个 seed；0/1/2/3 被另一位用户的任务占用（各 ~35GB，100% util）。

#### 6.1b 两个 seed 的断点续训：澄清（重要）

四个 seed 中 **s42 和 s7 确实做过断点续训**，但续的是**它们自己这次新 C0 的 epoch 0 checkpoint**，不是旧 C0（N32 depth6）的 checkpoint，也不是第一版 mixed-N 的 checkpoint。

证据：

| seed | 续训用的 checkpoint | epoch | global_step | optimizer_states | 当前 last.ckpt global_step |
|---|---|---:|---:|---|---:|
| s42 | `...seed-42 - 2026-11-11-01:45:46/last.ckpt` | 0 | 3125 | True | 109375（= 35 × 3125） |
| s7 | `...seed-7 - 2026-11-11-01:48:04/last.ckpt` | 0 | 3125 | True | 84375（= 27 × 3125） |

- 两份续训日志都打印 `Restored all states from the checkpoint`；
- global_step 正好是 3125 的整数倍，说明从 epoch0 之后连续计数，没有隐藏跳变；
- 旧 C0 checkpoint 在 `tmp/runs/C0-M4/checkpoints/`，配置是 `C0-M4-Nodes32-ADD-L4H16D384-DeBERTa`（N32/depth6/8 loops），与新 C0 的 `C0-M4-MixedN8-32-D12-ADD...`（N8–32/depth12/13 loops）不同目录、不同配置、不同数据；新 C0 没有加载它们；
- 第一版 mixed-N（01:29 启动）在 sanity check 阶段就被停掉，没有 checkpoint，也没有参与续训；
- s1234 / s2025 从未续训，从 01:48 起连续运行。

续训的代价与差异：

- s42 在 02:26 被杀时正在 epoch1 的 ~11%（约 353 步），s7 在 02:24 被杀时正在 epoch1 的 ~2%（约 48 步）；续训从 epoch0 末尾重新开始 epoch1，丢掉了这些 partial steps；
- DataLoader 的 shuffle 状态不一定随 checkpoint 恢复，因此续训后的 epoch1 数据顺序与“未中断地继续”可能不同；这是随机顺序差异，不是模型权重/优化器状态差异；
- 除此之外，config、seed、数据、optimizer state、epoch 计数都一致。若要求“四个 seed 都从 step0 连续跑满”才算严格可比，可以在本批结束后用同 seed 从零重跑 s42/s7；但当前没有必要，除非怀疑 resume 本身影响了轨迹。

### 6.2 最终状态（15:36，全部完成）

| seed | GPU | 状态 | e49 train / val | 完成时间 |
|---|---:|---|---:|---|
| 1234 | 5 | **50 epochs 完成** | 7.82 / **5.89** | 14:23 |
| 2025 | 7 | **50 epochs 完成** | 15.8 / **10.30** | 14:23 |
| 7 | 4 | **50 epochs 完成** | 18.7 / 15.60 | ~15:25 |
| 42 | 7 | **50 epochs 完成** | 2.41 / **2.01** | 15:36（14:33 迁到 GPU7 后完成） |

无 OOM、无 Traceback。

---

## 7. 新 C0 核心异常：loss 在好/坏盆地之间反复跳

### 7.1 完整 epoch-end 轨迹

下表是每个 epoch 完成时的 `total_loss/train_epoch / total_loss/val`（“—”表示该 seed 还没跑到该 epoch）：

| epoch | s42 train / val | s7 train / val | s1234 train / val | s2025 train / val |
|---:|---|---|---|---|
| 0 | 24.8 / 14.5 | 44.7 / 50.1 | 44.5 / 22.2 | 19.9 / 50.8 |
| 1 | 30.3 / 51.5 | 50.6 / 52.2 | 39.0 / 51.2 | 40.6 / 35.7 |
| 2 | 50.0 / 51.6 | 49.4 / 56.9 | 46.5 / 21.2 | 32.8 / 51.1 |
| 3 | 51.9 / 51.0 | 42.7 / 49.5 | 47.4 / 47.3 | 52.0 / 51.4 |
| 4 | 51.2 / 50.8 | 39.5 / 21.6 | 47.4 / 62.0 | 51.8 / 50.8 |
| 5 | 51.3 / 51.1 | 38.1 / 67.9 | 24.3 / 20.5 | 49.5 / 54.8 |
| 6 | 49.2 / 37.8 | 45.2 / 25.3 | 30.4 / 26.8 | 40.4 / 32.6 |
| 7 | 32.4 / 50.2 | 19.5 / 14.8 | 19.1 / 14.7 | 51.6 / 48.6 |
| 8 | 51.9 / 50.8 | 18.5 / 15.4 | 25.0 / 21.9 | 48.4 / 60.9 |
| 9 | 51.5 / 50.6 | 37.7 / 49.8 | 26.8 / 30.3 | 48.3 / 51.0 |
| 10 | 51.0 / 50.5 | 27.2 / 20.3 | 27.3 / 23.0 | 50.7 / 51.3 |
| 11 | 50.5 / 49.5 | 21.4 / 18.1 | 23.6 / 22.0 | 50.4 / 51.0 |
| 12 | 45.9 / 45.6 | 43.5 / 78.0 | 40.6 / 50.5 | 50.3 / 51.4 |
| 13 | 48.2 / 48.6 | 43.8 / 45.3 | 49.8 / 49.5 | 50.7 / 50.5 |
| 14 | 49.3 / 47.4 | 43.7 / 40.6 | 43.0 / 50.4 | 50.8 / 50.3 |
| 15 | 48.3 / 47.8 | 41.3 / 38.9 | 50.6 / 52.9 | 48.0 / 54.0 |
| 16 | 46.4 / 25.4 | 42.3 / 49.5 | 48.2 / 56.6 | 43.3 / 24.7 |
| 17 | 25.0 / 21.3 | 46.4 / 49.9 | 47.2 / 53.0 | 23.9 / 22.0 |
| 18 | 21.8 / 20.5 | 43.4 / 23.7 | 49.6 / 51.2 | 22.2 / 21.0 |
| 19 | 20.9 / 20.7 | 24.9 / 50.4 | 49.2 / 52.0 | 21.0 / 20.2 |
| 20 | 23.7 / 20.3 | 38.9 / 21.8 | 49.2 / 51.2 | 22.0 / 20.3 |
| 21 | 23.7 / 48.5 | 22.2 / 21.1 | 49.1 / 51.2 | 30.4 / 21.9 |
| 22 | 24.9 / 19.5 | 19.0 / 14.7 | 49.2 / 51.2 | 27.2 / 49.5 |
| 23 | 17.6 / 20.1 | 14.3 / 12.8 | 49.3 / 50.6 | 21.5 / 15.6 |
| 24 | 27.6 / 20.5 | 25.6 / 20.9 | 49.4 / 50.0 | 15.8 / 14.4 |
| 25 | 24.1 / 36.2 | 17.0 / 14.5 | 49.4 / 50.9 | 15.8 / 17.5 |
| 26 | 48.5 / 63.6 | 14.4 / 12.7 | 49.3 / 51.3 | 22.9 / 17.1 |
| 27 | 46.4 / 65.2 | 35.7 / 22.0 | 48.7 / 57.3 | 16.9 / 14.8 |
| 28 | 44.7 / 65.4 | 20.7 / 19.6 | 48.4 / 56.2 | 15.0 / 14.6 |
| 29 | 43.6 / 40.3 | 36.8 / 48.4 | 46.5 / 69.1 | 15.9 / 15.5 |
| 30 | 38.8 / 37.1 | 41.1 / 24.4 | 48.9 / 48.1 | 20.2 / 48.0 |
| 31 | 31.3 / 22.2 | 25.8 / 21.1 | 49.1 / 44.7 | 27.7 / 20.2 |
| 32 | 20.1 / 16.0 | 22.1 / 20.7 | 47.4 / 43.1 | 20.8 / 20.2 |
| 33 | — | 31.5 / 20.4 | 46.4 / 41.2 | 23.2 / 21.2 |
| 34 | — | — | 43.5 / 38.1 | 21.4 / 20.3 |
| 35 | — | — | 38.9 / 29.7 | 20.7 / 20.0 |
| 36 | — | — | 31.0 / 23.8 | 20.6 / 20.5 |
| 37 | 6.48 / 5.93 | 19.10 / 19.90 | 24.00 / 18.10 | 20.80 / 20.20 |
| 38 | 6.09 / 4.73 | 16.80 / 14.80 | 18.70 / 15.90 | 20.50 / 20.20 |
| 39 | **4.29 / 2.84** | 28.50 / 20.50 | 18.70 / 15.90 | 20.30 / 15.90 |
| 40 | — | — | 17.20 / 15.40 | 16.90 / 17.00 |
| 41 | — | — | 15.40 / 14.50 | **15.50 / 12.20** |
| 42 | — | — | **14.60 / 14.40** | 14.00 / 13.00 |
| 43 | — | — | 16.10 / 14.40 | 13.20 / 12.70 |

### 7.2 现象

- **非单调**：四个 seed 都在不同 epoch 到达 val 12.7–15.5，又在别的 epoch 反弹到 40–78；
- **s42**：e0 14.5 → e2–e15 长坏盆地（37–52）→ e16–e24 恢复（19.5–25.4）→ e26–e28 再次大坏（63.6/65.2/65.4）→ e31–e32 恢复（22.2/16.0）；
- **s7**：反复在 12.7–15.4 和 48–78 之间跳（e12 78.0、e29 48.4）；
- **s1234**：e7 14.7 后进入 e12–e33 的超长坏盆地（41–56），e34–e36 才开始恢复（38.1→29.7→23.8）；
- **s2025**：e24 14.4、e28 14.6 附近很好，但 e30 又跳到 48.0，e31–e36 回到 20.2–21.2；
- **四个 seed 在 e12–e16 都处于坏状态**（s42 e12–15、s7 e12–16、s1234 e12–33、s2025 e12–15），这可能不是巧合。
- **最终进展（15:36，全部完成）**：s42 e33 14.4 → e39 2.84 → e44 2.17 → e45 2.14 → e46 2.11 → e47 2.07 → **e48 1.96 → e49 2.01（完成）**；s1234 e36 23.8 → e41 14.5 → e44 21.7（小反弹）→ e45 12.3 → e48 8.51 → **e49 5.89（完成）**；s2025 e39 15.9 → e41 12.2 → e43 12.7 → e47 48.4（反弹）→ e48 14.7 → **e49 10.30（完成）**；**s7 e45 12.9 → e46 35.8（大反弹）→ e47 16.2 → e48 21.4 → e49 15.60（完成），best 仍是 e26 12.7**。三个 seed 在 e35–e49 找到稳定好盆地，50 epochs 对 3/4 seed 足够；s7 是唯一未明显收敛的 seed。

### 7.3 当前 best checkpoint

| seed | best checkpoint（epoch, val） | 其他 top-k |
|---|---|---|
| s42 | **e48 1.96**；另有 e0 14.5 | e47 2.07、e49 2.01 |
| s7 | e26 12.7 | e23 12.8、e45 12.9 |
| s1234 | e49 5.89 | e47 9.90、e48 8.51 |
| s2025 | e49 10.30 | e41 12.2、e44 11.9 |

注意：文件名统一显示 `val_loss=0.0000`，真实值来自日志；top-k 会轮转，坏状态没有被保存。

### 7.4 可能的解释（明确区分事实/假设）

**事实：**

1. 不是数据 bug 的典型表现：四个 seed 都能到达 val 12.7–15.5；数据审计通过；train 和 val 同步跳。
2. 不是评测 bug：训练和验证用同一个 `compute_batch_loss_nointerm`；train/val 同步变化；checkpoint monitor 用的是 `total_loss/val`，只是文件名占位错了。
3. 不是 resume bug：s1234 从未迁移，也在 e12 后进入坏盆地；s7/s42 迁移后都正常续训。
4. 不是 `max_train_loop` off-by-one：13 loops 覆盖 depth0–12，已验证。
5. 不是 OOM/资源竞争：四个日志无 OOM/Traceback；迁移后显存稳定。
6. 旧 C0（depth6）也有早期尖峰和 s2025 长期高 loss；新 C0 是同一不稳定性的放大版，不是全新故障类型。
7. **振荡是可恢复的**：s42/s1234/s2025 在 e35–e49 收敛到 final val 2.01/5.89/10.30（best 1.96/5.89/10.30），四个 seed 都跑完 50 epochs；s7 final 15.60、best 仍是 e26 12.7，晚段 e46/e48 又反弹到 35.8/21.4。坏盆地不是永久失败，至少三个 seed 能走出来。

**假设（未定论）：**

- **H1：C0 连续递归的误差累积/双盆地。** C0 没有离散 token 反馈，13 轮里模型吃自己的 hidden；一次更新把递归推入错误吸引子后，误差在后续 loop 累积，train/val 同时爆。好状态说明任务可学，但坏状态是自洽的错误递归。
- **H2：优化不稳定。** 常数 lr 1e-3、无 warmup、无 grad clip；`total_loss` 是 loop loss 直接相加，不同 batch 的 max depth 不同，梯度尺度随深度变化，容易把模型踢出好盆地。depth12/13 loops 比旧 depth6/8 loops 放大了这个问题。
- **H3：PAD/EMPTY/readout 边界崩。** CE 把 PAD 也算进去；如果某次更新让模型不再稳定预测 PAD/EMPTY，loss 会大幅上升，但不一定代表算术全错。坏状态 ~50 是“部分错”的量级，需要 per-factor/per-loop 拆解才能确认。
- **H4：数据混合/课程问题。** 四个 seed 在 e12–e16 同时坏，可能是模型开始认真拟合深节点时的共同相变；当前数据从第 0 个 epoch 就混合 depth1–12，没有浅→深课程，深节点可能在后期的梯度里突然占主导。

为什么目前不能直接验证坏状态：top-k 只保留最好的 3 个 checkpoint；坏状态的权重没有落盘。seed1234 现在 top-k 是 e41/42/43（val 14.4–14.5），e33 的 41.2 坏状态没有保存；s42 top-k e37/38/39，s7 e23/25/26，s2025 e41/42/43。因此现在不能直接对“坏状态”做 per-factor/per-loop 拆解，只能从日志轨迹和好 checkpoint 推断，或重跑一个可控的“坏状态复现”实验。

---

## 8. 想请 GPT 重点分析的问题

1. **机理判断**：H1（连续递归误差累积）vs H2（优化不稳定）vs H3（边界崩）vs H4（数据/课程），哪一个最可能？有没有我们没想到的解释？
2. **e12–e16 的共同坏状态**是否说明存在一个共同的相变点？如果是，最可能是什么触发的（深节点开始被拟合、梯度尺度变化、数据顺序、lr 过大）？
3. **是否延长到 100 epochs？** 旧 C0 是 60–100 epoch 才稳定；新任务 depth12 更难，50 epoch 可能只跑到震荡中段。延长 vs 先改优化，哪个优先？
4. **单项干预的优先级**：如果只允许改一个，应该选：
   - lr 3e-4 或 5e-4；
   - grad clip 1.0；
   - warmup；
   - loss 按 loop 数归一化 / 只对非 PAD 计算；
   - depth curriculum（先浅后深）；
   - 更大 batch？
   哪个最能区分 H1/H2，哪个最可能直接稳定训练？
5. **loss 实现是否有问题**：`total_loss` 按 loop 直接相加、CE 包含 PAD、`factor_loss_scales=[1,1,1,1,0,0]`，这些会不会放大了 depth12 的不稳定？是否应该改成 per-loop 平均 / 按非 PAD 平均？
6. **数据设计是否合理**：N 权重 10/20/30/40、每个 N×depth 等配额，导致 marginal depth 1/2 过采样（12,834 vs d11/12 5,833），N8 只有 depth1–2；100k/2k 是否足够？有没有更好的混合方式？
7. **评估策略**：应该用 top-k best checkpoint 还是最终模型？在“当前 loss 高、best checkpoint 低”的情况下，如何公平评估 strict full-graph、按 N/depth 分层、old N32/N128/deep_v2？
8. **AB 数据与 C0 的关系**：B 的“恢复”已被 parent 2×2 解释为父节点重算；C4 显示自然 hidden 替换能短暂传播。是否应该把 C+ 的目标改成“让 continuous hidden 也具备每轮重算”的归纳偏置，而不是显式纠错？
9. **有没有明显的 setup bug / 我们忽略的红旗？**

---

## 9. 关键文件与复现信息

### 9.1 路径

- 新 C0 配置：`experiments/configs/C0-M4-MixedN8-32-D12-ADD-L4H16D384-DeBERTa/`
- 新 C0 数据：`experiments/data/C0-MixedN8-32-D12-ADD/`
- 新 C0 数据生成：`tmp/scripts/gen_mixed_n_controlled.py`
- 新 C0 日志：
  - `tmp/runs/C0-M4-MixedN-controlled/logs/seed42_probe.log`（s42 e0）
  - `tmp/runs/C0-M4-MixedN-controlled/logs/seed42_resume.log`（s42 e1+）
  - `tmp/runs/C0-M4-MixedN-controlled/logs/seed7.log`（s7 e0）
  - `tmp/runs/C0-M4-MixedN-controlled/logs/seed7_resume.log`（s7 e1+）
  - `tmp/runs/C0-M4-MixedN-controlled/logs/seed1234.log`
  - `tmp/runs/C0-M4-MixedN-controlled/logs/seed2025.log`
- 新 C0 checkpoints：`tmp/runs/C0-M4-MixedN-controlled/checkpoints/`
- 新 C0 状态：`tmp/runs/C0-M4-MixedN-controlled/status.txt`
- 第一版 mixed-N（已停）：`tmp/runs/C0-M4-MixedN/logs/`
- 旧 C0 配置：`experiments/configs/C0-M4-Nodes32-ADD-L4H16D384-DeBERTa/`
- 旧 C0 数据：`experiments/data/Tr32Test128-ADD/`
- 旧 C0 checkpoints：`tmp/runs/C0-M4/checkpoints/`
- C0 诊断：`tmp/analysis/c0-trajectories/`、`c0-allfactor/`、`c0-decomposition/`
- matched-depth：`tmp/analysis/matched-depth/ADD/testdata_N32.pt`
- C4：`tmp/analysis/c0-natural-swap/`
- AB 注错：`tmp/analysis/inject-v2/`、`tmp/analysis/inject-discrete/`、`tmp/analysis/inject-continuous/`
- AB 评测：`tmp/analysis/Aprime-full/`、`B-full/`、`B-screen/`、`AL4fast-full/`、`Braw-check/`
- 记录：
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
  - `tmp/records/2026-11-11/day-summary.md`

### 9.2 启动命令（新 C0 单 seed）

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

### 9.3 续训命令

```bash
... python experiments/train.py ... \
  --resume_from_checkpoint "<checkpoint dir>/last.ckpt"
```

### 9.4 日志解析注意

- 日志是 tqdm 的 `\r` 进度条，直接 `grep` 会把整行显示得很长；建议 `tr '\r' '\n'` 后再 grep。
- 进度条上的 `total_loss/val` 属于**刚完成的 epoch**；正在运行的 epoch 显示的是上一 epoch 的值。
- 真实指标是 `total_loss/val`，不是文件名里的 `val_loss`。

---

## 10. 简短时间线

| 时间 | 事件 |
|---|---|
| 2026-11-07 | 发现 raw/clean aux 回灌 bug；16 档双口径全量结果 |
| 2026-11-08 | A/B 16 档、deep_v2、aux/margin 分析；A-L4fast、cyhB 对照 |
| 2026-11-09 10:50 | 旧 C0 三个 seed 启动，跑到 e2/e3/e5 |
| 2026-11-09 15:42/16:10 | 旧 C0 从 last.ckpt 续训，最终 e94/e96/e99 |
| 2026-11-09 晚 | A′ vs B-M6 单点 VALUE 注入（teacher/own，N32/N128，K4/Kall，4→5）；logit-steering 连续失败 |
| 2026-11-10 | parent 2×2；C0 诊断/失败模式；matched-depth；checkpoint sweep；C4；raw/clean 复核 |
| 2026-11-11 01:29 | 第一版 mixed-N 训练启动（bucket 数据，sanity check 阶段停止） |
| 2026-11-11 01:45–01:48 | 新 C0（controlled 数据）4 seed 启动 |
| 2026-11-11 02:24/02:26 | s7、s42 断点迁移到 GPU4/GPU6 |
| 2026-11-11 06:36 | 4h 错误监视结束，无 OOM/Traceback |
| 2026-11-11 10:36 | s42 e32 16.0、s7 e33 20.4、s1234 e36 23.8、s2025 e36 20.5 |
| 2026-11-11 12:20 | s42 e39 **2.84**、s1234 e43 14.4、s2025 e43 12.7；s7 仍 ~20；3/4 已收敛 |
| 2026-11-11 14:23 | s1234/s2025 跑完 50 epochs，final val 5.89/10.30 |
| 2026-11-11 15:10 | s42 e47 **2.07**（GPU7 跑 e48–e49）；s7 e48 21.4、best e26 12.7；s7 e49 即将完成 |
| 2026-11-11 15:36 | 四个 seed 全部完成 50 epochs：s42 e49 2.01（best e48 1.96）、s1234 5.89、s2025 10.30、s7 15.60（best e26 12.7） |

---

## 11. 一句话给 GPT

> 旧 C0（depth6）已经证明：三个 seed 会走向不同失败模式，train loss 下降不等于严格整图提升，最佳 checkpoint 在 e73 左右就平台化。新 C0（depth12、13 loops、mixed N、常数 lr 1e-3、无 clip、loss 按 loop 直接相加）在早期出现“四个 seed 都在 val 12.7–15.5 和 40–78 两个盆地之间反复跳”；但四个 seed 最终都跑完 50 epochs：s42 final 2.01（best 1.96）、s1234 5.89、s2025 10.30、s7 15.60（best e26 12.7）。请判断这是 C0 自由递归的固有误差累积，还是优化/loss 设计导致的不稳定，并给出最省算力的验证实验和下一步优先级。