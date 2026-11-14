# B 长轮数小规模探索（2026-11-14）

状态：**pilot only，不是最终 B 实验**。单 seed、32 张选出的 heldout N32 图、
42 个 primary chains；用于观察恢复时间和长循环再出错，并据此讨论正式观察
窗 H，不据此下 A′/B 的最终科学结论。

## 1. 权重、数据与选样

- A′：`A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa`，s42，
  `epoch=86-val_loss=0.0000.ckpt`；
- B：`B-M6-Nodes32-ADD-L4H16D384-DeBERTa`，s42，
  `epoch=92-val_loss=0.0000.ckpt`（recorrection run）；
- 数据：`tmp/analysis/heldout-seed42/ADD/testdata_N32.pt`，
  SHA256 `08f52382…ee707`；
- manifest：`tmp/analysis/b-horizon-pilot-20261114/manifest.json`，记录
  data/checkpoint/code 哈希与 A′/B 共同 early 资格。

selection 设 scan 上限 128，但前 32 张全部 common-complete 并选满 32 张后
停止；实际 **scanned=32，common-complete=32**。其中 27 张图有 primary
chain、5 张无 chain；共 42 chains（depth 1×21、2×15、3×2、4×4）；T=4–9，
n_loops=9–14。A′/B 的 own/teacher、before/after early 资格全部一致。
这是预算探索样本，后续独立验证另留样本。

## 2. 运行

六项均使用同一 manifest，own-state：

| 条件 | 干预 | 观察 |
|---|---|---|
| early after_u U-only | 起点 u=a+1，r 正确 | 64 轮 |
| early after_u R-only | 起点 r=r_true+3，u 正确 | 64 轮 |
| complete U-only | 整图完成后 u+1 | 64 轮 |
| complete R-only | 整图完成后 r+3 | 64 轮 |
| sustained_r | complete 起点强制 r 错误 4 轮后释放 | 释放后 64 轮（总 68） |

每项配同模型、同起点、同长度 clean。U 用 U+1、R 用 R+3（模运算）。
运行使用 GPT 的性能版 `24e586c`：CPU 侧整块读出、`--num_threads 4`、
JSONL 增量落盘；`run_serial.py` 串行执行，GPU4/5，训练 GPU0 未动。

v2 产物在 `tmp/analysis/b-horizon-pilot-v2-20261114/`。六项串行总耗时约
7–8 分钟（单项 early 约 75 s、late 约 78 s、controls 约 55 s），旧版同规模
early 约 53–55 分钟。early A′、late B（含 `leaf_constant` 新真值）和
controls B 均与旧代码做过小样本逐值等价检查，记录字段 0 diff。

## 3. 汇总口径

按 consensus 的四项选择：

- 全窗口逐轮为主；早期损伤参照同一轮 clean，双方都未算出的 EMPTY 不记为损伤；
- 稳定恢复 = 窗口末尾连续 2 轮正确（L=2）；曾恢复为辅；仅最后一轮回正单列；
- 逐图等权为主、节点加权为辅；零分母单独报告；
- VALUE 与前四属性同时正确并列。

直接注错点按模型**输出**判定：输入里的注错本身不记为损伤；若模型在同一
forward 内改回，则 direct 损伤为 0，而不是人为制造一次“恢复”。sustained
的 H 从释放后开始，强制保持窗口另写 `sustained_hold.csv`。

## 4. 覆盖

每个条件 42 targets / 27 图；no_clean / short 均为 0。H 增大时损伤分母会
增大（以 after_u_R 为例，H=2 的 256 增到 H=8 的 410；clean 尚未算出的节点
在早期不进入分母），H≥8 后分母稳定。B 的 direct 稳定恢复分母为 0（B 没有
direct 损伤），单独报告，不赋 0%/100%；A′ 的 direct 有损伤且全部未恢复。

## 5. 结果（first-four；VALUE 在本样本几乎完全一致）

### 5.1 H=64 的损伤与恢复（all_affected，逐图等权）

| 条件 | A′ damage | A′ stable | B damage | B stable |
|---|---|---|---|---|
| after_u U-only | 1.000 (368/368) | 0 | 0.660 (261/368) | 1.000 (261/261) |
| after_u R-only | 0.290 (121/410) | 0 | 0.847 (358/410) | 1.000 (358/358) |
| complete U-only | 0.141 (42/368) | 0 | 0.853 (323/368) | 1.000 (323/323) |
| complete R-only | 0.113 (39/410) | 0 | 0.859 (360/410) | 1.000 (360/360) |
| sustained_R | 0.113 (39/410) | 0 | 0.748 (322/410) | 1.000 (322/322) |

表中的 0.xxx 是逐图等权，括号内是 raw damaged/observed 或 stable/damaged；
两者不必相等。B 所有受损节点的 first recovery 都落在 **2–8 轮**，没有
never；A′ 在所有条件下都是 never（no-damage 目标除外）。各 H 中 B 的
only-last 最大为 complete_R H=8 的 0.005（H=16 及以后为 0），没有恢复后
再错。

### 5.2 直接点与下游

H=64，first-four：

- **A′**：direct 持续错——after_u U 42/42、R 39/42；complete U 42/42、
  R 39/42；sustained 39/42。complete/sustained 的误差只停在 direct，
  下游 d1/d2+ 损伤为 0；after_u U 则 d1 110/110、d2+ 216/216 全坏，
  after_u R 为 d1 14/110、d2+ 68/216。
- **B**：direct 全部 0（注入槽在同一 forward 被改回）；受影响的是
  target_u / d1 / d2+。after_u U 为 d1 90/110、d2+ 171/216；
  after_u R 为 target_u 41/42、d1 108/110、d2+ 209/216；
  complete 与 sustained 的 d1/d2+ 约 0.98–1.00，全部稳定恢复。

### 5.3 H 对照与 H 建议

B 的稳定恢复率（all_affected，first-four，逐图等权）：

| 条件 | H=2 | H=4 | H=8 | H=16 | H=32 | H=64 |
|---|---|---|---|---|---|---|
| after_u U | 0 | 0.727 | 1.000 | 1.000 | 1.000 | 1.000 |
| after_u R | 0 | 0.419 | 0.998 | 1.000 | 1.000 | 1.000 |
| complete U | 0 | 0.736 | 1.000 | 1.000 | 1.000 | 1.000 |
| complete R | 0 | 0.412 | 0.995 | 1.000 | 1.000 | 1.000 |
| sustained_R | 0 | 0.665 | 1.000 | 1.000 | 1.000 | 1.000 |

A′ 在所有 H 上都是 0。H=2/4 明显低估稳定恢复；H=8 已接近饱和，H=16 完全
饱和，H=32/64 不再变化。clean 分支在 after_u 从第 1 轮约 0.652 的计算进度
升到第 8 轮 1.000，之后 64 轮无漂移；complete/sustained 从第 1 轮就是 1.000。

**建议（供用户决定）：** 正式主观察窗取 **H=16**，理由是 B 在 H=8 已饱和、
16 轮留出余量并覆盖 after_u_R 在 H=8 尚未稳定的 1 个节点；同时保留 H=8
作敏感性检查。另设一个 **64 轮长期稳定性检查**（本批未见恢复后再错，但
不能由 16 轮外推无限稳定）。H 的选择依据是 B 的饱和和 clean 无漂移，不是
A′/B 差距最大。

### 5.4 sustained 的强制保持窗口

`forced` 定义为“clean 输入正确、注入输入错误”。direct r 在 4 轮保持中
forced=100%；A′ 的 d1 输入 forced=0、输出损伤也为 0，说明强制错父没有
传播到下游；B 的 d1 损伤从第 1 轮 0.290 升到第 4 轮 0.984，forced 也从
0 升到 0.984，释放后按 5.3 的曲线恢复到 1.000。A′ direct r 在保持和释放后
都保持 0.926 的损伤。

## 6. 典型案例

- **B 快速稳定**：after_u U-only，gid 0 / target 0；11 个受损节点，first
  recovery 2 轮，H=64 时 11/11 stable。
- **B 延迟稳定**：sustained_R，gid 25 / target 0；2 个受损节点，first
  recovery 4 轮，2/2 stable；属于本批唯一的 stable_delayed 目标。
- **A′ 未恢复**：after_u U-only，gid 0 / target 0；12 个受损节点，全部
  never，direct u 与下游一起持续错。
- **A′ 无损伤**：complete R-only，gid 1 / target 0；注入 r 错误但在这个
  target 上被改写回正确，all_affected 无受损节点；A′ 整体仍有 39/42 的
  direct 持续错，这是少数例外。

本批未出现 only-last 和 recover-then-error 的目标；H=64 的 only-last 最大
为 complete_R H=8 的 0.005。

## 7. 限制

- 单 seed、32 张选出的 common-complete 图、42 chains；不能外推到全部图或
  正式 B 预算；
- 只跑了 after_u / complete 的 U/R 和 4 轮 sustained；未跑 teacher start、
  unrelated、switch、N128、训练腐蚀审计、局部自洽；
- H 的分母随窗口变化；表中同时给了 n；VALUE 与 first-four 在本样本几乎
  相同，正式报告仍需并列；
- direct 的“模型输出判定”意味着 direct 为 0 不等于干预无效，而是注入槽
  在同一 forward 被改写；真正的传播在下游；
- 本批是优化后重跑，性能修复的等价检查只覆盖小样本；正式科学结论仍以
  用户确认的最终评估协议为准。

## 8. 产物

- 原始轨迹：`tmp/analysis/b-horizon-pilot-v2-20261114/*.json`（含 `.jsonl`
  和 `.meta.json`）；旧版 early 两份保留在 `tmp/analysis/b-horizon-pilot-20261114/` 作对照；
- 汇总：`tmp/analysis/b-horizon-pilot-v2-20261114/summary/`
  （`horizon_table.csv`、`per_round_curves.csv`、`per_round_accuracy.csv`、
  `target_outcomes.csv`、`events.csv`、`sustained_hold.csv`、`cases.json`、
  `coverage.json`）；
- 图：同目录 `figs/stable_vs_h.png`、`damage_curves.png`、`recovery_ecdf.png`、
  `clean_drift.png`；
- 汇总实现：`tools/b_mechanism/b_mechanism_horizon_summary.py`；
  性能版 runner：`24e586c`。