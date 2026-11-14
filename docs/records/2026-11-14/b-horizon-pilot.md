# B 长轮数小规模探索（2026-11-14）

状态：**pilot only，不是最终 B 实验**。单 seed、32 张选出的 heldout N32 图、
42 个 primary chains；用于观察恢复时间和长循环再出错，并据此讨论正式观察
窗 H，不据此下 A′/B 的最终科学结论。§9 是 2026-11-14 晚的 B1 CPU 补充：
分开 observed/damaged/stable、补 direct 首轮修回、逐节点全轨迹复发统计和
r→u→q1→q2 时序；§9 对 §4/§5/§6 的相应措辞作更正，原始结果不删除。

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

每个条件 42 targets / 27 图；no_clean / short 均为 0。每个 H 分开报告：
`observed` = 同轮 clean 至少一次正确的节点；`damaged` = 其中注入分支曾错的
节点；`stable` = damaged 中窗口末尾两轮正确。以 after_u_R 为例，observed 从
H=2 的 256 增到 H=8 的 410；A′ damaged 63→121，B 212→358；stable 的分母是
damaged，不是 observed。B 的 direct 稳定恢复分母为 0（B 没有 direct 损伤），
单独报告，不赋 0%/100%；A′ 的 direct 有损伤且在本批自身 damaged 集合中
全部未恢复。详细分 H 表见 §9。

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
两者不必相等；stable 的分母是 damaged，不是 observed。B 所有受损节点的
first recovery 都落在 **2–8 轮**，没有 never；A′ 在所有条件下都是 never
（no-damage 目标除外）。各 H 中 B 的 only-last 最大为 complete_R H=8 的
0.005（H=16 及以后为 0）。**注：这里的“没有恢复后再错”是 target 级分类
的旧说法；逐节点全轨迹统计见 §9.3，B complete_R/U 有少量节点回正后再错、
最终仍稳定，A′ 没有恢复。**

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

本批在 **target 级分类**中没有出现 only-last 和 recover-then-error 的目标
（H=64 的 only-last 最大为 complete_R H=8 的 0.005）。这**不能**推出逐节点
全程没有反复：§9.3 的完整轨迹统计显示 B 的 complete_R 有 20/360、
complete_U 有 7/323 个节点出现“回正→再错”，但最后都稳定；A′ 没有恢复。

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

## 9. B1 补充：分母、直接点与全轨迹复发（2026-11-14 晚）

本节是**对已有六份 v2 轨迹的 CPU 重分析，不新增模型运行**。样本仍是
32 张扫描图 / 27 张有链图 / 42 个 primary target、单 seed；脚本
`tools/b_mechanism/b_pilot_analysis.py`，输出
`tmp/analysis/b-main-20261114/b1/`（`b1_denominators.csv` 23,056 行、
`b1_direct_first_round.csv` 420 行、`b1_recurrence.csv` 15,728 行、
`b1_exceptions.csv` 3 行、`b1_sequences.json` 10 条）。除注明外，
下表用 `all_affected`、first-four、逐图等权之外的 raw 计数并列；宏平均
仍在 `b1_denominators.csv` 的逐图行上可复算。

### 9.1 各 H 的 observed / damaged / stable

`O/D/S` 分别表示 observed（同轮 clean 正确）、damaged（其中注入分支曾错）、
stable（damaged 中窗口末尾两轮正确）。**S 的分母是 D，不是 O。**
H=2 的 after_u 覆盖尚未算完，O/D 会随 H 增大；H=8 后本批分母不再变化。

A′：

| 条件 | H=2 | H=4 | H=8 | H=16 | H=32 | H=64 |
|---|---|---|---|---|---|---|
| after_u U | 214/214/0 | 344/344/0 | 368/368/0 | 368/368/0 | 368/368/0 | 368/368/0 |
| after_u R | 256/63/0 | 386/110/0 | 410/121/0 | 410/121/0 | 410/121/0 | 410/121/0 |
| complete U | 368/42/0 | 368/42/0 | 368/42/0 | 368/42/0 | 368/42/0 | 368/42/0 |
| complete R | 410/39/0 | 410/39/0 | 410/39/0 | 410/39/0 | 410/39/0 | 410/39/0 |
| sustained R | 410/39/0 | 410/39/0 | 410/39/0 | 410/39/0 | 410/39/0 | 410/39/0 |

B：

| 条件 | H=2 | H=4 | H=8 | H=16 | H=32 | H=64 |
|---|---|---|---|---|---|---|
| after_u U | 214/157/0 | 344/251/156 | 368/261/261 | 368/261/261 | 368/261/261 | 368/261/261 |
| after_u R | 256/212/0 | 386/338/122 | 410/358/357 | 410/358/358 | 410/358/358 | 410/358/358 |
| complete U | 368/248/0 | 368/321/209 | 368/323/323 | 368/323/323 | 368/323/323 | 368/323/323 |
| complete R | 410/240/0 | 410/359/128 | 410/360/358 | 410/360/360 | 410/360/360 | 410/360/360 |
| sustained R | 410/322/0 | 410/322/188 | 410/322/322 | 410/322/322 | 410/322/322 | 410/322/322 |

读法：A′ 的 D 在 after_u_R 从 63 增到 121，是 clean 早期未算完的节点逐步
进入分母；不是同一批节点在 64 轮里继续变坏。B 的 after_u_R H=8 为
357/358，H=16 才补齐最后 1 个；这是先前建议 H=16 而不是 H=8 的直接理由。

### 9.2 direct：实际注错数与首轮修回数

| 模型 | 条件 | 实际注错 | 首轮修回 | 首轮仍错 |
|---|---|---:|---:|---:|
| A′ | after_u U | 42 | 0 | 42 |
| A′ | after_u R | 42 | 3 | 39 |
| A′ | complete U | 42 | 0 | 42 |
| A′ | complete R | 42 | 3 | 39 |
| A′ | sustained R（保持期第 1 轮） | 42 | 3 | 39 |
| B | 上述五个条件，各 | 42 | 42 | 0 |

B 的 direct 输出损伤为 0，是因为注入槽在**同一 forward**被改回；这不是
“没有注入”或“恢复率 100%”。A′ 的 direct 有 39/42（R）或 42/42（U）
持续错，stable 分母非零且为 0。

A′ R-only 的 3 个首轮修回例外：`gid 1/t0/pos6`、`gid 9/t0/pos6`、
`gid 25/t0/pos14`；三者都是 **depth 0、`parents=[]`、有 3–5 个 children**
的 r 节点。但不能据此认定“叶/常数节点会被重写”：同样 depth 0 的 r 节点
另有 18 个没有修回，内部 r 节点也全部没有修回；这批数据只能把结构类型
作为待核实线索，不能给出充分条件。

### 9.3 逐节点全轨迹：回正→再错

对每个受损节点沿 64 轮统计（不是 target 分类）：B 的 after_u U/R 和
sustained_R 没有“回正→再错”；complete_R 有 **20/360**、complete_U 有
**7/323** 个节点出现至少一次回正后再错，但到 H=64 全部最终稳定。分角色：
complete_R 的 20 个是 downstream d1 2/110 + d2+ 18/216（target_u 0/42）；
complete_U 的 7 个是 d1 3/110 + d2+ 4/216。A′ 所有条件没有任何节点曾恢复，
因此也谈不上回正→再错。

| 模型 | 条件 | damaged | 曾恢复 | 回正→再错 | 最终稳定 |
|---|---|---:|---:|---:|---:|
| A′ | after_u U | 368 | 0 | 0 | 0 |
| A′ | after_u R | 121 | 0 | 0 | 0 |
| A′ | complete U | 42 | 0 | 0 | 0 |
| A′ | complete R | 39 | 0 | 0 | 0 |
| A′ | sustained R | 39 | 0 | 0 | 0 |
| B | after_u U | 261 | 261 | 0 | 261 |
| B | after_u R | 358 | 358 | 0 | 358 |
| B | complete U | 323 | 323 | 7 | 323 |
| B | complete R | 360 | 360 | 20 | 360 |
| B | sustained R | 322 | 322 | 0 | 322 |

第一次回正与最终稳定起点要分开：B 的 first recovery 在 2–8 轮（after_u U
为 2–6，after_u R 为 2–8，complete U 为 2–6，complete R 为 2–8，
sustained 释放后 2–7）；complete_R/U 的那 20/7 个节点因为中间又错，
最终稳定 run 的起点比 first recovery 晚。**“2–8 轮”是注错/释放后第一次
改对的轮次，不是每个节点的错误持续时间，也不是最终稳定所需时间。**
本批 B 的最终稳定 run 都在 H=8 内开始；H=16 只用于覆盖 after_u_R 的 1 个
H=8 未稳定节点和留余量。

### 9.4 r→u→q1→q2 时序与错值代入

以 gid 0 / target 0 为例（`r_pos=22, u=50, q1=120, q2=158`）。clean 的
u/q1/q2 = 17/12/20。按图的算术，若 r 被改成 20，则 u=20、q1=18、q2=9；
若 u 被改成 20，则 q1=15、q2=6；若 u 被改成 18，则 q1=13、q2=0。
实际逐轮输入/输出（原始轮号）摘要：

- **A′ after_u U**：u 输入 18、输出 18 持续；q1 从 EMPTY→13，q2→0。
  这与“错 u=18 被代入”的 13/0 完全一致，误差沿依赖向下游持续。
- **A′ after_u R**：r 输入 20、输出 20 持续；u 始终输入/输出 17，q1=12、
  q2=20。错 r 停在 direct，没有代入 u；这是局部错值，不是传播。
- **B after_u U**：u 输入 18，**同一 forward 输出 17**；q1/q2 随后按正确
  u 算出 12/20（第 4–5 轮）。
- **B after_u R**：第 1 轮 u 输入 17、输出 **20**（错 r 值被代入），r 输入
  20、输出 17；第 2 轮 u 输入 20、输出 17。q1/q2 随后为 12/20。说明 r 的
  错误值先被写进 u，下一轮再改回；这正是 B 的“下游先坏后恢复”的机制证据。
- **B complete R**：第 1 轮 u 17→20；第 2 轮 q1 12→**15**（与错 u=20 的
  预期一致），第 3 轮 q1 15→1，第 4 轮 1→12；q2 第 3 轮 20→EMPTY，第 4 轮
  EMPTY→9，第 5 轮 9→20。到第 6 轮 u/q1/q2 全对；对应 §9.3 的
  回正→再错节点。
- **B complete U**：u 18→17 同 forward 改回；q1 第 1 轮 12→13（错 u 的
  残留），第 2 轮 13→12；q2 第 2 轮 20→EMPTY，第 3 轮→20。第 4 轮全对。
- **B sustained R**：原始轮 1–4 强制 r=20，u=20，q1 12→15→18，q2 20→6→9
  （与错 u=20 的 15/6 一致，随后继续漂移）；原始轮 5 起释放，u 20→17，
  第 7 轮 q1 18→12，第 8 轮 q2 3→20。释放后按 §5.3 的 H 曲线恢复。

这些序列支持“错误值可以被写入父/下游槽位，再沿依赖恢复”，但不区分
“每轮按父值重算”与“检测到不一致后重算”两种实现；后者需要专门干预。

### 9.5 口径更正与可用的窄结论

- **A′ 的正确说法**：在本 pilot 的自身 damaged 集合内没有稳定恢复；
  不能写成“A′ 对所有注入都不恢复”，因为 3/42 的 R-only direct 注入在
  首轮就被改回，且这 3 个节点不属于它的 damaged 集合。
- **B 的正确说法**：first recovery 是注错/释放后第一次改对的轮次（2–8），
  不是每个节点的错误持续时间；complete_R/U 存在少量节点回正→再错，
  最终稳定 run 的起点可晚于 first recovery。不能把“target 级没有
  recover_then_error 标签”当作“逐节点全程无反复”。
- **H 的含义**：本批 H=8 已覆盖绝大多数最终稳定；H=16 覆盖 after_u_R 的
  最后 1 个 H=8 未稳定节点并留余量。H=16 不是根据 A′/B 差距选的，也
  不能保证 N128 或跨 seed；N128 仍需单独校准。

### 9.6 本节产物与边界

- 脚本：`tools/b_mechanism/b_pilot_analysis.py`（只读 v2 JSON，不写模型）；
- 输出：`tmp/analysis/b-main-20261114/b1/`（未跟踪，路径固定）；
- 轨迹：仍为 `tmp/analysis/b-horizon-pilot-v2-20261114/` 的六份 v2 JSON；
- 边界：单 seed、42 targets、27 图、每图 1–2 条 primary chain；不把本节
  数字外推到正式 B 预算，也不把 3 个 A′ 例外解释成已确认的结构规律。