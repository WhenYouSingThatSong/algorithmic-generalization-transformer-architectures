# 第一批 · 后训练 C 的同题重放、过程监督与传播（B 侧）

日期：2026-11-23。状态：**trained smoke/formal 均 exit 0；四模型 report 已生成。** 本批只读已有权重，不训练、不改模型结构。C 侧 original（C natural/history/fanout + A/B fanout）由 C 会话交付在 `docs/records/2026-11-23/first-batch-c/`；本交付只负责 trained：两种后训练 C 的标准/分层/扩展同题重放、fanout，以及过程监督第 10、30、50 周期的标出诊断。

## 1. 执行、数据与覆盖

```bash
export DATA_ROOT=/workspace/lab-xj/users/jinxi/code/yhy/management-20261120
export FIRST_BATCH_OUT=/workspace/lab-xj/users/jinxi/code/yhy/management-20261120/tmp/analysis/first-batch-20261123
CUDA_VISIBLE_DEVICES=2 bash tools/c_hard_graphs/run_first_batch.sh trained smoke
CUDA_VISIBLE_DEVICES=2 bash tools/c_hard_graphs/run_first_batch.sh trained formal
python tools/c_hard_graphs/first_batch_report.py --out "$FIRST_BATCH_OUT"
```

- `DATA_ROOT` 用管理树：`C_s42-update`/`C_s1234-update` 的 final 与 `epoch-10/30` 只在管理树；correction 通过已有只读 symlink `tmp/analysis/c-update-20261119` 指向原训练树；A/B/原始 C 为 plan 中的绝对路径。
- 覆盖（每模型）：standard 192 图、layered 32 图、expanded 64 图、fanout 64 题 × 3 fanout × 4 phase；mark 只测 update 两个种子的 final + epoch10/30，每权重 64 题。原始 C 与 A/B 的 raw 复用 C 侧交付/旧归档。
- 权重：`C_s42-correction` `88430930…`、`C_s1234-correction` `b9ad22aa…`、`C_s42-update` `f14f4bbb…`、`C_s1234-update` `693b6338…`；update epoch10/30 分别为 `227cecf8…`/`73334e8d…` 与 `3d9f102d…`/`bbaa6718…`。每个运行目录的 `identity.json` 与 `fit-source.json` 随 raw 保留。
- 报告代码在 `reproduction@f14c14b`；相对 9c13113 只改了一处汇总：`scenes-paired`/`fanout-paired` 现在同时输出 `common`、`common_clock`、`common_clock_fit`（fanout 无 fit 行，故为前两者），不再只留 common。原始模型数值和 raw 未变。

## 2. 干净拟合与训练来源

同题干净拟合（标准轮数后再四轮，节点数值；`same-graph-fit.csv` 含逐图 `value`/`first4`）：

| 模型 | 32/4 | 64/4 | 128/4 | 128/10（value / first4） |
|---|---:|---:|---:|---:|
| C_s42（原始） | 1023/1024 | 2047/2048 | 4091/4096 | 4025/4096 · 4023/4096 |
| C_s42-correction | 1024/1024 | 2048/2048 | 4096/4096 | 4088/4096 · 4088/4096 |
| C_s42-update | 1024/1024 | 2048/2048 | 4096/4096 | 4095/4096 · 4081/4096 |
| C_s1234（原始） | 1024/1024 | 2048/2048 | 4092/4096 | 4060/4096 · 4006/4096 |
| C_s1234-correction | 1024/1024 | 2048/2048 | 4096/4096 | 4096/4096 · 4096/4096 |
| C_s1234-update | 1024/1024 | 2048/2048 | 4096/4096 | 4095/4096 · 4081/4096 |

- 两种后训练在标准组 D+4 的四档都接近满分；原始 C 在 128/10 的 D+4 仍有少量缺口。标准组的同题比较使用 `common_clock_fit` 时只保留两臂都干净的题。
- 训练损失（逐 epoch 表 `training/training-loss-epochs.csv`）：update 末周期图加权 0.194094/0.290582，末 batch 0.125177/0.211051（C_s42/C_s1234）；correction 末周期图加权 0.002893/0.003604，末 batch 0.003287/0.002265。两者目标函数不同，**损失不能直接横比**。
- 验证集（`fit-sources/*.json`）：correction 的 D/D+4 数值与前四字段均 1.0，D+16 graph first4 all 0.9453/0.8516；update 的 D/D+4 数值 1.0，D+16 graph value all 0.9922/0.9766、graph first4 all 0.0/0.0625。update 的验证格式没有 `mean_first4`，已在源文件标注。

## 3. 目标修复：标准、扩展、分层

下表每格为“目标下一轮正确 / 共同合格；子节点额外损伤 / q_risk”，共同资格为同组 base+两个后训练模型（C_s42 或 C_s1234 各自一组）。

| 场景 | 种子 | 原始 | correction | update |
|---|---|---:|---:|---:|
| 标准，目标首次算对后只改目标 | 42 | 141/159 · 158/159 | 157/159 · 149/159 | 156/159 · 150/159 |
| 同上 | 1234 | 107/170 · 167/170 | 164/170 · 155/170 | 153/170 · 160/170 |
| 标准，目标与子节点均对后只改目标 | 42 | 160/185 · 179/185 | 184/185 · 185/185 | 181/185 · 185/185 |
| 同上 | 1234 | 92/191 · 172/191 | 170/191 · 189/191 | 155/191 · 189/191 |
| 标准，整图算完后只改目标 | 42 | 154/179 · 155/179 | 177/179 · 179/179 | 174/179 · 176/179 |
| 同上 | 1234 | 82/177 · 137/177 | 150/177 · 166/177 | 137/177 · 171/177 |
| 扩展，目标将算前只改内部目标 | 42 | 56/64 · 9/64 | 64/64 · 0/64 | 62/64 · 2/64 |
| 同上 | 1234 | 43/64 · 20/64 | 62/64 · 1/64 | 59/64 · 6/64 |
| 扩展，目标刚算完后只改内部目标 | 42 | 56/64 · 64/64 | 64/64 · 64/64 | 63/64 · 64/64 |
| 同上 | 1234 | 40/64 · 64/64 | 61/64 · 64/64 | 61/64 · 64/64 |
| 扩展，两父值改动相消 | 42 | 11/64 · 58/64 | 22/64 · 53/64 | 61/64 · 44/64 |
| 同上 | 1234 | 15/64 · 56/64 | 19/64 · 51/64 | 50/64 · 43/64 |
| 分层，目标将要计算前提前填错 | 42 | 28/32 · 4/32 | 32/32 · 0/32 | 31/32 · 1/32 |
| 同上 | 1234 | 17/32 · 9/32 | 32/32 · 0/32 | 28/32 · 4/32 |
| 分层，整图完成后再改内部目标 | 42 | 25/32 · 27/32 | 31/32 · 32/32 | 30/32 · 31/32 |
| 同上 | 1234 | 15/32 · 19/32 | 22/32 · 27/32 | 19/32 · 32/32 |

- 两种后训练都明显提高“已算错目标下一轮回到真值”的比例，correction 通常最高；update 在 C_s1234 上低于 correction。
- 但“目标回正”不等于下游无损：扩展组目标刚算完后只改内部目标时，三臂的子节点损伤都是 64/64（原 64/64），只是后训练目标更常回正；扩展组 `both_parents_cancel` 中 update 的目标回正从 11/15 升到 61/50，但子节点损伤仍为 44/43。
- 多位置同时注错（分层 `all_pending_wrong`/`all_just`、扩展 `parent_and_target_*`）时，后训练的目标下一轮仍多为 0/32 或接近 0；不能把单点修复率外推到多点。

## 4. 下游代价与方向相反的配对

`scenes-paired.csv` 的每一行是某一 `suite/size/depth/phase/condition/scope/base/trained/metric` 的成对计数：`common` 为共同合格题数，`base_only`/`trained_only`/`both` 为方向计数。三个 scope 都已输出；`common_clock_fit` 只对标准组存在（其他组没有标准拟合字段）。

少数直接读数（next_correct 越高越好，child_damage/child_specific 越高越差）：

| 配对 / 场景 | scope | common | base_only | trained_only | both |
|---|---|---:|---:|---:|---:|
| C_s42 → correction，标准 first_correct target next | common | 499 | 2 | 37 | 460 |
| 同上 | common_clock | 304 | 0 | 14 | 290 |
| 同上 | common_clock_fit | 284 | 0 | 14 | 270 |
| C_s1234 → correction，标准 first_correct target next | common | 524 | 3 | 103 | 409 |
| C_s42 → correction，标准 child_correct target child_damage | common | 377 | 2 | 51 | 322 |
| C_s42 → correction，扩展 input_source child_damage | common | 64 | 0 | 29 | 35 |
| C_s1234 → update，分层 later_target child_damage | common | 32 | 0 | 13 | 19 |

- 标准 `first_correct` 的目标修复方向明显偏向 trained（trained_only 37/103 vs base_only 2/3）；但在 `child_correct`/`graph_complete`，`child_damage` 的 trained_only 反而更大（例如 C_s42 correction 51 vs 2、94 vs 1；C_s1234 correction 69 vs 3、74 vs 1）。
- 扩展 `input_source`/`older_ancestor` 的目标下一轮基本不变，但 trained_only 的下游损伤更大（C_s42 correction input_source 29 vs base_only 0；C_s1234 update older_ancestor 8 vs 0）；分层 `later_target` C_s1234 update 的 child_damage 也是 13 trained_only vs 0 base_only。
- 因此本批的总图景是“后训练让目标更容易重算回真值”，同时“在子节点已填好/已消费的场景，后训练更常重新打开并损伤下游”。两者同时存在，不能只报目标修复率。

## 5. Fanout：一、二、四子节点传播

`fanout-counts.csv` 为各模型/深度/fanout/phase 的共同合格计数。下表取 fanout=1 与 4 的 `complete` 阶段；斜杠后为 **eligible 分母**，不是 attempts。C_s1234 三臂的 group common 都是 63/64（attempts=64；原始 C_s1234 individual 31/32，唯一不合格是 gid 422 depth8 complete，fanout 1/2/4；correction/update individual 32/32，但 group common 仍含该题，故为 63/64）；C_s42 家族为 64/64。A_s42/B_s42 individual 32/32；`primary` common 64/64，`four_with_C1234` common 63/64 同样由该题触发，不是 A/B 个体失败。

| 模型 | fanout | target_next | child_specific | descendant wrong node-rounds | output ever wrong |
|---|---:|---:|---:|---:|---:|
| C_s42 原始 | 1 | 59/64 | 19/64 | 293 | 37/64 |
| C_s42-correction | 1 | 64/64 | 61/64 | 191 | 64/64 |
| C_s42-update | 1 | 62/64 | 57/64 | 240 | 64/64 |
| C_s1234 原始 | 1 | 17/63 | 20/63 | 338 | 60/63 |
| C_s1234-correction | 1 | 52/63 | 30/63 | 207 | 63/63 |
| C_s1234-update | 1 | 30/63 | 50/63 | 252 | 63/63 |
| C_s42 原始 | 4 | 59/64 | 19/64 | 484 | 34/64 |
| C_s42-correction | 4 | 64/64 | 60/64 | 343 | 45/64 |
| C_s42-update | 4 | 62/64 | 61/64 | 551 | 64/64 |
| C_s1234 原始 | 4 | 17/63 | 30/63 | 809 | 58/63 |
| C_s1234-correction | 4 | 52/63 | 32/63 | 498 | 61/63 |
| C_s1234-update | 4 | 30/63 | 62/63 | 509 | 47/63 |

- 后训练在 `before` 阶段几乎消除了基础模型的 descendant wrong rounds（例如 C_s42/C_s1234 fanout=4 before：原始 358/365，correction 2/8，update 13/230）；`complete` 阶段则出现明显的 fanout 依赖：C_s42 update 的 descendant wrong rounds 从 240（fanout=1）升到 551（fanout=4），correction 从 191 升到 343；C_s1234 两臂也从 207/252 升到 498/509。
- `child_specific` 在 `children_correct`/`complete` 随 fanout 增大而增大，且后训练通常高于原始（尤其 update）。`descendant_empty_rounds` 在 C_s42-correction 的 `just_correct` 出现 5/17/39（fanout 1/2/4），报告保留。
- `output_ever_wrong` 只表示终点曾错过，不等于末轮仍错；末轮指标见 CSV 的 `output_wrong_end`。C_s42 原始在 complete 的 `output_wrong_end` 为 5/64（fanout=1），后训练为 0–1/64；C_s1234 原始为 1/63（eligible 63），后训练为 0/63。

## 6. 过程监督标出（mark，仅 update）

主阈值 0.5；完整 0.1/0.25/0.5/0.75/0.9 和逐图分数在 `mark-thresholds.csv.gz`，汇总在 `mark-counts.csv`。

| 权重 | new wrong_target 标出 | direct_conflict 目标标出（全 phase） | healthy 误报（target / 非 target） |
|---|---:|---:|---:|
| C_s42-update final | 48/64 | 104/512 | 0/128 · 1/9344 |
| C_s42-update epoch10 | 21/64 | 30/512 | 0/128 · 0/9344 |
| C_s42-update epoch30 | 56/64 | 119/474 | 0/120 · 28/8190 |
| C_s1234-update final | 63/64 | 148/512 | 0/128 · 0/9344 |
| C_s1234-update epoch10 | 63/64 | 125/508 | 0/128 · 0/9264 |
| C_s1234-update epoch30 | 63/64 | 136/507 | 0/128 · 0/9243 |

- 第 10 周期 C_s42 的“应新填/直接冲突”标出明显弱于 final 和 epoch30（21/64 vs 48/64 vs 56/64）；C_s1234 三个周期都接近饱和（63/64）。这符合“早期标出能力欠拟合”的诊断，但本批只作固定权重的逐周期对照。
- 0.5 阈值对 healthy 目标位置没有误报；非目标 healthy 误报在 C_s42 epoch30 为 28/8190，其余多为 0–1。非目标分母很大，不能把 28 当作整体错误率。
- 目标改变并不都由标记触发：final 的 target `corrected_unmarked` 为 C_s42 284、C_s1234 202（各 phase/条件汇总）；mark 分数与是否修改不是一一对应。`just_correct`/`complete` 的 `wrong_target` 标记接近 0，但目标仍常被重算（见 `mark-counts.csv`）。
- mark 只跑 update；correction 不在此表，不能把两臂标出能力混在一起。

## 7. C 侧 natural/history 与零覆盖

C 侧 original 交付（`docs/records/2026-11-23/first-batch-c/`）的覆盖与零覆盖已复核：

- natural：C_s42 337,614 个机会 / 217 个 node events / 306 条 edge records / 76 条 selected / 66 条 matched，donor fail 10；C_s1234 315,962 / 465 / 623 / 118 / 87，donor fail 31。
- history：C_s42 common 31/32(d4)、27/32(d8)，confirmed read 7/31、3/27；C_s1234 common 13/32、5/32，新读者首步恒为 24（空），`confirmed_read_opportunities=0`。这是**结构性零覆盖**，不是“答错”，也不是缺失；`same-value-counts.csv` 的空值保留为零分母。
- fanout：原始 C_s1234 的 individual 在 d8 complete 为 31/32（唯一 gid 422）；C_s1234 三臂的 group common 均为 63/64；C_s42 家族 common 仍为 64/64；A_s42/B_s42 individual 32/32，`primary` common 64/64，`four_with_C1234` common 63/64 是由原始 C_s1234 的 gid 422 触发的 group common 传播，不是 A/B 个体失败。报告保留该缺口。
- 自然/人工同错数的完整交叉表见 C 侧 `tables/natural-*.csv`；本报告的 `natural-coverage/pairs/counts` 复用同一 raw，不另造第二套表。

## 8. 文件与校验

- `tables/`：`scenes-counts.csv`、`scenes-paired.csv`（三 scope）、`same-graph-fit.csv`、`fanout-counts.csv`、`fanout-paired.csv`、`fanout-topology-pairs.csv`、`natural-coverage.csv`、`natural-counts.csv`、`natural-pairs.csv`、`same-value-counts.csv`、`same-value-new-reader.csv`、`mark-counts.csv`；另附 `scenes-graphs.csv.gz`、`fanout-graphs.csv.gz`、`mark-thresholds.csv.gz`（逐图/逐阈值原始表压缩）。
- `training/training-loss-epochs.csv`、`training/validation-summary.csv`：从已有训练日志/`fit-source.json` 生成；correction 的 loss 来自 `steps.jsonl`，update 来自 `docs/records/2026-11-21/fit-history`，两者目标不同。
- `fit-sources/`：四个后训练模型的 `fit-source.json` 副本。
- `first-batch-trained-raw.tar.gz`：22 个 trained 目录的全部 raw/identity/fit-source（1,836 个文件；archive SHA256 `1e9912a3…`）；逐文件哈希在 `first-batch-trained-raw-manifest.sha256`。C 侧 raw 在 C 交付目录，A/B 与原始 C 的大批轨迹沿用 2026-11-22 归档。
- 完整 report 运行根：`tmp/analysis/first-batch-20261123/`；raw 不搬到 docs。报告脚本 `reproduction@f14c14b`。
- C 侧复核：`docs/records/2026-11-23/first-batch-c/verification.md`（`3281ba2`）；五项数值可复算，两处分母标签（fanout eligible 与 A/B `four_with_C1234` common）已在本 README 修正。

## 9. 边界

- 固定权重、固定题面、固定输出；不是训练新模型，也不把后训练差异单独归因于某一目标函数或“纠错训练”。两种后训练的损失目标不同，验证格式也不同。
- 目标回正 ≠ 下游未受损；本批多处显示 trained 目标更高但 child_specific/descendant 损伤也更高；方向计数同时列出 base_only 和 trained_only。
- C_s1234 history 的零覆盖、原始 C_s1234 fanout individual 的 1/32 不合格（group common 63/64，A/B 的 `four_with_C1234` common 也随之为 63/64，individual 仍 32/32）、A/B 标准组较低的基础拟合（见上一批 `abc-input-scenarios-results`）都必须与结果一起读。
- 候选五（逐位置测试时自适应计算）另立批次，不混入本批结论。

## 10. 分工

本会话 `yhy-57` 是 Agent B：只写 trained 的 mark/standard/layered/expanded/fanout；C 会话是 Agent C：写 original 的 natural/history/fanout 和 A/B fanout，raw 在 `first-batch-c/`。两边共用 `FIRST_BATCH_OUT`，模型目录不重叠；本侧未改 C 原始 raw 和模型目录。
