# B 问题①：已有答案的更新（repair study 首批）

日期：2026-11-15
运行代码：`95611a1`（`tools/repair_study/run_b.py`、`summarize.py`）
权重：A′ s42 e86（`0b66e196...`）；B-M6 s42 e92（`ef58ba26...`）
数据：heldout N32（`08f52382...`），同一 main256 manifest
状态：**数据报告；未重训，未启动 M3/M4/C+，不作机制结论。**

## 1. 样本与运行

- 主样本固定 `graph32–95` 64 张候选；实际有合格 target 的图为 56 张。
- 每个模型、每个时点 92 个 target；A′/B 共同合格交集 184
  （`just_solved` 92 + `whole_graph_complete` 92）。
- 时点：
  - `just_solved`：该模型第一次实际输出目标正确值的时点；
  - `whole_graph_complete`：原 manifest 中两模型首次共同整图数值全对的时点。
- 要求 target 和全部数值父节点在该时点正确；不满足则判不合格，不补样。
- **主样本的单父限制**：`run_b.py` 沿用 manifest 的 chains，而 manifest
  选择默认只保留单数值父节点（`parents=1`）；因此主表 92 target/时点
  只覆盖单父节点。graph142/128 是三父节点，仅作为 anchor 诊断，
  不并入主表，也不能作为“多父节点行为已重复验证”的证据。
- 五分支：`unchanged`、`correct_write`、`wrong_plus_1`、`wrong_plus_3`、
  `empty`；只改 target VALUE 一次，之后自由续算 16 轮。
- anchor：graph142，`after_original_fault`，target 128，单独 cohort，
  不并入主表。
- 运行：GPU1，先 model_a 后 model_b；A′/B 各 185 条记录
  （184 固定 + 1 anchor），全部 `eligible`，`meta.status=complete`。

命令：

```bash
python tools/repair_study/run_b.py \
  --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json \
  --model model_a --include_anchor \
  --out_dir tmp/analysis/b-value-update-20261115/main_A
# model_b 同理；summary 用 summarize.py
```

## 2. 目标 VALUE 的原始计数（fixed_candidates）

每个时点/模型 92 个 target；表中为 `first output` 与 `final two rounds`
正确数；逐图等权在唯一值上与计数一致（全部 1.0 或 0.0）。

| 模型 | 时点 | 分支 | initial | first output | final two |
|---|---|---|---:|---:|---:|
| A′ | just_solved | unchanged | 92/92 | 92/92 | 92/92 |
| A′ | just_solved | correct_write | 92/92 | 92/92 | 92/92 |
| A′ | just_solved | empty | 0/92 | 92/92 | 92/92 |
| A′ | just_solved | wrong_plus_1 | 0/92 | 0/92 | 0/92 |
| A′ | just_solved | wrong_plus_3 | 0/92 | 0/92 | 0/92 |
| A′ | whole_graph_complete | unchanged | 92/92 | 92/92 | 92/92 |
| A′ | whole_graph_complete | correct_write | 92/92 | 92/92 | 92/92 |
| A′ | whole_graph_complete | empty | 0/92 | 92/92 | 92/92 |
| A′ | whole_graph_complete | wrong_plus_1 | 0/92 | 0/92 | 0/92 |
| A′ | whole_graph_complete | wrong_plus_3 | 0/92 | 0/92 | 0/92 |
| B | just_solved | unchanged | 92/92 | 92/92 | 92/92 |
| B | just_solved | correct_write | 92/92 | 92/92 | 92/92 |
| B | just_solved | empty | 0/92 | 92/92 | 92/92 |
| B | just_solved | wrong_plus_1 | 0/92 | 92/92 | 92/92 |
| B | just_solved | wrong_plus_3 | 0/92 | 92/92 | 92/92 |
| B | whole_graph_complete | unchanged | 92/92 | 92/92 | 92/92 |
| B | whole_graph_complete | correct_write | 92/92 | 92/92 | 92/92 |
| B | whole_graph_complete | empty | 0/92 | 92/92 | 92/92 |
| B | whole_graph_complete | wrong_plus_1 | 0/92 | 92/92 | 92/92 |
| B | whole_graph_complete | wrong_plus_3 | 0/92 | 92/92 | 92/92 |

`ever_output_wrong`：A′ 的 wrong_plus 分支为 92/92；B 的对应分支为
0/92。empty 分支在两个模型上 first output 都为 92/92。

## 3. 下游与其它区域（第16轮，VALUE，fixed_candidates）

**观察分母说明**：主表每个模型 92 个 target；`all_nodes` 2944 =
92 次干预 ×32 个节点观察（不是 2944 个独立节点/独立图）。`downstream`
541 是这些干预下重复观察到的下游节点数，不是 541 个独立 target。
以下先列第16轮，再列 `node_curves.csv` 的逐轮临时损伤；第16轮无损伤
不能概括为全程无损伤。

第16轮：
- B 五分支：`all_nodes` 2944/2944，`downstream` 541/541，
  `outside_local_region` 2219/2219，`parents` 92/92；harmed 0。
- A′：
  - `just_solved` + wrong_plus_1/3：`all_nodes` 2311/2944，
    `downstream` 0/541，`outside_local_region` 2219/2219，
    `parents` 92/92；harmed 633（target 92 + downstream 541）。
  - `whole_graph_complete` + wrong_plus_1/3：`all_nodes` 2852/2944，
    `downstream` 541/541，`outside_local_region` 2219/2219，
    `parents` 92/92；harmed 92（target 本身）。
  - unchanged/correct_write/empty：all_nodes 2944/2944，downstream 541/541，
    outside 2219/2219，parents 92/92。

逐轮（`node_curves.csv`，VALUE）：
- B 的 wrong_plus_1/3 和 empty 分支在第1–7轮存在临时受损，第8轮起
  all_nodes harmed 为 0。以 all_nodes harmed 为例：
  - just_solved wrong_plus_1：`133, 228, 143, 72, 26, 10, 2, 0...`；
  - just_solved wrong_plus_3：`136, 245, 150, 78, 27, 10, 2, 0...`；
  - just_solved empty：`356, 312, 188, 98, 34, 10, 2, 0...`；
  - whole_graph wrong_plus_1：`195, 251, 140, 64, 20, 7, 2, 0...`；
  - whole_graph wrong_plus_3：`210, 265, 152, 72, 25, 8, 2, 0...`；
  - whole_graph empty：`176, 233, 147, 65, 20, 5, 1, 0...`。
- 这些临时受损也出现在 downstream/outside_local_region 的逐轮行中；
  `parents` 逐轮 harmed 为 0。完整逐轮列在
  `docs/records/2026-11-15/b-value-update-node-curves.csv`。

## 4. 具体轨迹

### 4.1 B:32:78:just_solved，wrong_plus_1，target 78，真值 4

- A′：initial `5`，16 轮 target 输出全为 `5`；downstream 首轮
  `[11,24]`，最后 `[11,11]`。
- B：initial `5`，第 1 轮输出 `4`，之后 16 轮全为 `4`；downstream
  首轮 `[24,24]`，最后 `[10,10]`。
- 同一 case 的 `unchanged`：两模型均 initial `4`、输出全 `4`，
  downstream 最后 `[10,10]`。

### 4.2 graph142 anchor，target 128，真值 2，parents `[92,64,22]`

- A′：unchanged initial `2`、输出全 `2`；correct_write `2`；
  wrong_plus_1 `3→3...`；wrong_plus_3 `5→5...`；empty initial `24`，
  输出回到 `2`。
- B：unchanged initial `8`、输出全 `8`；correct_write initial `2`、
  输出全 `2`；wrong_plus_1 `3→2...`；wrong_plus_3 `5→2...`；
  empty initial `24`、输出全 `2`。

anchor 目标初始值/父值按实际准备结果记录；没有强行改成旧现象。

## 5. 我认为

以下是我的看法，和上面的计数分开：

1. 我认为 B 的 wrong_plus 分支在 first output 就回到正确值，而 A′
   在同一时点持续 16 轮错误，是这批数据最直接的差异；但这是行为
   计数，不能直接解释为内部机制。
2. 我认为 graph142 anchor 的 B unchanged 从 `8` 保持到 `8`，而其它分支
   回到 `2`，与旧异常现象相容；A′ 的 empty 能回到 `2`、wrong_plus
   不能，值得 GPT 单独核查。
3. 我认为第16轮下游/其它区域没有新增损伤（B all_nodes 2944/2944、
   downstream 541/541），但逐轮表显示 wrong_plus/empty 在第1–7轮
   有临时受损，第8轮后才归零；不能把第16轮结果说成全程无损伤，也
   不能把 2944/541 当成独立节点样本数。该结果只覆盖本批单父主样本
   和 16 轮窗口，不能外推到其它 seed、N128、多父节点或训练腐蚀。

## 6. 产物与边界

- 原始记录：`tmp/analysis/b-value-update-20261115/main_A/records.jsonl`、
  `main_B/records.jsonl`（各约 8 MB，含 16 轮全部节点四属性）。
- 汇总：`tmp/analysis/b-value-update-20261115/summary/`；
  仓库内小表：
  `docs/records/2026-11-15/b-value-update-target-summary.csv`、
  `docs/records/2026-11-15/b-value-update-coverage.csv`、
  `docs/records/2026-11-15/b-value-update-node-curves.csv`、
  `docs/records/2026-11-15/b-value-update-anchor-trajectories.csv`。
- 本报告只覆盖 B 问题①、主 manifest 子集和 graph142 anchor；不包含
  C 的局部修复、M3/M4、C+ 或问题②。C 的对应结果另见
  `docs/records/2026-11-15/c-local-repair.md`，不与本报告合并成单一
  分数。