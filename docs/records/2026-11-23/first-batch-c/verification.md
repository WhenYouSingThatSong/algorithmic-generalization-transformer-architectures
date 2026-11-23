# 第一批四模型 report · C 侧 CPU 复核

日期：2026-11-23。复核对象：`docs/records/2026-11-23/first-batch-b-trained/`（提交 `de5d2f4`，报告代码 `reproduction@f14c14b`）与生产 report 根 `tmp/analysis/first-batch-20261123/formal/tables/`。本复核只读，不启动训练、不占用 GPU，不改 B 的模型目录与 raw；不改 C 已交付的 natural/history/fanout raw。

结论先行：**要求复核的五项数值均能复算，未发现算术错误。** 发现两处口径需要写清楚：`fanout-counts` 的 `/64` 是 attempts 分母（C_s1234 各臂 eligible 为 63）；`C_s1234 d8 complete 1/32` 是原始 C_s1234 的 individual 不合格，后训练两臂 individual 仍为 32/32，但它们的 group common 分母同样被原始 C 的该题拉成 63/64。这不改变 report 数值，只影响分母标签。

## 1. 交付一致性与身份

- B 提交的 12 张表与生产 report 根逐文件一致（gzip 表解压后比较）：`scenes-counts`、`scenes-paired`、`same-graph-fit`、`fanout-counts`、`fanout-paired`、`fanout-topology-pairs`、`natural-coverage`、`natural-counts`、`natural-pairs`、`same-value-counts`、`same-value-new-reader`、`mark-counts`。
- `first-batch-trained-raw.tar.gz` 的 SHA-256 为 `1e9912a3c19b81fc289bb8ebb4fe1a518680f569b95b2116e9969fdebc6afb76`，与 `first-batch-trained-raw-archive.sha256` 一致；manifest 1,836 行，与 `provenance.json` 的 `raw_archive.files` 一致。
- 归档内 22 个 `identity.json` 与生产目录逐字节一致。
- mark 身份与 `provenance.json`/README 一致：C_s42-update final `f14f4bbb…`、epoch10 `227cecf8…`、epoch30 `73334e8d…`；C_s1234-update final `693b6338…`、epoch10 `3d9f102d…`、epoch30 `bbaa6718…`；plan 均为 `b3c4b1be…`。每个 mark 权重目录 64 个图文件。
- 格式小注：B 提交的 CSV 为 CRLF，C 交付为 LF；同一表内容除换行符外一致，不影响数值。后续若编辑这些 CSV，建议统一为 LF，避免 `git diff --check` 的尾随字符问题。

## 2. same-value-counts：C_s1234 是结构性零覆盖

- 直接读 C 侧 `raw/history/C_s1234` 的 64 个题例：新子节点在 stable / first_after_empty / recovered / wrong_control 四臂的**首步读数全部为 24（空）**；`reader_gold` 为 0–22 的普通数值（抽查 400/401/402 为 7/17/4），因此 `clean_reader_ok=false`、阳性对照 `positive_control_used=false`。
- 复算结果与 B 表完全一致：C_s1234 四臂 `confirmed_read_opportunities` 为空（0）；这不是缺失，也没有可补的分母。最终报告应显式写 **0 个可确认读取机会**，不能把空值读成“答错”或“没有跑”。
- C_s42 复算：d4 common 31/32、d8 27/32；confirmed read 7/31、3/27；stable/first_after/recovered 下一轮正确 7/7、7/7、6/7（d8 3/3、3/3、3/3）；wrong_control 下一轮 0/7、0/3。与 `same-value-counts.csv` 一致。

## 3. fanout-counts：分母与 1/32 不合格的来源

- `fanout-counts.csv` 不含 depth 列；其 `eligible` 是跨深度汇总。C_s1234 原始、correction、update 三臂在 `complete` 的各 fanout 行均为 **63/64 eligible**；分母 64 是 attempts。
- 从 raw 逐题复核（`formal/fanout/{model}`）：
  - 原始 C_s1234：depth 8 / complete 只有 **gid 422** 一条不合格，fanout 1/2/4 各为 31/32；depth 4 全 32/32。individual 因此为 31/32。
  - C_s1234-correction、C_s1234-update：individual 在 depth 8 / complete 均为 **32/32**；它们的 63/64 只出现在 group `common`，因为 group 的 common 资格仍包含原始 C_s1234 的 gid 422。
  - A_s42/B_s42：individual 全 32/32；在 group `primary` 的 common 为 64/64，但在 group `four_with_C1234` 的 common 也是 63/64（同样由原始 C_s1234 触发）。这不是 A/B 自身的个体不合格。
- 因此 B README “C_s1234 d8 complete 1/32 不合格、common 63/64；其余 64/64”应读作：**原始 C_s1234 的 individual 是 1/32；C_s1234 三臂的 common 均因此为 63/64；C_s42 家族 common 仍为 64/64。** A/B 在 `four_with_C1234` 的 common 行不适用“其余 64/64”。
- README 的 fanout 表把分母写成 `/64`，但分子只统计 eligible 行。例如 C_s1234-correction fanout=1 complete 的 `target_next=52`，按 eligible 应为 **52/63**，不是 52/64。建议最终报告改用 eligible 分母或在表注写明“斜杠后为 attempts，C_s1234 行实际 eligible=63”。本复核不修改 B 的表。
- 表内其它 fanout 汇总可复算：例如 C_s42 原始 fanout=1/4 complete descendant wrong 293/484，correction 191/343，update 240/551；C_s1234 原始 338/809，correction 207/498，update 252/509；与 README 表一致。

## 4. scenes-paired：三 scope 方向计数

- 用 `scenes-graphs.csv` 逐图重算 `scenes-paired.csv` 的全部 **16,000** 行（common / common_clock / common_clock_fit × base_only/trained_only/both），0 处不一致；`scenes-counts.csv` 4,704 行同样 0 处不一致。`fanout-paired.csv` 1,152 行、`fanout-counts.csv` 504 行也全部复算一致。
- `common_clock_fit` 的分母另与 `same-graph-fit.csv` 对照：按各 group 的 arms 在该图 `round=d+5` 的 `value==nodes` 复算，fit 合格的图数 × 该 group 可用 metric/pair 数，与表内 common 合计逐项吻合（如 C_s42 first_correct target source 为 176 个 fit 合格图，组内 2 pair × 3 metric = 1,056，对应表内 common 合计 1,056）。该 scope 只在标准组存在，符合报告说明。
- 标准组、`target` 条件的 child_damage（group=每一原始种子，scope=common；格式 common / base_only / trained_only / both）：

| 模型 → 后训练 | phase | common | base_only | trained_only | both |
|---|---|---:|---:|---:|---:|
| C_s42 → correction | child_correct | 377 | 2 | 51 | 322 |
| C_s42 → correction | graph_complete | 512 | 1 | 94 | 255 |
| C_s1234 → correction | child_correct | 383 | 3 | 69 | 301 |
| C_s1234 → correction | graph_complete | 523 | 1 | 74 | 235 |
| C_s42 → update | child_correct | 377 | 0 | 53 | 324 |
| C_s42 → update | graph_complete | 512 | 5 | 100 | 251 |
| C_s1234 → update | child_correct | 383 | 0 | 77 | 304 |
| C_s1234 → update | graph_complete | 523 | 0 | 111 | 236 |

- 同一 child_damage 切片在 common_clock / common_clock_fit 下方向不变，例如 C_s42 correction child_correct 115/0/22/93（clock）、109/0/21/88（fit）；C_s1234 correction child_correct 120/1/27/89（clock 与 fit 相同）；C_s42 correction graph_complete 182/0/36/84；C_s1234 correction graph_complete 229/0/38/95。
- 标准组 `first_correct`、`target` 的 next_correct（越高越好）方向偏训练后：C_s42 correction 499/2/37/460（group C_s42 common）、C_s1234 correction 524/3/103/409；与 B README 的 499/2/37/460、524/3/103/409 一致。
- 扩展组 `before_input_source` 的 child_damage：C_s42 correction 64/0/29/35、C_s42 update 64/0/26/35；C_s1234 correction 64/6/13/22、C_s1234 update 64/3/29/25。`before_older_ancestor`：C_s42 correction 64/0/7/57、C_s42 update 64/0/7/57；C_s1234 correction 64/1/8/55、C_s1234 update 64/0/8/56。B README 的 29 vs 0、8 vs 0 均能对上。
- 分层组 `complete` + `later_target` 的 child_damage：C_s42 correction 32/0/5/27、C_s42 update 32/0/4/27；C_s1234 correction 32/2/10/17、C_s1234 update 32/0/13/19。B README 的 C_s1234 update 13 vs 0 能对上。
- 方向一致性：上述 key child_damage 切片在逐 size/depth/scope 行中**没有出现 base_only > trained_only** 的反向格；trained_only 在各格均占优。全表 3,560 个聚合键中确有 392 个同时出现两个方向的非零计数（异质性存在），但本批 README 引用的具体切片没有方向反转。`specific` 等指标也有 base_only > trained_only 的格，注意 `specific` 是“采用注入错值”的害指标，不能按 next_correct 的方向解释。

## 5. mark-counts：0.5 阈值、健康误报与 epoch 分母

- 从 90,605 行 `mark-thresholds.csv` 逐图重算 `mark-counts.csv` 1,555 行，0 处不一致；又从 raw 的 `score>0.5` 直接复算关键格，全部一致：

| 权重 | new/wrong_target 标出 | direct_conflict target 标出 | healthy target 误报 | healthy 非 target 误报 |
|---|---:|---:|---:|---:|
| C_s42-update final | 48/64 | 104/512 | 0/128 | 1/9344 |
| C_s42-update epoch10 | 21/64 | 30/512 | 0/128 | 0/9344 |
| C_s42-update epoch30 | 56/64 | 119/474 | 0/120 | 28/8190 |
| C_s1234-update final | 63/64 | 148/512 | 0/128 | 0/9344 |
| C_s1234-update epoch10 | 63/64 | 125/508 | 0/128 | 0/9264 |
| C_s1234-update epoch30 | 63/64 | 136/507 | 0/128 | 0/9243 |

- final 的 target `corrected_unmarked` 为 C_s42 284、C_s1234 202，与 README 一致；epoch30 的 eligible 分母下降（如 healthy 非 target 8,190），表中已保留。
- `mark` 只覆盖 update 两 seed 的 final + epoch10/30；correction 无 mark 表，不能横比。

## 6. natural：覆盖与 donor fail

- 直接对 C 侧 384 个 natural raw 求和：C_s42 337,614 opportunities / 217 node_events / 306 edge_records / 76 selected / 66 matched，skip 全为 `no_exact_same_clock_wrong_donor` 10；C_s1234 315,962 / 465 / 623 / 118 / 87，donor fail 31；没有 `archived_transition_not_reproduced`。与 B 的 `natural-coverage.csv` 合计和 README §7 完全一致。
- C 侧 `natural-coverage/pairs/counts` 与 B 报告同名表除 CRLF/LF 外内容一致；B 报告复用同一 C raw，未另造第二套。

## 7. 复核边界

- 本复核验证“数值能否从 raw/逐图表复算、分母是否显式、身份是否一致”，不对机制、训练因果或候选五作结论。
- 未逐格重算 README 的所有叙述性场景表（如分层 all_pending_wrong、扩展 parent_and_target_*），只覆盖 B 指定的五项与标准/扩展/分层的 key 切片；这些表的底层 `scenes-counts` 已全表复算。
- 未改 B 的 `first-batch-b-trained/`、未改 `tmp/analysis/first-batch-20261123/formal/{mark,standard,layered,expanded}/C_*` 与后训练 fanout 目录，也未改 C 的既有 raw。
