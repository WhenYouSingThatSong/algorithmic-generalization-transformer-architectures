# C 同值/异值来源对照：early / complete / late 时点

日期：2026-11-15 16:08 CST；17:13 CST 按 1636 审阅补 coverage 来源、共同状态表与分母，未重跑。
状态：**10 项 counterfactual CPU 测试、stage smoke、两 seed 正式 main 已完成；不是 C+，也不是总体准确率**。  
代码：`reproduction@572cc10`（固定运行版本；运行期间未 pull/改代码）。  
交接：[C stages/joint](../../../ai-collab/handoff/2026-11-15-b-multiparent-c-stages-joint.md)。

## 1. 运行与覆盖

```bash
python -m unittest discover -s tools/counterfactual_control/tests -v

CUDA_VISIBLE_DEVICES=2 python tools/counterfactual_control/run.py \
  --weight s42_e143 --stage_study --strata N32_shallow N128_deep \
  --n_graphs 2 --horizon 4 --out_dir tmp/analysis/c-stages-20261115/smoke_s42 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=2 python tools/counterfactual_control/run.py \
  --weight s42_e143 --stage_study --n_graphs 8 \
  --out_dir tmp/analysis/c-stages-20261115/s42 --device cuda:0
CUDA_VISIBLE_DEVICES=2 python tools/counterfactual_control/run.py \
  --weight s1234_e143 --stage_study --n_graphs 8 \
  --out_dir tmp/analysis/c-stages-20261115/s1234 --device cuda:0

# all / common early+complete+late / strict common
python tools/counterfactual_control/summarize.py ...
```

- smoke：N32_shallow/N128_deep 各 2 图，8 个结构候选 ×3 stage =24 条；23 eligible、1 ineligible（late 的 q=0）。
- 正式 main：每 seed 6 档 × 前 8 图，每图至多 2 个结构候选；共 576 条 stage 记录
  （2 seeds × 6 strata × 8 graphs × 2 candidates × 3 stages）。
  - stage eligible（单时点 `status`）：early 185/192、complete 184/192、late 173/192。
  - 提交的 `c-stages-coverage.csv` 来自 `summary_matched`；其中 `included=True`
    是**三阶段共同交集**，不是单时点 `status=eligible`。两种口径不能混用：
    | stage | included（共同交集） | eligible 但不在交集 | ineligible | unavailable | 总记录 |
    |---|---:|---:|---:|---:|---:|
    | early | 169 | 16 | 7 | 0 | 192 |
    | complete | 169 | 15 | 0 | 8 | 192 |
    | late | 169 | 4 | 11 | 8 | 192 |
    `complete`/`late` 的 unavailable 是首次整图完成超出 64 轮预算/无法取得；
    `early`/`late` 的 ineligible 是 q 或源资格失败。未补样。
- 三阶段共同交集：**169 个目标，每个时点 169 条，共 507 条**；跨阶段固定同一目标、
  同一 q，每阶段 287 个 q。严格来源（三阶段来源 target first-four 全对）：
  **72 个目标、每时点 72 条（216 条）**，每阶段 128 个 q。
- 原状态节点观察率（按目标重复计数图内节点；**条件诊断样本，不是总体准确率，
  也不是独立图准确率**）：
  | scope | stage | records | nodes | VALUE | first4 |
  |---|---:|---:|---:|---:|---:|
  | 共同交集 `included` | early | 169 | 12096 | 6980 (57.71%) | 6980 (57.71%) |
  | 共同交集 `included` | complete | 169 | 12096 | 12096 (100%) | 12086 (99.92%) |
  | 共同交集 `included` | late | 169 | 12096 | 11956 (98.84%) | 9040 (74.74%) |
  | 单时点 `status=eligible` | early | 185 | 13632 | 8015 (58.80%) | 8015 (58.80%) |
  | 单时点 `status=eligible` | complete | 184 | 13632 | 13632 (100%) | 13594 (99.72%) |
  | 单时点 `status=eligible` | late | 173 | 12544 | 12390 (98.77%) | 9342 (74.47%) |
  | 所有记录（unavailable 按 0 节点） | early | 192 | 14336 | 8551 (59.65%) | 8551 (59.65%) |
  | 所有记录（unavailable 按 0 节点） | complete | 192 | 13632 | 13632 (100%) | 13594 (99.72%) |
  | 所有记录（unavailable 按 0 节点） | late | 192 | 13632 | 13336 (97.83%) | 9868 (72.39%) |
  - 只有共同交集三行是同一批目标、同一批节点的匹配状态；`status=eligible`
    三行组成不同（late 的 173 条对应 12544 个节点），不能当作同一组图的漂移曲线。
  - 之前写的 0.588/1.000/0.988 与 late first4=0.745 来自 `status=eligible`
    子集；所有可用状态口径则是 early 8551/14336、complete 13632/13632、
    late 13336/13632。完整数值见 `c-stages-common-state.csv`。
- stage 绝对轮次：late = complete + 16；early 使用 `d(u)+1` 的预定状态。
  smoke 实测 early 3 / complete 7 / late 23，符合预期。

## 2. 第一轮结果（q 为直接子节点）

| stage | q | same correct | same changed | diff correct d1/d3 | diff expected d1/d3 | diff changed d1/d3 |
|---|---:|---:|---:|---:|---:|---:|
| early | 311 | 310 | 1 | 0 / 0 | 307 / 310 | 311 / 311 |
| complete | 316 | 316 | 0 | 70 / 25 | 125 / 115 | 246 / 291 |
| late | 300 | 296 | 4 | 110 / 83 | 44 / 49 | 190 / 217 |

- 同值来源在三个时点都几乎没有改变 q（early 1/311、complete 0/316、
  late 4/300 changed）。
- 目标 u 第一轮 forward 回到原值的数量：early +1 130/185、+3 135/185；
  complete 121/184、114/184；late 116/173、105/173；同值/unchanged 约
  368/370、368/368、346/346。q 的异值命中不等于 u 自身保持异值。
- 异值来源的效果随阶段明显减弱：early 的 shifted q 正确 0，complete 为
  70/316（+1）和 25/316（+3），late 为 110/300 和 83/300。
  q 命中局部预期的比例也从 early 的约 99% 降到 complete/late 的约
  15–40%。更准确的说法是：**早期的特定数值传播最清楚，完成后这一关系改变**；
  不能只说“异值影响消失”或“晚期更稳定”。例如 late/+1 仍有 177/287 个 q
  错误，其中只有 42 个符合注入数值的局部预期，其余 135 个是其它错误或无数值；
  late/+3 为 204 个错误，其中 46 个符合预期、158 个其它。局部预期命中率下降
  的一个**候选解释**是周围其它状态分摊了影响；本数据没有验证这个机制，不能
  把它写成已证结论。
- 这不表示 late 更“稳定”：late 的 same correct 296/300，但周围节点 first4
  在 `status=eligible` 子集已降到 9342/12544=0.745；VALUE 主表与四属性严格表
  必须分开。

### 共同交集与严格子集

| set | stage | q | same correct | diff correct d1/d3 | diff expected d1/d3 |
|---|---|---:|---:|---:|---:|
| 三阶段共同（169 targets） | early | 287 | 287 | 0 / 0 | 285 / 286 |
| 三阶段共同 | complete | 287 | 287 | 69 / 25 | 114 / 105 |
| 三阶段共同 | late | 287 | 286 | 110 / 83 | 42 / 46 |
| 严格来源（72 targets） | early | 128 | 128 | 0 / 0 | 126 / 127 |
| 严格来源 | complete | 128 | 128 | 32 / 15 | 50 / 49 |
| 严格来源 | late | 128 | 128 | 48 / 35 | 21 / 20 |

严格来源子集仍显示同样的阶段顺序：early 影响最大，complete 次之，late
最小。同值来源的正确数要按交集分开读：共同交集为 early 287/287、
complete 287/287、late 286/287（1 个 changed）；只有严格子集在三个时点都是
128/128 全对。不能说“所有交集同值来源都全部 correct”。

### 逐轮（共同交集，delta 1）

| stage | round | q | same correct | diff correct | diff expected |
|---|---:|---:|---:|---:|---:|
| early | 1 | 287 | 287 | 0 | 285 |
| early | 2 | 287 | 287 | 191 | 4 |
| early | 4 | 287 | 287 | 284 | 1 |
| complete | 1 | 287 | 287 | 69 | 114 |
| complete | 2 | 287 | 287 | 228 | 0 |
| complete | 4 | 287 | 287 | 281 | 0 |
| late | 1 | 287 | 286 | 110 | 42 |
| late | 4 | 287 | 282 | 204 | 2 |
| late | 16 | 287 | 257 | 200 | 1 |

读法：early 的异值 q 到 round4 已基本回到原值；complete 更快（round2
已 228/287），late 的 same 分支本身在 round16 也只 257/287，说明周围
状态仍在漂移。逐轮值不是持续恢复率；不能用 round16 的切片替代逐例跟踪。

## 3. 边界

- stage 样本是条件于目标/来源/q 资格的诊断样本，不是总体发生率；未完成、
  q 空和源资格失败均保留在 `coverage.csv`。
- early 不是“第一次猜对”，而是 `d(u)+1` 的预定状态；complete 是原图整图
  VALUE 首次全对，不是四属性全对；late 是 complete+16。
- 同值与异值来源仅匹配两个叶改动位置/次数，hidden 距离未匹配；不把
  VALUE 主表解释成剥离了变量名/其它信息。严格四属性子集只作补充。
- 没有 C+、N256、祖先回滚或训练；没有覆盖旧 C4 百分比。

## 4. 产物

- smoke：`tmp/analysis/c-stages-20261115/smoke_s42/`
- main：`.../s42/`、`.../s1234/`
- summaries：`.../summary_all/`、`summary_matched/`、`summary_strict_matched/`
- 提交小表：`c-stages-coverage.csv`、`c-stages-summary-all.csv`、
  `c-stages-summary-matched.csv`、`c-stages-summary-strict_matched.csv`、
  `c-stages-endpoint-summary.csv`、`c-stages-common-state.csv`
- 完整 records、case_curves、paired_effects 留在 tmp。
