# C 同值/异值来源对照：early / complete / late 时点

日期：2026-11-15 16:08 CST。  
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
  - stage eligible：early 185/192、complete 184/192、late 173/192；
  - 非 included：18 ineligible + 16 unavailable；complete 有 8 个、late 有 8 个
    unavailable（首次整图完成超出 64 轮预算/无法取得），early 有 7 个 ineligible，
    late 有 11 个 ineligible（q 或源资格失败）。未补样。
- 三阶段共同交集：**169 个 target/stage 记录（507 条），每阶段同一目标、同一 q**；
  严格来源（三阶段来源 target first-four 全对）：**72 个 target/stage（216 条）**，
  每阶段 128 个 q。
- 原始状态：
  - early：原图全节点 VALUE 8015/13632 = 0.588（early 尚未算完）；
  - complete：13632/13632 = 1.000，first4 0.997；
  - late：VALUE 0.988、first4 0.745（VARIABLE 已漂移，但 VALUE 多数仍对）。
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
  同时，q 命中局部预期的比例也从 early 的约 99% 降到 complete/late 的约
  15–40%；early 的“局部影响”最强，complete/late 已有较多其它状态可分摊。
- 这不表示 late 更“稳定”：late 的 same correct 296/300，但周围节点 first4
  已降到 0.745；VALUE 主表与四属性严格表必须分开。

### 共同交集与严格子集

| set | stage | q | same correct | diff correct d1/d3 | diff expected d1/d3 |
|---|---|---:|---:|---:|---:|
| 三阶段共同（169 target/stage） | early | 287 | 287 | 0 / 0 | 285 / 286 |
| 三阶段共同 | complete | 287 | 287 | 69 / 25 | 114 / 105 |
| 三阶段共同 | late | 287 | 286 | 110 / 83 | 42 / 46 |
| 严格来源（72 target/stage） | early | 128 | 128 | 0 / 0 | 126 / 127 |
| 严格来源 | complete | 128 | 128 | 32 / 15 | 50 / 49 |
| 严格来源 | late | 128 | 128 | 48 / 35 | 21 / 20 |

严格来源子集仍显示同样的阶段顺序：early 影响最大，complete 次之，late
最小；同值来源在所有交集里都是 correct=全部（仅 late 1 个 changed）。

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
  `c-stages-endpoint-summary.csv`
- 完整 records、case_curves、paired_effects 留在 tmp。
