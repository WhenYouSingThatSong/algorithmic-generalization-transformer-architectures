# C 通用同值/异值来源对照：首批执行

日期：2026-11-15 12:37 CST；15:35 补 1431 审阅的 CPU 案例/口径。  
状态：**6 项 CPU 测试、8 case smoke、两 seed 正式 main（各 96 图/192 候选）已完成；不是 C+ 训练，也不是总体准确率估计**。  
代码：`reproduction@6b4f32c`（GPT 新增 `tools/counterfactual_control/`；本轮未改语义）。  
交接：[generic-source-control](../../../ai-collab/handoff/2026-11-15-generic-source-control.md)。

## 1. 执行与覆盖

```bash
python -m unittest discover -s tools/counterfactual_control/tests -v

# smoke (GPU1)
CUDA_VISIBLE_DEVICES=1 python tools/counterfactual_control/run.py \
  --weight s42_e143 --strata N32_shallow N128_deep --n_graphs 2 --horizon 4 \
  --out_dir tmp/analysis/c-source-control-20261115/smoke_s42 --device cuda:0

# main, same GPU serial
CUDA_VISIBLE_DEVICES=1 python tools/counterfactual_control/run.py \
  --weight s42_e143 --out_dir tmp/analysis/c-source-control-20261115/s42 \
  --device cuda:0 --num_threads 4
CUDA_VISIBLE_DEVICES=1 python tools/counterfactual_control/run.py \
  --weight s1234_e143 --out_dir tmp/analysis/c-source-control-20261115/s1234 \
  --device cuda:0 --num_threads 4

python tools/counterfactual_control/summarize.py \
  --inputs tmp/analysis/c-source-control-20261115/s42/records.jsonl \
           tmp/analysis/c-source-control-20261115/s1234/records.jsonl \
  --out_dir tmp/analysis/c-source-control-20261115/summary
```

- 6 项 `counterfactual_control` CPU 测试通过。
- smoke：s42 在 N32_shallow/N128_deep 各 2 图、4 轮，8 个结构候选均 eligible；
  检查了 `initial`、两张叶常数改动、三份来源目标 a/a+1/a+3、同位置同轮次、
  hidden 行替换、sham 与 unchanged 相同，以及 local-q 预期与完整来源 q 真值分离。
- 正式 main：每个权重 6 档 × 前 16 图 = 96 图；每图至多 2 个结构候选，
  共 192 候选/权重。s42 191 eligible、s1234 180 eligible；合计 **371/384
  eligible**。13 个 ineligible 中：11 个至少一项源四属性资格失败；另 2 个
  源资格全部通过但 q 集合为空（父输入或 unchanged 第一轮 q 条件不满足）。
  两者都在 `coverage.csv` 保留，未补样、未换图。
- 覆盖：fan-in 1/2/3 = 97/136/138；真实 depth 1–10（中位 4），
  depth_band lower/upper = 191/180；原值覆盖 0–22（4 只有 1 个 case），
  delta +1/+3 均执行。六档每档图数、q 数见下表。
- 同值来源 hidden 向量**没有** exactly-equal 的样本（0/371）；L2 距离范围
  0.044–5.646（中位 1.861）。异值来源距离更大（+1: 2.07–10.50，中位 7.43；
  +3: 2.07–10.12，中位 7.26）。距离只记录、不匹配。

## 2. 主结果（第一轮，q 为直接子节点）

每行是 eligible case 的直接子节点 q 观测；`same correct` 是同值来源分支
q 保持原真值，`diff expected` 是异值分支 q 等于局部预期
`(a+delta+c) mod 23` 的数量。分母 q 由源资格和 unchanged 条件固定，
不因分支输出改变。

| weight | stratum | cases | q | same correct | same changed | diff expected d1 | diff changed d1 | diff expected d3 | diff changed d3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 32 | 54 | 54 | 0 | 54 | 54 | 54 | 54 |
| s42_e143 | N32_deep | 32 | 46 | 46 | 0 | 46 | 46 | 46 | 46 |
| s42_e143 | N64_shallow | 32 | 64 | 64 | 0 | 64 | 64 | 64 | 64 |
| s42_e143 | N64_deep | 32 | 54 | 54 | 0 | 54 | 54 | 54 | 54 |
| s42_e143 | N128_shallow | 32 | 89 | 89 | 0 | 89 | 89 | 89 | 89 |
| s42_e143 | N128_deep | 31 | 49 | 49 | 0 | 47 | 49 | 49 | 49 |
| s1234_e143 | N32_shallow | 32 | 54 | 54 | 0 | 54 | 54 | 54 | 54 |
| s1234_e143 | N32_deep | 32 | 46 | 45 | 1 | 46 | 46 | 46 | 46 |
| s1234_e143 | N64_shallow | 31 | 62 | 62 | 0 | 62 | 62 | 62 | 62 |
| s1234_e143 | N64_deep | 27 | 47 | 46 | 1 | 47 | 47 | 47 | 47 |
| s1234_e143 | N128_shallow | 29 | 75 | 75 | 0 | 74 | 75 | 74 | 75 |
| s1234_e143 | N128_deep | 29 | 43 | 42 | 1 | 41 | 43 | 43 | 43 |

图等权（同一批 case/同一 q 的逐图平均，第一轮）：

| weight | stratum | same correct GE | diff expected d1 GE | diff expected d3 GE |
|---|---|---:|---:|---:|
| s42_e143 | N32_shallow | 1.000 | 1.000 | 1.000 |
| s42_e143 | N32_deep | 1.000 | 1.000 | 1.000 |
| s42_e143 | N64_shallow | 1.000 | 1.000 | 1.000 |
| s42_e143 | N64_deep | 1.000 | 1.000 | 1.000 |
| s42_e143 | N128_shallow | 1.000 | 1.000 | 1.000 |
| s42_e143 | N128_deep | 1.000 | 0.974 | 1.000 |
| s1234_e143 | N32_shallow | 1.000 | 1.000 | 1.000 |
| s1234_e143 | N32_deep | 0.969 | 1.000 | 1.000 |
| s1234_e143 | N64_shallow | 1.000 | 1.000 | 1.000 |
| s1234_e143 | N64_deep | 0.969 | 1.000 | 1.000 |
| s1234_e143 | N128_shallow | 1.000 | 0.969 | 0.969 |
| s1234_e143 | N128_deep | 0.969 | 0.906 | 1.000 |

合计（第一轮，683 个 q 观测）：

- 同值来源：q 仍正确 680/683，改变/受损 3/683；这 3 个都在 s1234。
- 异值 +1：q 仍正确 **0/683**，改变/受损 **683/683**，等于局部预期
  678/683（99.3%）。
- 异值 +3：q 仍正确 **0/683**，改变/受损 **683/683**，等于局部预期
  682/683（99.8%）。
- 同值分支 q 的“等于异值局部预期”为 0/683；所以同一张图、同两个叶常数
  的非零同值改动没有把 q 推向异值预期。

目标 u 的第一轮 forward：同值/unchanged/sham 的 u 原值正确 370/371；
异值 +1 分支 u **不再读出注入值**，其中 261/371 回到原值、110/371 是其它
读出；+3 分支 269/371 回到原值、102/371 是其它读出（`u_shifted` 第一轮
均为 0）。因此不能把“离开注入值”写成“已正确重算”，q 的异值命中也不能
解释成 u 自身持续保持异值。

### 逐轮曲线（全档合计，delta 1/3）

| round | q | same correct | same changed | diff expected d1 | diff correct d1 | diff expected d3 | diff correct d3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 683 | 680 | 3 | 678 | 0 | 682 | 0 |
| 2 | 683 | 679 | 0 | 4 | 458 | 1 | 451 |
| 4 | 683 | 678 | 0 | 1 | 674 | 0 | 677 |
| 8 | 683 | 678 | 0 | 0 | 676 | 0 | 676 |
| 16 | 683 | 677 | 0 | 0 | 676 | 0 | 676 |

读法：异值来源在**第一轮**把 q 大量推到局部预期；第二轮约 2/3 q 回到
原值，第四轮后大部分回到原值。同值来源几乎没有影响（3 个 case 在
第一轮受损，之后无新增 changed/harmed）。这些是逐轮切片，不是逐节点
持续恢复率：第 16 轮同值分支仍有 6 个错误 q 观测，异值分支相对不改动
分支仍有 1 个差异；不能把“round16 约 676/683”写成永久恢复或“99%恢复”。

## 3. 具体案例

完整机器可读版本见 `c-source-control-examples.csv`，其中有全部叶系数/值、
value-token 位置、q 表达式和逐轮分支读出。下面只摘要。

### 案例 A：`s42_e143 / N32_shallow:0 / u66`

- 目标表达式（全部贡献）：`u = 1*x2 + 1*x10 (mod 23)`；叶值 x2=11
  （value token 位置 0）、x10=2（位置 8），11+2=13，与 `a=13` 一致。
- 三份来源改同两个叶：same x2 11→13、x10 2→0；shift1 x2 11→13、
  x10 2→1；shift3 x2 11→13、x10 2→3。来源 u 读出 13/14/16。
- q 表达式：`q104=x30+u`（x30=11），`q112=x30+x2+u`；原真值
  q104=1、q112=12。局部预期 +1=[2,13]，+3=[4,15]。
- unchanged/sham/same 第一轮 q=[1,12]；shift1=[2,13]；shift3=[4,15]。
  u 的 first-forward 输出 13/13/6；shift 分支 q 在 round3 回到 [1,12]。
  同值来源 hidden L2 距离 2.91，q 仍保持正确。

### 案例 B：`s42_e143 / N32_shallow:0 / u132`

- 目标表达式（全部贡献）：`u = 2*x2 + 3*x14 + 1*x18 + 1*x22 (mod 23)`；
  叶值 x2=11、x14=18、x18=2、x22=19，2*11+3*18+2+19=97≡5，与 `a=5` 一致。
- 来源改动：same x2 11→12、x18 2→0；shift1 x2→12、x18→1；
  shift3 x2→12、x18→3；来源 u 读出 5/6/8。
- q 表达式：`q152=u+x18+x80`（x80=17）；原真值 q152=1；局部预期
  +1=[2]、+3=[4]。
- unchanged/same 第一轮 q=[1]；shift1=[2]；shift3=[4]；u 的
  first-forward 输出 5/5/21；shift1 的 q 在 round3 回到 [1]，shift3 在
  round2 回到 [1]。这是多父节点、不同原值的例子；同值来源 L2 距离 4.23，
  也没有改变 q。

这里的 x 是数值节点位置；value-token 位置只在 `c-source-control-examples.csv`
中单列，不把两者混称。

两例都显示：局部向量在**第一轮 forward**产生特定影响，随后若干轮回到
原轨迹/原值；不能把第一轮命中写成持续修复。

## 4. 边界

- 这是**条件于结构/源资格/unchanged q 条件**的诊断样本；不是总体准确率，
  也不能把 371 个候选当成独立图。一个图可有两个 q，重复使用同一
  unchanged/同值轨迹，不作独立重复样本。
- 同值来源与异值来源的 leaf 改动位置/次数匹配，但 hidden 向量距离没有匹配；
  同值向量不是 exactly-equal（L2 0.044–5.65）。
- 异值分支的第一轮 q 命中局部预期，支持“局部向量携带与 VALUE 相关的
  计算信息”这一行为解释；仍不能说明整个向量只包含 VALUE，也不能替代
  内部算法/因果证明。
- 13 个 ineligible 中 11 个是源四属性资格失败、2 个源资格通过但 q 集合为空；
  分布保留在 `coverage.csv`，没有为了凑数换图或改区间。
- 没有在同一数据上重跑旧 C4，也不把旧 C4 的百分比和本轮拼表；没有 C+
  训练、B 机制解释或跨位置/跨图来源。

## 5. 产物

- smoke：`tmp/analysis/c-source-control-20261115/smoke_s42/`
- main：`.../s42/`、`.../s1234/`（各 192 candidates；191/180 eligible）
- summary：`.../summary/`（`coverage.csv`、`case_curves.csv`、
  `paired_effects.csv`、`summary.csv`、`full_reference_curves.csv`）
- 提交的小表：`docs/records/2026-11-15/c-source-control-coverage.csv`、
  `c-source-control-summary.csv`、`c-source-control-examples.csv`
- 完整 records、hidden/轨迹大文件留在 tmp，不进入 git。
