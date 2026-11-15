# C 通用同值/异值来源对照：首批执行

日期：2026-11-15 12:37 CST。  
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
  eligible**。13 个 ineligible 全部是 q 资格为 0（源四属性或 unchanged 第一轮
  q 条件不满足），未补样、未换图。
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

合计（第一轮，683 个 q 观测）：

- 同值来源：q 仍正确 680/683，改变/受损 3/683；这 3 个都在 s1234。
- 异值 +1：q 仍正确 **0/683**，改变/受损 **683/683**，等于局部预期
  678/683（99.3%）。
- 异值 +3：q 仍正确 **0/683**，改变/受损 **683/683**，等于局部预期
  682/683（99.8%）。
- 同值分支 q 的“等于异值局部预期”为 0/683；所以同一张图、同两个叶常数
  的非零同值改动没有把 q 推向异值预期。

目标 u 的第一轮 forward：同值/unchanged/sham 的 u 原值正确 370/371；
异值 +1 分支 u 回算为原值 261/371、+3 为 269/371（`u_shifted` 第一轮均为 0，
说明写回行在 immediate readout 正确，但 forward 已重算）。因此 q 的异值
命中不能解释成 u 自身持续保持异值。

### 逐轮曲线（全档合计，delta 1/3）

| round | q | same correct | same changed | diff expected d1 | diff correct d1 | diff expected d3 | diff correct d3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 683 | 680 | 3 | 678 | 0 | 682 | 0 |
| 2 | 683 | 679 | 0 | 4 | 458 | 1 | 451 |
| 4 | 683 | 678 | 0 | 1 | 674 | 0 | 677 |
| 8 | 683 | 678 | 0 | 0 | 676 | 0 | 676 |
| 16 | 683 | 677 | 0 | 0 | 676 | 0 | 676 |

读法：异值来源在**第一轮**把 q 大量推到局部预期；第二轮约 2/3 q 回到
原值，第四轮后约 99% 回到原值。同值来源几乎没有影响（3 个 case 在
第一轮受损，之后无新增 changed/harmed）。这描述的是“单次局部向量替换后
的短程传播”，不是持续修复率。

## 3. 具体案例

### 案例 A：`s42_e143 / N32_shallow:0 / u66`

- 原值 `a=13`，fan-in=2；目标系数 `leaf2:1, leaf10:1`（u = x2 + x10 mod 23）；
  叶常数 x2=0、x10=8。
- 三份来源只改两个叶：same `x2+2, x10+21`；shift1 `x2+2, x10+22`；
  shift3 `x2+2, x10+1`；来源 u 读出分别 13/14/16。
- 直接子节点 q=[104,112]，原真值 [1,12]；局部预期 +1=[2,13]，+3=[4,15]。
- unchanged/sham/same 的第一轮 q 都是 [1,12]；shift1 第一轮 [2,13]，
  shift3 第一轮 [4,15]。u 的 first-forward 输出是 13/13/6；shift 分支的
  q 在 round3 回到 [1,12]。同值来源的 hidden L2 距离 2.91，仍保持 q 正确。

### 案例 B：`s42_e143 / N32_shallow:0 / u132`

- 原值 `a=5`，fan-in=3；目标系数 `leaf2:2, leaf22:1, leaf14:3, leaf18:1`；
  叶常数 x2=0、x18=16。
- 来源改动：same `x2+1, x18+21`；shift1 `x2+1, x18+22`；
  shift3 `x2+1, x18+1`；来源 u 读出 5/6/8。
- q=[152]，原真值 [1]；局部预期 +1=[2]、+3=[4]。
- unchanged/same 第一轮 q=[1]；shift1=[2]；shift3=[4]；u 的
  first-forward 输出为 5/5/21；shift1 的 q 在 round3 回到 [1]，shift3 在
  round2 回到 [1]。这是多父节点、不同原值的例子；同值来源 L2 距离 4.23，
  也没有改变 q。

两例都显示：局部向量只在一个 forward 内明显影响直接子节点，随后系统
回算/回稳；不能把第一轮命中写成持续修复。

## 4. 边界

- 这是**条件于结构/源资格/unchanged q 条件**的诊断样本；不是总体准确率，
  也不能把 371 个候选当成独立图。一个图可有两个 q，重复使用同一
  unchanged/同值轨迹，不作独立重复样本。
- 同值来源与异值来源的 leaf 改动位置/次数匹配，但 hidden 向量距离没有匹配；
  同值向量不是 exactly-equal（L2 0.044–5.65）。
- 异值分支的第一轮 q 命中局部预期，支持“局部向量携带与 VALUE 相关的
  计算信息”这一行为解释；仍不能说明整个向量只包含 VALUE，也不能替代
  内部算法/因果证明。
- 13 个 ineligible 全部 q=0；源四属性或 unchanged 第一轮 q 条件失败的分布
  在 `coverage.csv`，没有为了凑数换图或改区间。
- 没有在同一数据上重跑旧 C4，也不把旧 C4 的百分比和本轮拼表；没有 C+
  训练、B 机制解释或跨位置/跨图来源。

## 5. 产物

- smoke：`tmp/analysis/c-source-control-20261115/smoke_s42/`
- main：`.../s42/`、`.../s1234/`（各 192 candidates；191/180 eligible）
- summary：`.../summary/`（`coverage.csv`、`case_curves.csv`、
  `paired_effects.csv`、`summary.csv`、`full_reference_curves.csv`）
- 提交的小表：`docs/records/2026-11-15/c-source-control-coverage.csv`、
  `c-source-control-summary.csv`
- 完整 records、hidden/轨迹大文件留在 tmp，不进入 git。
