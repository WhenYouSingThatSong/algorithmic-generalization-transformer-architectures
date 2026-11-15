# C 持续错父与噪声剂量校准（首批）

日期：2026-11-15 20:55 CST。  
状态：**四个现有权重的持续错父 L=2/4 与噪声强度校准已完成；仅评估，没有新训练**。  
代码：`reproduction@3320449`（运行期间固定，未 pull/改配置）。  
权重：initial=s2026/e143；hold/noise/semantic=腐蚀 pilot 的 epoch3 final。  
数据：128 张 N32 开发验证图；噪声只用固定前 64 图；均为已查看诊断数据，不是独立最终测试。

## 1. 执行与覆盖

- 7 项新增 stdlib/torch 测试通过；sustained/noise smoke 各 4 图、两模式汇总通过；正式运行无 NaN/OOM。
- sustained：4 权重 ×128 图 ×2 时点 ×L=2/4 ×8 分支；共 1024 个 graph×timing×duration 记录。
- noise：4 权重 ×64 图 ×2 时点 ×15 分支；两方向是同一图的重复方向，不是独立图或 seed。
- sustained 共同 VALUE 资格：256/256 个 graph×timing（L=2/4 共用交集）；common strict 167/256。
- noise 共同 VALUE 资格：128/128 个 graph×timing；common strict 94/128。
- 所有固定候选都有结构 u→q；没有 no_candidate。运行 meta、source/export 哈希见 `c-stress-audit.json`。

## 2. 持续错父（sustained）

每个 L 先强制写 u 的源行 L 轮；`q_wrong` 只在第 1 次输入把 q 改错一次，之后不再写 q。释放后不写任何行，观察 16 轮。`free`、`original`、`same` 是 clean/源回放/同值源对照；`original` 不是全程 sham（历史状态已经分叉）。

### q 的释放期恢复（common VALUE；`wrong_q_clean`）

| timing | L | arm | release r1 | r2 | r4 | r16 |
|---|---:|---|---:|---:|---:|---:|
| hold | 2 | initial | 68/128 | 119/128 | 124/128 | 127/128 |
| hold | 2 | hold | 60/128 | 121/128 | 124/128 | 128/128 |
| hold | 2 | noise | 57/128 | 119/128 | 124/128 | 128/128 |
| hold | 2 | semantic | 39/128 | 117/128 | 124/128 | 128/128 |
| late | 2 | initial | 60/128 | 106/128 | 110/128 | 111/128 |
| late | 2 | hold | 59/128 | 122/128 | 124/128 | 128/128 |
| late | 2 | noise | 54/128 | 117/128 | 122/128 | 126/128 |
| late | 2 | semantic | 42/128 | 117/128 | 124/128 | 128/128 |

- L=4 的 release r1 略低（hold/noise 49–56，semantic 34–36），r4 已到 124，r16 与 L=2 接近；强制更久主要延迟最初几轮，不是释放后永久失败。完整 L=2/4 表在 CSV。
- `free`/`original`/`same` 的 q 在 forced/release 都是 128/128；`original` 首轮与 free 一致。`q_wrong` 分支（只在第一次输入改 q）与 `wrong_q_clean` 的 release 曲线几乎相同：q 的初始错误没有造成明显额外持续损伤。
- 关键对照：强制写错 u 时，**目标 u 的输出仍可在 92–108/128 个 case 读回正确**，但同一批的 q 输出只有 0–5/128 正确；释放后 q 的中位首次恢复约 2 轮。这说明“target 解码看起来对”不等于下游已恢复；本批只能作行为观察，不证明 hidden 是否被读取。
- 事件：initial late 的 q 释放后仍有约 110–111/128 末两轮正确，三组后训练为 128（hold/semantic）或 126（noise）；initial 的额外错误轮数明显更高。semantic 在 release r1 低于 hold，r4/r16 接近 hold；本批没有显示 semantic 的持续错父恢复更快。
- descendants 只有少数残留；详细 release/full 窗口、first recovery、reerror 见 `c-stress-sustained-events.csv` 与 `target_events` 原始文件（tmp）。

## 3. 噪声强度校准

每个模型只改 u，随机方向在 0.5/1/2/3/4/6 倍语义替换 L2 下测量；两方向共享同一方向、只在强度上缩放。`same`/`wrong` 是参照。下表的 `new error` 是“原本 correct→注入后 VALUE 错”，不是所有 after-wrong。

### 保持时点（hold；common VALUE 64 图）

| condition | 实际 L2（hold） | new VALUE error /128 | children harmed（hold / semantic） |
|---|---:|---:|---:|
| free | 0 | 0 | 0 / 0 |
| same | 0.40 | 0 | 0 / 0 |
| wrong | 5.78 | 64/64 | 124 / 107 |
| noise 0.5× | 2.89 | 0/128 | 0 / 0 |
| noise 1× | 5.78 | 1/128 | 1 / 0 |
| noise 2× | 11.55 | 17/128 | 46 / 27 |
| noise 3× | 17.33 | 50/128 | 83 / 44 |
| noise 4× | 23.11 | 73/128 | 88 / 46 |
| noise 6× | 34.66 | 98/128 | 88 / 46 |

- 同 L2 的 `wrong`（语义异值源）造成 64/64 目标读出错误和 124 个子节点损伤；1× 噪声虽同 L2，只有 1/128 目标读出错误、0–1 个子节点损伤。**L2 匹配不等于错误剂量或功能损伤匹配。**
- 随强度增加，目标读出错误率约从 0/1% 升到 13–20%（2×）、39–50%（3×）、56–65%（4×）、75–79%（6×）；四模型趋势相近，具体数字见 `c-stress-noise-dose.csv`。
- 在高强度下 semantic 权重的 children harmed 明显低于 hold（6×：46 vs 88；late：56 vs 92），但这是单 seed 的观测，且 donor/norm 各模型不同；不能把它当作已确定的训练收益或机制。initial 的 children harmed 更高（6×：130/151），说明这也不是“所有后训练”共性。
- late 时点趋势相同但整体损伤更高：2× children harmed hold 74、semantic 51；6× hold 92、semantic 56。完整表见 `c-stress-noise-events.csv` / `c-stress-noise-accuracy.csv`。

## 4. 两个可核对轨迹

实际 u/q 表达式（完整展开和局部预期见 `c-stress-example-expressions.json`）：

- graph89：`u=x1 = x60(21)+x64(16) mod23=14`；`q=x99 = x76(12)+x78(10)+x1(14) mod23=13`。
- graph0：`u=x53 = x79(1) mod23=1`；`q=x25 = x125(21)+x53(1) mod23=22`。

### sustained graph89 / late / semantic / L=2

| condition | u 输入→输出（前几轮） | q 输入→输出（前几轮） |
|---|---|---|
| free_q_clean | 14→14,14,14 | 13→13,13,13 |
| wrong_q_clean | 22→20,20,14,14 | 13→13,21,13,13 |
| wrong_q_wrong | 22→20,20,14,14 | 21→1,21,13,13 |

第 1/2 轮是 forced，之后是 release；`wrong_q_clean` 的 q 在第 2 轮被错误的 u 拉成 21，释放后回 13；`wrong_q_wrong` 的 q 首轮就是 1，之后也回 13。`free` 始终正确。完整逐轮数值在 `c-stress-example-values.csv`。

### noise graph0 / late / semantic

| condition | u 输入→输出（前几轮） | q 输入→输出（前几轮） |
|---|---|---|
| noise 1× | 1→1,1 | 22→22,22 |
| noise 3× | 1→1,1 | 22→22,16 |
| noise 6× | 6→1,1 | 22→22,21 |
| wrong | 6→18,1 | 22→10,16,22 |

这是单图示例，不是平均；它显示同一强度档可以有不同 target/q 模式，不能用一个节点的读出代替完整损伤。

## 5. 解释与边界

- 本批只评估现有四权重，没有新训练、三种子、真实改题、持续污染训练、层/头干预或新的 noise/semantic 训练组。
- 持续错父的 `wrong` 每个强制轮写入的是**该绝对时点**的源向量；`original` 也不是全程 sham。源题来自分别构造的 donor，局部 VALUE 关系匹配不等于整个 hidden 状态自洽。
- `q_matches_input_parent_readouts` 只是行为一致性；目标 u 读出正确不代表 q 已经使用该读数。持续实验说明“u 输出回正”不保证下游立即停止受影响，不能据此定位 inner readout 或独立计算通道。
- L2 相同不代表错误数量/功能损伤相同；噪声方向是同一图的重复测量，不把两方向当独立图或 seed。高噪声只是校准结果，不自动推荐某个训练强度。
- 128 图/64 图开发集已查看；单 seed，不作总体模型排名或最终泛化结论。
- 继承的 `reduce_batch` factor-PAD 裁剪缺口未修；评估 mask 排除 PAD。

## 6. 产物

- 运行：`tmp/analysis/c-sustained-noise-20261115/{sustained,noise}/`
- 汇总：`.../{sustained,noise}-summary/`
- 提交小表：
  - `c-stress-sustained-accuracy.csv`、`c-stress-sustained-events.csv`、
    `c-stress-sustained-dose.csv`、`c-stress-sustained-coverage.csv`
  - `c-stress-noise-accuracy.csv`、`c-stress-noise-events.csv`、
    `c-stress-noise-dose.csv`、`c-stress-noise-coverage.csv`
  - `c-stress-example-expressions.json`、`c-stress-example-values.csv`
  - `c-stress-audit.json`
- 原始 records/trajectory/target_events 和权重留 tmp；不提交。GPU2 已释放。