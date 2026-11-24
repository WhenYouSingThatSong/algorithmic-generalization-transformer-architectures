# C 持续错父与噪声剂量校准（首批）

日期：2026-11-15 20:55 CST。  
状态：**四个现有权重的持续错父 L=2/4 与噪声强度校准已完成；仅评估，没有新训练**。  
代码：`reproduction@3320449`（运行期间固定，未 pull/改配置）。  
权重：initial=s2026/e143；hold/noise/semantic=腐蚀 pilot 的 epoch3 final。  
数据：128 张 N32 开发验证图；噪声只用固定前 64 图；均为已查看诊断数据，不是独立最终测试。

审阅更正（2026-11-15 21:32）：按提交 CSV 核对并修正记录总数、对照范围、强制阶段时点和噪声分母；补充高噪声长期残留。原始数值未改、未重跑 GPU，详见 `ai-collab/handoff/2026-11-15-2132-c-stress-review.md`。

## 1. 执行与覆盖

- 7 项新增 stdlib/torch 测试通过；sustained/noise smoke 各 4 图、两模式汇总通过；正式运行无 NaN/OOM。
- sustained：4 权重 ×128 图 ×2 时点 ×L=2/4 ×8 分支；共 2048 个权重×graph×timing×duration 记录，每条包含 8 分支。
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
- 三组后训练的 `free_q_clean`/`original_q_clean`/`same_q_clean` 在所提交检查点的 q 都是 128/128；initial late 释放 r16 分别为 L2 的125/128、L4 的124/128，存在自然漂移。`original` 首轮与 free 一致。在持续错 u 的前提下，`wrong_q_wrong` 与 `wrong_q_clean` 的释放曲线接近，部分首轮有差异；不能将此表述套到不写 u 的 `free_q_wrong` 分支，也不能用总数接近断言逐例相同。
- 关键对照：**最后一个强制轮** u 的输出有 92–108/128 正确，q 只有 0–5/128 正确。第一强制轮的 `wrong_q_clean` q 则有8–40/128正确，不能笼统说强制期间一直0–5。释放 r2 多数回正（具体首次恢复时间见原始 target_events，不用稀疏检查点替代）。注意每个强制轮的输入 u 都被重新写错，q 与 u 正确输出属于同一 forward；这个现象本身不证明存在独立信息通道或检测机制。
- 事件：initial late 的 q 释放后仍有约 110–111/128 末两轮正确，三组后训练为 128（hold/semantic）或 126（noise）；initial 的额外错误轮数明显更高。semantic 在 release r1 低于 hold，r4/r16 接近 hold；本批没有显示 semantic 的持续错父恢复更快。
- descendants 只有少数残留；详细 release/full 窗口、first recovery、reerror 见 `c-stress-sustained-events.csv` 与 `target_events` 原始文件（tmp）。

## 3. 噪声强度校准

每个模型只改 u，随机方向在 0.5/1/2/3/4/6 倍语义替换 L2 下测量；每图抽两个固定方向，每个方向在六档强度间复用，不是两个方向彼此相同。`same`/`wrong` 是参照。下表的 `new error` 是“原本 correct→注入后 VALUE 错”，不是所有 after-wrong。

### 保持时点（hold；common VALUE 64 图）

| condition | 平均实际 L2（hold） | new VALUE error（hold） | 首轮 children harmed（hold / semantic） |
|---|---:|---:|---:|
| free | 0 | 0/64 | 0/132 / 0/132 |
| same | 0.40 | 0/64 | 0/132 / 0/132 |
| wrong | 5.78 | 64/64 | 124/132 / 107/132 |
| noise 0.5× | 2.89 | 0/128 | 0/264 / 0/264 |
| noise 1× | 5.78 | 1/128 | 1/264 / 0/264 |
| noise 2× | 11.55 | 17/128 | 46/264 / 27/264 |
| noise 3× | 17.33 | 50/128 | 83/264 / 44/264 |
| noise 4× | 23.11 | 73/128 | 88/264 / 46/264 |
| noise 6× | 34.66 | 98/128 | 88/264 / 46/264 |

噪声的128目标试次=64图×两方向；264子节点试次=132个直接子节点×两方向。控制/语义替换只有一遍，分母不同，不能直接横比损伤计数。上表子节点比例为 pooled counts，主表的逐图等权统计另见原始完整汇总。

- 同 L2 的 `wrong`（语义异值源）造成 64/64 目标读出错误和 124 个子节点损伤；1× 噪声虽同 L2，只有 1/128 目标读出错误、0–1 个子节点损伤。**L2 匹配不等于错误剂量或功能损伤匹配。**
- 随强度增加，目标读出错误率约从 0/1% 升到 13–20%（2×）、39–50%（3×）、56–65%（4×）、75–79%（6×）；四模型趋势相近，具体数字见 `c-stress-noise-dose.csv`。
- 在高强度下 semantic 权重的首轮 children harmed 低于 hold（6×：46/264 vs 88/264；late：56/264 vs 92/264）。这是当前单seed在未用于训练的强噪声上的迁移观察，值得保留，但不是总体训练收益或特定机制证明。initial 更高（6×：保持130/264、晚期151/264），三组后训练均有减少，semantic 在这个指标上又低于 hold/noise；不能从 initial 更差反推“不是后训练共性”。各模型 donor/norm 仍不同。
- late 时点趋势相同但整体损伤更高：2× children harmed hold 74、semantic 51；6× hold 92、semantic 56。完整表见 `c-stress-noise-events.csv` / `c-stress-noise-accuracy.csv`。

### 首轮损伤少，不等于后续容易恢复

保持时点、选中的 q（每图一个；噪声两方向）：

| 注入 | hold 首轮正确 / 第16轮正确 | semantic 首轮正确 / 第16轮正确 |
|---|---|---|
| 异值源替换 | 4/64 → 64/64 | 15/64 → 64/64 |
| 3×噪声 | 93/128 → 120/128 | 108/128 → 122/128 |
| 6×噪声 | 90/128 → 90/128 | 106/128 → 95/128 |

6×噪声的首轮传播少于异值替换，但后续有延迟损伤和持久残留。semantic 的6×受损q末两轮正确36/69（hold33/71）；这些是各自受损集合，不是同分母恢复率对照。逐图等权的全部后代额外错误轮数，6×保持为hold5.398、semantic4.091，晚期6.489、5.385；3×晚期却是hold2.084、semantic2.093，不能写成semantic所有场景都降低累计损伤。

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

这是单图示例，不是平均。3×时u在输入及所有16轮输出都读作正确的1，但q第2–4轮为16、第5轮才回22；6×时u首轮输出已回1，q从第2轮到第16轮一直为21。这里没有持续覆盖。这说明局部VALUE读出不足以判定计算影响是否消失，但不能据此定位编码/注意力头；q其他父节点或结构字段也可能受到影响。当前示例未导出noise的clean分支，clean配对核对在聚合表中，逐例完整对照留原始records。

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
