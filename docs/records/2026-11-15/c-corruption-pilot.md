# C 单节点腐蚀后训练三组小比较：hold / noise / semantic

日期：2026-11-15 18:55 CST。
状态：**三组训练、128 图两时点评估与汇总已完成；单 seed、复用已查看过的验证集，不是最终泛化或机制结论**。
代码：`reproduction@0f2e619`（运行期间未 pull、未改配置/输入）。
起点：原始 C0 s2026/e143，SHA256
`40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780`。
交接：[C 腐蚀后训练三组](../handoff/2026-11-15-c-corruption-pilot.md)。

## 1. 设计、数据与执行

- 复用上一批新生成的 1024 train / 128 val（官方 N32/depth≤6，固定 seed，manifest 为
  long-pilot 的 `data/manifest.json`）；没有用 val 选 checkpoint，也没有扩大 N。
- 三组都从原始 s2026/e143 出发，均训练 2D 轮、3 遍、batch8、lr=1e-4，各自新建
  同配置 AdamW；损失为求解段 0.5×均值 + 保持段 0.5×均值，前四字段权重 1，aux 0。
- 计划：1024 个 train 图均有一个固定结构目标，所有目标在 3 遍中调度 1489 次；
  每图每遍独立哈希 50% 尝试；每次至多一个位置。结构目标均有父/直接子/可匹配
  反事实，delta 为 1–22 的固定非零模偏移；train 的 D 分布为 4/5/6/7，
  target depth 1–5。
- 三组预算相同：393 optimizer steps、3072 图曝光、recipient sample×loop 40284、
  recipient token×loop 10312704、recipient attention 2640052224；donor 额外
  sample×loop 12298。完整哈希/预算见 `c-corruption-audit.json`。
- 9 项 stdlib/torch 测试通过；三组各 4 batch smoke、各 8 图评估、汇总和标准
  checkpoint 加载均通过；正式训练无 NaN/OOM。smoke 仅作实现检查，未从 smoke
  权重继续训练。

### 实际训练注入剂量

| arm | scheduled | injected | before VALUE correct | donor VALUE intended | after first readout wrong | changed readout |
|---|---:|---:|---:|---:|---:|---:|
| hold | 1489 | 0 | 1489/1489 | 1488/1489 | 0 | 0 |
| noise | 1489 | 1489 | 1481/1489 | 1489/1489 | 32/1489 | 44/1489 |
| semantic | 1489 | 1489 | 1485/1489 | 1489/1489 | 1489/1489 | 1489/1489 |

- `before` 是注入前目标 VALUE 是否正确；`after` 是替换后、下一 forward 前的读出。
- semantic 的所有注入均命中指定异值；noise 的随机方向只偶尔改变读出，
  即使 L2 与该位置的 semantic 替换相同。hold 的 `actual_norm=0`。
- 三组 donor 计划/参考 forward 相同；平均 semantic reference norm 约 6.14–6.25。
- 有 4 个 semantic 调度目标的 `before` 已不正确；这些仍保留在计划中，不按
  行为补样。完整逐条注入日志在 `tmp/.../{hold,noise,semantic}/injections.jsonl`。

## 2. 干净长程保持（无干预）

128 图，节点数 4096；`first4` 是四字段同时正确，`VARIABLE` 的下降是主要来源：

| arm | D+16 VALUE / first4 | D+32 VALUE / first4 | D+64 VALUE / first4 | abs64 VALUE / first4 |
|---|---:|---:|---:|---:|
| initial | 0.9995 / 0.7979 | 0.9893 / 0.1995 | 0.9146 / 0.0293 | 0.9255 / 0.0403 |
| hold | 1.0000 / 0.9998 | 0.9995 / 0.9272 | 0.9873 / 0.3506 | 0.9902 / 0.4497 |
| noise | 1.0000 / 0.9998 | 0.9995 / 0.8816 | 0.9878 / 0.2883 | 0.9915 / 0.3743 |
| semantic | 1.0000 / 0.9998 | 1.0000 / 0.9304 | 0.9927 / 0.3523 | 0.9954 / 0.4468 |

- 四个 endpoint 的 SYNTAX/OPERATION 在所有时点都是 1.0000；first4 的下降
  全部由 VARIABLE 解释。完整逐字段曲线见 `c-corruption-clean-summary.csv`。
- 与 initial 相比，三组后训练在 D+16/32/64 都有明显的 VALUE/first4 长程改善；
  这是 +D 后训练配方本身的效应，不是本批注错组的额外收益。
- hold 与 semantic 的干净指标接近（D+64 VALUE 0.9873 vs 0.9927，first4
  0.3506 vs 0.3523；abs64 方向也不完全一致）；noise 在 D+64/abs64 的 first4
  较低（0.2883/0.3743），但这是单次训练、单 seed，不能据此认定噪声训练损伤。
- 没有看到三组相对于 hold 的一致“干净保持受损”签名；也没有看到 semantic
  一致优于 hold 的干净指标。

## 3. 固定干预评估

### 资格

- 128 图 × 2 个固定时点 = 256 个 graph×timing；四个模型（initial/hold/noise/
  semantic）的 `value_qualified` 均为 128/128，即 **256 个共同 VALUE 资格**。
- `common_strict` 为 201/256；初始模型在 late 只有 73/128 满足严格四属性，
  hold/noise/semantic 在 hold/late 均为 128/128。分母不是同一批图时，不能把
  strict 子集的改善直接推广到全部 128 图。
- 评估分支共享原始起点：unchanged、same（同值异源）、semantic（目标异值）、
  noise（L2 匹配随机方向）、noise3（3×幅度，仅评估）。所有分支的 target/child
  资格按同一图/同一时点固定，不按成功与否丢样。

### 目标与直接子节点（VALUE，common 256 个事件）

| timing | arm | target r1 / r2 / r16 | children r1 / r2 / r4 / r16 | children r1 harmed |
|---|---|---:|---:|---:|
| hold | initial | 108 / 128 / 128 | 17 / 169 / 252 / 263 | 249 / 266 |
| hold | hold | 105 / 128 / 128 | 22 / 161 / 249 / 263 | 244 / 266 |
| hold | noise | 98 / 126 / 128 | 25 / 158 / 248 / 263 | 241 / 266 |
| hold | semantic | 91 / 126 / 128 | 67 / 151 / 248 / 263 | 199 / 266 |
| late | initial | 109 / 128 / 127 | 30 / 154 / 225 / 233 | 236 / 266 |
| late | hold | 106 / 128 / 128 | 22 / 162 / 249 / 263 | 244 / 266 |
| late | noise | 99 / 126 / 128 | 26 / 157 / 247 / 262 | 240 / 266 |
| late | semantic | 93 / 126 / 128 | 77 / 159 / 246 / 263 | 189 / 266 |

- `same` 分支在 `common_value` 主视图的 target/children/descendants 与 unchanged
  基本相同；`all_fixed` 或个别 late 切片存在 1 个节点级的差异（例如
  noise late children r16 为 266 vs unchanged 265）。它来自另一条源题的
  hidden 向量，不是 exact unchanged，不能当作零影响对照；其后期差异也不能
  当作干预收益。`noise` 的匹配幅度通常不足以改变目标读出；`noise3` 只作压力评估。
- semantic 使目标第一轮 VALUE 更常错误（hold 时 91 vs hold 105；late 93 vs 106）；
  目标到第 2 轮已基本恢复（126–128/128）。因此没有看到 semantic 比 hold
  更快修复目标。
- semantic 的直接子节点在**第一轮**较少受损（hold 时 67/266 正确 vs hold
  22/266；late 77 vs 22）；但到第 2 轮 hold 已反超（161 vs 151；late 162 vs 159），
  第 4 轮和末轮两分支接近。即“即时影响范围更小”，不是“恢复更快”或“最终更稳”。

### 完成后事件与额外错误轮数

- target：四个模型的 semantic 分支都 128/128 受损、128/128 曾恢复；semantic
  各有 1 个 reerror（hold 无），末两轮均为 128/128。没有目标持续错误差异。
- children：hold 时 semantic 的 damaged=227/266，hold=255/266；但
  final2 among damaged 为 224/227 vs 252/255，graph-equal final2 为 0.9947 vs
  0.9952；late 类似。semantic 的 graph-equal excess wrong rounds 在 children
  为 1.33（hold 时）/1.27（late），hold 为 1.45/1.45；差异小且没有重复 seed。
- descendants 的方向相同，semantic 的即时 damaged 数和 excess wrong rounds
  略低，但末两轮 graph-equal 仍接近；不能据此声称下游恢复机制改善。
- 完整 event_summary / intervention_summary（含 all_fixed、common、strict 与
  first4）在 CSV；报告正文只列 common VALUE 主视图。

### 时点与噪声压力评估

- hold 时点和 late(D+16) 的方向相同：semantic 目标首轮更易错、子节点首轮
  损伤更少，但没有末两轮净改善；不是“越晚越受益”。
- noise3 只在评估中，不加入训练。其高幅度会造成较大的子/后代损伤；
  semantic 模型在 noise3 下的 damaged 数和 excess wrong rounds 低于 hold/
  noise（例如 hold 时 children damaged 65 vs 96/101，late 76 vs 102/105）。
  这可以作为一个待验证的鲁棒性观察，但不能说成本批训练出了噪声鲁棒性：
  noise3 没有参与训练，且模型权重/干净状态不同。

## 4. 两个完整例子

例子是固定选的，不是随机抽样；完整 17 轮（含初始读出）见
`c-corruption-examples.csv`。

| case | 分支 | target VALUE（真值：初始/首轮/末轮） | children 正确（r1/r4/r16） | descendants 正确（r1/r4/r16） |
|---|---|---|---|---|
| graph89 / late / semantic | initial | 14: 22 / 20 / 14 | 3/3/3 | 6/6/6 |
| graph89 / late / semantic | hold | 14: 22 / 20 / 14 | 1/3/3 | 4/6/6 |
| graph89 / late / semantic | semantic | 14: 22 / 20 / 14 | 3/3/3 | 6/6/6 |
| graph68 / late / semantic | hold | 6: 0 / 6 / 6 | 2/1/2（n=5） | 13/5/3（n=16） |
| graph68 / late / semantic | semantic | 6: 0 / 6 / 6 | 4/2/2（n=5） | 15/12/3（n=16） |

- graph89：目标本身在三个模型都被注入成 22（真值 14），但 semantic 训练后的
  直接子节点首轮 3/3、后代 6/6，hold 为 1/3、4/6；到 r4 两者都恢复。
  这是“即时下游损伤更小”的例子，不是最终恢复更好。
- graph68：目标从错误初值恢复到真值 6，但 hold 和 semantic 的子节点末轮都只有
  2/5，后代只有 3/16；说明目标修复不等于整条依赖链恢复，且这里 semantic
  训练没有消除该残留。是否由结构/数值吸引域造成，本批没有机制证据。

## 5. 解释与边界

- 三组共享计划、预算、起点和数据；差异主要在注入方式（hold 无、noise 随机方向、
  semantic 当前模型反事实源向量）。但三组训练权重不同，评估时的源向量/干净状态
  也不同；这是实际干预实现差异，不能仅凭胜负判定“内部错误检测/纠错机制”。
- 本批只验证一次注入后的 16 轮；没有反复钳制、多个节点、不同错误幅度搜索、
  人工注错、自然错误重放、混合训练、N64/N128 训练或从头训练。
- semantic 的目标首轮更易错、下游即时损伤更少，但没有稳定/复现的恢复增益；
  这不支持“semantic 注错训练已带来纠错能力”，也不支持“noise 训练更好”。
- 128 图验证集在上一批已查看过，不能称最终独立测试；单 seed，不能做总体排序。
- 保留继承的 `reduce_batch` factor-PAD 裁剪缺口；三组一致，评估节点 mask 排除
  PAD。本批没有修原训练 recipe。
- 输出目录、权重、完整 records/NPZ 留 tmp，不提交。报告中的 `common_value`
  与 `all_fixed` 分母不同，CSV 分开列出；空分母为 NA，不当作成功。

## 6. 产物与哈希

- 训练：`tmp/analysis/c-corruption-pilot-20261115/{hold,noise,semantic}/`
  （meta、steps、injections、final.ckpt、日志）
- 评估：`.../evaluation/`（clean_curves、coverage、records、逐图 NPZ、meta）
- 汇总：`.../summary/`（clean、intervention、event、dose、budget、node_events）
- 提交小表：
  - `c-corruption-clean-summary.csv`
  - `c-corruption-intervention-summary.csv`
  - `c-corruption-event-summary.csv`
  - `c-corruption-training-dose.csv`
  - `c-corruption-coverage.csv`
  - `c-corruption-evaluation-dose.csv`
  - `c-corruption-budget.csv`
  - `c-corruption-examples.csv`
  - `c-corruption-audit.json`
- 源码/依赖/数据/计划哈希和 checkpoint SHA 见 `c-corruption-audit.json`；
  budget 三组相同，checkpoint 不同但评估规则固定。
- 原始 NPZ/权重/完整 records 不提交；GPU2 已释放。
