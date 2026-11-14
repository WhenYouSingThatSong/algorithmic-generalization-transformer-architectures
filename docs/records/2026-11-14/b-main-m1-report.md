# B 主实验 M1 阶段结果：early own 2×2、teacher 对照、long32 H=64

日期：2026-11-14
代码基线：`reproduction@9df92a8`
状态：**阶段结果，等待 GPT/用户检查；M2 complete 尚未启动。**

## 1. 本阶段跑过什么

固定候选与资格沿用 M0：`graph_id 32–287`，256 张候选，无 top-up；256 张
common complete，218 张有链，365 条主链。teacher 子集为前 64 张候选
（32–95，92 条链）；long32 子集为前 32 张候选（32–63，49 条链）。

| 子集 | 模型/起点/时点 | 图 | target/条件 | 记录数/文件 | 文件数 |
|---|---|---|---:|---:|---:|
| early own | A′/B × before_u/after_u | 256 | 365 | 1,460 | 4 |
| teacher64 | A′/B × before_u/after_u | 64 | 92 | 368 | 4 |
| long32 early | A′/B × before_u/after_u | 32 | 49 | 196 | 4 |

共 8,096 条轨迹记录。early/teacher 的 H=16 轨迹同时提供 H=8 截取；long32
只跑 own，H=64。条件均为 `clean,u,r,ur`；使用 `--value_mode a_plus_1`。
所有 12 个 runner 输出均为 `complete=true`，`processed` 与 manifest 图数
一致，`skipped=0`，没有缺 clean 或记录数异常。

模型权重：A′ `A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa` s42 e86；B
`B-M6-Nodes32-ADD-L4H16D384-DeBERTa` s42 e92。数据：
`tmp/analysis/heldout-seed42/ADD/testdata_N32.pt`。

## 2. 指标口径

- `all_affected` = direct（实际注入位置）∪ target_u ∪ downstream d1 ∪
  downstream d2+。只有该节点在 clean 轨迹对应轮次正确时才计入观测。
- `damage`：在 H 轮窗口内，clean 正确而注入轨迹至少错过一次的节点比例。
- `stable`：在 `damage` 的节点中，H 的最后连续 2 轮都正确的比例。
  这是条件在“已经受损”的节点上的恢复率，不是全体节点正确率。
- `node_weighted` 为主要汇总，`graph_equal` 同时保留；下表用
  node-weighted，graph-equal 在关键结论上与之一致（见
  `b-main-m1-key-metrics.csv`）。
- 由于不同模型/条件的受影响节点集合和 clean 可观测集合不同，`damage`
  不能跨模型直接当作“谁更差”比较；主要比较 `stable`。

## 3. early own 2×2

A′/B 各自 own before_u/after_u，H=16；B 的 H=8 为同轨迹截取。

| 时点 | 条件 | A′ damage | A′ stable | B damage | B stable H=8 | B stable H=16 |
|---|---|---:|---:|---:|---:|---:|
| after_u | U | 100.0% | 0/2,677 | 71.5% | 1,895/1,911 (99.2%) | 1,913/1,913 (100%) |
| after_u | R | 27.0% | 0/820 | 87.0% | 2,573/2,646 (97.2%) | 2,648/2,648 (100%) |
| after_u | U+R | 98.9% | 0/3,008 | 87.3% | 2,580/2,654 (97.2%) | 2,656/2,656 (100%) |
| before_u | R | 98.9% | 0/3,009 | 83.3% | 2,457/2,528 (97.2%) | 2,533/2,533 (100%) |
| before_u | U | 100.0% | 0/2,677 | 0.0% | —（无受损） | —（无受损） |
| before_u | U+R | 98.7% | 0/3,003 | 82.4% | 2,432/2,502 (97.2%) | 2,507/2,507 (100%) |

- B 在五个有受损的条件上 H=8 稳定率为 97.2%–99.2%，H=16 全部 100%。
- A′ 六个条件在 H=8/H=16 的 stable 都是 0。
- 12 个模型的 clean 分支在 H=16 全节点正确率均为 100%。

### 直接位置首轮改回（own early H=16）

| 模型 | U-only | R-only | U+R（两个 direct 位置） |
|---|---:|---:|---:|
| A′ | 0/365 改回，365 错 | 33/365 改回，332 错 | 33 改回，697 错（共 730） |
| B | 365/365 改回，0 错 | 365/365 改回，0 错 | 365 改回，365 首轮仍错（共 730） |

B 的 U+R 首轮有一个 direct 位置仍错，但到 H=16 全部稳定恢复；单看
U-only 或 R-only 时两个 direct 条件都在首轮改回。

### `before_u_U` 的边界

该条件在 early own/teacher64/long32 中 `actual=0`（没有替换已存在的数值），
而是 `empty_to_value`：own 365、teacher 92、long49。B 的 `all_affected`
damage 为 0，因此没有 stable 分母；这**不能解读为 B 的恢复成功**。观测上
B 首次前向没有保留这个空槽写入的可观察错误，A′ 在同一条件下仍有 100%
受影响节点受损。后续机制解释需要单独检查该时点的输入/输出，不应把它和
after_u 的“已有错误值修复”混在一起。

## 4. teacher64 对照

teacher start 使用 64 张候选/92 条链，H=16；条件与 own 相同。

| 模型/条件 | damage | stable H=8 | stable H=16 |
|---|---:|---:|---:|
| A′ after_u U | 100.0% | 0/633 | 0/633 |
| A′ after_u R | 23.6% | 0/171 | 0/171 |
| A′ after_u U+R | 98.5% | 0/714 | 0/714 |
| A′ before_u R | 98.5% | 0/714 | 0/714 |
| A′ before_u U | 100.0% | 0/633 | 0/633 |
| A′ before_u U+R | 98.3% | 0/713 | 0/713 |
| B after_u U | 69.8% | 440/442 (99.6%) | 442/442 (100%) |
| B after_u R | 87.0% | 617/631 (97.8%) | 631/631 (100%) |
| B after_u U+R | 87.3% | 619/633 (97.8%) | 633/633 (100%) |
| B before_u R | 81.1% | 574/588 (97.6%) | 588/588 (100%) |
| B before_u U | 0.0% | —（无受损） | —（无受损） |
| B before_u U+R | 80.0% | 566/580 (97.6%) | 580/580 (100%) |

teacher start 与 own 的定性模式一致：A′ 无稳定恢复，B H=8 为 97.6%–
99.6%、H=16 为 100%。但 teacher64 只有 92 条链，且不是与 own 的 365 条
链做配对；这里只作为状态起点对照，不做因果配对结论。

## 5. long32 H=64

固定 32 张候选/49 条链，own before_u/after_u，四个条件，直接跑 H=64。

| 模型/条件 | B stable H=8 | B stable H=16 | B stable H=64 | A′ stable（H=8/16/64） |
|---|---:|---:|---:|---:|
| after_u U | 260/261 (99.6%) | 261/261 | 261/261 | 0 |
| after_u R | 343/354 (96.9%) | 354/354 | 354/354 | 0 |
| after_u U+R | 344/355 (96.9%) | 355/355 | 355/355 | 0 |
| before_u R | 321/332 (96.7%) | 332/332 | 332/332 | 0 |
| before_u U | —（无受损） | —（无受损） | —（无受损） | 0（A′ 全受损） |
| before_u U+R | 320/331 (96.7%) | 331/331 | 331/331 | 0 |

B 的 H=8 稳定率为 96.7%–99.6%，H=16 和 H=64 都为 100%；A′ 在所有
可用 H 上都是 0。clean 分支在 H=64 仍为 100%。
这支持“主 H=16”而不是 H=8：H=8 仍有少量不稳定，H=16 已饱和，H=64
长期保持。

## 6. 本阶段能说什么、不能说什么

可以说的：

1. 在这个 N32 heldout 固定候选集、s42、A′ e86/B e92 的设置下，B 在
   after_u/before_u 的 U、R、U+R（除 before_u_U 无受损外）都出现
   “先受损、H=16 稳定恢复 100%”的模式；A′ 没有稳定恢复。
2. teacher start 与 own start 的定性结果一致；long32 的 H=64 结果没有
   出现“恢复到 H=16 后又漂移”。
3. clean 输入在同样轮数下没有漂移，当前 clean 对照为 100%。

不能说的：

- 不能把 `before_u_U` 的 0 damage 当作 B 的恢复能力；该条件没有实际替换
  既有值。
- 不能把 `damage` 比例直接当作模型好坏；受影响集合不同，且 R-only 的
  下游传播范围不同。
- 不能外推到 N128、其它 seed、C0/C+、switch/held-state 或训练腐蚀。
- 不能把 teacher64 与 own 当作同一批 target 的配对实验。
- 不能据此确认 first-4 指标有效：本批三份汇总中 `value` 与 `first4` 的
  all_affected 行逐值完全一致（3,060 对）；这需要单独确认是指标定义
  在这个数据上退化，还是记录实现问题。

## 7. 产物与复现入口

原始轨迹：

- `tmp/analysis/b-main-20261114/early/`
- `tmp/analysis/b-main-20261114/teacher/`
- `tmp/analysis/b-main-20261114/long32/`

汇总：

- `tmp/analysis/b-main-20261114/summary/early_own/`
- `tmp/analysis/b-main-20261114/summary/teacher64/`
- `tmp/analysis/b-main-20261114/summary/long32_early/`
- 本报告附带 `b-main-m1-key-metrics.csv`（84 行，node-weighted 与
  graph-equal 关键行）。

汇总入口：

```bash
python -I tools/b_mechanism/b_main_summary.py \
  --inputs <4 个同子集 JSON> --out_dir <summary 子目录> \
  --horizons 8,16
```

long32 使用 `--horizons 8,16,32,64`。所有 runner 使用 `--num_threads 4`，
GPU3 串行，未动 GPU0 的 seed2026；C 线未参与本阶段。

## 8. 闸门

M2 complete 目标类型、M3 sustained、M4 D1/D2/D3 均未启动。等待 GPT/用户
检查本报告、汇总代码和 key-metrics CSV 后再继续。