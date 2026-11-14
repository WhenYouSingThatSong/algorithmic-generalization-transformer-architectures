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
- `graph_equal`（逐图等权）为主要汇总；`node_weighted` 和原始分子/分母
  作为辅列（见 `b-main-m1-key-metrics.csv`）。零受损分母记为 NA。
- 由于不同模型/条件的受影响节点集合和 clean 可观测集合不同，`damage`
  不能跨模型直接当作“谁更差”比较；主要比较 `stable`。

## 3. early own 2×2

A′/B 各自 own before_u/after_u，H=16；B 的 H=8 为同轨迹截取。

| 时点 | 条件 | A′ damage (ge) | A′ stable | B damage (ge) | B stable H=8 (ge / nw) | B stable H=16 |
|---|---|---:|---:|---:|---:|---:|
| after_u | U | 1.0000 | 0/2,677 | 0.6515 | 0.9953 / 0.9916 | 1.0000 |
| after_u | R | 0.2578 | 0/820 | 0.8479 | 0.9801 / 0.9724 | 1.0000 |
| after_u | U+R | 0.9871 | 0/3,008 | 0.8480 | 0.9794 / 0.9721 | 1.0000 |
| before_u | R | 0.9872 | 0/3,009 | 0.8066 | 0.9798 / 0.9719 | 1.0000 |
| before_u | U | 1.0000 | 0/2,677 | 0.0000 | —（无受损） | —（无受损） |
| before_u | U+R | 0.9856 | 0/3,003 | 0.7988 | 0.9797 / 0.9720 | 1.0000 |

表中 damage/stable 的 ge 为逐图等权；stable 的 nw 为节点加权辅列。
B 的 H=16 分母分别为 1,913、2,648、2,656、2,533、0、2,507；A′ 的
stable 分母与 damage 分母相同。

- B 在五个有受损的条件上 H=8 的 graph-equal stable 为 0.9794–0.9953，
  H=16 全部 1.0000。
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
damage 为 0，stable 分母也为 0，报告为 NA；这里只记录该计数，不计算
恢复率。观测上 B 首次前向没有保留这个空槽写入的可观察错误，A′ 在同一
条件下仍有 100% 受影响节点受损；该差异如何解释待后续检查，不并入
after_u 的既有错误值修复统计。

## 4. teacher64 对照

teacher start 使用 64 张候选/92 条链，H=16；条件与 own 相同。

| 模型/条件 | damage ge (nw) | stable H=8 ge (nw) | stable H=16 |
|---|---:|---:|---:|
| A′ after_u U | 1.0000 (1.0000) | 0/633 | 0/633 |
| A′ after_u R | 0.2268 (0.2359) | 0/171 | 0/171 |
| A′ after_u U+R | 0.9825 (0.9848) | 0/714 | 0/714 |
| A′ before_u R | 0.9825 (0.9848) | 0/714 | 0/714 |
| A′ before_u U | 1.0000 (1.0000) | 0/633 | 0/633 |
| A′ before_u U+R | 0.9800 (0.9834) | 0/713 | 0/713 |
| B after_u U | 0.6496 (0.6983) | 0.9979 (0.9955) | 100% (442/442) |
| B after_u R | 0.8452 (0.8703) | 0.9795 (0.9778) | 100% (631/631) |
| B after_u U+R | 0.8466 (0.8731) | 0.9796 (0.9779) | 100% (633/633) |
| B before_u R | 0.7926 (0.8110) | 0.9784 (0.9762) | 100% (588/588) |
| B before_u U | 0.0000 (0.0000) | —（无受损） | —（无受损） |
| B before_u U+R | 0.7831 (0.8000) | 0.9784 (0.9759) | 100% (580/580) |

teacher start 与 own 的定性模式一致：A′ 无稳定恢复，B 的 H=8
graph-equal 稳定率为 0.9784–0.9979、H=16 为 100%。但 teacher64 只有
92 条链；下面的同链配对显示它在这批链上与 own 轨迹重合。是否把它作为
独立起点对照，需要由 GPT/用户判断；这里只保留原全集和配对结果。

### 4.1 teacher64 与 own 的同链配对

按同图、同链（`event_id`）、同 timing、同 condition 筛取 own 对应记录
（`b_subset_records.py`），每个 timing 得到 92 链 × 4 条件 = 368 条，
与 teacher64 完全对齐。A′/B 各 before_u/after_u 四个组合均为 368/368
记录，且 `rounds` 与 teacher 记录逐值相同；H=8/H=16 的 damage/stable
也逐值相同（`summary/teacher_pair_own/teacher_vs_own_subset.csv`）。

因此在这批 92 条链上，own rollout 与 teacher 记录的 `rounds` 逐值相同；
H=8/H=16 的 damage/stable 也相同。这里只记录该事实，是否把 teacher
视为独立起点对照由 GPT/用户判断；原 teacher 与 own 全集结果保留作
可复现记录。

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

B 的 H=8 节点加权稳定率为 96.7%–99.6%；按 graph-equal 为
0.9627–0.9983；H=16 和 H=64 都为 1.0000。A′ 在所有可用 H 上都是 0。
clean 分支在 H=64 仍为 100%。H=8 到 H=16 的差异来自少量未稳定节点；
H 的最终选择待 GPT/用户判断，这里保留 H=8/16/64 的原始截取。

### 5.1 第 17–64 轮反复检查

现有 summary 的 `stable` 是 H 窗口最后两轮正确；这不能证明第 16 轮正确
的节点在 17–64 轮从未再错。`b_long_window.py` 直接从 long32 逐轮记录
统计（all_affected，VALUE）：

| B 条件 | 第16轮正确 | 17–64 再错 | 曾恢复→再错 | 末尾两轮稳定 | 最终连续正确段起点 |
|---|---:|---:|---:|---:|---:|
| after_u R | 404 | 0/404 | 0/354 | 354/354 | 中位数 4，全部 ≤16 |
| after_u U | 355 | 0/355 | 0/261 | 261/261 | 中位数 3，全部 ≤16 |
| after_u U+R | 404 | 0/404 | 0/355 | 355/355 | 中位数 4，全部 ≤16 |
| before_u R | 404 | 0/404 | 0/332 | 332/332 | 中位数 4，全部 ≤16 |
| before_u U+R | 404 | 0/404 | 0/331 | 331/331 | 中位数 4，全部 ≤16 |

B 在该 64 轮窗口内没有“第 16 轮已正确、17–64 又错”的节点；所有曾恢复
的节点也没有再错。A′ 的 stable 仍为 0；A′ 第 16 轮未正确的节点大多
没有最终连续正确段。该结果只能表述为“在本观察窗口内保持”，不能外推
永久稳定；详见 `summary/long32_early/long_window.csv` 与
`long_window_nodes.csv`。

## 6. 数据观察与待判断项（不作机制结论）

这里只列出可直接从记录核对的观察；解释、机制和最终口径由 GPT/用户判断。

1. early own：B 的五个有受损条件在 H=8 的 graph-equal stable 为
   0.9794–0.9953，H=16 为 1.0000；A′ 六个条件在 H=8/H=16 都为 0。
2. teacher64 与 own 的同图/同链子集：A′/B × before_u/after_u 的
   368/368 条记录 `rounds` 逐值相同，H=8/H=16 的 damage/stable 也相同。
3. long32：B 的 H=8 graph-equal stable 为 0.9627–0.9983，H=16/H=64 为
   1.0000；A′ 全为 0。第 17–64 轮复错统计见 5.1，均未观察到复错；
   这只是本观察窗口内的记录。
4. clean 分支在 H=16/64 全节点正确率均为 100%。
5. 字段统计：5,349,376 个 node-rounds 中 SYNTAX/VARIABLE/OPERATION
   错误均为 0，VALUE 错误 793,656；“VALUE 对、first4 错”为 0；
   保存标志与 `pred4/true4` 重算不一致为 0。该计数只说明本批记录中
   VALUE 与 first4 的汇总相同，是否视为指标退化或实现问题待判断。
6. `before_u_U`：`actual=0`、`empty_to_value>0`、B 的 all_affected
   damage=0，stable 分母为 0（NA）；A′ 同条件 all_affected damage 为
   100%。这是时点/状态差异的记录，不计算 B 的恢复率。
7. 范围仅覆盖当前 N32 heldout 固定候选、s42、A′ e86/B e92 及本阶段
   条件；未覆盖 N128、其它 seed、C0/C+、switch、训练腐蚀。

## 7. 产物与复现入口

原始轨迹：

- `tmp/analysis/b-main-20261114/early/`
- `tmp/analysis/b-main-20261114/teacher/`
- `tmp/analysis/b-main-20261114/long32/`

汇总：

- `tmp/analysis/b-main-20261114/summary/early_own/`
- `tmp/analysis/b-main-20261114/summary/teacher64/`
- `tmp/analysis/b-main-20261114/summary/long32_early/`
- `tmp/analysis/b-main-20261114/summary/teacher_pair_own/`
- `tmp/analysis/b-main-20261114/summary/long32_early/long_window.csv`
- `tmp/analysis/b-main-20261114/summary/long32_early/long_window_nodes.csv`
- `tmp/analysis/b-main-20261114/summary/m1_field_errors.csv`
- 本报告附带 `b-main-m1-key-metrics.csv`（84 行，graph-equal 与
  node-weighted 关键行）。

汇总入口：

```bash
python -I tools/b_mechanism/b_main_summary.py \
  --inputs <4 个同子集 JSON> --out_dir <summary 子目录> \
  --horizons 8,16
```

long32 使用 `--horizons 8,16,32,64`。所有 runner 使用 `--num_threads 4`，
GPU3 串行，未动 GPU0 的 seed2026；C 线未参与本阶段。

## 8. 闸门

M1 数据和 M2 complete 数据都已落盘；M3 sustained、M4 D1/D2/D3 未启动。
本报告只保留 M1 数据、计数和待判断项，M2 计数见
`b-main-m2-report.md`。是否继续后续阶段由 GPT/用户判断。