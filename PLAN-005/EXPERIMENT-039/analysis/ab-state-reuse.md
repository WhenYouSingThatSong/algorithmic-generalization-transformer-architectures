# A/B 连续重新赋值与两个死槽位复用：首批结果

- 日期：2026-11-19。
- 状态：**使用已有 A/B 权重；固定 N32 开发图 32–63；repeat/reuse 各自 H8、H16 独立序列；未训练、未扩图、未进入 >128 节点或通用槽位分配**。
- 运行版本：`reproduction@6ef63e36f51113fa5c60f95d6d107c8521df4482`；本批 AB-main 在 GPU0 串行完成。C 的 state-reuse lane 由 C 执行，不在本报告合并。
- 来源：上一批已审阅 `reassign-main-A/B` 的首个 delta 完成快照；A `A-L4D384fast` s42/e86，B `B-M6` s42/e92；数据 SHA `08f52382…`，manifest SHA `19c99264…`。这是同一批开发图和已见来源，不是独立最终测试，也不是多种子复现。

## 1. 执行与覆盖

- AB smoke：先跑 graph32–33 / H4，8/8 队列任务完成；真实 tokenizer、起跑编辑、独立真值、阶段继承、死槽位和六分支检查通过。检查记录见 [smoke-checks](ab-state-reuse/smoke-checks.md)。
- AB main：repeat 与 reuse 各 32 图；两模型均 32/32 `selected`，无补图。repeat 每模型 576 条记录（32 图 × 2 schedule × 9 个阶段 × 2 个预算），reuse 每模型 128 条（32 图 × 2 阶段 × 2 个预算）；两模式分别有 576 和 128 个共同 case。
- `H8` 与 `H16` 是两次从同一源快照独立起跑的序列，**H16 的前 8 轮不是 H8 序列的延续**。每阶段预算固定，失败不等待、不补样。
- repeat 的 schedule：同输入 `+1,+3,+7,-11` 四阶段；交替输入 `a:+1,b:+3,a:+7,b:-3,a:-8` 五阶段；各阶段都从上一段末态继续。reuse 先回收 a，再回收 b，b 的新方程引用 a:v1；旧 a/b 是 sink，无旧消费者。
- 起跑 `state_round` 为源快照的 `T+1`，分布 5–9，均值 6.78；repeat/reuse 都保留该预计算成本。warm/clear/warm_pinned 从此状态开始，scratch 每阶段从当前完整新题 tokens 的 EMPTY 结果开始。

| 项目 | repeat | reuse |
|---|---:|---:|
| 图/模型 | 32/32 | 32/32 |
| 每模型记录 | 576 | 128 |
| 共同 case | 576 | 128 |
| H8/H16 | 独立 | 独立 |
| 分支 | warm/clear/scratch/warm_pinned/clean/sham | 同左 |

## 2. repeat：目标叶与真值改变下游（H16 末两轮）

每格为 `目标叶正确/32` 与 `changed nodes 正确/该阶段 changed 分母`；changed 分母在阶段间因模 23 抵消为 481–482。`clean`/`sham` 在全部阶段为 target 32/32、constants 192/192；`warm_pinned` 与 `warm` 的目标/changed 计数逐阶段相同，未观察到 pinning 改善。

| model | schedule | ep | warm target/changed | clear target/changed | scratch target/changed | warm_pinned target/changed |
|---|---|---:|---:|---:|---:|---:|
| A | same_input | 1 | 4/32 / 4/482 | 32/32 / 482/482 | 32/32 / 482/482 | 4/32 / 4/482 |
| A | same_input | 2 | 4/32 / 4/482 | 32/32 / 482/482 | 32/32 / 482/482 | 4/32 / 4/482 |
| A | same_input | 3 | 3/32 / 3/482 | 32/32 / 482/482 | 32/32 / 482/482 | 3/32 / 3/482 |
| A | same_input | 4 | 28/32 / 478/482 | 32/32 / 482/482 | 32/32 / 482/482 | 28/32 / 478/482 |
| A | alternating_inputs | 1 | 4/32 / 4/482 | 32/32 / 482/482 | 32/32 / 482/482 | 4/32 / 4/482 |
| A | alternating_inputs | 2 | 3/32 / 5/481 | 32/32 / 481/481 | 32/32 / 481/481 | 3/32 / 5/481 |
| A | alternating_inputs | 3 | 3/32 / 7/482 | 32/32 / 482/482 | 32/32 / 482/482 | 3/32 / 7/482 |
| A | alternating_inputs | 4 | 30/32 / 207/481 | 32/32 / 481/481 | 32/32 / 481/481 | 30/32 / 207/481 |
| A | alternating_inputs | 5 | 28/32 / 478/482 | 32/32 / 482/482 | 32/32 / 482/482 | 28/32 / 478/482 |
| B | same_input | 1 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 |
| B | same_input | 2 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 |
| B | same_input | 3 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 |
| B | same_input | 4 | 32/32 / 466/482 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 466/482 |
| B | alternating_inputs | 1 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 |
| B | alternating_inputs | 2 | 32/32 / 481/481 | 32/32 / 481/481 | 32/32 / 481/481 | 32/32 / 481/481 |
| B | alternating_inputs | 3 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 |
| B | alternating_inputs | 4 | 32/32 / 481/481 | 32/32 / 481/481 | 32/32 / 481/481 | 32/32 / 481/481 |
| B | alternating_inputs | 5 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 | 32/32 / 482/482 |

H8 的 target 计数与上表的所有 repeat 行相同，但 H8 的 changed/unchanged 计数更低；例如 A clear/scratch 最后阶段 changed 为 475/465，而 H16 为 482。H8 不是 H16 的前缀，详细值在 [ab-stage-summary](ab-state-reuse/ab-stage-summary.csv) 的 `horizon=8` 行。

## 3. repeat：旧答案保留、新答案接受与原题损伤（H16 末两轮）

每格是 `retain_previous/accept_new/extra_damage` 的节点数；retain/accept 是 changed nodes 的互斥读数，extra_damage 是原题真值未变保护节点相对同一轮 clean 的新增错误，不是第三类 changed 读数。分母见表 1 的 changed 数。H8 的同口径全表在 `ab-retention-summary.csv`。

| model | schedule | ep | warm | clear | scratch |
|---|---|---:|---:|---:|---:|
| A | same_input | 1 | 475/4/0 | 0/482/0 | 0/482/0 |
| A | same_input | 2 | 3/4/0 | 0/482/0 | 0/482/0 |
| A | same_input | 3 | 1/3/0 | 0/482/0 | 0/482/0 |
| A | same_input | 4 | 1/478/4 | 0/482/0 | 0/482/0 |
| A | alternating_inputs | 1 | 475/4/0 | 0/482/0 | 0/482/0 |
| A | alternating_inputs | 2 | 206/5/0 | 0/481/0 | 0/481/0 |
| A | alternating_inputs | 3 | 4/7/0 | 0/482/0 | 0/482/0 |
| A | alternating_inputs | 4 | 6/207/2 | 0/481/0 | 0/481/0 |
| A | alternating_inputs | 5 | 0/478/6 | 0/482/0 | 0/482/0 |
| B | same_input | 1 | 0/482/0 | 0/482/0 | 0/482/0 |
| B | same_input | 2 | 0/482/0 | 0/482/0 | 0/482/0 |
| B | same_input | 3 | 0/482/0 | 0/482/0 | 0/482/0 |
| B | same_input | 4 | 5/466/16 | 0/482/0 | 0/482/0 |
| B | alternating_inputs | 1 | 0/482/0 | 0/482/0 | 0/482/0 |
| B | alternating_inputs | 2 | 0/481/0 | 0/481/0 | 0/481/0 |
| B | alternating_inputs | 3 | 0/482/0 | 0/482/0 | 0/482/0 |
| B | alternating_inputs | 4 | 0/481/0 | 0/481/0 | 0/481/0 |
| B | alternating_inputs | 5 | 0/482/0 | 0/482/0 | 0/482/0 |

读法上的关键观察：

- repeat 期间，A 的 `warm` 在中间阶段多数保留旧答案，target 只在 3–5/32 左右；B 的 `warm` 在完整 H16 中每个阶段都接受新 target（32/32）。`clear` 和 `scratch` 在 H16 的 target 均 32/32。
- 回到原题阶段（same_input ep4 / alternating ep5）A warm target 为 28/32，仍有失败；B warm target 32/32，但 same_input ep4 的 changed 为 466/482 且有 **16 个保护节点 extra damage**，alternating ep5 为 482/482、damage 0。不能只拿 target 判断谁“恢复得更好”。
- H16 的 constants 在全部 repeat/reuse 条件和分支均为 192/192；changed 以外的 `unchanged` 区域不一定全对，例如 A warm same_input ep2 为 337/543、ep3 为 336/542；这些值与 `extra_damage` 的保护节点口径不同，不能混写。
- `warm_pinned` 与 `warm` 的轨迹和计数逐阶段相同；本批没有观察到 pinning 改善。H16 target 的 `first_all_correct` 与至少两次输出的稳定后缀计数与上表 target 相同；没有 target 回正后再错，详见 `ab-event-summary.csv`。

## 4. reuse：两个死槽位的结果

H16 末两轮的 target/changed 如下；changed 为 30/29（模 23 后两个/三个槽位新值碰巧等于旧值，未进入 changed 分母）。H8 的 target 与 H16 相同，除 ep2 scratch 为 29/32；H8 scratch 的 H8 changed 和 damage 也较低，完整值见 CSV。

| model | ep | warm target/changed | clear target/changed | scratch target/changed | warm_pinned target/changed |
|---|---:|---:|---:|---:|---:|
| A | 1 | 2/32 / 0/30 | 32/32 / 30/30 | 32/32 / 30/30 | 2/32 / 0/30 |
| A | 2 | 3/32 / 0/29 | 32/32 / 29/29 | 32/32 / 29/29 | 3/32 / 0/29 |
| B | 1 | 32/32 / 30/30 | 32/32 / 30/30 | 32/32 / 30/30 | 32/32 / 30/30 |
| B | 2 | 32/32 / 29/29 | 32/32 / 29/29 | 32/32 / 29/29 | 32/32 / 29/29 |

旧答案保留/新答案接受（H16）：

| model | ep | warm retain/accept/damage | clear retain/accept/damage | scratch retain/accept/damage |
|---|---:|---:|---:|---:|
| A | 1 | 30/0/0 | 0/30/0 | 0/30/0 |
| A | 2 | 29/0/0 | 0/29/0 | 0/29/0 |
| B | 1 | 0/30/0 | 0/30/0 | 0/30/0 |
| B | 2 | 0/29/0 | 0/29/0 | 0/29/0 |

- `clear` 在每个 reuse 阶段只清空目标结果槽（affected=target，1 个位置）；repeat 的 `clear` 则清空外部已知依赖闭包，实际每阶段 2–24 个节点（均值 15.06）。
- H16 时 A/B 的 `scratch` target 都 32/32；H8 的 ep2 `scratch` 为 29/32，且 extra_damage 为 A 16、B 19。H8 和 H16 是两个独立序列，不能把 H8 的失败当作 H16 的前缀结果。
- 在 reuse 中，A 的 `warm` changed 始终为 0/30 或 0/29（保留旧槽），B 为 30/30、29/29；`clear`/`scratch` 在 H16 均写出新 target。A/B 的 `warm_pinned` 与 `warm` 相同。

### 4.1 接口写入

- reuse 32/32 图均有结构计划；每图 2 个死 sink，a<b；旧 a/b 在原图中没有任何消费者。
- 每个 H16 模型有 136 个 `symbolic_edits` 操作数行（H8 同样 136；两阶段合计），其中 3 行是“符号编辑但字节未变化”的合法行；`raw_task` 的实际变化始终是这些行的子集，方程边界和操作数变量身份由 `core.expression_positions` 通过。
- b 的新 `new_parents` 全部包含新 a；archive 每模型每 horizon 96 条（每图 1+2），只保存模型实际输出，未回灌到任何分支起点；token 长度始终 1024，无重复定义或长度变化。
- 同值更新单列：target 旧/新值相同的图为 a 2/32、b 3/32；这些不进入 changed 分母，也不应写成“旧值残留”。

## 5. 资源成本

- 队列总耗时约 674.3 秒（GPU0 顺序执行；receipts：repeat-A 248.2 s、repeat-B 250.0 s、repeat-summary 28.8 s、repeat-delivery 11.8 s、reuse-A 61.7 s、reuse-B 61.6 s、reuse-summary 9.6 s、reuse-delivery 2.6 s）。
- 记录中的累计 forward（包含源快照预计算，不含外部数据加载）：
  - repeat：warm/clear/warm_pinned 均值 40.11（范围 13–89），scratch 均值 33.33（8–80）。
  - reuse：warm/clear/warm_pinned 均值 24.78（13–41），scratch 均值 18（8–32）。
- warm/clear/warm_pinned 的起点已包含源 `T+1` 预计算；scratch 每阶段从 0 重解。H8/H16、schedule 和 episode 的原始累计值在 `ab-forward-cost.csv` 与 records 中保留，不能用平均 forward 代替逐阶段预算。
- 每任务日志、receipt、plan、records/coverage hash 在 tmp；小表与 audit 进 git。

## 6. 案例

自动结果包按结果类别选了 6 条/模式，包含实际 inputs→outputs、目标和关键方程；它们只作阅读入口，不是总体发生率：

- repeat：`repeat-examples.jsonl`；A graph32 ep2（warm 失败、clear/scratch 成功）、A graph33 ep1（失败）、A graph32 ep1（warm 成功）、A graph33 ep4（回到原题成功）、B graph32/33 ep1（成功）。
- reuse：`reuse-examples.jsonl`；A graph32 ep1（warm 失败、clear/scratch 成功）、A graph33 ep2（失败）、A graph33 ep1（warm 成功）、A graph36 ep2（成功）、B graph32/33 ep1（成功）。
- 每条含 `plan`、旧/新真值、节点实际 inputs/outputs；完整轨迹和所有图仍在 `records.jsonl`/全量 summary（tmp），不把例子当选样。

## 7. 支持的观察与仍未区分的解释

- 可支持的行为描述：A 在完成态上启动的 `warm` 大多不接受重复的新题面；B 大多接受。`clear` 外部依赖清空和 `scratch` 从 EMPTY 重解在 H16 均能写出新 target。reuse 中 A 的 warm 不写目标槽，B 写；clear/scratch H16 成功。以上都是解码 VALUE/前四属性计数。
- H8/H16 的差异表明后续轮数会影响 changed/未改区域和少量失败，不能把 H8 当作 H16 前缀；H8 scratch reuse 的 29/32 需要与 H16 的 32/32 分开报告。
- 仍未区分：旧答案保留是“不能使用新常数”“旧结果被写回/未被覆盖”“下游更新失败”还是 readout/寻址/运算差异；clear 成功不等于模型自己知道依赖闭包；scratch 成功不等于同一内部更新路径。A/B 的权重与训练设置不同，且本批只有一个 seed 的开发图，不能作架构因果结论。
- 不能把 reuse 的 2 个 sink 结果写成通用槽位分配、128 节点突破或容量结论；外层归档/调度由实验者提供。

## 8. 证据与限制

- 运行/审计：`ab-state-reuse-audit.json`、`main-structure-evidence.json`；smoke 检查 `smoke-checks.md`。
- 小表：`ab-stage-summary.csv`（1320 行，H8/H16 × stage × region）、`ab-retention-summary.csv`（264 行）、`ab-event-summary.csv`、`ab-resource.csv`、`ab-forward-cost.csv`、`ab-reuse-interface.csv`、两份 delivery 的 `key-results.csv` 和 `examples.jsonl`。
- 完整 records、coverage、summary、events、delivery 源目录仍在 `tmp/analysis/state-reuse-exec-20261119/AB-main/`；本报告不改写任何旧 W 结果。
- 本批是 32 图固定开发集的单 seed 行为计数，不是独立最终测试，也不是四种子；H8/H16 不是独立图（同一批图的两个序列），不能按图做独立重复统计。

## 9. 文件 hash

完整文件清单与 hash 见 `ab-state-reuse/ab-state-reuse-audit.json`；关键小表：

| 文件 | SHA-256 |
|---|---|
| `ab-state-reuse-audit.json` | `7d3df854f6c9205490ae334ae606d200b05005c16ecfad600249a9f0b229d54c` |
| `ab-report-audit.json` | `e45ecb552d5ab9dafa6fb7ec27a323379c342f4324cb19b0dc7ec6e8eb902424` |
| `ab-stage-summary.csv` | `e927cebf8836f310ab904bd0958a2c7f9038ef1a58ab3bbbf9396eb3e2b06543` |
| `ab-retention-summary.csv` | `4ba749e7bf2668c141f52345689973335052cccc5650eb38cfb007a8668d88ca` |
| `ab-event-summary.csv` | `7f3aa63b3d11e8cdb929229cd1df7041aca48e8a6c0ff60ecc29946cce54cb90` |
| `ab-reuse-interface.csv` | `0088c7e3dbee1a38bf5afbbb4dd9d17bbc4f37f04b13ddec4c15a2a2a5e37a8c` |
| `ab-resource.csv` | `559cd6d2cb5641f941e22ad617d6e528ea95a412199d353c0bfe0c0b88048d89` |
| `ab-forward-cost.csv` | `c65ee31f67d62d0c7ad3e85ea52c7f9ca412b8820c99907f930ac84604832e05` |
| `main-structure-evidence.json` | `16c6c8cbd4cb98f8effea7bb4d386fb714fbf0f124ee3fbbe89435824b1b4c3d` |
| `repeat-examples.jsonl` | `4962b5df5ce9bc80c9c6032889038256e644e32ac5130a0fe7e9ca3056b5dd4f` |
| `reuse-examples.jsonl` | `8298235bde836d0a272878526213f8e3f1c241eecfd5029da9bada146644be16` |
