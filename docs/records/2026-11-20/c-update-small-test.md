# C_s2026 UPDATE 小测（dev-50 + 2 图 smoke）

日期：2026-11-20。状态：**训练队列中途小测，不是正式 evaluate，也不是最终测试**。  
代码：`reproduction@35edb21`（C_s2026 update 训练与小测使用的代码/plan 身份一致）。  
plan：`tmp/analysis/c-update-20261119/data/plan.json`，SHA256 `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`。  
C_s1234 correction 仍在训练；正式五原权重+四 final 评估仍由 Agent B 在全部最终权重交接后执行。

## 1. 产物与身份

- 训练输出：`tmp/analysis/c-update-20261119/run/train/C_s2026-update/`
- final：`final.ckpt`，SHA256 `c6ba26a0faed6185eed1685266bcbffd847c699215a44baf377eb659c0e7d107`
- last：`last.ckpt`，SHA256 `7fd353c2fa9d6b18c5b2dc2c554229fcb115fc67b9a2559336c450968c69f578`
- 训练审计：`training-audit.json`；50 epoch，78,200 batches，2.5M 图呈现。
- 小测：
  - `dev-50.json`：训练脚本在 epoch50 自动运行的 128 dev 图评估；
  - `run/small-eval/C_s2026-update/`：从 `final.ckpt` 重新加载的 2 图/N32_shallow smoke（只验证加载/评估链路，不是能力估计）。
- 原始 records、完整 dev 记录和权重都在 tmp，不提交。

## 2. 实际污染剂量

| 项目 | 值 |
|---|---:|
| batches | 78,200 |
| enabled batches | 3,879（4.96%） |
| graphs presented | 2,500,000 |
| candidates | 80,000,000 |
| attempts | 1,985,808 |
| accepted | 1,964,581 |
| source acceptance | 98.93% |
| 改前 EMPTY | 873,015 |
| 改前 numeric | 1,091,566 |
| already intended wrong | 49,205 |

这是实际注入日志的计数，不是假设比例；`before numeric` 较多表示更新训练中一部分结果位置在注入时已读成数字，不能把它当成“总是从 EMPTY 注入”。

## 3. dev-50 clean 行为

每格为节点平均（图等权）；第一列为 first4，括号为 VALUE。

| 模型 | D | D+4 | D+16 | D+16 graph value / first4 all |
|---|---:|---:|---:|---:|
| C_s2026 correction | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 / 1.000 |
| C_s42 correction | 1.000 (1.000) | 1.000 (1.000) | 0.998 (1.000) | 1.000 / 0.945 |
| C_s2026 update | 0.998 (0.998) | 0.9995 (0.9995) | 0.761 (0.998) | 0.977 / 0.000 |

- update 在 D/D+4 的 VALUE 和 first4 都接近 1；D+16 的 VALUE 仍为 0.998，但 VARIABLE 字段漂移使 first4 降到 0.761、整图 first4 全对为 0。这是长窗口身份字段漂移，不是算术 VALUE 大幅失败。
- C_s2026 correction 在 dev-50 的 D+16 仍全对；C_s42 correction 有轻微 VARIABLE 漂移（first4 0.998、整图 first4 0.945），VALUE 仍 1.000。

## 4. UPDATE 读出/执行（dev-50 clean 条件）

`update` 行统计的是同一批节点轮次；`plan`=第二层 UPDATE 头预测要更新，`changed`=末层 VALUE 实际改变，`reference`=按同步局部标签应更新。

| 条件 | observed | reference | plan | changed | value_correct | cleared | tp / fp / fn | plan_execution_agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 92,352 | 4,249 | 4,255 | 4,256 | 92,279 | 92,352 | 4,223 / 32 / 26 | 92,297 |

- 计划与执行在 92,297/92,352（99.94%）个节点轮次上一致；末层清零 92,352/92,352。
- tp 4,223、fp 32、fn 26 是当前 dev-50 的定位/执行计数；不能把 `plan_execution_agree` 当成“定位头因果控制了内部计算”，它只衡量第二层计划与末层 VALUE 是否改变。

## 5. 固定原 C 功能探针（dev-50 的 64 图子集）

`origin=original_C` 是未做本批后训练的原 C 接收；`origin=current_C` 是本批向量。两者都用冻结接收端；同值/原图来源是控制，不能用单侧失败当机制。

`complete_budget`、`wrong_value` 的关键计数（selected / receiver_intended / first_wrong / first_expected_wrong）：

| origin | selected | receiver_intended | first_wrong | first_expected_wrong |
|---|---:|---:|---:|---:|
| original_C | 64 | 64 | 57 | 34 |
| current_C | 64 | 63 | 56 | 29 |

- 同值 `same_value`：original_C 43/43 receiver_intended，first_wrong 0；current_C 43/43，first_wrong 0。
- `source_original` 与 `receiver_exact`：两 origin 均 first_wrong 0。
- `early` 时点：original_C wrong_value 43/43 receiver_intended、first_wrong 43；current_C 43/43、first_wrong 43（first_expected_wrong 42）。
- 解释：在这个小测子集里，错误源向量确实能让原 C 的下游 q 首轮读错；同值/原图/原样写回对照没有同样错误。但 `first_expected_wrong` 低于 `first_wrong`，说明“注入后 q 首轮错”与“恰好等于错父值代入结果”不是同一个量；完整分母和跨权重混杂需看 B 的正式表。

## 6. 2 图 smoke（只验证加载/链路）

从 `final.ckpt` 重新加载的 `N32_shallow` 2 图 smoke：

- D：first4/value 1.000/1.000；
- D+4：1.000/1.000；
- D+16：first4 0.719、value 1.000；
- D+64：first4 0.031、value 0.969。

与 dev-50 方向一致：VALUE 长窗口仍高，VARIABLE/first4 在更长窗口漂移。2 图数字不能作为总体估计。

## 7. 边界

- 这是开发集小测；正式 test、五个原权重+四个 final 的统一表仍由 B 执行，C 不重复正式 evaluate。
- C_s1234 correction 还在训练；本报告不把它或四份 final 当已完成。
- 不把 UPDATE 的 `plan`、`changed`、`tp/fp/fn` 写成硬门控或因果执行证明；没有 gate，中间 mask 不回写向量。
- 不把原 C 接收探针的“未传播”直接写成向量无效；同值/原图控制与接收端读出必须并读。
- 原始记录、dev JSONL、权重在 tmp，不提交；正式比较分母由 B 的统一 evaluate 产生。
