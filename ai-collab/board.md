# 工作看板

> 最近更新：2026-11-14 15:00 CST · [gpt]
> 活看板：任务可以移动；状态变化在 `log.md` 记一行。条目格式 `[YYYY-MM-DD HH:MM][author]`。

## 当前决策

- [2026-11-14 15:02][user，gpt 记录] 后续 GPT 负责与用户商量和设计，其他 agent 负责执行和小验证；GPT 不需要过度防御性验证，具体疑点与验证点交执行 agent 处理。此长期分工已按用户明确要求写入协作协议与 consensus。
- [2026-11-14 15:00][user，gpt 记录] 后续统一在 **`reproduction`** 协作。GPT 已将修复 `c91b6fe` 及交付记录 `dc61d12` 快进并入本地 reproduction，本次通过 reproduction 提交推送。此前审阅分支交付方式由本条取代；指标/评估仍未定稿，全量实验状态不变。
- [2026-11-14 14:01][claude] `consensus.md` 先留空；具体口径、结果和当前状态放在本看板，待用户确认后再上升为共识。
- [2026-11-14 14:25][claude] GPT review 已完成：R1/R2/R3 是确定的实现缺陷，暂不进入正式 B 实验；先修实现，再讨论指标和跑 Phase 0。
- [2026-11-14 14:41][claude] R1/R3/R2/R7/R8 代码修复和 4 图 pipeline smoke 完成；未启动 B 全量实验，R4–R6/R9 指标仍待讨论。

## 需要 review（GPT）

- [x] **B 机制代码 v2 + 指标实现**（GPT 已完成；发现 P1 缺陷，见“Review 结论”）
  - 审查对象：`origin/reproduction` 的 B 包提交 `ab80594`；review 修复提交 `66440b8`（分支最新提交另含本协作目录）
  - **先读日志**：`log.md` 的 2026-11-14 11:35 / 13:44 条目，了解为什么改、改了哪些
  - 代码：`tools/b_mechanism/`
  - 设计与边界：`docs/records/2026-11-14/b-mechanism-design.md`
  - 变更/修复清单：`docs/records/2026-11-14/b-mechanism-changelog.md`
  - pipeline smoke（非结果）：`docs/records/2026-11-14/b-mechanism-smoke.md`
  - 重点：A′/B 共同资格、single-parent + q2、early 2×2、late D1/D2/D3、sustained/switch、audit `S[:-1]`、summary N0/N1/N2、manifest 哈希验证
  - 回复：在下方“Review 结论”追加，并在 `log.md` 记一行；不要直接改 `consensus.md`

## 进行中

- [2026-11-14 10:12][claude] 旧 C0 seed2026 150 epoch，GPU0；输出 `tmp/runs/oldC0-newseed-150ep/`；14:42 检查 epoch 61/150，约 5.9 it/s，val 1.540。不要停止/重启。
- [2026-11-14 13:44][claude] B review 包：commit `ab80594`；GPT review 完成，R1/R2/R3/R7/R8 已修，当前不启动实验。

## 待办 / 下一步

- [x] [gpt] 按上节 review B 代码和文档，给出可执行结论。
- [x] [claude] 先修 R1（early 资格键统一）和 R3（selection 搜索/保存预算一致），再生成 manifest；验收含“一边合格一边不合格”和换 batch_size 的用例。
- [x] [claude] 修 R2（summary 保留运行身份、完整配对键和分组键），覆盖 A/B 同图、own/teacher、before/after 和输入文件顺序交换。
- [x] [claude] 处理 R7（controls condition/clean/汇总）和 R8（D1/D2/D3 新真值参照与汇总）接口缺口；仅 pipeline smoke。
- [ ] [user/claude] 讨论 R4–R6/R9：全窗口统计、曾恢复 vs 持续恢复、节点/图聚合、节点级指标；指标未定稿前不称为正式验收。
- [ ] [user/claude] 决定是否/何时启动 B Phase 0；当前只做了 4 图 smoke，未跑全量。
- [ ] [claude] 实现缺陷修复且用户批准后先跑 B Phase 0（A′ s42 + B-M6 s42 的 common-complete manifest），报告合格 n；保留完整原始轨迹。
- [ ] [claude] Phase 0 合格数足够后再跑 Phase 1（early 2×2、late R-only/U-only、late 四格、unrelated），单卡串行。
- [ ] [claude] Phase 2 sustained/switch；Phase 3 训练腐蚀审计；局部自洽仍不跑。
- [ ] [claude] seed2026 完成后用当前 eval 实现评旧 N32/N128，并与 s42/s1234/s2025 对照；指标未定稿，先保留原始结果。
- [ ] [user/claude] 决定是否/何时重新设计局部自洽错误控制。

## 阻塞 / 等待

- B 全量实验：实现缺陷已修；等待 R4–R6/R9 指标讨论和用户批准。
- 局部自洽控制：构造可能不自洽，DEFERRED / NOT RUN。

## 最近完成

- [2026-11-14 14:41][claude] B review 修复：R1/R3/R2/R7/R8 代码完成，提交 `66440b8`，4 图 pipeline smoke 通过；未改训练、未启动全量实验。
- [2026-11-14 14:25][claude] 合并 GPT 审阅分支 `review/b-mechanism-5137014`；审阅文档进入 `handoff/2026-11-14-1406-gpt-b-mechanism-review.md`。
- [2026-11-14 13:40][claude] 项目备份 `tmp/archive/project-backup-20261114.tar.gz`，SHA256 `7037e0ce2c1cc09b4494031327ce68e959950bec2ba6b2ab1eaa39095a73517a`。
- [2026-11-14 13:44][claude] 提交 `ab80594`：B 工具/文档迁出 `tmp/`、指标统计实现（未定稿）、`--save_top_k`、无效 C0 配置清理。
- [2026-11-14 13:45][claude] 推送 `origin/review/b-mechanism`（后重命名为 `reproduction`）。
- [2026-11-14 11:35][claude] B 代码 v2 smoke 全路径跑通；仅 pipeline smoke。
- [2026-11-14][claude] 旧 C0 150ep 的 N32/N128 结果已整理；数字见下方“共享发现”。
- [2026-11-13][claude] parent 2×2 和 teacher round 索引 bug 修复并重跑；旧 p1 数字作废。

## 背景与口径（参考）

- 当前主线：旧 C0 150ep + B 机制；离散线 A / A′ / B，连续线 no-interm C0。
- 命名：**C0** = 旧 C0（旧窄分布，50k，32 节点、depth ≤ 6，129 词表，seed 42/1234/2025，150ep）；**C0′** = 新 MixedN（8/16/24/32 节点、depth 1–12，100k，129 词表，100ep，已冷存）；**vocab33** = 独立错误配置，已废弃，不得引用。详见 `docs/records/2026-11-14/c0-naming-vocab.md`。
- 指标现状（未定稿）：现有 **Value Node Acc**、**Graph Value All**、**Strict First-4** 三个统计；但指标与评估尚未最终确定，旧三指标只作辅助/历史口径，不能称为正式验收标准。B common 的 numeric mask 与正式 `metric_utils` 的 numeric 范围不同，不能直接混称。
- 当前活跃：C0 旧线 + B 主对照 A′ s42 / B-M6 s42；C0′ 和 vocab33 不活跃；cyhB 三 seed 放第二批。

## 共享发现 / 证据

- [2026-11-14 14:41][claude] Post-review fix smoke（4 图）：selection 9 chains / 16 sinks / 16 leaves；batch 4 与 8 的 `(T, n_loops)` 一致；early 36 条、late 66 条、controls 9+9 clean；summary 7 行，A′/B 来源分离、输入顺序不变、`duplicate_cleans=0`。**仅 pipeline smoke**。
- 旧 C0 150ep 旧 N32（e143，Value / Graph / Strict，按当前 eval 实现口径，非最终验收）：s42 `0.9992 / 0.9873 / 0.9844`；s1234 `0.9990 / 0.9854 / 0.8770`；s2025 VALUE 约 0.48，后两项为 0。
- 旧 C0 150ep 旧 N128（e143）：s42 `0.9912 / 0.8379 / 0.0156`；s1234 `0.9677 / 0.2910 / 0`。
- 旧 C0 原始结果：`tmp/analysis/oldC0-resume-150ep/`；旧数据配方，不能直接与 B 的 heldout/matched-depth 样本混用。
- B smoke：4 张 heldout 图 selection 4/4 common-complete；early 28 条、late 53 条、leaf 16 条、controls 7+7；audit 128 candidates + 692 次替换的 pre-corruption 全 EMPTY。**不是结果**。
- 腐蚀审计初步信号：`last` 策略下 candidate 腐蚀前基本是 EMPTY，而非“已填值被改坏”；仅 1 个 batch，需全量确认。

## 开放问题

- B 是“按父值每轮重算”“检测父子不一致再重算”，还是“优先保留已有槽位”？需要 early/late 时点对照和 sustained/switch 控制。
- 训练腐蚀覆盖的是 `EMPTY→随机值` 还是 `已填值→错误值`？决定 `4→5` 是训练内行为还是跨时机泛化。
- 旧 C0 N128 的 strict 归零主要来自 VALUE 错误、VARIABLE 漂移，还是整图 conjunction 的尾部效应？
- C0′ 是否值得在修正数据/词表后重训，当前没有排期。

## Review 结论

- [2026-11-14 14:06][gpt] 已复核 `reproduction@5137014`，**未通过正式 B 实验验收**。本条取代上方“暂无”，保留此前记录。7 个 B Python 脚本在换行归一化后与 v2 压缩包完全一致，上一轮问题尚未修复。
  - P1 实现缺陷：early 标记 `early_own_after` 与读取的 `early_own_after_u` 不一致；summary 的 clean 配对键/汇总分组缺运行身份，可能串模型和时点；selection 搜索按 batch 预算、manifest 却保存单图预算。
  - P2 未完成/口径问题：只统计首轮受损、any-later 被叫恢复、未按图等权；controls 无 clean 且 mode 不被 summary 识别；D1/D2/D3 正确标记仍按旧 label。
  - 用户最新指标要求：两个按节点核心量——VALUE 准确率、前四属性同时正确率。旧整图三指标可保留为辅助；正式 metric_utils 仍含题面常量且 strict 仍为整图，须区分，不直接混用。
  - 局部自洽及依赖它的切换仍暂缓。独立持续错父后释放可补齐基线后推进；现有 switch_held_r 不等价于旧自洽切换，不纳入本阶段必跑项。
  - 证据、修复建议、验收样例见 [Git 复核交接](handoff/2026-11-14-1406-gpt-b-mechanism-review.md)。源码审查与包对照已完成；未独立运行模型 smoke，未修改实验代码或 consensus。

- [2026-11-14 14:12][gpt] 更正上述审阅措辞：用户明确**指标和评估尚未完全确定**。上条“用户最新指标要求”“未通过正式 B 实验验收”过度表述了协议的确定程度，以本条及修订后的交接文档为准。R1–R3 的资格键、运行身份配对、预算不一致仍是确定的实现缺陷；首轮/全窗口、曾恢复/持续恢复、节点/图聚合、两项节点核心指标属于待讨论选择，不能因尚未实现就认定为 bug。controls 汇总接口及 D1/D2/D3 真值参照另列为接口缺口或解释边界。未改 consensus。审阅分支目前仅在本地，尚未推送。

- [2026-11-14 14:25][claude] 已接收并合并 GPT 审阅分支。对齐结论：四层探索框架（早期时点 / 整图完成 / 更新依据 / 训练腐蚀）与当前设计一致；但当前实现只部分覆盖“逐轮传播与恢复”的三情形，R4/R5 是观察窗口与恢复定义待定，R1/R3/R2 是确定实现缺陷。下一步先修 R1/R3，再修 R2；不启动 B 实验。局部自洽和 `switch_held_r` 的旧“稳定错误切换”解释继续暂缓。

- [2026-11-14 14:41][claude] R1/R3/R2/R7/R8 已修复（`66440b8`）并通过 4 图 pipeline smoke；仍未启动 B 全量实验。R4–R6/R9 的指标/聚合/窗口定义继续待用户讨论。

- [2026-11-14 14:53][gpt] 复核 `e27cc6b`：R1/R2/R3 修复通过源码与 CPU 回归检查，controls 汇总接口已接通。发现两处残留：合法 switch 被 hold 分支遮蔽，第二错值从未写入；D1/D2/D3 的新真值漏改题面常数。用户授权必要修复后，已在 `review/b-mechanism-e27cc6b` 修正为 N 轮固定、第 N+1 轮一次切换并释放，同时补齐常数新标签。8 项 CPU unittest、8 文件语法编译和 diff 检查通过；没有运行 GPU 或改变服务器任务。指标/评估仍未定稿，switch 不据此升级为正式必跑实验。详细证据与受影响结果见 [后续复核与修复](handoff/2026-11-14-1453-gpt-b-mechanism-followup.md)。本条补充并修正上条“全部已修”的范围，历史记录保留。
- [2026-11-14 14:56][gpt] 交付状态：修复提交 `c91b6fe` 已推送至 `origin/review/b-mechanism-e27cc6b`，未合并 reproduction。服务器 agent 可从该分支核对两处修复及 CPU 回归，再补模型小样本检查；没有授权或启动全量实验，指标方案仍待用户讨论。
- [2026-11-14 15:00][gpt] 分支更正：按用户最新确认，以上修复和记录统一并入 reproduction，服务器 agent 后续从 reproduction 更新即可。旧分支及日志作为历史保留；后续提交不再另建审阅分支。
