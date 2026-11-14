# tmp/records 索引

本目录是复现过程的**工作记录**（非论文正文、非最终交付物），按时间累积，
不删除历史结论；被推翻的结论保留并在原文中划除。
**目录结构**：自 2026-11-09 起按日期分子目录（`2026-11-06/` … `2026-11-09/`），
本 README 保留在根目录作为索引；新增记录按日期放入对应子目录，并在下表登记。
最近整理：2026-11-09 23:45 CST · Claude Code

## 建议阅读顺序

1. `2026-11-08/experiment-results-20261108.md` — 当前实验结果总表（六张表 + 表注）；
2. `2026-11-08/margin-saturation-finding.md` — 对 raw/clean 机制的最终解释；
3. `2026-11-07/aux-channel-finding.md` — raw/clean 双口径的由来；
4. `2026-11-08/cyh-weights-check.md` — 同学权重的核查结论；
5. `2026-11-08/replication-report-20261108.md` — 阶段性总报告；
6. `2026-11-09/inject-discrete-finding-20261109.md` — **最新核心发现**：
   离散 VALUE 注入实验（纠错训练 vs 无纠错，u/q 恢复率）；
7. `2026-11-09/injection-design-for-gpt.md` — 注错实验的完整设计与待 GPT 审查的问题；
8. `2026-11-09/daily-report-20261109.md` — 11-09 工作总汇报（C0 提速、切换、ETA）；
9. `2026-11-10/next-phase-plan.md` — 当前执行计划：离散线扩样+机制、连续线采错误轨迹；
10. `RUN_CHECKLIST.md` — **训练前/初期必须逐项核查的清单**（配置、词表、数据、loop、
    checkpoint 语义、前 100 step、第一个 epoch、跑后归档）；
11. `2026-11-12/checkpoint-registry.md` — C0 各阶段 checkpoint 总表
    （best_logged / best_available / final_logged / last.ckpt 分开）。

## 文件索引（按日期）

### 2026-11-06/

| 文件 | 内容 | 状态 |
|---|---|---|
| `env-setup.md` | conda `yhy11.6` 环境安装、源测速、依赖 | 历史记录，仍有效 |
| `data-gen.md` | 训练/验证数据生成：50,000×32 / 1,000×128、文件大小核对 | 历史记录，仍有效 |

### 2026-11-07/

| 文件 | 内容 | 状态 |
|---|---|---|
| `val-protocol.md` | 验证集口径：固定 128 节点、train/val 隔离、验证续算随机性 | 协议事实，仍有效 |
| `issues.md` | 已确认源码问题登记（DAG_train 残留导入、sys.path 依赖等） | 问题台账，部分待处理 |
| `scripts.md` | GPU 监控/调度脚本 v3.x、误判事故与修复 | 历史记录，仍有效 |
| `for-gpt-review.md` | A 线排查包（供外部审查的快照） | 历史快照，结论已被后续更新 |
| `ood-extension-facts.md` | OOD 扩展三事实（128 名词表天花板、位置编码、深度偏浅）+ deep pilot | 事实记录，仍有效 |
| `aux-channel-finding.md` | REGRET/MODIFICATION 无监督辅助因子 + raw 回灌污染；引入 raw/clean 双口径 | 核心发现，仍有效 |

### 2026-11-08/

| 文件 | 内容 | 状态 |
|---|---|---|
| `replication-report-20261108.md` | 阶段性总报告（A 线闭环、B 线归因中） | 阶段报告，部分被表 7/8 更新 |
| `cyh-weights-check.md` | 同学 B-M6 三 seed 权重核查：配置一致、未训满、clean 高/raw 低 | 核查记录，仍有效 |
| `experiment-results-20261108.md` | 六张大表 + 结论速览（含 A-L4fast 消融、cyh 对照） | **主结果表** |
| `margin-saturation-finding.md` | raw 满分的机制：argmax 饱和、margin 尾部；修正“自行重算”解释 | **机制结论** |

### 2026-11-09/

| 文件 | 内容 | 状态 |
|---|---|---|
| `launch-Aprime-C0-20261109.md` | A′（上游无纠错 L4H16D384+fast）×3 与 C0（连续 M4 nointerm）×3 启动记录 | 历史记录 |
| `c0-optimization-proposal.md` | C0 提速方案：compile 漏编、watch、日志、OMP、D1–D4 | 方案 + 证据 |
| `eval-nointerm-fix.md` | `eval_nointerm_model.py` 修补（导入/ckpt/config/save/画图）+ C checkpoint 演习 | **已走通** |
| `speedup-report-20261109.md` | C0/A′ 提速复盘：compile 漏编、watch、日志、OMP、CPU 争用；六 run ETA | 复盘报告 |
| `daily-report-20261109.md` | 11-09 工作总汇报：C0 提速、评估链修复、六 run 切换/迁移、ETA、后续计划与待 GPT 问题 | 供外部 AI 分析 |
| `inject-discrete-finding-20261109.md` | 离散 VALUE 注入实验：A′×3 vs B-M6，u/q 恢复率、轮边界纠错机制 | **最新核心发现** |
| `injection-design-for-gpt.md` | 注错实验的完整设计、连续线方案、效度威胁与给 GPT 的问题 | 待外部审查 |
| `status.tsv` | 11-09 六 run 进程/GPU/状态快照 | 历史快照 |

### 2026-11-10/

| 文件 | 内容 | 状态 |
|---|---|---|
| `next-phase-plan.md` | 下一阶段执行计划：离散线扩样（N=32/128、256 图/档、teacher/own-state、rounds 1/2/4）+ parent-corruption 2×2；连续线五个候选、先采 C0 自然错误轨迹 | **当前执行计划** |

### 2026-11-12/

| 文件 | 内容 | 状态 |
|---|---|---|
| `checkpoint-registry.md` / `.json` | C0 三阶段 checkpoint 总表：旧 C0-100ep（vocab129）、新 C0-50ep、新 C0-100ep（vocab33）；区分 `best_logged` / `best_available` / `final_logged` / `last.ckpt` | **当前记录** |
| `vocab-data-correction-20261112.md` | **重要更正**：官方生成器从全部 128 个变量名不放回抽样重命名；旧数据覆盖 128 个名字，新 MixedN8–32 数据只覆盖 32 个；新数据生成器 `n_vars=32` 是错误，数据必须用 129 词表重新生成；含 N128 heldout 与 preflight 记录 | **新 C0/C+ 必须遵守** |
| `../RUN_CHECKLIST.md` | 训练前/初期核查清单：配置 diff、运行时词表、数据 OOV/N-A、loop、loss 口径、checkpoint 语义、前 100 step、第一 epoch、跑后归档 | **必须使用** |
| `perf-bench-a-class-20261112.md` | A 类性能补丁 300-step 对比：A1+A3 约 7%，A4 无额外收益且不采纳；未重启正式 run | 记录 |
| `../runs/C0-M4-MixedN8-32-D12-Vocab128-2Val-100ep/run_record.md` | 新 C0 双验证 run 的完整记录：数据、配置、选模规则、preflight、smoke、ETA、启动命令 | **训练中** |

### 2026-11-13/

| 文件 | 内容 | 状态 |
|---|---|---|
| `loss-curves-20261113.md` / `loss_history.csv` | C0 2Val run 的 epoch loss 快照：三 seed 的 val / val_n128 / train_epoch，best epoch，早期 spike | 训练中快照 |
| `migration-loss-drop-check.md` | 核查迁移后 loss 快速下降：数据/验证集未变（hash、mtime、checkpoint hparams）；resume 重置 shuffle；seed2025 误用 last=e53 而 e55 top-k 存在 | 结论 + 流程错误记录 |
| `b-checkpoint-fingerprints.md` | P0：所有 B 权重（B-M6 + cyh 三 seed）的 `last` mask、vocab、params、路径盘点 | 已完成 |
| `p1-pilot-20261113.md` | P1 pilot：B 128 图 R-only/EMPTY5 轨迹 + C0 中期 eval smoke；4→4 待做 | **旧数作废**：parent 脚本行/列索引 bug，见下条 |
| `code-review-fixes-20261113.md` | **代码修复记录**：parent 2×2 `st[Rg]`→`st[:,Rg]`；v2 teacher-state 轮次下标 `loop`→`r-1`；受影响结果清单 + smoke + 全量重跑结果 | **修复完成，重跑完成** |
| `metric-and-tokenizer-fixes-20261113.md` | 指标/tokenizer 修复：统一 `build_factored_tokenizer`；分因子/位置/`last_var` 排除 PAD；新增数值节点、整图数值全对、strict 前三指标；batch=8/1 一致性 smoke | **代码修复完成，smoke 通过** |
| `../runs/C0-M4-MixedN8-32-D12-Vocab128-2Val-100ep/migration/MIGRATION_PLAN.md` | 三 seed 迁移记录：old/new GPU/PID、resume ckpt、丢失 epoch、速度 | 进行中 |

### 2026-11-14/

| 文件 | 内容 | 状态 |
|---|---|---|
| `c0-naming-vocab.md` | **命名/词表更正**：C0=旧 C0；欠拟合 C0′=新 MixedN 数据、129 词表、100ep；vocab33 是独立错误配置、已废弃，不得当作 C0′ 50ep | **必须遵守** |
| `b-mechanism-design.md` | B 机制的完整设计：early/late 三时点、共同合格样本、late 四格、分母、控制、训练腐蚀审计、代码与执行顺序 | **设计稿，待审查；未执行** |
| `b-local-self-consistent-deferred.md` | 局部自洽错误控制：**DEFERRED / NOT RUN**；当前构造可能不自洽，不进入结果 | **搁置** |
| `b-mechanism-changelog.md` | B 代码 v2：审查修复清单、局部自洽搁置、smoke 状态 | **代码审查用** |
| `b-mechanism-smoke.md` | 4 图/1 batch 的流水线 smoke：selection/early/late/controls/audit；非结果 | **smoke 完成** |

## 当前核心结论（截至 2026-11-09 晚）

1. 离散潜空间 + 中间监督的规模/深度泛化成立：clean 口径下 A/B 系在 N=8…128、
   纯链深度 122 均接近或达到 100%。
2. 论文图 4 的 M5→M6 差距，大半来自**评估协议噪声（raw/clean）+ 容量混淆**，
   不能直接归因于纠错训练。
3. "纠错训练带来 raw 鲁棒性"**已被否定**：无纠错的 A-L4fast（训满）raw 全满分；
   有纠错但未训满的 cyh 三 seed raw 仅 21–96%。
4. raw 满分的机制是 **argmax 决策饱和**，不是主动纠错：
   logits 会随 aux 扰动移动，但 margin（平均 28–35 nat）足够大，argmax 不翻。
   决定 raw 成败的是 margin 分布的**尾部**（A-M5 s42 尾部 0.20 → raw 0%）。
5. **离散 VALUE 注入**（2026-11-09）给出了“纠错训练是否真有用”的直接证据：
   - A′（无纠错，3 seed×47 候选）：u 恢复 **4.3%**，下游 q 最终正确 **0.7%**；
   - B-M6（有纠错，47 候选）：u 恢复 **100%**，q 最终正确 **100%**；
   - 纠错发生在**轮边界**：同轮 q 要么推迟一轮、要么先错后改，从不与 u 的纠正在
     同一 forward 内完成。
   - 尚未排除“B-M6 是从父节点重算，而非检测错误再改回”的替代解释；
     需 parent-corruption 2×2 实验（见 `injection-design-for-gpt.md` §2）。
6. **连续 M4（C0）** 在 N≥40 的整图全对率接近 0（N=32 为 0.553），
   而离散 A′ clean 在 16 档 ~100%——连续状态在 OOD 规模上明显退化。
7. 未复现：M1–M3 基线；论文机制分析；B/A-L4fast 各仅单 seed。

## 维护约定

- 新记录按**创建日期**放入 `YYYY-MM-DD/` 子目录，并在本 README 对应日期段登记一行；
- 结论被推翻时**划除并注明**，不要静默改写历史；
- 数字类结论必须能由 `tmp/analysis/` 的原始 `.pt` 复算，并注明脚本；
- 区分三种口径：论文报告 / 源码核查 / 复现安排（见 `2026-11-07/for-gpt-review.md` 开头）。
