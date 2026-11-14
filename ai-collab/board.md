# 工作看板

> 最近更新：2026-11-14 13:56 CST · [claude]
> 活看板：任务可以移动；状态变化在 `log.md` 记一行。条目格式 `[YYYY-MM-DD HH:MM][author]`。

## 需要 review（GPT）

- [ ] **B 机制代码 v2 + 正式指标修复**
  - 审查对象：`origin/reproduction` 的 B 包提交 `ab80594`（分支最新提交另含本协作目录）
  - **先读日志**：`log.md` 的 2026-11-14 11:35 / 13:44 条目，了解为什么改、改了哪些
  - 代码：`tools/b_mechanism/`
  - 设计与边界：`docs/records/2026-11-14/b-mechanism-design.md`
  - 变更/修复清单：`docs/records/2026-11-14/b-mechanism-changelog.md`
  - pipeline smoke（非结果）：`docs/records/2026-11-14/b-mechanism-smoke.md`
  - 重点：A′/B 共同资格、single-parent + q2、early 2×2、late D1/D2/D3、sustained/switch、audit `S[:-1]`、summary N0/N1/N2、manifest 哈希验证
  - 回复：在下方“Review 结论”追加，并在 `log.md` 记一行；不要直接改 `consensus.md`

## 进行中

- [2026-11-14 10:12][claude] 旧 C0 seed2026 150 epoch，GPU0；输出 `tmp/runs/oldC0-newseed-150ep/`；上次检查 epoch 24/150，约 6.3 it/s。不要停止/重启。
- [2026-11-14 13:44][claude] B review 包：commit `ab80594`，等待外部 review。

## 待办 / 下一步

- [ ] [gpt] 按上节 review B 代码和文档，给出可执行结论。
- [ ] [claude] review 通过后先跑 B Phase 0（A′ s42 + B-M6 s42 的 common-complete manifest），报告合格 n。
- [ ] [claude] Phase 0 合格数足够后再跑 Phase 1（early 2×2、late R-only/U-only、late 四格、unrelated），单卡串行。
- [ ] [claude] Phase 2 sustained/switch；Phase 3 训练腐蚀审计；局部自洽仍不跑。
- [ ] [claude] seed2026 完成后用正式三指标评旧 N32/N128，并与 s42/s1234/s2025 对照。
- [ ] [user/claude] 决定是否/何时重新设计局部自洽错误控制。

## 阻塞 / 等待

- B 全量实验：等待外部 review 和用户批准。
- 局部自洽控制：构造可能不自洽，DEFERRED / NOT RUN。

## 最近完成

- [2026-11-14 13:40][claude] 项目备份 `tmp/archive/project-backup-20261114.tar.gz`，SHA256 见 `consensus.md`。
- [2026-11-14 13:44][claude] 提交 `ab80594`：B 工具/文档迁出 `tmp/`、正式指标修复、`--save_top_k`、无效 C0 配置清理。
- [2026-11-14 13:45][claude] 推送 `origin/review/b-mechanism`（后重命名为 `reproduction`）。
- [2026-11-14 11:35][claude] B 代码 v2 smoke 全路径跑通；仅 pipeline smoke。
- [2026-11-14][claude] 旧 C0 150ep 的 N32/N128 结果已整理；数字见 `consensus.md`。
- [2026-11-13][claude] parent 2×2 和 teacher round 索引 bug 修复并重跑；旧 p1 数字作废。

## 共享发现 / 证据

- 旧 C0：s42/s1234 在 N32 接近饱和；N128 的 Graph Value All 明显下降、Strict First-4 接近 0。原始结果 `tmp/analysis/oldC0-resume-150ep/`。
- B smoke：4 张 heldout 图 selection 4/4 common-complete；early 28 条、late 53 条、leaf 16 条、controls 7+7；audit 128 candidates + 692 次替换的 pre-corruption 全 EMPTY。**不是结果**。
- 腐蚀审计初步信号：`last` 策略下 candidate 腐蚀前基本是 EMPTY，而非“已填值被改坏”；仅 1 个 batch，需全量确认。

## 开放问题

- B 是“按父值每轮重算”“检测父子不一致再重算”，还是“优先保留已有槽位”？需要 early/late 时点对照和 sustained/switch 控制。
- 训练腐蚀覆盖的是 `EMPTY→随机值` 还是 `已填值→错误值`？决定 `4→5` 是训练内行为还是跨时机泛化。
- 旧 C0 N128 的 strict 归零主要来自 VALUE 错误、VARIABLE 漂移，还是整图 conjunction 的尾部效应？
- C0′ 是否值得在修正数据/词表后重训，当前没有排期。

## Review 结论

- 暂无。
