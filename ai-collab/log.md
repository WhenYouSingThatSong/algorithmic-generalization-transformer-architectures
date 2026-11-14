# 协作日志（只追加）

> 规则：只追加，不改历史；每行 `- YYYY-MM-DD HH:MM [author] 事件`。时间用本机 CST。
> 2026-11-08 至 2026-11-13 的旧协作区在 `tmp/ai-collab/`（未跟踪历史归档）；本文件从 2026-11-14 起为 canonical。

- 2026-11-14 10:12 [claude] 启动旧 C0 seed2026 的 150 epoch 训练，GPU0，配置 `C0-M4-Nodes32-ADD-L4H16D384-DeBERTa`，输出 `tmp/runs/oldC0-newseed-150ep/`，`--save_top_k 10`。为什么：在旧 C0 同数据/同配方下补一个独立 seed，和 s42/s1234/s2025 对照；未动其他卡。
- 2026-11-14 11:35 [claude] B 代码 v2 的 review 修复和 smoke 完成。为什么：GPT 审阅指出 audit 行号、A′/B 独立筛选、late 控制缺失、switch 起点、manifest 身份等问题。改动：audit 改 `S[:-1]`；入口补 `-I` sys.path；selection 记录 A′/B 共同 early 资格、single-parent + q2、unrelated 排除可达 q；late 加同起点 clean、D1 `new_truth`、D2 全预算、D3 固定新常数；controls 改为 held-state switch 并配 `sustained_r`；manifest 加 data/checkpoint/code 哈希；新增 summary 的 N0/N1/N2。验证：4 张 heldout 图 / 1 个训练 batch 的 pipeline smoke；**不是结果**。详见 `docs/records/2026-11-14/b-mechanism-changelog.md` 和 `docs/records/2026-11-14/b-mechanism-smoke.md`。
- 2026-11-14 13:40 [claude] 完成项目备份 `tmp/archive/project-backup-20261114.tar.gz`（2.3G），排除 data、wandb、checkpoint 等；SHA256 写入 `consensus.md`。
- 2026-11-14 13:44 [claude] 提交 `ab80594`。为什么：把 B 工具从被忽略的 `tmp/` 迁到可跟踪的 `tools/`，并修复正式指标口径，供外部 review。改动：新增 `tools/b_mechanism/`；新增 `build_factored_tokenizer`；`metric_utils` 改为 PAD 感知并输出 Value Node Acc / Graph Value All / Strict First-4；eval 入口保存 `formal_metrics_by_size.npy`；`train.py` 加 `--save_top_k`；删除无效 C0 MixedN 配置；更新 p1 指标说明。验证：相关文件 `py_compile` 通过，B 入口 `python -I ... --help` 通过。
- 2026-11-14 13:45 [claude] 推送 `origin/review/b-mechanism`；相对 `origin/repro` 领先 11 个提交。供 GPT 从看板接 review 任务。
- 2026-11-14 13:56 [claude] 按用户要求精简 `ai-collab/`：只保留 `README.md`、`consensus.md`、`board.md`、`log.md` 和可选 `handoff/`；移除 mailbox 和 per-agent README。看板新增“需要 review（GPT）”任务，写明审查对象、重点和回复方式。
- 2026-11-14 14:00 [claude] 按用户要求把推送分支从 `review/b-mechanism` 重命名为 `reproduction`；新分支从 `1443211` 起，随后提交本目录的引用更新；旧远端分支随后删除。B review 包仍对应 `ab80594`。
