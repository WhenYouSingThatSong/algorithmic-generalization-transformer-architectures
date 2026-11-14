# 共享共识

> 最近更新：2026-11-14 13:56 CST · [claude]
> 只写已确认的事实、定义和用户拍板的决策；不确定的内容放 `board.md` 的开放问题。

## 项目与角色

- 复现论文 *Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning*（arXiv 2510.14095v1）。
- 当前主线：旧 C0 的 150 epoch 训练 + B 机制分析；离散线为 A / A′ / B，连续线为 no-interm C0。
- 用户是最终决策者；Claude Code 负责本地训练/评测/代码，GPT 做外部 review，Kimi Code 可参与并行分析。
- 不修改别人正在运行的任务、进程、GPU 或输出目录；B 机制代码/实验的启动和变更先经用户同意。
- 不把 smoke、单 batch 或 pipeline 检查写成机制结论。

## 仓库与环境

- 仓库：`/workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures`
- 工作分支：`reproduction`，已推送 `origin`；B review 包对应提交 `ab80594`。
- 环境：conda `yhy11.6`，Python 3.12。
- 数据：`experiments/data/`；权重/输出：`tmp/runs/`、`tmp/analysis/`、`tmp/archive/`、`tmp/trash/`（均 gitignored）。
- 旧协作区：`tmp/ai-collab/` 仅作历史归档。

## 命名与数据口径

以 `docs/records/2026-11-14/c0-naming-vocab.md` 为准。

- **C0（旧 C0）**：旧窄分布数据，50k 训练，固定 32 节点、depth ≤ 6；词表 129；seed 42/1234/2025；原 100 epoch，后补到 150 epoch。
- **欠拟合 C0′**：新 MixedN 数据（8/16/24/32 节点、depth 1–12，100k），词表 129，100 epoch；已冷存到 `tmp/archive/newC0-underfit-20261114/`，短期不用。
- **vocab33 错误配置**：独立错误代，变量名只有 x0..x31、词表 `(7,33,2,25,2,2)`；在 `tmp/trash/2026-11-13-c0-superseded/`，不得用于评测或引用。
- 2026-11-13 旧记录曾把新的 2Val MixedN run 称为“C0”；2026-11-14 已更正：C0 只指旧 C0。

## 指标

正式验收使用三个指标：

1. **Value Node Acc**：非 PAD 数值节点的 VALUE token 准确率。
2. **Graph Value All**：每张图所有数值节点 VALUE 都正确。
3. **Strict First-4**：每张图所有非 PAD 位置的 syntax / variable / operation / value 都正确。

后两个是整图 conjunction，非平滑；必须同时报告分子/分母和图数。正式 eval 输出 `formal_metrics_by_size.npy`。旧 `VALUE acc` / `last_var_fully` 可能使用旧口径，只能作 pipeline smoke。

## 已确认结果

- 旧 C0 150ep，旧 N32 测试（e143）：s42 `0.9992 / 0.9873 / 0.9844`，s1234 `0.9990 / 0.9854 / 0.8770`，s2025 VALUE 约 0.48、后两项为 0。
- 旧 C0 150ep，旧 N128 测试（e143）：s42 `0.9912 / 0.8379 / 0.0156`，s1234 `0.9677 / 0.2910 / 0`。
- 以上顺序均为 Value Node Acc / Graph Value All / Strict First-4；原始结果在 `tmp/analysis/oldC0-resume-150ep/`。
- B 机制代码 v2 在 `tools/b_mechanism/`；6 个入口 `py_compile` 和 `python -I ... --help` 通过。当前只做了 4 张 heldout 图 / 1 个训练 batch 的 **pipeline smoke**，不是机制结果。
- B 全量 selection / early / late / controls / audit 尚未运行；local self-consistent 控制为 **DEFERRED / NOT RUN**。

## 边界与未决

- B 全量实验等待外部 review 和用户批准；不擅自启动。
- 局部自洽错误控制暂缓：当前构造可能不自洽，不进入结果；见 `docs/records/2026-11-14/b-local-self-consistent-deferred.md`。
- C0′ 和 vocab33 不活跃；不要与 C0 混表。
- 旧 C0 seed2026 150ep 正在 GPU0 训练；不要停止/重启。
- 旧 C0 的旧 N32/N128 数字是旧数据配方，不能直接与 B 机制的 heldout/matched-depth 样本混用。
- B 主对照：A′ s42 + B-M6 s42；cyhB 三 seed 放第二批。

## 关键路径

- 命名：`docs/records/2026-11-14/c0-naming-vocab.md`
- B 设计：`docs/records/2026-11-14/b-mechanism-design.md`
- B 变更：`docs/records/2026-11-14/b-mechanism-changelog.md`
- B smoke：`docs/records/2026-11-14/b-mechanism-smoke.md`
- 局部自洽暂缓：`docs/records/2026-11-14/b-local-self-consistent-deferred.md`
- B 工具：`tools/b_mechanism/b_mechanism_README.md`
- 历史记录索引：`tmp/records/README.md`
