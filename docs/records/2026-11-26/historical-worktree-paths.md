# 历史路径说明：`management-20261120/`

- 日期：2026-11-26

## 结论

仓库里有若干记录文件仍写着 `management-20261120/...` 的绝对路径。**这些是历史事实，不要改写。**

## 背景

`management-20261120/` 曾是同一 git 仓库的一个 worktree（分支 `b-attribution-review`）。2026-11-25 的收敛操作把它删除了，其 `tmp/analysis/` 下的 13G 分析产物**移动**进主树 `tmp/analysis/`，其余受版本控制的内容仍在 `b-attribution-review` 分支上。同日 `well-ordered/` 与 `tmp/old-968129a/` 两个 worktree 也已删除，`well_ordered_experiments` 成为主树唯一分支。

所以：**这些路径在记录写下时是真实存在的**，只是现在解析不到了。

## 现状

- 30 个已跟踪文件仍含 `management-20261120` 字样（17 个 json + 13 个 md，无脚本）。
- 其中 **29 个被各自的 `files.tsv` 用 SHA-256 钉住**。改写它们会让清单失效。
- 另有一个 `tmp/analysis/` 下的未跟踪文件集（约 56 个）同样保留旧路径，未纳入本说明。

## 为什么不改写

1. **违反文档自己的规则。** `Research Record.md` 1.6 节要求"已有记录原则上不得直接删除或覆盖"、"不得改变其原始含义和发生时的状态"。
2. **这些字段是来源记录。** 例如 `fit-sources/*.json` 的 `path` 记录的是当时那份 checkpoint 的实际位置；改成新路径就不再是当时的事实。
3. **会破坏校验清单。** 29 个文件在 `files.tsv` 里有归档 SHA-256。

因此采用"原地保留 + 本说明"的方式。

## 若要按旧路径复跑

有两种可行做法，都不需要改历史文件：

- 在 `yhy/` 下建一个兼容软链 `management-20261120 -> algorithmic-generalization-transformer-architectures`；但注意当时该 worktree 里是 `b-attribution-review` 分支的内容，与主树并非逐字节相同，软链只适用于读写 `tmp/analysis/` 的场景。
- 或按 `PLAN-001/repository-files.tsv` 的文件去向重定位；该清单给出了基线全部跟踪文件的旧位置、新位置与 SHA-256。

## 相关

- 同批次收敛的核对记录见 `Research Record.md` 的 2026-11-25 各条。
- 另一处旧路径问题（`docs/records/2026-11-25/abc-readable-examples/README.md` 里的 `cd .../yhy/well-ordered`）已于 2026-11-26 直接改正，因为那是操作命令而非历史记录。
