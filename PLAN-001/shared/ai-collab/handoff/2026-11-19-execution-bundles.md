# 两位执行agent：直接执行的队列与结果包

日期：2026-11-19 12:22 CST。用户要求进一步实现并推送供执行。
承接[实验设计与验收](2026-11-18-state-reuse.md)，不增加新训练、节点规模或样本范围。
本次把任务拆成可直接运行的队列，并提供统一的结果整理代码，避免再次手写命令或只交无法复核的二次汇总。

## 1. 分工和执行入口

- Agent B：负责AB lane，包含repeat、reuse两实验，两模型各一次运行及各自摘要/结果包。
- Agent C：负责C lane，四版本分别运行，最后共同汇总及生成结果包。
- 旧CPU补交已审阅完成，不再把它当等待项。同步后可以直接进入已授权小验证。
- 共享服务器工作树按原约定串行交接，空闲时pull，运行中不pull。选空闲GPU；下列GPU2仅是命令示例，不表示当前GPU2空闲。

新增入口：`tools/state_reuse/batch.py`、`tools/state_reuse/deliver.py`。
prepare只写计划，不访问GPU；run逐个执行，不并行启动模型。固定源码hash、每任务日志/回执、第一次失败即停止，保留所有输出。

## 2. Agent B命令

在仓库根目录，确认空闲并同步后：

```bash
python tools/state_reuse/batch.py prepare --lane AB --phase smoke \
  --out-dir tmp/analysis/state-reuse-exec-20261119/AB-smoke
python tools/state_reuse/batch.py run \
  --bundle tmp/analysis/state-reuse-exec-20261119/AB-smoke --gpu 2
```

AB smoke有8个串行任务：repeat-A、repeat-B、repeat摘要/结果包、reuse-A、reuse-B、reuse摘要/结果包。模型部分都是前两固定候选、H4。
默认source仍是上一批`reassign-main-A/B`。实际目录不同可在prepare用`--source-a/--source-b`，必须对应同一已审阅来源；main必须传相同source覆盖值。

按原交接核对真实tokenizer、完整起跑编辑、独立真值、死槽位引用、阶段继承、六分支等；写真实检查记录，不写空模板。

```bash
python tools/state_reuse/batch.py accept-smoke \
  --bundle tmp/analysis/state-reuse-exec-20261119/AB-smoke \
  --notes tmp/analysis/state-reuse-exec-20261119/AB-smoke-checks.md
python tools/state_reuse/batch.py prepare --lane AB --phase main \
  --smoke-bundle tmp/analysis/state-reuse-exec-20261119/AB-smoke \
  --out-dir tmp/analysis/state-reuse-exec-20261119/AB-main
python tools/state_reuse/batch.py run \
  --bundle tmp/analysis/state-reuse-exec-20261119/AB-main --gpu 2
```

main仍固定32候选，H8/H16各自独立序列，没有新增样本。

## 3. Agent C命令

```bash
python tools/state_reuse/batch.py prepare --lane C --phase smoke \
  --out-dir tmp/analysis/state-reuse-exec-20261119/C-smoke
python tools/state_reuse/batch.py run \
  --bundle tmp/analysis/state-reuse-exec-20261119/C-smoke --gpu 2
```

C smoke有6个串行任务：四训练版本各前2图/H4、共同摘要、结果包。
按原交接检查10路的实际向量修改、donor不来自未来、输入输出时序、same/noise剂量、连续反馈与共同资格；写检查记录。

```bash
python tools/state_reuse/batch.py accept-smoke \
  --bundle tmp/analysis/state-reuse-exec-20261119/C-smoke \
  --notes tmp/analysis/state-reuse-exec-20261119/C-smoke-checks.md
python tools/state_reuse/batch.py prepare --lane C --phase main \
  --smoke-bundle tmp/analysis/state-reuse-exec-20261119/C-smoke \
  --out-dir tmp/analysis/state-reuse-exec-20261119/C-main
python tools/state_reuse/batch.py run \
  --bundle tmp/analysis/state-reuse-exec-20261119/C-main --gpu 2
```

main四版本各32固定图/H16，仍同源seed2026，不称四种子。

**accept-smoke是执行agent记录已完成协议检查，不是向用户请求第二次批准。**全部自动任务退出正常并不自动证明实验语义正确，所以不会自动生成该记录。小验证零记录只代表覆盖检查，不能据此通过GPU/实验操作验收；按原交接报告，不改候选凑数。

## 4. 断点与失败

- 同一bundle再run，会跳过具有本队列complete回执、且产物hash未变的任务，不重复已完成运行。
- 失败/运行中断的半成品不删除、不覆盖、不偷偷续到内部轮次。检查日志，明确失败点后另起输出目录；若需只补某一任务，可直接用plan内该任务argv换新目录，之后汇总引用真实完成目录并记录变更。
- 重新执行前不改运行中工作树。源码改变时计划拒绝继续；适配修正提交后新建计划，对受影响路径补小验证。不要用修改hash的方式绕过。
- main要求已完成的同lane smoke及其检查记录；不是根据模型分数决定是否通过。模型无收益仍可执行/交付。
- 队列不监控别人的训练、不选择GPU、不停止任何外部进程。

## 5. 自动结果包及报告补充

每个`*-delivery/`包含：

| 文件 | 用途 |
|---|---|
| key-results.csv | AB共同样本H16逐阶段、C各时点/来源lag2的关键10路中5路对照；全体与共同输入资格分列 |
| report.md | 可直接阅读的有限窗口结果入口，不自动写机制结论 |
| examples.jsonl | 每模型、每种观察到的结果类型，最小两个graph_id的完整相关节点输入/输出轨迹 |
| meta.json | 来源、代码、文件hash；验证本结果包从哪份摘要生成 |

例子按结果类别挑选，只作说明，不用于计算总体发生率。C例子固定late/semantic/lag2；其他条件并没有被结果包否定，应查原始摘要全表。AB例子同图多阶段只保留最早匹配阶段，完整序列留records。
smoke H4时AB主表可能无H16行，这是预算筛选结果，不是数据丢失；检查records和smoke摘要。

交付时把结果包小文件复制到对应`docs/records/2026-11-19/`新目录，原始records/隐藏快照/全量CSV留tmp。另附：
1. 真实小验证检查笔记、命令/版本/权重/数据来源；每任务回执的耗时和预算。
2. 扫描/可选/共同资格/实际受损/恢复各分母，零例与缺组；别只交漂亮主表。
3. AB多次更新是否退化、回到原题是否恢复；reuse新版本是否被第二槽消费；外部归档/清空代价。
4. C pre操作在clean中的成功率、u读对q仍错的分层、restore_u是否额外改善、正确节点是否受损。
5. 支持的观察、尚未区分的解释；不要把这批开发集小试称独立最终测试。

新增环境适配/导出代码一起提交；显示名继续A/B/C及C中文训练版本。

## 6. 本地检查范围

新增3项针对性CPU检查：队列范围/CPU参数、不完整或被修改产物的拒绝、摘要条件与分母不混；CLI入口正常。未启动服务器/GPU。
不重复跑旧实验测试矩阵。执行agent负责真实环境检查与执行，回传后GPT审阅源码和结果。
