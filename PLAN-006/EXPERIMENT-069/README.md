# EXPERIMENT-069　多子节点传播的跨模型重放

所属计划 [PLAN-006](../../Research%20Record.md#plan-006)。旧台账 74。

[主文档中的实验设计、结果及当时分析](../../Research%20Record.md#experiment-069)

## 优先查阅

[fanout-counts.csv](results/first-batch-b-trained/tables/fanout-counts.csv) · [verification.md](analysis/first-batch-c/verification.md) · [操作细节](analysis/first-batch.md)（[原版第 12 行](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch.md#L12)）

阅读报告中的历史相对链接，可从原目录打开。[verification.md](../../docs/records/2026-11-23/first-batch-c/verification.md) · [first-batch.md](../../docs/records/2026-11-23/first-batch.md)

## 文件

[方案、报告与复核说明](analysis)、[代码快照](code)、[结果与原始轨迹](results)。逐文件来源和校验值见[文件清单](files.tsv)。

后训练模型的分项共用一个压缩包。[归档及其校验文件所在目录](../EXPERIMENT-066/results/first-batch-b-trained)。

C 侧原始文件按 natural、history、fanout 分入各自实验。[共同校验清单](../EXPERIMENT-067/results/first-batch-c/raw-manifest.sha256)。

代码按基线版本留档。运行所需的模块结构、相对导入和调用方法以旧位置及其交接文档为准。
