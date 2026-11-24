# EXPERIMENT-048　正常窗口 C 的正式六项评测

所属计划 [PLAN-004](../../Research%20Record.md#plan-004)。旧台账 48、49、50、51、52、53。

[主文档中的实验设计、结果及当时分析](../../Research%20Record.md#experiment-048)

## 优先查阅

正式六档评测①：未注错求解与长程保持　[c-selected-eval-results.md](analysis/c-selected-eval-results.md) · [01-基础拟合与干净轨迹.csv](results/selected-tables/01-%E5%9F%BA%E7%A1%80%E6%8B%9F%E5%90%88%E4%B8%8E%E5%B9%B2%E5%87%80%E8%BD%A8%E8%BF%B9.csv) · [操作细节](analysis/c-selected-evaluation.md)（[原版第 26 行](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L26)）

正式评测②：第二层是否标出应更新的位置　[c-selected-eval-results.md](analysis/c-selected-eval-results.md) · [02-第二层更新定位.csv](results/selected-tables/02-%E7%AC%AC%E4%BA%8C%E5%B1%82%E6%9B%B4%E6%96%B0%E5%AE%9A%E4%BD%8D.csv) · [操作细节](analysis/c-selected-evaluation.md)（[原版第 27 行](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L27)）

正式评测③：末层实际执行了什么　[c-selected-eval-results.md](analysis/c-selected-eval-results.md) · [03-末层执行与任务后果.csv](results/selected-tables/03-%E6%9C%AB%E5%B1%82%E6%89%A7%E8%A1%8C%E4%B8%8E%E4%BB%BB%E5%8A%A1%E5%90%8E%E6%9E%9C.csv) · [操作细节](analysis/c-selected-evaluation.md)（[原版第 28 行](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L28)）

正式评测④：注错后的目标与下游　[c-selected-eval-results.md](analysis/c-selected-eval-results.md) · [04-恢复与传播.csv](results/selected-tables/04-%E6%81%A2%E5%A4%8D%E4%B8%8E%E4%BC%A0%E6%92%AD.csv) · [操作细节](analysis/c-selected-evaluation.md)（[原版第 29 行](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L29)）

正式评测⑤：同图同时有空位、错值与健康位置　[c-selected-eval-results.md](analysis/c-selected-eval-results.md) · [05-同图混合.csv](results/selected-tables/05-%E5%90%8C%E5%9B%BE%E6%B7%B7%E5%90%88.csv) · [操作细节](analysis/c-selected-evaluation.md)（[原版第 30 行](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L30)）

正式评测⑥：修回后的新计算与冻结接收模型　[c-selected-eval-results.md](analysis/c-selected-eval-results.md) · [06-修回后的使用.csv](results/selected-tables/06-%E4%BF%AE%E5%9B%9E%E5%90%8E%E7%9A%84%E4%BD%BF%E7%94%A8.csv) · [操作细节](analysis/c-selected-evaluation.md)（[原版第 31 行](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L31)）

阅读报告中的历史相对链接，可从原目录打开。[c-selected-eval-results.md](../../docs/records/2026-11-21/c-selected-eval-results.md) · [c-selected-evaluation.md](../../docs/records/2026-11-21/c-selected-evaluation.md)

## 文件

[方案、报告与复核说明](analysis)、[代码快照](code)、[结果与原始轨迹](results)。逐文件来源和校验值见[文件清单](files.tsv)。

代码按基线版本留档。运行所需的模块结构、相对导入和调用方法以旧位置及其交接文档为准。
