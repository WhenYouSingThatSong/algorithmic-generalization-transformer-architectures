# 复现仓库：内容、状态与入口

核查：2026-11-19 22:48。实验状态基于远端 `reproduction@f9e10e4` 与已提交报告；未连接服务器运行进程，也未重跑模型。这是仓库结构/交付检查，不是整库逐行正确性认证。

## 从哪里开始

2026-11-20补充：最新实现为[C多时点局部纠错＋过程监督UPDATE](../tools/c_multitime/README.md)，设计/身份分析见[记录](records/2026-11-20/c-identity-and-multitime-design.md)；旧批已交[dev小测](records/2026-11-20/c-update-small-test.md)。下方11月19日盘点数字与状态保留为历史快照，当前执行看板为准。

1. 当前要执行什么：[看板](../ai-collab/board.md)。
2. 最新训练方案/命令：[C纠错与UPDATE](../tools/c_update/README.md)。
3. 实验代码分类：[工具索引](../tools/README.md)。
4. 模型称呼与身份：[A/B/C命名](model-names.md)。
5. 最近完成的共同实验：[ABC工作空间报告](records/2026-11-19/abc-workspace.md)。
6. 协作规则与历史：[约定](../ai-collab/README.md)、[日志](../ai-collab/log.md)。

## 仓库包含什么

| 位置 | 内容 | 当前读法 |
|---|---|---|
| `experiments/` | 原论文模型、tokenizer、图生成、训练/评估、配置、基线和解释性分析 | 包含后续修复；不能笼统称为未经修改的官方代码 |
| `Simtransformer/` | Transformer层、注意力、位置编码等底层组件 | 多个实验共用，不是新实验队列 |
| `tools/` | 11组新增实验工具，训练、干预、汇总和检查入口 | 以各目录README定位具体协议/批次 |
| `docs/records/` | 11月14/15/16/18/19的方案、报告、CSV/JSON、小型证据与复核材料 | 日期/批次必须保留，不能把旧计划当最新状态 |
| `ai-collab/` | 看板、共识、日志、交接、历史交付包 | 当前任务看board顶部；log保留原始时间线 |
| `tmp/records/`、`tmp/scripts/` | Git仍追踪的早期记录和一次性脚本，共12文件 | 历史兼容区，不是当前新代码默认入口 |

统计基准f9e10e4：495个被Git追踪的文件，约65.9 MiB当前文件内容（不含Git历史大小）；其中tools137文件、docs220文件、experiments62文件、ai-collab44文件。32个文件位于各工具的tests目录，不能把文件数当成测试已全部通过。

## 实验推进到哪里

| 工作 | 仓库中已有的交付 | 当前边界 |
|---|---|---|
| 原模型与复现基础 | 图任务生成器、离散/连续训练、评估、配置与解释性代码 | 原始基线目录存在，不代表其中每一种基线都已复现 |
| 离散A/B机制 | early/complete、父子污染、多父节点、重复赋值等代码及阶段结果 | 结论必须注明图规模、种子、起点和受损分母 |
| C能力与机制探索 | 六档能力、漂移、反事实替换、同值来源、联合修复、读出/使用诊断 | 读出恢复不能直接等同下游功能恢复 |
| C早期后训练 | s2026起点的长程保持小试、hold/noise/semantic小试及评估报告 | 多版本来自同一起点，不是多种子证据；不是本批50epoch训练 |
| ABC共同任务 | 已交赋值任务和20窗口长程序结果；A/B各主seed，C三个原始seed | C跨窗口使用离散数值提交，不是纯连续突破128变量或原生526节点图 |
| 最新C纠错/UPDATE | 训练/功能探针/固定评估代码，验收、四次50epoch任务与分工 | 截至核查未收到执行agent的新训练/评估结果提交；服务器是否已经启动不能仅凭Git判断 |

本批负责人：Agent C生成唯一plan、技术小验证、三种子纠错和单种子UPDATE训练及阶段评估；训练完成后Agent B接收同一plan和四份final，正式比较原A/B/C与新权重；GPT设计和审阅。不会新训练A/B，也不自动添加普通续训对照。

## 数据与权重是否也在Git里

不是完整实验资产仓库。训练数据、当前主力权重、大多数原始轨迹/运行输出仍在服务器 `tmp/` 或数据目录；单纯clone通常不足以直接复跑，需要交接中记录的文件及SHA。

已追踪的历史例外：

- `experiments/model_interp/demo_checkpoint/last.ckpt`，约22.4 MiB：原main就已有的解释性示例权重，不是当前C后训练结果。
- `ai-collab/artifacts/c0-capability-20261114-raw.zip`、`c0-capability-s2026-e143-raw.zip`：历史能力结果交付包。
- docs中也有数MiB级CSV/JSON/案例文件，不都是短报告。

`.gitignore`包含tmp/data等规则，但**忽略规则不会自动取消已经被Git追踪的文件**。本次保留这些历史资产及路径，没有删除数据、重写Git历史或移动运行依赖。

## 本次发现并处理的整理问题

1. 原根README只介绍论文，`tools/README`只列B工具，容易找错入口：已补复现入口和完整工具索引。
2. 命名页缺最新两个C版本：已补C（纠错后训练）和C（UPDATE后训练），标清批准范围及尚待结果。
3. B工具README还停在4图smoke、C评估README尾部仍写“尚未运行GPU”：已标明历史说明并链接后续报告，没有改历史实验数据。
4. 各目录README是各批次设计，状态分散。当前任务以board顶部为准；旧记录中的“本批不训练”不限制新批准的训练批次。

## 分支清理记录

现保留：`reproduction`（统一工作分支）、`main`（旧默认分支）、`project_page`（网页内容，含index.html/static/blog_content）。GitHub默认分支仍为main，本次未改默认分支。

用户授权后，删除以下远端及本地分支引用；删除前均核对相对reproduction独有提交为0，提交历史仍可从reproduction访问：

| 删除分支 | 原提交 |
|---|---|
| `repro` | `52dd950dce8d32a8e72513999cac5a310e01ca53` |
| `review/b-mechanism-5137014` | `6823fb6197cecc56c4d9727ac7f550782c4172b4` |
| `review/b-mechanism-e27cc6b` | `dc61d122ea17667b5746ea15a8b04eaa25819b6a` |

不再创建新的审阅分支；后续按协作约定在reproduction交接。
