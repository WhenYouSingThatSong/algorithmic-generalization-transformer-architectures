# 工作看板

> 最近更新：2026-11-22 14:13 CST · [gpt]
> 活看板：任务可以移动；状态变化在 `log.md` 记一行。条目格式 `[YYYY-MM-DD HH:MM][author]`。

## 当前首项 · 未经纠错后训练的原始 C 如何恢复

- [ ] [2026-11-22 14:13][gpt → Agent C] 按[权重、数据、执行与验收](../docs/records/2026-11-22/original-c-recovery.md)运行新入口。先单图，再做两个原始 C 和 A′、B 的同题比较。实际写入、下游曾使用错值、传播后持续恢复分别报分母。按训练日志和同题拟合分层，人工错与自然错分别交付。
- [ ] [2026-11-22 14:13][gpt → Agent B] C 交付后，复核共同资格、零覆盖、同值替换有无实际改变、父值干预与释放的逐轮轨迹。正式新图结果待服务器执行。

## 当前交接 · 同题机制比较与计算空间复用

- [2026-11-22 11:38][claude] **B 复核完成（CPU，未重跑）**：4 个 stage 的 delivery/table/provenance hash 与行数全部匹配；四条件主表 A 0、B 32、C_s42 28/29/25、C_s42-correction 32/32/31 可复算，target 实际写入均为 32/32。补齐下游 `specific/damage/stable` 分列和 s1234/update 同事件计数：B just_solved 目标 32/32 但 specific/damage 10/32；C_s42 later 25/32 但 12/32；s1234 为 17/17/15，correction 32/31/22，update 28/28/19。父值屏蔽阳性对照与 workspace extra=0 表也复算一致。**发现两项报告缺口**：`natural_replay` 行数为 0，C 报告未显式写零覆盖（natural 新错仅 C_s1234 14、C_s1234-update 15）；四条件报告未并列下游分母和 s1234 分种子。复核/表 `docs/records/2026-11-22/attribution-b-review.md` + `attribution-b-review/`；未改 C 产物、未用 GPU。
- [2026-11-22 13:08][claude] C 已按 B 复核补记 C 报告（不改原始表/权重）：`c-attribution-delivery.md` 增加四条件 `specific/damage/stable` 分列（含 s1234/correction/update）、明确 `four-way-common` 192 行全为 common=1，并显式写 `natural_replay=0` 零覆盖（natural 新错仅 C_s1234 14、C_s1234-update 15）。未新增 GPU/训练；B 的小表保持原引用。
- [x] [2026-11-22 00:40][Agent C] 按[执行顺序、入口与分母](../docs/records/2026-11-22/state-repair-attribution.md)在独立工作树串行接续。最新安排暂缓方案二。已在运行的工作按原计划交付；尚未启动旧队列时，用新入口的 `--include-prior-mechanisms` 直接接上六项既有观察评测。随后依次运行同题四条件、父值屏蔽、恢复来源、变量复用。再交架构原型小测和跨任务参考评分。复用已有权重。**C 完成（2026-11-22 01:43）**：完整队列 exit 0；六项既有观察 + 四新阶段 + 架构两步 smoke + 跨任务参考检查均完成；交付 `docs/records/2026-11-22/attribution/`、`c-attribution-delivery.md`、`attribution/provenance.json`。四条件首轮修回：A 0/32，B 32/32，C_s42 28/29/25，C_s42-correction 32/32/31（pending/just_solved/later）；父值屏蔽仅父值受损、无关位置 0；计算空间复用见交付表。
- [2026-11-22 01:30][claude] C 在 GPU6 串行执行新入口：`attribution_queue.py --include-prior-mechanisms`。smoke 全过（prior transitions/signals/identity/fanout + reader versions/readers + 四新阶段，八份权重，四个 stage delivery 均生成）；formal 已启动，日志 `tmp/analysis/c-attribution-formal.log`，输出 `tmp/analysis/c-attribution-v1/`。旧停止小试未启动，按新入口跳过；`c-update-20261119` 已以只读 symlink 接到管理数据根。B 复核待交付后开始。
- [x] [2026-11-22 00:40][Agent B] 复核同图四格的资格交集、实际写入数和下游曾受损且可跟踪的分母、自然错值与回放对照、屏蔽父值阳性对照、复用题的最近定义。两份连续种子分别报告，同题基础拟合与训练曲线随表提交。**完成，见 11:38 条；natural_replay 零覆盖和下游/s1234 分列作为待补项记录。**

## 前次交接 · 停止更新与四组后续机制评测（串行）

- [ ] [2026-11-22 00:09][Agent C] 上述队列按原顺序执行；其 fanout 正式交付后，继续[第六、七项及服务器命令](../docs/records/2026-11-22/c-reader-continuation.md)。独立入口 reader_queue.py 串行运行三次同名赋值的版本追踪，再运行上游数值回正后的历史／新增读者配对。种子 42、1234。每项先小测再正式；单独输出。Agent B 复核旧版响应、共同正确对照和未入选原因。

- [ ] [2026-11-21 23:59][Agent C] 唯一执行者按[串行队列、命令与验收](../docs/records/2026-11-21/c-research-sequence.md)接续。顺序为已有停止更新小试 → 已存轨迹的数值与身份变化 → 置信与第二层更新预测 → 身份干预后的两个新读者 → 错值分支传播与汇合。两个种子分表，每个阶段小测、正式、交付后顺次推进。若上一小试正运行，先让其结束，随后队列复用它的完成回执；未启动可直接用总入口。每个阶段交 delivery、两张 CSV；传播另交三档共同合格图配对表。按下方停止小试原约定交其四表。
- [ ] [Agent B] 随阶段产物复核计数和共同分母，特别是四种身份供体是否同题合格、1/2/4 条路径对照是否同题受理。无需再启动队列。执行进度和产物身份更新 board/log。

## 当前交接 · 现有 C 权重的停止更新小试

- [ ] [2026-11-21 23:41][Agent C] 按[结果复核、运行命令与验收](../docs/records/2026-11-21/c-stop-pilot.md)评测种子 42、1234。新题号，同图比较全图更新、整图停止和逐定义位置停止，另测冻结位置受扰时的代价。先小测再执行两份正式权重，各 272 道基础题，独立输出目录。交基础拟合、准确率和逻辑预算、下游消费者、传播四张表。Agent B 可复核配对分母。继续使用已有权重。

## 当前交接 · 现有连续纠错权重的难图评测

- [x] [2026-11-21 22:30][Agent B] 在独立工作树按[难图执行与验收](../docs/records/2026-11-21/c-hard-graphs.md)评测正常窗口后训练 C 的 42、1234 两份末周期权重。每种子 544 道基础题，覆盖长短分支、等工作量配对、同名复用和 166 次定义压力档；上游替换有同起点干净对照。运行两种子小测和正式评测，交五表、有效分母、权重及产物身份。Agent C 已交原六档的六表，可直接引用。两任务独立目录、单写入者，无新训练。**C 代执行完成（2026-11-21 22:48）**：smoke 两 seed 通过；formal 各 544 题/1,600 输入/256 次干预尝试，五表在 `docs/records/2026-11-21/hard-graphs/`，身份/哈希在 `hard-graphs-provenance.json`，汇总见 `c-hard-graphs-results.md`。
- [2026-11-21 22:48][claude] 难图正式评测两 seed 完成（GPU0/1，exit 0）：各 544 题、1,600 输入、256 次干预尝试。独立命名标准轮数整图 512/512；复用两臂明显下降（reuse4 数值 0.832/0.760、reuse2 0.663/0.594，按 544 图）。上游替换首轮目标修回 C_s42 232/243、C_s1234 189/252。五表在 `docs/records/2026-11-21/hard-graphs/`，身份/哈希在 `hard-graphs-provenance.json`，汇总 `c-hard-graphs-results.md`。B 请勿重复启动同一目录。

## 当前交接 · 正常窗口后训练 C 正式评测仅选 42 和 1234

- [x] [Agent B] 六表生成已实现。按[执行文档末节](../docs/records/2026-11-21/c-selected-evaluation.md)运行 `report`，提交六份 CSV 及来源身份。旧版重点分档缺新字段时用 `focused` 单独补采该档，已完成的标准评测直接复用。CPU 合成轨迹已验证交叉计数、四臂分母与旧产物补充路径，实测数以服务器输出为准。**C 代执行完成（2026-11-21 22:28）**：两 seed assess 均 complete，六档 `selected_detail` 全覆盖，无需 focused 补采；六份 CSV 在 `docs/records/2026-11-21/selected-tables/`，来源身份/哈希在 `selected-eval-provenance.json`，汇总见 `c-selected-eval-results.md`。

- [x] [2026-11-21 21:23][Agent B] 本批范围取代下方历史三种子正式评估安排。盘点现有结果，按[两种子执行与六表交付](../docs/records/2026-11-21/c-selected-evaluation.md)领取 42、1234。保留 2026 拟合记录。交各模型各档分母、单图混合及局部／全局成功分表。Agent C 本轮无需新增训练。**C 代执行完成（2026-11-21 22:28）**：只评 42、1234，各档分母在六表内；C_s2026 未进入本批六表。
- [2026-11-21 21:38][claude] 按用户指示，C 已接管本轮两 seed 的 assess：C_s42 在 GPU0、C_s1234 在 GPU1，输出 `tmp/analysis/c-local-update-normal-v1/eval/{C_s42,C_s1234}`；完成后由 C 运行 `combine` 和 `report`，交六份 CSV、来源身份和分母。B 请勿重复启动同一两目录；如已有部分产物请告知路径，优先合并。GPU2/3 仍为外部占用。
- [ ] [研究储备] [24 项研究问题和方案二](../docs/records/2026-11-21/research-checklist.md)。方案二待讨论，另行安排。

## 当前交接 · 报告用语与拟合证据

- [2026-11-21 11:12][user/gpt] GPT、Agent B、Agent C 的正文、表头和口头汇报统一按[团队用语与拟合规则](../docs/model-names.md)。依赖深度、末端角色、训练周期、推理循环轮数和末次观测须写全。新规则适用于两位 agent 的下一次交付。
- [x] [Agent C] 随下一次回传确认采用上述规则。优先从三份正常窗口后训练已有日志导出逐周期训练损失、末周期均值、末批损失、各检查点同图基础验证成绩。仓库根目录执行 python tools/c_local_update/checks/fit_history.py --run <单种子训练目录> --output <该种子独立JSON>，三份结果分别附回传。其他配方注明目标及平均方式，缺项标缺；共同验证损失有则列明。无需重训。**已补交（2026-11-21 11:21）**：三份 `fit_history.py` JSON 在 `docs/records/2026-11-21/fit-history/`（C_s42 `ea9f820e…`、C_s1234 `89170f7c…`、C_s2026 `123617ba…`）；末周期加权损失 0.1941 / 0.2906 / 0.5993，最后批次 0.1252 / 0.2111 / 0.5274；C_s2026 最低在周期 6（0.3226），末周期不是最低。逐图重算的再运行16轮节点四字段联合计数为 3,066 / 3,647 / 3,195（分母 4,096），旧概括“0.98–0.99”已按实际分母更正。汇总与边界见 `c-local-update-fit-history.md`。
- [ ] [Agent B] 随下一次回传确认采用上述规则。正式 50 周期结果照原协议报告。讨论机制时并列同种子、相同基础拟合水平的补充比较，先用验证数据登记匹配指标、容差和并列选择规则；没有可比权重就说明。深度和角色分别给出组内错数/节点数，检查交集和干净起点，勿从单图或单个不充分拟合的权重推广。

## 当前优先 · 再赋值、正常窗口 C 三种子、分类评测

- [ ] [2026-11-20 23:11][user/gpt] **本批按顺序交接**。① B 交原始 A/B/C 再赋值小测 → ② C 固定清单，在 8 张 dev 图跑 s2026 初始化版最小评测（节点 VALUE/四字段、UPDATE 分类、下游恢复）及最大深度反传 smoke → ③ C 并行启动原 C s42/s1234/s2026 正常 D+1 窗口后训练，各 50 epoch；每种子 e1 交训练后首次 dev，继续训练 → ④ B 收 e50 做正式分层评估、三种子汇总。每步由负责人回填状态/路径/样本分母。同版本已完成项复用。命令见下方交接；新入口 tools/c_local_update/checks/minimal_dev.py。
- [2026-11-20 23:03][user/gpt] [执行清单、数据规则、命令和验收](handoff/2026-11-20-inline-rebinding-and-c-local.md)。此项批准三份原始 C 的新配方，各 50 epoch；此前只做 s2026 的长窗口配方另留记录。新版只在正常 D+1 次前向内训练，长程保持只评估。
- [ ] [Agent B] 首先在原始 A_s42/B_s42/C_s42、s1234、s2026 完成一次输入、顺序绑定的再赋值小测。228 对匹配图、4 张 139 次定义的单臂图。按复用版本、消费者、图深、宽和控制题表现报告，保存实际定义/使用索引和缺失对照。重点展示同名被多次复用、保存的旧计算能否继续使用，以及真实随机 DAG 中只在最后一次使用之后回收名字。
- [ ] [Agent C] 冻结 dev/六测试分层与代码，最大深度 smoke 后在空闲卡分别从三个原始 C 启动训练；保留旧队列，避免重复写入。交 dev 曲线、真实剂量、正式 final/hash。
- [ ] [Agent B] 按已冻结的方案对三份新 final 分类评估；训练期间可用 dev 验证入口。单列新算、局部冲突、健康、间接错，报告同图混合和真实完成、三种子分表、自然漂移和功能使用的联合结果。CPU 汇总代码已交付。看板更新执行状态后由 GPT 复核。
- [2026-11-21 01:48][claude] **C 侧进度**：B 五模型再赋值产物已落盘；C 8 图 minimal dev 与最大深度三模式 smoke 通过，三种子已出 e1 dev，训练在 GPU1/2/3（约 epoch 14/50）。旧长窗口 multitime C_s2026 已完成 e50，final `c4c8b3c8…`，但 dev clean 明显退化（D 整图 0.5625、D+16 0.4375；best-dev 仍是 e1 1.0）；证据 `docs/records/2026-11-21/c-multitime-final.md`/`.json`，提交 `ccf383c`。GPU5 已空出；旧结果与新三种子分开，正式评测仍归 B。
- [2026-11-21 05:25][claude] **C 三种子 50 epoch 全部完成**（exit 0，00:29–05:18）：C_s42 final `f14f4bbf…`，C_s1234 `693b6338…`，C_s2026 `53ba33c6…`；meta 全 complete、78,200 步、manifest `a91fd53e…`。dev-50 clean：C_s42 / C_s1234 D、D+4 均 1.0，C_s2026 D 0.5547（节点 mean 0.9451）保留该 seed；D+16 三种子 VALUE 0.99/0.98/0.59、四字段近 0（VARIABLE 漂移，仅观察）。实际剂量、曲线、final/dev/meta 哈希：`docs/records/2026-11-21/c-local-update-finals.md`/`.json`，提交 `d0001be`。GPU1/2/3 与 GPU5 已空；正式分类评估归 B。

## 当前：旧批评估交付 + 新多时点局部纠错/过程监督UPDATE

- [2026-11-20 19:05][claude] **C multitime smoke 已通过，训练在 GPU5 运行**：先 GPU3 因外部 74.50 GiB 占用 OOM，换空闲 GPU4 以同一 micro32/donor32 通过；六步 max-depth 验收、GPU 标签、forward、两轮 BPTT、UPDATE 梯度、reload/dev2/冻结原 C probe 全部通过。smoke 分模式：frontier 496 accepted/483 changed、middle 241/241（237 个注入前已正确，证明无接收正确性过滤）、late_budget 528/528；late smoke batch 全部已完成，不能外推到主批。证据/身份/耗时/显存：`docs/records/2026-11-20/c-multitime-smoke.md` + `.json`/acceptance/audit/timings（本提交）。正式 s2026 训练在 GPU5，当前约 epoch4/50、step5,959；peak_allocated 45.8GB（交接转述的~73GB未在本次分配峰值重现），约 7–8h ETA；GPU5 独占，GPU4 B 的 A 评估继续，GPU6/7 留给 B。

- [2026-11-20 18:35][user/gpt] **用户批准隔离工作树并行，要求将全面评估分工推送。** [执行要求](handoff/2026-11-20-parallel-evaluation.md)。此条更新旧“所有任务一律串行等待”的限制；运行树不pull/改源码、输入只读、输出唯一写入者仍有效。用户转述新smoke已过、GPU5新训练已启动、GPU4旧A评估已开始；服务器receipt尚待提交，不重复smoke、不把口头汇报当已核查产物。
- [ ] [Agent B，评估调度唯一负责人] 保留当前A任务；先排除原execute总队列自动重复领取，再用单模型evaluate分三条互斥队列：GPU4 A/B；GPU6 s2026原始/correction/update；GPU7 s42原始/correction、s1234原始/correction。同plan、标准模型输出目录，已完成项不重跑；全部完成后单进程compare。不得同时启动三份完整execute。卡号启动前复核，具体安全交接见执行要求。
- [ ] [Agent B] 优先交同seed s2026三版对照、普通纠错三个种子对照；基础能力/纠错/定位执行/修复后使用四组结果，早晚/联合/持续/N32与N128分层。用已有dev先做CPU受扰汇总，不等九份test全完成；源通过/受损/恢复分母和VALUE/first4分开，完整清单见交接。
- [ ] [Agent C] GPU5继续当前s2026新训练，不改batch/配方、不在同卡加任务；补交已完成smoke的receipt和代码身份（含外部占用OOM记录），不重复跑。分模式报告有效污染与EMPTY/错误/正确起点、定位/执行、显存、训练/保存/dev耗时，特别检查较晚污染是否大量源拒绝；低通过如实报告，不暗换规则。首epoch后更新ETA。
- [ ] [两agent] 独立固定代码树/输出运行，读取本次文档不更新运行树；在管理工作树更新board/log并提交小表/证据身份。新版本仍只s2026，待阶段结果再讨论扩种子；本次不新增训练组或变量名干预。

- [2026-11-20 17:51][claude] **旧 c_update 四份 final 完成并交接 Agent B**：`C_s2026 correction`、`C_s2026 update`、`C_s42 correction`、`C_s1234 correction` 均 50 epoch 完成；plan `tmp/analysis/c-update-20261119/data/plan.json` SHA `b3c4b1be…`，run 根目录 `tmp/analysis/c-update-20261119/run`。final SHA：s2026-correction `e78c8ac7…`、s2026-update `c6ba26a0…`、s42-correction `88430930…`、s1234-correction `b9ad22aa…`。四份 final/meta/audit/dev-50 的完整 hash、实际 dose 和 B 的精确 evaluate 命令见 `docs/records/2026-11-20/c-update-training-handoff.md`（提交 `40da6df`）；C_s2026 update dev 小测另见 `96bdde6`。B 现在可以按同一 plan 跑五个原权重+四 final 的正式 evaluate；C 尚未启动新 multitime smoke。未重跑旧训练、未改旧 c_update 源码。

- [2026-11-20 13:57][user/gpt] 用户确认不按接收模型先正确筛选注错，要求记录命名/身份漂移分析并实现优化版。[设计记录](../docs/records/2026-11-20/c-identity-and-multitime-design.md)、[代码与验收](../tools/c_multitime/README.md)、[交接](handoff/2026-11-20-c-multitime.md)。统一新名**C（多时点局部纠错＋过程监督UPDATE）**；先原C s2026/e143、50epoch上限，旧任务不迁移、不打断。新入口独立，旧c_update源码未改。
- [x] [gpt] 新核心：三时点结构候选、无接收正确性过滤、同步局部GPU标签、第二层定位/末层正确性与对齐、全BPTT、默认micro32/供体32、训练/保存/dev分项计时、严格身份断点、同plan评估与三种同seed C对比。普通teacher标签另提供向量化等价实现，未切换旧训练。命名和历史别名已记录。
- [x] [Agent C] 先交接/保留旧四次训练与阶段评估，不在运行树pull；空闲后按新README执行六步最大深度smoke和verify。交GPU标签/loss梯度等价、两轮BPTT、加载、源质量与实际注入、显存和耗时；技术检查不是能力结果。默认32，OOM才以新目录测16/8并记录；不静默改精度/损失/概率。**smoke完成，见 19:05 条；GPU3 OOM为外部占用，GPU4重试通过。**
- [ ] [Agent C] 验收后按单seed50epoch范围执行；原50k/原e143/旧plan不变，main不从smoke或旧UPDATE final续训。新模式等概率、5%batch×50%结构候选，late固定D+4不保证完成；注错前状态只作记录。交每模式实际剂量/成本/训练曲线、dev、final/hash；未批准自动扩种子或新增消融。
- [ ] [Agent B] **旧批任务保留优先交付**：从已有dev JSON/JSONL先补普通C前沿污染纠错的目标恢复、下游损伤、再错及UPDATE受扰条件定位表；提交小表/原始汇总JSON与来源hash，不能只报clean准确率。完成旧五原权重+四final正式评估，不重复已有阶段GPU评估。
- [ ] [Agent B] 后续接收新final，用同plan的evaluate_new和compare_new比较s2026原C/前沿污染纠错/前沿污染＋过程监督UPDATE；VALUE/first4分开，clean/受扰定位分开，共同资格只用于评估分层而非训练筛选。固定预算与实际完成区别明确。
- [ ] [gpt] 审阅服务器适配/验收与结果，再讨论变量身份信息是否丢失、UPDATE收益及是否扩种子。变量名干预、只读题面结构本轮仅分析待办，不新增训练任务。

## 上一批授权与历史状态：C纠错三个种子 + 中间UPDATE一个种子，各50epoch

- [2026-11-19 22:50][gpt] 用户授权清理历史分支：远端/本地`repro`和两个`review/b-mechanism-*`已删除，删除前独有提交均为0；`main`/`project_page`保留，统一工作分支不变。新增[仓库内容导航](../docs/repository-map.md)，补根入口、全部工具索引和最新C版本命名，更正两处README历史状态提示。检查到远端f9e10e4，无执行agent新结果提交；不据此推断服务器是否已启动。仅整理文档，未改运行代码、数据或权重。

- [2026-11-19 22:20][user/gpt] 用户批准具体实现；新增要求“源向量读成错值还不够，要在未做纠错后训练的原C里检查是否真的让下游用错”。[执行交接](handoff/2026-11-19-c-correction-update.md)、[方案/代码/命令](../tools/c_update/README.md)。范围：原50k训练图，C纠错s42/s1234/s2026 + C UPDATE先s2026，各50epoch；不加普通续训对照，不重训A/B。此项更新下方历史“第四步训练暂缓”的限制，其余方向不自动扩展。
- [x] [gpt] 新目录`tools/c_update/`：连续污染、原B剂量核对、第二层UPDATE读出、同步标签和正确性/对齐联合loss、固定评估、未做纠错训练原C接收功能检验、串行队列和共同资格汇总。9项stdlib针对性测试通过；torch/真实权重验证留执行agent。尚未在服务器运行，不能把实现完成当训练结果。
- [2026-11-19 22:26][user/gpt] **两agent分工已定：Agent C负责训练阶段；Agent B负责训练交接后的全部模型正式评估和结果整理；GPT负责设计与审阅。** Agent名称不是模型范围：Agent B本轮评估原A/B/C及四份新C。共享服务器仍串行交接，任一时刻只有一位队列负责人；C训练期间B不改运行树、不启动另一套评估、不重生成数据。
- [x] [执行agent C，训练阶段队列负责人] 空闲交接同步reproduction，生成唯一plan，按README prepare→smoke。核对新可算位置/源年龄、teacher目标、无硬gate、BPTT与对齐梯度、新下游保持旧hidden。smoke和main分目录，main仍从原e143开始。
- [x] [执行agent C] 技术smoke通过后执行四次50epoch；原C功能探针64dev图先做，每检查点再做。epoch1/3/5/10/15/20…50小评测，性能一般仍继续到50，代码/数值故障停下记录。不得临时改源筛选或挑传播成功样本训练。
- [x] [执行agent C] 训练阶段交付四份final路径/SHA、唯一plan路径/SHA、运行提交及代码hash、训练曲线、实际污染剂量、自动阶段评估和原C功能探针、异常/恢复记录。完成后确认队列退出，在board/log写明交给Agent B；不重复启动正式evaluate。**完成，见 17:51 条。**
- [ ] [执行agent B，交接后评估阶段队列负责人，已可开始] 接收C的同一plan和四份final，核对交接身份后运行README的`execute.py --phase evaluate`：五个原权重+四份final共同正式评估。整理节点VALUE/first4、更新定位与执行、各分母、共同资格端点、原C接收的特异错误传播与同值对照；给统一比较表和具体轨迹案例。读取现有阶段小评测，不重复运行训练中的自动评估。raw/权重留tmp；报告和小表进docs/records；更新board/log。发现影响实验含义的问题先交GPT讨论，不自行换协议。
- [ ] [gpt] 审阅服务器适配、小验证、功能探针和训练结果；随后与用户讨论UPDATE是否扩到另两个种子。当前不先假定注错有效或纠错获益。

## 已交付历史：ABC共同任务、复杂长计算、边界诊断（前三步）

- [2026-11-19 20:13][gpt] **ABC main 审阅至6bf760e**：[审阅记录](handoff/2026-11-19-abc-workspace-review.md)。本批前三步已完成；11产物/9源码hash匹配，重聚合程序600条与共同24图阶段结果。C program carry/reset/scratch最终成功s2026=1/17/36、s42=0/7/33、s1234=0/2/29（各/40）。报告更正s1234 scratch首阶段23/24、主分支后续输入可能不同；补入不改题s4仍正确15/19/9与改题4/7/4、同输入probe条件正确率。未重放服务器权重/records，无需GPU重跑；第四步仍不启动。
- [ ] [执行agent C] 空闲交接仅补交既有 `verify_c_smoke.py` main适配版和 `check_hidden_rows.py` 源码/说明，供审阅现有核验依据；不新增GPU、小验证矩阵或训练。B本批补交关闭。

- [2026-11-19 19:55][claude] **B verifier CPU 补交完成**：按 17:52 审阅要求，提交 `tools/abc_workspace/checks/verify_ab_records.py` 和小结果 `docs/records/2026-11-19/ab-workspace/ab-verifier.json`（提交 `8ce841e`）。脚本只读 records/manifest，不加载模型/不跑 GPU；在主批上 0 errors，核对 9,080 条记录、290,560 个节点真值、4,800 条 reference 记录、5,400 条 conditional 记录、600 个 probe。代码和报告明确分开 reference 程序真值与 actual-input conditional 真值，两者不要求一致；同时补充 terminal 宽度、520 执行/结构祖先操作措辞和 A probe 的 assignment 188/0、program 228/25 分列。AB 结果未重跑，C 结果未改。
- [2026-11-19 19:17][claude] **C workspace smoke+main 完成，ABC 合并表已生成**：使用 B 生成的同一 `tasks.json` b13f87df…（80 任务）/`models.json` 73e99dcb…（5 权重）；C smoke 7/7（660 条，独立检查 0 错，GPU0），C main 7/7（GPU0）。C main 13,520 条：三种子 assignment 各 1,940；program 2,575/2,585/2,540（carry 非数值提交使诊断减少，失败/无效记录保留分母）。独立 verifier 重建 13,520 条，0 errors。common24：C_s2026/C_s42/C_s1234 的 assignment scratch 四阶段均 24/24；carry 从 23/13/19 降到 4/7/4，reset_results 从 6/9/11 降到 0/0/0；all80 回原题 scratch 80/80/76，carry 11/11/9，reset 0/1/0。20 窗口 526 逻辑节点/520 祖先操作 program 最终成功：C_s2026 scratch 36/40、reset 17/40、carry 1/40；C_s42 33/7/0；C_s1234 29/2/0；carry invalid records 198/90/278，final-stage nonnumeric 1/1/0。AB 结果见 17:31 条；合并 ABC 小表/报告 `docs/records/2026-11-19/abc-workspace.md` + `abc-workspace/`，提交 `399038e`。未训练、未扩图/种子、未做第四步；C 三 seed 分列，不作机制或通用容量结论。

- [2026-11-19 17:52][gpt] **AB交付审阅至d92915a，C待交**：[审阅](handoff/2026-11-19-ab-workspace-review.md)。运行代码未改，15源码/config hash与6CSV源hash一致，1460组及240程序首次失败/最终成功交叉核对通过，未重放原始records。A/B carry 1/40、39/40；外部reset/scratch均40/40。B唯一失败窗口11，12–20 conditional全对但原程序仍错。纠正terminal随w=2/4/6变化，单父链属复制控制。**Agent B仅CPU补交既有verify_ab_smoke.py及小检查结果，区分reference与actual-input核验；无需GPU重跑。Agent C继续同清单三种子既有任务，无新批准等待；仓库当前尚无本轮C结果。**不增AB矩阵或第四步训练。
- [2026-11-19 17:31][claude] **AB workspace 主批完成**：共享 80 任务清单，GPU7 串行 5/5；两模型 assignment 各 1,940 条、program 各 2,600 条，共 9,080 条记录，独立复核 0 处不一致。A/B 原题基线均 80/80 四属性合格；assignment 的 A carry 终轮 terminal macro 0.220/0.134/0.138，回到原题为 1.000 但 node 0.989；B carry 为 1.000/1.000/0.988/0.997；reset_results/scratch 两模型四个阶段均 1.000。20 窗口 program 最终成功：A carry 1/40、B carry 39/40，reset_results/scratch 均 40/40；A carry 39 个首次失败在第 2 窗口，B 的 1 个在第 11 窗口；无自然 invalid commit。固定 probe 中 A 的 reset/scratch/refresh-and-reset 各修回 416、改坏 25 个节点-事件，refresh-task 为 0/0；B 各 probe 均 0/0。报告 `docs/records/2026-11-19/ab-workspace.md` + 小表和检查，提交 `839c93b`；队列实际 HEAD 为 `3c360ec`（C 文档/导出器提交），AB 源码 hash 未变。未训练、未改结构；C lane 未运行，不作机制结论。
- [2026-11-19 16:01][user/gpt] 用户批准前三步实施，第四步训练/新结构暂缓。[交接](handoff/2026-11-19-abc-workspace.md)和[代码/命令](../tools/abc_workspace/README.md)已完成：共同新任务、ABC合法重新赋值、固定32位置长程序、固定阶段诊断、串行队列与共同汇总。范围更新此前“本批关闭、不新增GPU”的限制。A/B沿用s42主权重；C为s42/s1234/s2026三个原始e143权重。赋值每形状16图，程序每形状8图、20窗口；5形状均保留失败，无补样。**长程序有模型实际数值的离散提交边界，C标为混合接口，不声称始终连续或原生526节点整图能力。**只有题面更新任务保留C其余hidden、不全局离散重嵌入。
- [x] [gpt] 新目录`tools/abc_workspace/`实现任务生成、真实tokenizer接口、连续/离散运行、无真值回填的跨窗口提交、独立原程序/actual-input评分、固定成功失败共同诊断、最终/末两轮分列、失败分母、配对修回/改坏、共同起点汇总。6项针对性CPU检查及CLI/编译通过；本地未加载权重/跑GPU。
- [x] [执行agent B] 空闲交接同步后生成唯一共享tasks/models清单，按README运行AB smoke，核对交接的实际写入/独立真值/错误提交/分叉预算，然后自行进入AB main。完成交结果、代码适配和验收记录；不训练、不修改运行中的共享树。**完成，见 17:31 条。**
- [x] [执行agent B] 按 17:52 审阅提交独立复核脚本和小型结果，分别说明 reference 程序与 actual-input conditional 的核验方式；只做 CPU 补交，不重跑 GPU。**完成，见 19:55 条。**
- [x] [执行agent C] 使用B生成的完全相同清单；共享树串行交接后运行C三个种子smoke，重点检查局部题面embedding写入、其他hidden不变、C整行reset与离散VALUE reset区别、程序离散提交接口。通过后自行进入C main。模型失败是结果，不因成功少改题或补图；接口/代码错误先修并回报。**完成，见 19:17 条。**
- [x] [最后交接agent] 两lane完成后`merge_lanes.py`生成ABC共同表；同输入条件正确率与原程序正确率分列，不能把各lane自己的common当ABC common。交固定案例、程序首次失败/最终成功/全阶段成功、累计长度/祖先计算/预算和配对诊断，不写统一机制结论。**完成，见 19:17 条。**
- [x] [gpt] 审阅执行适配与结果（见20:13记录）；围绕能力边界的下一步留用户讨论，第四步后训练、UPDATE门控/定位结构、纯连续跨窗口方案均未启动。

## 已完成：A/B连续赋值与死槽位复用；C恢复状态重新使用

- [2026-11-19 15:02][gpt] **两批结果审阅至5d07e08**：[审阅记录](handoff/2026-11-19-state-reuse-review.md)。30个产物与11个可对应源码hash匹配，C十路/资格/损伤总表、AB干预176行changed/retention计数一致；未发现要求GPU重跑的新增运行缺陷，未重放服务器完整records。B死槽真值改变时30/30、29/29，A保留旧态0/30、0/29而clear成功；B重复赋值回原题仍有16个下游错误。C共同hold/noise2/lag2为原始27/31、三个后训练31/31；仅换q对末两轮VALUE无改善，整clean状态对照不能定位机制。报告公式、restore_u修回/净改善、原始clean分母、token位置示意标签及跨表同名字段已在审阅澄清。**Agent B/C本批关闭，空闲交接同步即可，不新增GPU/训练任务**；下一步扩大复用、C分块恢复等留用户讨论。
- [2026-11-19 14:31][claude] **C-main 完成**：固定 `6ef63e3`，GPU2 串行 6/6；smoke 6/6 已按真实 tokenizer/连续反馈/十路状态重建/donor 时序接受（记录 `docs/records/2026-11-19/c-state-reuse/c-smoke-checks.md`）。四版本各 32 固定图 ×36 条件（3 时点×4 来源×3 lag）×H16，共 4,608 条 records；每版本 188,397 sample / 48,229,632 token forwards，main 约 513 s。q last-two 主口径为 VALUE，first4 单列；`input_qualified` 881/1138/1131/1127，`common_input_qualified` 870/1,152。可归因扰动的持续 q VALUE 损伤只出现在 noise2：early/noise2 unique 图 10/6/7/6，hold/noise2 initial 4 图而三个后训练版本 0；observed 下 keep/pre/solved 的 q VALUE 逐版本相同，clean receiver 修回 124/124，pre donor 对 VALUE 无增量；restore_u 仅 lag1 修回 10 条、同时改错 5 条。q 输入正确的损伤 initial 11 条全为 g16 late clean-path、noise 1 条为真实单条；没有可重复的“u/q 输入读数都对、q VALUE 仍错”。自动包 late/semantic/lag2 的 first4 3→29 是 VARIABLE 字段漂移，VALUE 29→29（initial 仅 12/32 input_qualified），不能当算术恢复。报告/小表/audit：`docs/records/2026-11-19/c-state-reuse/c-functional-reuse.md`，提交 `21a8c10`；原始 records/隐藏快照留 tmp。未训练、未扩图、未做层/头干预；四版本同源 seed2026，不是四种子，不作机制或最终测试结论。
- [2026-11-19 13:39][claude] **AB-main 完成**：固定 `6ef63e3`，GPU0 串行 8/8；smoke 已按真实 tokenizer/独立真值/阶段继承/死槽位检查接受。repeat/reuse 各 32/32 图，两模型分别 576/128 条记录、576/128 个共同 case；H8/H16 是独立序列。H16 末两轮：A warm target 4/4/3/28（同输入）及 4/3/3/30/28（交替），B warm 32/32；A/B clear 与 scratch 均 32/32；reuse A warm 2/32、3/32，B warm 32/32，clear/scratch 32/32；H8 reuse scratch ep2 为 29/32。A/B 的 warm_pinned 与 warm 轨迹相同，本批未见 pinning 改善。报告 `docs/records/2026-11-19/ab-state-reuse.md` + 小表/audit，提交 `5e8fff1`；原始 records/回执留 tmp。未训练、未扩图，不作机制或通用槽位结论；C lane 未动。
- [2026-11-19 12:22][user/gpt] 用户要求多完成执行准备再交agent。已增加[串行队列与结果包](handoff/2026-11-19-execution-bundles.md)：batch.py分别生成AB/C的小验证及main计划、固定源码、每任务日志/完成回执、跳过已完成任务；deliver.py自动关键表/说明/完整案例。新增3项CPU检查及CLI检查通过，未运行GPU。**Agent B/C现在按此入口直接执行**，完成原交接真实检查后自行记录accept-smoke并进入原定32候选main，不再等待重复用户批准；零资格/协议错误须报告。共享树仍串行，失败半成品不覆盖。模型/数据/预算/实验条件未扩张。
- [2026-11-19 12:09][gpt] [CPU补交审阅完成](handoff/2026-11-19-cpu-supplement-review.md)，同步至05d64b7。18个产物/源码hash匹配、10752行联合计数及312行focal事件算术通过；A/B重导数字不变，C交集更正289/414、持续446/606获支持。旧stable_return允许单末轮、旧导出器空示例报错及来源声明限制已记录，不改变本批主要数字，也不影响新state_reuse脚本。上一批CPU待办关闭，两agent可按既定授权继续小验证及固定范围执行，不需重新等待批准。仓库未见新GPU任务结果，不能把CPU补交当新试验完成。
- [2026-11-18 17:28][user/gpt] 用户认可六方向下的两个优先后续并要求实现与充分验收。见[新交接](handoff/2026-11-18-state-reuse.md)和[代码](../tools/state_reuse/README.md)。本条更新此前“新hidden干预/复用尚未授权”的范围，仅授权交接内固定开发集小验证及执行；不启动新训练/大规模扩节点/新架构。
- [x] [gpt] 独立目录实现A/B连续重新赋值（同输入/交替输入、回到原题）、两个死输出槽先重写再消费、六分支H8/H16；C三时点×四来源×三检查延迟、3×3接收状态/历史q操作加sham；实际输入输出、独立真值、资格交集、图均权、配对效果/稳定段/旧值残留导出。9项针对性CPU检查已通过；真实GPU/权重尚未运行。
- [x] [执行agent B] 先完成下方上一批CPU补交，再同步固定版本，按新交接做前两候选H4小验证，之后A/B各固定32候选repeat与reuse，H8/H16独立序列，串行执行；不按成功补样。交接口diff、版本/资源成本、共同分母、阶段结果及新增适配代码。**已完成，见上方 13:39 条。**
- [x] [执行agent C] 先完成下方联合统计/导出代码补交，再做前两图H4小验证；通过后前32图、C原始/保持/噪声/语义四版本、H16。固定延迟全保留，资格不按治疗输出筛；交十路结果、donor校准、实际输入/输出案例与共同分母。无自然失错也应报告，不擅自延长/换权重。**已完成，见上方 14:31 条。**
- [x] [gpt] 两agent回传后的新代码、协议记录、计数与案例审阅完成（见15:02条）；下一阶段方案仍待讨论。服务器共享树仍串行交接，运行中不pull；此前CPU待办保留，不覆盖历史数据。

## 审阅完成至ad0315e：结果保留，CPU补交与统一命名

- [2026-11-18 16:22][gpt] [审阅与补交](handoff/2026-11-18-priority-review.md)：两运行核心源码/hash与记录一致，C六CSV与7368行四格算术通过，A/B七产物LF hash与主表交叉计数一致；未重放服务器records/模型。A保留旧态时新下游0/454、空白454/454；B为454/454（delta1/3/7）及448/454（delta11），常数漂移0。C报告的受控TF分母、原始版本事件范围、2–6×噪声分布及因果措辞已更正；旧293/414联合比例暂撤回。下方历史完成记录中对应说法以本条/审阅为准。
- [2026-11-18 16:22][user/gpt] **后续统一模型展示名A/B/C**，执行agent必须遵守[命名规范](../docs/model-names.md)。C写中文训练版本；旧名字集中[映射](../docs/records/2026-11-18/model-name-aliases.md)。现象不再用A/B/C编号；原始路径/机器ID/hash不改。
- [x] [gpt] 报告导出器取消pinned/free及VALUE/四属性相同的结果断言，去掉32样本及额外检查已执行的硬编码；1图/H2带差异的端到端回归通过。当前旧CSV不覆盖，无需GPU重跑。
- [x] [执行agent C] CPU补交小表导出脚本、输入/输出联合计数及focal事件范围，按审阅具体口径复核旧百分比；展示用C中文版本与现象①②③。不增加新hidden干预/训练。
- [2026-11-18 20:01][claude] **C CPU补交完成**：固定 `3454e07`，未加载权重/未跑 GPU。新增 `tools/priority_pilots/export_function_audit_tables.py`，从既有 function-audit 大表重导 8 份 v2 小表（source-summary/quadrants/situation-counts/matched/events/events-focal/joint/examples）+ audit，首版 CSV/hash 保留；展示名映射 C（原始/保持/噪声/语义后训练），现象改①②③，机器 arm 保留。联合计数核对：pair/u_wrong/hold/r2 完整 controlled 1013、controlled TF 363、q输入正确44/错319、输出TF且u输入正确 **289/414**、现象③32（旧293/414是输入轴边际，已更正为289/414=69.8%）；持续/wrong_q_clean/hold/L2/R1 完整 controlled 973、controlled TF 515、q输入正确0、u输入也正确 **446/606**。focal 事件表按模型/时点/持续长度分列直接q注错、首轮输出对/错、曾错返回/未返回/再错；例如 C（原始）initial/hold q_wrong 直接注错128条、首轮对116/错12、再错0，both_wrong 首轮对3/错125、再错2，u_wrong 无直接q注错、曾错123。14项测试通过（含2项导出helper）；未改 runner/records/选点，未开始 C state-reuse GPU 任务。提交 `f1e96c2`，报告 §9 与 v2 audit 含源/代码/导出 SHA。
- [x] [执行agent B] 用修复后导出器新目录CPU重导，对比旧表数字并补scratch/tokenizer一次性核验代码/证据；展示统一A/B。原runner/数据/选点不动，不重跑GPU。
- [2026-11-18 18:03][claude] **B 上一批CPU补交完成**：修复后导出器在 `report-tables-review` 重导，6 个 CSV 与 ad0315e 历史文件逐字节相同；新 audit 如实记录 `pinned_free_identical`/`value_equals_first4` 及“scratch/aux 由独立审计负责”。新增 `verify_scratch_start.py`：A/B 各 128 case、完整原始序列（1024 token）× 六字段共 1,572,864 格，只有 256 个预期常数 VALUE 格不同，0 unexpected；scratch audit 与 re-export 证据已提交。报告统一 A/B、修正 r8=450/454 与 r16/末两轮=454/454。12 项测试、5 个 CLI help 通过；未重跑 GPU、未改 runner/选点/数据；新 state-reuse GPU 小验证尚未启动。

## 当前优先：C输入对齐诊断 + A/B合法重新赋值

- [2026-11-18 12:56][user/gpt] 用户要求选两个方向，具体化首步、实现并给验收后推送。选E2/E4+M1、W1/W2；[代码](../tools/priority_pilots/README.md)与[执行/验收交接](handoff/2026-11-18-priority-pilots.md)已完成。此项更新此前“六方向仅讨论”中这两个首步的状态，其余方向不自动启动。
- [x] [gpt] C的现有pair/stress输入对齐CPU导出与四格/图均权汇总；A′/B七分支重新赋值runner与新旧真值/共同资格/末两轮/同后续预算汇总。11项stdlib针对性测试、3个CLI加载、语法检查通过。未加载服务器权重，未运行真实GPU实验。
- [x] [执行agent C] 领取F：确认空闲后同步，按交接处理既有pair/sustained/noise记录；核对输入/输出轮次、匹配q污染参照、受控子集与来源分母；交小表、固定排序案例、hash和报告。CPU任务，不重跑模型，不把观察四格写成因果机制。
- [2026-11-18 14:07][claude] **F 输入对齐诊断完成**：同步 `f89677b`，11 项测试通过；仅 CPU 读取 pair/sustained/noise 既有记录（1024/2048/512 条，全部 selected/value_qualified；1,172,736 edge-rounds），未加载权重/未跑 GPU。手工核对 pair/持续/噪声的 post1/post2 输入来源、释放轮计法、匹配参照，四格和图均权全部通过。关键观察：pair 第2轮 293/414 条“u输出对、q错”已有 u 输入正确，但 controlled 内 q 输入正确仅 44/363；持续释放 R1 的 446/606 已具备 u 输入正确，controlled 残差 q 输入正确为 0/515；噪声 r1 的 s3/s6 中 q 输入读数几乎全对但 q 输出错（165/165、160/160）。仅 sparse C（u/q 输入输出及控制均对但 q 错）：pair 114、持续 6（graph107）、噪声 2,230 个 edge-rounds；不能写成已证明机制。报告 `docs/records/2026-11-18/priority-function-audit.md` + 6 小表和 audit，提交 `bdc480f`；未训练新模型。
- [x] [执行agent B] 领取W：按交接先A′/B各2图×4轮小验证，再固定graph32–63、每图1叶×delta1/3/7/11×七分支、H16。沿用manifest指定权重，严格旧题前四属性资格，不补样；检查新真值只用于统计、scratch无oracle答案、pinned只固定常数VALUE。交共同资格及自身资格结果，old/new和常数漂移分开。
- [2026-11-18 15:03][claude] **W 完成**：固定 `f89677b`，A′/B 各 32/32 图严格资格、128 条 case、128 条共同 case；先 2 图×H4 smoke 后运行主批。A′ `new_warm` 仅 3/486 changed nodes 达新真值（但 scratch 486/486），B `new_warm` δ=1/3/7 为 486/486、δ=11 为 480/486；两模型 scratch 均 486/486；pinned 与 free 全轨迹相同；常数漂移 0/28,672；未改节点末两轮新增损伤 0/538。报告 `docs/records/2026-11-18/priority-reassignment.md`、6 份 CSV+audit；大 records 留 tmp；未训练、未改结构、未扩图/槽位。下一步建议先做“已有答案更新”诊断，不进入槽位复用。
- [2026-11-18 12:56][gpt] 两任务按共享树串行同步约定交接；B两模型串行执行。范围仅交接定义的首批，不训练C+、不改更新结构、不扩图/槽位复用。无改善可验收；协议/真值/时序错误不可验收。GPT待两份结果与新增环境适配代码回来后审阅。

## 研究推进索引：六个方向，按编号讨论

- [2026-11-18 11:33][gpt] 用户要求将研究内容按相对正交方向拆分、逐项推进。新增[六方向待办](../docs/records/2026-11-18/research-directions-todo.md)：E定义评估、M现有机制、W工作空间复用、T训练配方、U更新结构、G架构归因与推广，各6项；附已有基础、完成标准、依赖、分阶段建议及前文想法覆盖表。旧草稿和既有待办保留。
- [ ] [user/gpt] 第一轮优先讨论E1/E2/E4、W1；M1先复用已有证据，G1/G2先设计公平对照。随后再选最小诊断与训练/结构候选，不同时变更全部因素。
- [2026-11-18 11:33][gpt] **执行agent说明**：36项是研究索引，不是36项已授权作业；目前不新增运行任务。后续交接引用编号，按“已定方案→实现→小验证→执行→审阅”推进；服务器串行约定不变。

## 待讨论：潜状态错误、节点复用与定位—提交

- [2026-11-18 11:24][gpt] 按用户要求新增[设计草稿](../docs/records/2026-11-18/loop-correction-design-draft.md)，核对远端至d4addb9无新结果。汇总已有B/C证据，区分读出异常/功能损伤/功能恢复；提出原题污染、合法重新赋值、固定槽位复用的独立协议，明确活跃宽度、旧定义/引用、序列与恢复证据限制。核对REGRET/MODIFICATION当前未训练启用，给出同前向、双前向、流水式定位提交候选，以及不改架构C+与控制架构消融。
- [ ] [user/gpt] 讨论R0重新赋值与R1固定槽位的范围、功能检查口径、双前向原型及UPDATE/INVALID语义；确定后再拆代码/执行任务。优先级为草稿建议，不是已批准计划。
- [2026-11-18 11:24][gpt] **给执行agent B/C：本次仅讨论文档，不新增训练、GPU实验或待领取执行项。** 保留既有授权与运行；空闲交接时同步，运行工作树不pull。未改共识、源码、数据和旧报告。

- [2026-11-16 15:49][gpt] 按用户提醒补清总览/1542审阅：输出读出正确、下一轮离散VALUE字段正确、内部正确读取并使用是三个层次；本批B只核对前两项，C隐藏反馈更不能将读出等同计算作用。核对完成后注错原表：同192目标的+1/+3/+7下游末两轮795/795，+11为760/795，目标均192/192；新补交是此旧实验失败子集，非相反的新实验。图44 complete的q从原本正确6变17并持续。仅文档澄清，不新增任务。

- [2026-11-16 15:42][gpt] **B残留CPU补交审阅至3e59aa3完成**：[复核](handoff/2026-11-16-1542-b-residual-review.md)。新导出脚本hash与meta一致，3328行/208节点轨迹、10摘要及原35/795计数一致；未重放全量源records/权重。进一步确认早期五个持续错误子节点在第2–16轮全部父输入VALUE正确却输出仍固定错误，不能仅用传播延迟解释；同图另有三父节点子节点第2轮恢复。原B CPU待办关闭；总览摘要/结果/下一步已同步，删除过时“尚待B补交”状态。下一轮直接写错/清空/上游传播对照留用户讨论，不新增GPU或C任务。

- [2026-11-16 15:15][gpt] 按用户要求，在[总览](../docs/records/2026-11-16/project-status-overview.md)摘要后补“下一步要明确什么，准备怎么做”：B更新/残留失败、C读出与计算、C+防漂移与修错收益、共同评估基础。先补已有轨迹分析、固定评估，再讨论小试与三seed；B既有CPU待办不变，其余不因该说明自动启动。

- [2026-11-16 15:11][gpt] 用户要求明确raw是否bug并补开篇摘要。已对作者原始提交`52dd950`、当前`508afdd`及本地论文v1完成只读核查：[证据链](../docs/records/2026-11-16/raw-aux-bug-review.md)。确认原始离散独立评估器将零任务loss、训练/模型内续算禁用的REGRET/MODIFICATION预测回灌，属于实现bug，不只是可任选协议；论文定义四字段。总览已明确更正并新增摘要。作者Figure 4实际运行路径仍缺证据；当前B机制force_aux_false已规避，C隐藏向量反馈不直接受此bug影响。旧默认eval路径仍保留该行为，本次未改代码/重跑，不新增agent任务。

- [2026-11-16 14:49][gpt] 新增[项目现状总览](../docs/records/2026-11-16/project-status-overview.md)，以`a2e9eb8`及已审阅记录为基线，汇总主力/历史/废弃权重、两批C后训练端点、B/C实验操作与结果、数据/指标、已修复问题及延期项。本次fetch未发现新实验提交；这是资料整理，不是新增11月16日实验。B五个残留案例CPU补交仍待完成；C现有批次已交付，正式三seed/中等噪声训练仍待讨论。不新增执行任务，不改共识/实验代码，不需要重跑GPU。以后可从总览进入具体批次，历史看板状态不得覆盖较新完成记录。

- [2026-11-15 22:03][gpt] 按用户要求新增[今天B/C实验汇总](../docs/records/2026-11-15/bc-daily-summary.md)：权重、实验具体操作、样本/结果、观察和必要例子。涵盖B单/多父错值分析，C来源/阶段/自然错误修复、s2026能力、长程与注错后训练、父子对照及持续/噪声校准。采用截至6b765c9已核对口径；原简报保留，不改数据/代码、不新增执行任务。

## 当前执行：C持续错父与噪声剂量校准（2026-11-15 20:33）

- [2026-11-15 21:32][gpt] **审阅至8d70b3f完成**，见[审阅](handoff/2026-11-15-2132-c-stress-review.md)。执行源码未改；源码/10份声明导出hash、10560行准确率配对计数一致，96个wrong条件剂量均实际强制预期错值；未重放服务器权重。已更正记录数、强制末轮与首轮、initial late对照漂移、噪声132/264分母和方向描述，补强噪声长期残留。无需重跑GPU。
- [2026-11-15 21:32][gpt] **给Agent C**：本批已交付，空闲后同步审阅；不自动启动新训练/新矩阵。单次保护优势在第二次强制后基本消失；semantic释放恢复不更快；强噪声有目标始终读对但下游延迟出错案例。2–3×中等噪声小训练、现有records的CPU联合归因、三seed范围留用户讨论。Agent B既有CPU工作不变。

- [2026-11-15 20:33][user/gpt] 用户认可2016审阅后的下一步并要求实现。已实现[执行/汇总](../tools/c_evaluation_suite/README.md)，详见[交接与命令](handoff/2026-11-15-c-sustained-noise-execution.md)。本条授权这两项现有权重评估，小验证通过后可执行；替代历史“持续污染/噪声强度尚不执行”的范围限制，不授权新训练或三seed正式实验。
- [x] [gpt] 核心代码：四权重、128图两时点、持续写u 2/4轮×q原状/单次改错，配free/干净源回放/同值源控制，释放16轮；噪声校准固定前64图、两时点、六档强度×两方向，配free/同值/异值。逐轮输入输出/剂量、强制与释放分开、全固定/四模型跨L共同资格、clean及只改q参照、逐图等权、末两轮和目标真实恢复时刻。基础评估协议保留，不重跑已完成批次。
- [x] [执行agent C] 领取后确认共享树与GPU队列空闲，记录同步SHA；运行新增5项stdlib+2项torch及两模式各4图smoke/汇总，重点核对u写L次、q只写一次、释放后无写入、同绝对时点donor、噪声同方向缩放。通过后按交接串行执行128图持续实验与64图校准；不得因结果差而换图/强度/资格。首次执行已获授权，无需重复询问启动。
- [x] [执行agent C] 交覆盖/实际剂量/阶段曲线/恢复事件/健康节点损伤、少量真实方程和每轮数值；噪声原本已错不算新增，读出未变但下游受损另列；原始记录留tmp，提交子表须记source/export哈希及过滤规则。不得把两方向当独立图或seed，不自动选训练强度。
- [2026-11-15 20:56][claude] **C 持续错父与噪声校准完成**：同步 `3320449`，7 项测试、两模式 smoke/汇总通过，无 NaN/OOM。sustained：128 图×2 时点×L2/4；common VALUE 256/256、strict 167；强制写错 u 时 q 输出 0–5/128，u 输出仍可 92–108/128 正确，释放后 q 中位首次恢复约 2 轮，r4 110–124、r16 127–128（initial late 110–111）；L4 的 release r1 稍低但 r4 接近，semantic 初期恢复更慢，没有更快恢复证据。noise：64 图×2 时点；1×（同 L2）new VALUE error 1/128，而语义 wrong 64/64；2×/3×/4×/6× new error 约13–20%/39–50%/56–65%/75–79%。semantic 高强度的 children harmed 低于 hold（6× 46 vs88；late56 vs92），但属单 seed、不同 donor/norm，不能当训练收益；L2 不等同功能损伤。报告 `docs/records/2026-11-15/c-stress.md` + 10 小表/审计，提交 `a97debe`；GPU2 已释放，未训练新模型。
- [2026-11-15 20:33][gpt] 本地5项stdlib（含两模式真实汇总CLI）通过，CLI加载通过；2项torch交服务器，未加载权重/未跑GPU。原训练/腐蚀试验源码、既有结果未修改。Agent B继续既有CPU任务，不新增GPU；运行工作树不pull或切版本。

## 已交付：C基础评估首批（2026-11-15 19:45）

- [2026-11-15 20:16][gpt] **审阅至4ed7331完成**，见[审阅与下一步讨论](handoff/2026-11-15-2016-c-basic-review.md)。新提交未改执行源码；24个声明导出hash、运行源码hash、11520行配对计数核对一致，未独立重放服务器权重/NPZ。旧CPU补交关闭。本轮报告已更正semantic首轮保持的反向措辞、恢复分母/残留与clean混淆、总数相同推断无反向影响、graph68真实恢复轮次。无需为这些文档更正重跑GPU。
- [2026-11-15 20:16][gpt] **给执行agent C**：本批已交付，工作树空闲时同步审阅即可；不因本条启动新训练/持续污染。下一步与用户讨论固定独立评估、持续错父×子节点是否已错、noise实际损伤校准。**给agent B**：这三次提交没有新B结果；保持既有CPU任务，未授权新GPU矩阵。

- [2026-11-15 19:45][user/gpt] 用户要求先做基础实现并交agent执行。已实现[tools/c_evaluation_suite](../tools/c_evaluation_suite/README.md)，详见[首批执行交接](handoff/2026-11-15-c-basic-evaluation-execution.md)。此条授权下面三项评估，替代历史“尚不执行新矩阵”；不授权任何新训练。
- [x] [gpt] CPU已有records联合分析；现有initial s2026/e143及hold/noise/semantic四权重六档统一基本能力/到每图D+64；128图u→q两点2×2及同值来源/原样写回对照。节点VALUE/四属性、角色/深度/父数、完整读出轨迹、完成/失稳分母、输入父值与输出比较。原训练/腐蚀代码不改。
- [x] [执行agent C] 先确认共享工作树和GPU队列空闲，固定本提交，运行6项stdlib+1项torch及4图smoke；成功后按交接串行执行六档各64图/模型和128图父子两时点评估。已获用户授权，无需再次问启动。若源构造/时序/资格全失败则反馈，不为得到好结果换样。
- [x] [执行agent C] 交能力/长程曲线、全固定与共同资格的u/q恢复和传播、健康节点损伤；两个完整方程+实际输入输出案例；审计区分源文件与提交子表哈希。复用新CPU分析补1904联合表，原strict/unrelated/all与导出哈希补交仍保留。
- [2026-11-15 20:05][claude] **C 基础评估首批完成**：同步 `7f25a42`，7 项测试、clean/pair smoke 和汇总通过，无 NaN/OOM。clean 4 权重×6 档×64 图（到每图 D+64）：D+64 VALUE/first4 均值约 initial 0.796/0.022、hold 0.977/0.344、noise 0.968/0.276、semantic 0.968/0.343；first4 整图 all 在 D+64 均为 0。pair 128 图×2 时点全部有 q，value 资格 256/256，common strict 185；q_wrong 的 q 首轮 97–116、r2 128，both_wrong 首轮 0–3、r2 61–74、r4 111–126、r16 112–128；u_wrong 的 q 首轮 8–40，r2 71–84。semantic 的 u/q 首轮错误更多，未显示更快恢复；其下游 damaged 数较少。报告 `docs/records/2026-11-15/c-basic-evaluation.md` + 13 小表/审计，提交 `bfa33dc`；GPU2 已释放，未训练新模型。
- [执行agent B] 继续既有CPU工作，不新增GPU；共享树/GPU按既定串行交接，运行中不pull/改配置。
- 范围：只评估当前四权重，不训练三种子，不运行真实改题、持续污染、层/头干预或原延期的祖先局部自洽控制。这些留后续讨论。
- 本地6项stdlib通过、语法检查；1项torch由服务器验证。尚未在本地加载模型或运行GPU评估。

## 当前讨论：先明确C评估，再定正式后训练（2026-11-15 19:14）

- [2026-11-15 19:14][user/gpt] 用户提出先建立基本能力、纠错能力、机制的完整评估。已记录[评估框架讨论稿](handoff/2026-11-15-c-evaluation-framework-draft.md)，尚未定稿/未授权新矩阵执行。
- 研究原则：解码头读出与下一轮Transformer使用隐藏状态是两条不同信息路径；解码错不等于下游实际用了该错值。下游不是已确认存在的另一专门计算头；具体编码/机制仍需证据。
- 设计重点：固定能力基准与探索性机制实验分开；正常求解、长程保持、干预有效性、恢复/传播、健康状态损伤并列；用“下游已对/也错/尚待算”和真实改题适应性区分保留答案与重建能力。
- 先讨论/利用现有权重和CPU证据设计评估，不启动三种子新后训练或新GPU矩阵；既有CPU补交继续，已有任务不打断。测试集/样本量/时长和矩阵待用户确认。

## C腐蚀三组审阅与CPU补交（2026-11-15 19:04）

- [x] [gpt] 审阅至267bf40，源码未变且审计源码hash匹配；预算与主CSV计数一致。semantic减少子节点首轮传播但目标恢复不更快、末端接近；noise仅32/1489注入后读出错误，不能称等错误剂量。3倍噪声下semantic较少下游错误是值得复核的迁移观察。详见[审阅](handoff/2026-11-15-1904-c-corruption-review.md)。
- [x] [gpt] 更正报告字段归因（VARIABLE主导而非全部）、initial late末两轮127/128、链接、已做人工注错状态及噪声解释；未改代码/重跑。下方历史“未训练C+”应读为尚未证明纠错机制，不是没有做后训练。
- [x] [执行agent C] 仅CPU补完整strict/unrelated/all汇总及源/导出hash映射；现有两提交小表hash不匹配audit，先明确是否过滤子集。补目标×下游联合逐图配对与实际方程/值轨迹，训练noise的新增错误数；不跑GPU。具体见审阅单。
- [2026-11-15 19:30][claude] **1904 CPU补交完成**：提交 `b9e934b`；干预/event 改为完整源汇总，新增 noise-dose、joint-pairs/highlights、example-values/expressions；audit 记录 source→export SHA 与过滤条件。noise before-correct→after-wrong 真实新增 24/1481（1.62%）；旧“32 after wrong”含 8 个 before wrong。未重跑模型。
- 本轮没有新训练授权；第二seed/更强噪声训练方案与用户讨论后再执行。B既有任务不变。

## 当前授权：C腐蚀后训练三组小比较（2026-11-15 18:29）

- [2026-11-15 18:29][user/gpt] 用户授权先核对B注错规模，再实现C三组并交执行；时机/位置按C设计，不照搬B。具体见[本批方案与命令](handoff/2026-11-15-c-corruption-pilot.md)、[代码入口](../tools/c_corruption_pilot/README.md)。此条替代下方历史“不得新增后训练”，仅授权本批hold/noise/semantic。
- [x] [gpt] B配置/代码/256batch审计记录核对：batch5%×候选50%，last为新槽EMPTY；每图每次约0.7897替换/0.6954错误数字，有批级聚集，不是已填值注错。未重跑B审计。
- [x] [gpt] 核心实现：固定s2026/e143、复用1024/128、三组均+D/3遍/batch8/lr1e-4；每图每遍50%尝试一个内部节点，保持阶段前半单次注错。在线当前模型反事实源，同计划同计算预算；noise匹配L2，三倍噪声仅评估。非零模偏移1–22，记录实际语义成功/原值正确与损伤规模。原训练代码不变。
- [x] [执行agent C] 领取后确认共享树/GPU空闲，固定提交；先计划生成、三组4batch smoke、各8图评估汇总、3项torch检查。小验证通过即按方案串行三组正式训练与128图两时点评估，首次执行已获用户授权，无需再问启动。若语义替换没有有效错值/共同资格，先交证据、不偷偷改参数。
- [x] [执行agent C] 交报告/小表/元数据哈希与预算、实际训练dose、all_fixed及共同资格、节点VALUE/四属性与分字段、配对损伤/恢复/再错/末两轮。原始JSON/NPZ/权重留tmp；不按val挑checkpoint，模型更新后不改测试规则。
- [2026-11-15 18:56][claude] **C 腐蚀后训练三组完成**：同步 `0f2e619` 后固定运行；9 项测试、三组 4-batch smoke/8 图评估和标准加载通过，无 NaN/OOM。1489 次调度、semantic 1489/1489 注入命中指定异值，noise 32/1489 首读出错误；三组预算相同（393 steps、3072 图曝光、recipient sample×loop 40284）。干净 D+64 VALUE/first4：hold 0.9873/0.3506、noise 0.9878/0.2883、semantic 0.9927/0.3523；SYNTAX/OPERATION 全 1.0，收益均来自 VARIABLE。共同 VALUE 资格 256 个 graph×timing，common strict 201。semantic 目标首轮 VALUE 更易错（hold 时 91 vs hold105；late 93 vs106），但直接子节点首轮损伤更少（67 vs22；late77 vs22），到 r4/末轮与 hold 接近；没有稳定恢复增益。完整报告/小表/哈希见 `docs/records/2026-11-15/c-corruption-pilot.md`，提交 `eace08d`；GPU2 已释放，未训练 C+。
- [执行agent B] 不新增GPU任务；继续原CPU补交，和C按串行共享树协议交接。运行中不pull、改配置或输入。
- 本地6项stdlib（含真实汇总CLI样例）通过，3项torch留服务器；尚未加载模型或运行本批训练。当前源/配置/数据指纹检查用于对齐本批，不扩展防御性验证。

## C 长程结果审阅（2026-11-15 18:05）

- [x] [gpt] 审阅至0e6aefd；执行源码未变，核对预算/节点/事件/配对/六档/轨迹小表。+D/+2D 两套预算均改善长程保持；+2D不是数值准确率处处优于+D。新验证first4仍全部图最终失稳，不能称纠错机制已建立。详见[审阅记录](handoff/2026-11-15-1805-c-long-results-review.md)。
- [x] [gpt] 更正报告末轮76/128与末两轮75/128混用、优化器措辞和字段归因边界；未改执行代码或重跑。继承的reduce_batch不能裁剪factor PAD已记录，暂不改变训练recipe。
- [x] [执行agent C] 仅CPU补本批manifest/train meta/eval meta来源与预算审计摘要；若现有NPZ支持，补分字段曲线/首次失错时间，不支持则说明，不重跑GPU。不提交权重/原始训练数据。
- [2026-11-15 18:56][claude] **C 长程审阅CPU补交完成**：提交 `51d88ff`；新增 `c-long-pilot-audit.json`（source/data/train/validation 哈希、参数、checkpoint/预算）及 `c-long-pilot-field-curves.csv`/`c-long-pilot-field-drift.csv`。validation NPZ 的 predictions 含四字段，直接确认 SYNTAX/OPERATION 全 1.0、first4=VARIABLE；VALUE D 后失错图：initial 71/127、base equal 85/128、+D equal 4/128、+2D 3/128；VARIABLE 失错中位约 8.5–31 轮。未重跑模型。
- 本轮不新增训练。第二seed、人工注错和长程比例设计继续与用户讨论；B既有CPU任务不受影响。

## 当前新增执行：C长程小比较（2026-11-15 17:36）

- [2026-11-15 17:36][user/gpt] 用户授权核对s2026并实现额外D/2D小规模比较；具体见[方案与执行命令](handoff/2026-11-15-c-long-pilot.md)、[代码入口](../tools/c_long_pilot/README.md)。这条授权替代下方历史“不得开始任何C后训练”，但仅限普通续训/+D/+2D三组，不启动人工注错、自然错误重放、混合或从头训练。
- [x] [gpt] s2026/e143报告、CSV与归档身份已核对：384图曾VALUE完成383；H16六档整图63–64/64，H64 N128两档整图0，节点VALUE仍69.36%–76.61%。归档hash匹配，未独立重放NPZ/模型。固定此权重作为小试起点，不换best/final。
- [x] [gpt] 实现tools/c_long_pilot：官方新N32/depth≤6数据1024 train/128 val、同D固定batch、base/+D/+2D完整连续BPTT、固定阶段权重、两套预算端点、新验证自由续算至D+64、节点VALUE/四属性与漂移/恢复、逐节点配对改善损伤。原model.py/训练配置未改。stdlib6测试通过；torch1项留服务器，不代表实际训练已验证。
- [x] [执行agent C] 领取并确认共享工作树空闲后固定本提交。先生成新数据、三组各2 batch smoke（含原loss/D等价与新ckpt标准eval load），正常后串行base→plus_d→plus_2d→新验证→三个等计算量端点六档诊断。命令见交接单；不得边跑边pull。首次执行本小试已授权，无需重复问启动。
- [x] [执行agent C] 提交短报告、budget/summary/paired/event小表和少量轨迹；同时呈现2遍数据的同曝光端点，以及base6遍/+D3遍/+2D2遍的等循环计算代理量端点。参数统一lr1e-4、batch8；不得按有利口径或验证效果换checkpoint。N64/N128只诊断，不训练；异常/需要改参数时反馈。
- [2026-11-15 17:58][claude] **C 长程 pilot 完成**：同步 `08e8264`，未边跑边 pull；7 项 CPU/torch 测试通过，smoke 的 base loss/D 等价和标准 eval load 通过，正式训练无 NaN/OOM。新数据 1024/128（manifest status=complete，8 个排重参考）；三组 equal_compute 的 optimizer steps 786/393/262，sample/token/attention 代理量同为 40284/10312704/2640052224。新验证：D+64 节点 VALUE/first4：initial 0.9143/0.0293，base equal 0.8862/0.0332，plus_d equal 0.9980/0.4185，plus_2d 0.9985/0.4663；整图 VALUE D+64：initial 57/128、base equal 44/128、plus_d equal 124/128、plus_2d 125/128。plus 组 VALUE 完成后失稳仅 3–4/128，但 first4 仍 128/128 出现完成后错误、末两轮整图 0。旧六档 H64：plus_2d 节点 VALUE 0.983–0.991、first4 0.559–0.595、整图 VALUE 0.688–0.969，较 initial 的 0.694–0.944、0.021–0.044、0–0.641 明显改善；H16 正常 D 和 tV/t4 中位数保持。单 seed/128 图新验证，旧六档非独立测试，不能归因机制或称正式配方。报告 `docs/records/2026-11-15/c-long-pilot.md` + 7 小表，提交 `8934d0a`；GPU2 已释放，未训练 C+。
- [执行agent B] 继续上轮已交办的CPU残留链条补交，不新增B GPU实验。C训练期间不更新共享运行树；按既定串行交接。

## 当前审阅：三项首批已完成，补现有数据解释（2026-11-15 16:36）

- [2026-11-15 16:36][gpt] 已审阅至26bf873，详见[三项结果审阅](handoff/2026-11-15-1636-bc-results-review.md)。主计数与仓库CSV核对一致；未复算服务器完整records。B/C runner未改，唯一源码改动为跨时点汇总资格显式化及测试，当前样本不受影响。
- 主要结果：B目标2400/2400修回，但+11下游35/795末两轮仍错；原graph142在干净起点写8也保持16轮。C共同287个q的+1首轮局部预期命中285→114→42（early/complete/late），late仍177个错误，不能解释成免疫。联合首轮增益集中在parent_wrong类（0→17/22），末两轮无净收益。
- [x] [执行agent B] 仅CPU按审阅单修正采样分母/末轮措辞，导出5个残留案例完整链条的输入输出及clean配对，追踪首个持续错误位置；不重跑GPU、不新增机制干预。
- [2026-11-16 15:32][claude] **B 1636 CPU 补交完成**：报告 §3 改为“末两轮（15/16）”并写明四错值合并 795、仅采样分母 679；outside 改为仅末两轮正确，补 B whole/+1 在 B:94:72 有 2 个曾错。新增 `tools/repair_study/export_b_residual.py`（14 项 repair-study 测试通过）和 `b-multiparent-residual-{chains,summary}.csv`/meta：5 target×2 时点、每个时点最终 35 个下游错误节点集合相同；5 个 target 首轮/末两轮均正确，首个持续错误都在第1轮的直接子节点（76/112/68/122/118，深度 2/3/2/4/4）。未重跑 GPU、未改选点/时点。
- [x] [执行agent C] 仅CPU注明阶段coverage的included来自共同交集、状态率来自各自eligible子集；补共同169目标状态表、联合三类表，更正“其它状态分摊”假设和7/7单位。导出联合1改善/1恶化与仅父分支的逐例身份和轨迹；不重跑模型。
- [2026-11-15 17:15][claude] **C 1636 审阅补交完成**：提交 `34803c3`（文档/小表）与 `a62732c`（s2026 原始包目录归一）；s2026 集成在 `b1370b7`。阶段报告补 coverage 来源表（included=三阶段共同交集 169/阶段，另有 eligible-not-common 35、ineligible 18、unavailable 16）和固定 169 目标状态表；共同交集 VALUE/first4：early 6980/12096、complete 12096/12096 与 12086/12096、late 11956/12096 与 9040/12096，并标明 status=eligible 分母不同（late 12544）。联合报告补三类首轮表（parent_wrong 0/8/3/17，parent_correct 0/0/0/0，variable_only 12/12/12/12）和末两轮表（6/22、7/23、12/12）；1 改善为 N32_shallow:14:170:53，1 恶化为 N64_deep:0:306:52，二者末两轮与 parents 相同，joint 相对 parents 首轮 +14/0、末两轮 0/0；all-node 7/7 改为 case×节点末两轮端点观察（5 case、位置集合与 parents 完全相同）。新增 5 个 CSV；28+10+12 项 CPU 测试通过，未重跑 GPU。
- 本轮未开启C+或后续GPU实验。已完成实验条目保留；下阶段设计与用户讨论后再安排。

## 当前执行入口：B多父节点、C时点对照与联合修复（2026-11-15 15:00）

- [2026-11-15 15:00][user/gpt] 用户已授权这三项实验；C+训练暂缓。具体方案、命令、样本和交付要求见[本批交接单](handoff/2026-11-15-b-multiparent-c-stages-joint.md)。本条取代下方历史“无新增GPU实验”的调度限制，既有CPU补交仍保留。
- [x] [gpt] 已实现核心代码与配对汇总；本地21项相关stdlib测试通过，语法/差异检查通过。未加载服务器权重、未运行模型；服务器agent负责小验证及执行。
- [x] [执行agent B] 先补完既有CPU报告更正，再在空闲工作树同步固定提交。按交接单小验证后运行：同架构A′/B、固定64候选图、每图按父数1/2/3各选至多1个目标，首次算对/整图完成两个时点；多个错值，固定前8图穷举全部错误数值。原graph142异常另列，同时比较相同绝对轮次的干净/受污染上下文。提交资格覆盖、逐轮与末两轮VALUE/四属性、图等权及同目标跨时点表。不得按行为换样补样。
- [2026-11-15 15:53][claude] **B 多父节点首批完成**：运行 `8907018`，A′/B 各 386 条（384 主 + 2 anchor），全部 eligible、无缺图；64 图×父数 1/2/3×两个时点=192 target/时点。目标 VALUE 错值：A′ 0/2400（枚举+采样），B 2400/2400；下游 A′ just_solved 错值 0/795、whole_graph 795/795，B +11 有 35/795 末两轮不正确（5 target；枚举 +11/+17 各 7/116）。报告 `docs/records/2026-11-15/b-multiparent.md` + coverage/target/endpoint/round/cross-timing/anchor 小表；汇总修正 `--require_timings` 按每模型分别要求时点，12 项 CPU 测试通过，提交 `7b499e7`。GPU2 已释放，C 可按交接单开始时点对照。
- [x] [执行agent C] 先完成下方来源对照CPU补例；B释放实验队列后，执行C时点对照小验证与首批：现有两e143权重、六档各固定前8图、每图至多2目标；早期/首次整图数值全对/再过16轮。每时点保留同值来源与异值+1/+3，报告各时点资格及三时点同目标同子节点交集，VALUE资格主表与严格四属性子集分开。
- [2026-11-15 16:08][claude] **C 时点对照完成**：`572cc10`；10 项 CPU 测试；576 stage records，eligible early 185/complete 184/late 173；三阶段共同交集 169 target/stage、严格来源交集 72。第一轮 q：early 311（same正确310，diff正确0，局部预期307/310），complete 316（same316，diff正确70/25，预期125/115），late 300（same296，diff正确110/83，预期44/49）。原图 early VALUE 0.588，complete 1.000，late 0.988/first4 0.745。报告 `docs/records/2026-11-15/c-stages.md` + 小表，提交 `b697927`。
- [x] [执行agent C] 随后执行联合修复小验证与首批：复用原C局部修复的57事件及健康对照、保存的hidden和同一批历史正确向量，一次同时替换目标及直接父节点。复用原单点/仅父分支；新增不改动重放、联合原样写回与联合替换。不得重新选事件或换donor；交付相对不改动/仅目标/仅父的逐例改善和损伤，保留自然错误类别及健康对照。
- [2026-11-15 16:08][claude] **C 联合修复完成**：复用 57 事件/114 records，全部 eligible；smoke 2 事件×2 seed、main s42 54/s1234 60 records。natural target VALUE first：unchanged 12/57、target 20、parents 15、joint **29/57**；final2 六分支均 25/57。healthy joint first 53/57 vs unchanged 46，final2 29 vs 27。all-node final2 helped/harmed joint 7/7，target 1/1，没有持续净改善。报告 `docs/records/2026-11-15/c-joint-repair.md` + 小表，提交 `b697927`。未训练 C+；GPU2 已释放。
- 调度：共享工作树和GPU实验队列串行（B→C时点→C联合）；既有训练不打断。运行期间不pull/编辑运行树或输入；交接时记录固定commit、命令与适配。用户已授权这批小验证及正式首批，无需重复请求启动；若发现真实语义问题先记录反馈，不擅自换设计。各agent只提交本人产物，正常fetch/整合/push reproduction。
- 边界：不训练C+，不扩N256，不做祖先整片回滚/局部自洽/额外机制矩阵。“末两轮正确”仅指观察窗口内，不等同永久稳定；未完成或不合格样本保留分母并说明。

## 前批：C通用同值/异值来源对照（2026-11-15 11:59）

- [2026-11-15 14:31][gpt] 已审阅至f007600，无新增执行代码；384行coverage/9344行summary中的主视图计数核对一致，C局部修复逐例改善/恶化0/0与1/1已核实。详见[1431审阅](handoff/2026-11-15-1431-source-results-review.md)。同值与异值的特定下游效应已跨六档/父数/原值出现，仍是条件诊断、非C+收益。
- [x] [执行agent C] 仅CPU补充来源对照两例完整方程/全部叶贡献/题面修改与实际节点轨迹，纠正案例A的0+8≠13算术矛盾；把u离开注入值与回正确值分开，主表并列图等权，13个不合格拆源失败11/q条件失败2。修正“只一轮影响”及后续切片=持续恢复的措辞。无需重跑模型，不追加新实验。
- [2026-11-15 15:36][claude] **C 1431 审阅补交完成**：新增 `c-source-control-examples.csv`，从原 records/data 导出两例完整目标表达式、全部叶系数/值、value-token 位置、q 表达式/真值和各分支实际轨迹；案例 A 修正为 x2=11、x10=2（11+2=13），不再把 value-token 位置当叶值。报告把 u 首轮拆成“回原值 261/371、其它读出 110/371”（+3 为 269/102），主表并列 graph-equal，13 个不合格拆为源资格失败 11 / q 为空 2；把“只一轮影响/99%恢复”改为“首轮特定影响、后续逐轮回到原轨迹”，并说明 round16 同值仍有 6 个错误 q 观测。未重跑模型。提交 `0b56202`。

- [2026-11-15 11:59][user/gpt] 用户授权补齐同值异来源对照，并明确不限4/5，需在不同数据下构造通用场景。新代码`tools/counterfactual_control/`按原值a构造a→a、a→a+1/a+3（模23），同图/同位置/同轮次，匹配同两个叶常数的非零改动。六档数据、两e143权重、父数/深度分组取样；旧C4百分比不与本轮拼表。详见[设计与边界](../tools/counterfactual_control/README.md)、[执行交接](handoff/2026-11-15-generic-source-control.md)。
- [x] [gpt] 核心构造、连续运行、同分母配对汇总已实现；本地6项针对性stdlib测试及语法检查通过，未实际加载权重/运行GPU。已验证任意原值0–22及模回绕，不把完整来源题的q真值误当局部替换预期。
- [x] [执行agent C] 空闲工作树同步固定提交，先N32_shallow/N128_deep各2图×4轮小验证；通过后六档各固定前16图、每图至多2目标、两权重串行、16轮。交付源资格/覆盖/向量距离、同值与异值配对差及具体例子。原值不筛选、不按干预成功换样，缺组/不能构造如实记录。此任务已授权，不需要再次确认启动。
- [2026-11-15 13:47][claude] **C 通用来源对照首批完成**：同步 `6b4f32c`；6 项 CPU 测试通过。smoke s42 在 N32_shallow/N128_deep 各 2 图×4 轮，8/8 eligible；main 两 seed 各 96 图、192 候选，s42 191、s1234 180 eligible（合计 371/384；13 个 q=0 不合格，未补样）。覆盖 fan-in 1/2/3=97/136/138，depth 1–10，原值 0–22。第一轮 683 q：同值来源 q 正确 680/683，改变/受损 3/683；异值 +1/+3 q 正确均 0/683，改变/受损均 683/683，命中局部预期 678/683 与 682/683。异值 q 在 round2 恢复约 2/3，round4 后约 99% 回到原值；u 第一轮 forward 多数回算原值（+1 261/371，+3 269/371）。同值来源 hidden 无 exactly-equal（L2 0.044–5.65）。报告 `docs/records/2026-11-15/c-source-control.md`，小表 `c-source-control-coverage.csv` / `-summary.csv`，提交 `ff5da08`。GPU1 已释放，未训练 C+。
- [2026-11-15 13:47][claude] **C 1138 审阅 CPU 补交完成**：提交 `target_cases.csv`、`target_summary.csv`（保留 stratum/selection_group）和末两轮改善/恶化小轨迹 `c-local-repair-review-node-curves.csv`；报告补充 parent 类别取 t−1、干预在 t、自然/干预窗口不同；调度补记为 C 用 GPU6/7、B 用 GPU1，C 运行期间未 pull/改代码。提交 `abe61aa`。
- [ ] [执行agent B] 继续下方1138审阅要求的已有结果CPU补交，本次没有新增B GPU实验。
- 调度：共享工作树与GPU任务串行；运行目录不pull/改输入。C先完成已有CPU补交，再执行本任务。不重训C+，不动既有训练，不扩N256或其它来源构造。

## 前批：B问题①、C问题③（2026-11-15 01:35）

- [2026-11-15 11:38][gpt] 已审阅至602a27f：两线首批完成，本轮新增提交没有执行代码变更；CSV主计数核对一致，未独立复算服务器完整轨迹。详见[首批结果审阅](handoff/2026-11-15-1138-repair-results-review.md)。B主92目标沿用单父节点筛选，原异常却是三父节点，覆盖范围需明确；C短期改善不等于末两轮持续改善，也不等于C+无效。
- [x] [执行agent B] 仅CPU补交现有node_curves和graph142/128五分支目标/父/下游小轨迹；修报告“第16轮无损伤”与“全程无损伤”混用，注明单父节点主样本及重复节点观察分母。不重跑GPU、不自动扩多父节点。
- [2026-11-15 14:45][claude] **B CPU补交完成**：`b-value-update.md` 增加主样本单父节点限制、`2944/541` 重复节点观察分母说明；§3改为“第16轮”并补逐轮临时损伤（B wrong_plus/empty 第1–7轮有 harm，第8轮起 all_nodes harmed=0），§5.3相应限定。新增仓库内 `b-value-update-node-curves.csv`（719KB）和 `b-value-update-anchor-trajectories.csv`（640行：graph142/128五分支×target/全部parents×16轮）。提交 `330940d`。未重跑GPU、未扩多父节点。
- [x] [执行agent C] 仅CPU补交现有target_cases及保留stratum/selection_group的target_summary、报告案例及末两轮改善/恶化案例小轨迹；明确parent分类取t−1、干预取t、原自然结局窗口与干预窗口不同。不重放模型、不训练C+。
- [x] [执行agent B/C] 补记本轮实际并行GPU调度是否使用独立固定工作树、运行中是否有更新；之后恢复既定串行交接。此项不预设结果失效或要求重跑。
- [2026-11-15 14:45][claude] **B/C调度补记**：C 报告已记 smoke/main 使用 GPU6/7、B 使用 GPU1，均在同一共享 worktree 的 `95611a1`，运行期间未 pull/改代码，输出目录独立；C 报告明确“没有证据表明该并行调度污染数据”。本轮实际并行与交接单“GPU串行”不一致，记录供 GPT/用户判断；此后恢复串行安排。

- [2026-11-15 01:35][user/gpt] 用户选择先研究①B已有答案怎样更新、③C自然错误能否局部修复，暂缓②无依赖分支影响来源；授权GPT实现并上传、执行agent小验证后运行。核心代码在`tools/repair_study/`；详见 [行为定义](../tools/repair_study/README.md) 和 [执行交接](handoff/2026-11-15-0135-bc-repair-study.md)。本条优先于旧“等待下一步设计”，不包含C+训练或原M3/M4全矩阵。
- [x] [gpt] B固定样本/五分支、C事件分类/选样/四分支hidden替换、配对汇总已实现；本地仅stdlib针对性测试及语法检查，无权重/模型运行。原案例与其它样本分开，不按修复结果挑样。
- [x] [执行agent B] 同步固定提交，先2候选图×4轮并带graph142诊断；正常后A′/B各固定64候选图×16轮，报告共同合格数、覆盖/恢复行为及下游影响，保持anchor单列。命令见交接，不扩图、不调错误幅度、不重训。
- [2026-11-15 03:05][claude] **B 已有答案更新首批完成**：A′/B 各 185 条记录（184 固定 + 1 anchor），全部 eligible；固定样本 56 图、每时点/模型 92 target、共同合格 184。目标 VALUE final-two：A′ wrong_plus_1/3 为 0/92，B 为 92/92；unchanged/correct_write/empty 两模型均 92/92。A′ just_solved wrong_plus downstream 0/541、all_nodes 2311/2944；whole_graph downstream 541/541、all_nodes 2852/2944；B 五分支 all_nodes 2944/2944、downstream 541/541。报告 `docs/records/2026-11-15/b-value-update.md` + 两个小 CSV，提交 `47ffaad`。GPU1 已释放，未重训、未启动 M3/M4/C+。
- [x] [执行agent C] 从原六档NPZ做CPU事件分类和固定选样，报告缺来源/缺健康对照；再两案例×4轮小验证，确认原轨迹重放一致及原样写回一致后，按同一清单运行两个seed各16轮续算。保存hidden快照在tmp，按自然结局/场景分开汇总。
- [2026-11-15 02:57][claude] **C 局部修复首批完成**：8 项 repair-study CPU 测试通过；CPU 选样 48,523 events / 40,014 selectable，固定 57 case，缺健康对照 0；3 个自然缺组如实记录、不补样。s42 27 case/54 records、s1234 30 case/60 records，smoke 2×4 轮和 main 16 轮均 complete，无 replay failure。初步结果：restore_target 的 initial 常被立即读对，但 parent_wrong 的 first-forward 0→8/22 后 final-two 仍 6/22（同 unchanged）；parent_correct 的 restore_target first 0/23、final2 7/23（同 unchanged）；variable_only 的 VALUE 四分支 12/12，first4 的 initial 12/12 但 first-forward/final2 均 0/12。健康对照也出现首轮变化，故首轮提升不能当错误特异修复。报告 `docs/records/2026-11-15/c-local-repair.md`，小表 `c-local-repair-key-metrics.csv` / `-selection-events.csv` / `-selection-groups.csv`，提交 `81ac77a`。GPU6/7 已释放，B 仍在 GPU1；C 不再启动 GPU 任务。
- [x] [gpt] 首批报告/CSV及执行代码差异已审阅；后续小轨迹补交再复核逐例变化，跨场景扩展待用户讨论。
- 调度：共享工作树和GPU任务串行，B小验证/主运行优先；C CPU选样可提前，GPU排其后。现有训练不打断、运行工作树不pull；两位agent各自只提交本人产物和必要适配。

## B / C 精简汇总（2026-11-15）

- [2026-11-15][gpt] 按用户要求新增 [B/C实验简报](../docs/records/2026-11-15/bc-experiments-brief.md)：固定权重、已做实验、主要数字、真实轨迹及结论边界。B使用修复后主实验和五案例核查，C使用六档报告及用户提供的NPZ包复核结果；不覆盖原报告，不新增运行安排。

## 当前交接：GPT补齐证据入口，执行agent只做CPU导出（23:22）

- [2026-11-14 23:22][user/gpt] 用户认可由GPT先写核心实验/统计逻辑，执行agent负责服务器适配、小验证、运行和反馈；本次先处理剩余三个核查点。具体见 [交接单](handoff/2026-11-14-2322-b-evidence-export.md)。不改变既定实验条件，不启动M3/M4/C+。
- [x] [gpt] 源码与窄CSV核对：graph142/pos128在第2–16轮保持8（正确2）；unrelated12节点事件全部短暂受损后恢复；teacher/own比较脚本漏clean。前两者机制未定，不能把“窄CSV已导出”当作依赖/因果核查完成。
- [x] [gpt] 写好完整5案例导出工具与固定清单，比较入口纳入clean；原数据图结构与manifest分别保留。新增4项针对性CPU测试；无服务器模型运行。
- [x] [执行agent B] 空闲工作树同步后按交接单CPU导出5个完整pair及原数据图关系，提交小型review_evidence；原始tensor bundle留tmp。重新运行teacher/own比较（含clean）并更新M1报告。仅路径适配，语义改动先反馈。
- [2026-11-14 23:40][claude] **B证据导出完成**：`b_export_review_bundle.py` 对5个固定case（142/late_u/0；146/149/275/unrelated/0；276/unrelated/1）导出完整 clean/intervention 16轮pair、dataset_graphs（含第三父/叶常数关系）和 provenance；`review_evidence.json` 2.8MB 提交为 `docs/records/2026-11-14/b-main-review-evidence.json`，`review_bundle.json` 3.0MB留tmp。含clean的teacher/own比较输出16行：clean/r/u/ur各92交集、`rounds_identical=92`、missing/duplicate/conflict=0；M1 §4.1已更新。graph142 pos128第1轮2、第2–16轮8（真值2），parents [92,64,22]；未重跑模型、未启动M3/M4。
- [x] [gpt] 完整证据核查已完成到行为层面，见 [五案例核查](../docs/records/2026-11-14/b-review-evidence-audit.md)。5图160节点父/子关系与原数据导出结构一致；12个exclusive受损节点均不在注入点下游。graph142/128第3–16轮父输入均正确仍输出8，排除“父值持续错误”解释，不能据此确定内部复制/吸引态机制。
- [2026-11-14 23:51][gpt] 新增可复跑CPU核查脚本及派生CSV/JSON；teacher/own含clean共1472条交集轨迹一致。日志“missing=0”限定为missing_in_own；missing_in_teacher每行273是own额外样本，非缺失故障。未重跑模型，未新启动B/C实验。C这轮无新提交，待下一步设计。
- [2026-11-14 23:22][gpt] 下方旧“B结果补充已完成”仅指窄CSV导出及入口提交，完整证据核查以上方待办为准。graph142未进入long32，不能据其它图的H64结果推断其最终恢复。

## 上轮审阅：M2已交付，先补CPU核查（22:36）

- [2026-11-14 22:36][gpt] 审阅 `bc3f226..7584f45` 两线新增代码/报告，B已完成M1补充与M2，C已交现有NPZ分析。详情见 [结果与代码审阅](handoff/2026-11-14-2236-bc-results-code-review.md)。未发现需推翻整批模型运行的证据；新增辅助脚本有明确缺陷，不能照原样扩用。
- [x] **B优先**：修 `b_long_window.py` late多目标clean键碰撞、checkpoint_round被忽略、偶数median；补source/目标身份。现有M1 early默认16/64不因此作废；M2完整窗口复错分析修后用现有轨迹做，不重跑模型。
- [x] **B结果补充**：导出graph142/pos128未稳定事件；unrelated exclusive 12次损伤按完整干预身份/父关系/逐轮输入输出核对。明确sink首轮512/512修回、输出damage0与恢复NA并不矛盾。补teacher/own逐值配对可复跑入口，M1首轮表注明after_u。
- [2026-11-14 23:02][claude] **B review 修复完成**：`b_long_window.py` 已改公共 `target_key`、重复 clean 校验、参数化 checkpoint、`statistics.median`，per-node 补 source/event/target/pos；新增 `b_compare_teacher_own.py`、`b_export_events.py`、`test_long_window.py`，B CPU 测试 26 项通过。M1 early 计数不变；M2 long32 完整窗口 B late_r 13/349、late_u 6/300、leaf 41/817 曾恢复→再错，均在第16轮前、17–64 为0；graph142/pos128 与 unrelated exclusive 12 次事件已导出。报告和仓库内小 CSV 更新，未重跑模型、未启动 M3/M4。
- [x] **C优先**：将“VALUE与VARIABLE保持”改为独立统计，不遗漏延迟完成但保持的图；保留互斥primary_event。修测试夹具预测值/正确标志不一致，现有NPZ重汇总即可。辅助逐字段sink表注明micro或补macro。
- [x] **C解释补充**：要讨论两事件先后，用同图两事件时间差及未发生分母；首次受损节点positions需补导出才能深入追踪。不要从H64切片说更早失稳，不把tV时VARIABLE错写成必然永远无t4。
- 本地只做clean键最小复现及214行CSV计数/比例核对，未取回服务器全量原始数据、未重跑模型。本轮不新增M3/M4/C+启动安排。下方20:44入口为上阶段授权/历史，M2“待执行”以本条完成状态为准。

## 当前执行入口：B 继续 M2，C 整理现有轨迹（20:44 更新）

- [2026-11-14 20:44][user，gpt记录] 用户要求在审阅未发现阻塞问题后继续推进，并将两位agent的提醒写入看板推送。审阅基线为 `6bffe85`：B M1已交付（`52654f0`）；C六档探索与小验证已交付（`8c55605`）。**在已审阅的执行路径及本次改动中，没有发现阻止下述工作的新代码缺陷；不是“所有代码无bug”或已独立复算服务器原始结果。**
- 这条更新优先于下方历史“B主实验未启动”“C未验证”“M2等待确认”等状态。原始记录保留；无需为下述已交办事项再次请求启动确认。
- GPT本轮只更新交接，不运行服务器任务。执行agent负责小验证与运行，GPT后续继续审阅新增执行/分析代码和结论；不重复已经通过的整套检查，不为文案修正重跑模型。

### Agent B：补正 M1 口径，并继续 M2 complete

- [ ] **M1报告修正（现有轨迹CPU即可）**：主表改回逐图等权，节点加权和原始分子/分母作为辅列。各H保留 observed / damaged / stable，零受损分母记NA；不同模型在不同受损集合上的恢复率不能单独当作同分母的因果效应。
- [ ] **长轮数补充**：H16/H64末尾两轮正确不等于第17–64轮从未再错。用long32已有逐轮数据统计“第16轮已正确节点中，17–64轮曾再次出错的数/分母”，另列曾恢复→再错、最终连续正确段起点。若没有反复再写“本观察窗口内保持”；不外推永久稳定。清楚区分首次恢复与窗口末尾两轮正确。
- [ ] **起点配对**：teacher64是own全集的固定子集；按同图、同链、同timing、同condition、同H筛取own对应92条链，核对共同资格后做配对。保留原全集结果；不能用64图与256图总体差异归因起点。
- [ ] **两项指标与直接点**：现有代码分别计算VALUE和first4，相同汇总不自动说明bug。直接从记录统计前三字段各自错误数及“VALUE对、first4错”的节点轮次，解释是否本批仅有数值错误。`before_u_U`保留EMPTY→数值、首轮输出正确数；无输出损伤时不填恢复率100%。直接位置表注明before/after，避免两种时点混表。
- [ ] **启动 M2**：按既有 `b-main-execution-plan.md` 的complete第一部分执行：A′ s42 e86 / B-M6 s42 e92，同现有main256 manifest（候选32–287），`late_r,late_u,sink,leaf_value,unrelated`及同起点clean；主H16并报H8。固定long32子集（32–63）跑同条件H64；不补样、不换权重、不扩展条件。模型端已有相关小验证，不必重做全套；若新发现具体问题，只做对应小检查。
- [ ] **M2交付**：保留原始输入/输出轨迹，报告各目标类型资格数、actual injection与输出损伤、逐轮传播/恢复、clean自身漂移；unrelated同时报告原目标、受注入分支、共享后代与exclusive区域。early/complete比较用同图同链同条件交集，不把共同完成样本的结果推广到未完成图。输出阶段报告、关键CSV、命令/代码SHA/manifest与权重身份，更新看板日志。
- M1补充可与M2运行安排衔接，文案问题不单独阻塞M2；若发现会改变实验含义的实现错误则先处理。M2交付后讨论M3持续错父/M4改题面；本次不自动扩展到switch、局部自洽、N128或跨seed。

### Agent C：补可复核分析，整理 C+ 设计所需证据

- [x] C修复同步、小验证及首批六档运行完成：服务器报告12+2项CPU测试、生成/重载与2图模型小验证通过；s42/s1234各384图×64轮。详见 `c0-capability-smoke.md` / `c0-capability-report.md`，代码交付 `8c55605`。此处更新旧交接勾选状态，不代表GPT代跑了验证。
- [2026-11-14 21:26][claude] **C 现有轨迹复分析完成**：新增 `tools/c_capability/analyze_existing.py`（只读现有 NPZ，不加载模型/不重跑数据）及 8 项 CPU 测试，提交 `4cb9bf0`；输出逐字段/错误深度/每图错误数/模差/完成延迟/逐图事件/事件汇总/例子，产物 `tmp/analysis/c0-capability-20261114/analysis/`。报告定义已修正：`stable_recovered` 为退化后末尾连续 2 轮正确；末轮正确率不再与从未退化保持率混用；所有完成图均有后续观察（tV≤61）。事件支持数：690 张完成图中没有一张同时保持 VALUE 与 VARIABLE；s42 六档无 stable recovery，s1234 仅 N32_deep 1 张、N64_deep 2 张；VARIABLE 新错误平均比 VALUE 首次退化早。source_index 已保留，但未做错父代入/下游归类，报告不作因果机制结论。
- [2026-11-14 22:57][claude] **C review 修复完成**：提交 `eec76ca`。`value_preserved`/`variable_preserved`/`value_and_variable_preserved` 改为独立布尔统计，`primary_event` 仍作互斥案例分类；补“延迟完成且稳定”合成测试。字段辅助表补 graph-equal（sink 与 micro 可不同）；`value_error_*` 改用 `completed_by_H`（tV≤H）队列；`events_by_graph.csv` 增收首次受影响 positions 和同图事件顺序；测试夹具改为先构造预测再派生正确标志，`test_c_analysis` 14 项通过。现有 NPZ 重汇总：690 完成图 VALUE preserved=197、VARIABLE preserved=0、两者独立均保持=0；493 张两类事件均发生的图中 VARIABLE 先 479、同轮 1、VALUE 先 13，另 VARIABLE-only 196、VALUE-only 0、neither 1。报告已把 H64 切片“更早失稳”改为“H64 正确率更低”，并明确 tV 时 VARIABLE 错不等于永远无 t4。未重跑模型/C+。
- [x] **把辅助分析脚本提交**：逐字段正确率、错误节点数、按深度错误、模差、完成延迟及典型轨迹目前有报告但缺完整可复跑入口。提交读取现有NPZ的CPU脚本与实际命令、输入身份、输出字段定义；无需重新运行模型或生成新数据。若新增正式summary输出，保留现有列语义并注明新列。
- [x] **修正报告定义**：`stable_recovered`是退化后末尾连续2轮正确，非仅末轮；final/completed应称“末轮正确比例（在曾完成图中）”，不与从完成后从未退化的retention混用。完成后无观察的图单列；相对轮次曲线每轮列可观察分母。表中mean与正文median统一，不把平均写成中位数。
- [x] **收紧结论**：“没有稳定不动点”改为“所测轨迹在64轮窗口内出现完成后退化”；“计算能力不是主要瓶颈”限定到具体权重/档位，保留s1234 N128的完成缺口。H64按深度错误率高说明深层在该时刻更易错，不能单凭切片说深层最先出错。短轮数能力与超训练轮数的稳定性分开；“表示离开训练流形/忘记身份/换固定点”仅是待检验假设。
- [x] **指标范围**：全数值节点VALUE/first4两项核心指标继续保留，internal与sink作为分解。固定N只保证全节点（以及叶数固定时非叶节点）的宏/微平均相同，sink/深度桶节点数随图变化，不能笼统宣称所有指标宏微一致。
- [x] **整理失败事件与例子（仅现有数据）**：分为从未整图算完/延迟完成、完成后VARIABLE字段退化但VALUE保持、完成后VALUE退化，以及退化后恢复/再次出错。逐图保存首次完成、首次字段/数值退化、受影响节点/深度、首次恢复及最终连续正确段；按权重/档位报告支持数，避免用少数例子替代整体统计。可用已记录source index关联原图追踪节点及直接下游，错父代入仅在其它父值可解释时归类，多处同时出错单列。
- [x] **供讨论的交付**：更新C报告并附事件表、几条可复核轨迹；说明哪些故障是完成能力、哪些是保持能力，哪些有自然恢复空间。VARIABLE读出先错不证明它导致VALUE错误；hidden仍是唯一回灌状态，本轮只能报告时间关联，不能作因果机制结论。
- 不为此重跑六档，不训练C+、不扩展人工注错/probe；s2025仍不测。s2026完成后按已有同协议补入安排，不预先宣称完成。下一轮C+机制与训练方案由用户/GPT依据证据讨论。

### 同步与执行纪律

- 统一 `reproduction`；各agent在空闲工作树 `git pull --ff-only origin reproduction`，固定运行代码版本、独立产物目录；共享工作树/Git交付串行，不在训练运行目录中pull。各线单任务、4线程，使用确认空闲GPU，不打断现有训练。
- 领取与完成时更新本人行及append-only日志，写代码SHA/目录/实际任务状态。只推本人负责内容，推前fetch整合，不包含对方未完成改动。不要再让旧看板标题覆盖最新阶段状态。

## 本次修复交接（历史，状态见上方当前执行入口）

- [2026-11-14 19:13][gpt] 用户明确授权直接修复；1857审阅所列 B/C 问题已落到代码。B 23项CPU测试及B/C语法检查通过；C依赖测试/模型小验证未在本机运行。详见 [修复与同步单](handoff/2026-11-14-1913-bc-review-fixes.md)。下方历史“待实现”以本条为更新，原记录保留。
- [x] Agent B：空闲工作树同步 reproduction；核对 independent early eligibility、unrelated protected 分支及 D1/D2/D3 新旧真值专用汇总。现有256图全完成的manifest可复用；不重复跑B1/M5。验证后更新执行单和结果。
  - [2026-11-14 19:22][claude] 已同步 `5669677` 并补 `18d6230`（`b_new_truth_summary.py` CLI 的 sibling import 修复）。`unittest discover` 23/23 通过；pilot 主汇总 1,380 行与旧 horizon 汇总逐值一致。unrelated smoke H=2：注入 pos 28、protected target 50，protected_target damage 0/7、protected_shared_descendants 4/5、protected_exclusive 0/41。D1/D2/D3 用 `fix-test/B_late_v2.json` 验证，四个 CSV 有数据，D2 无 clean，old/new var 0→3、new const 3。early-only 合成 manifest（图32，T=None）：early 成功输出 2 条，late/control 均 `skipped_no_common_complete=1`。CPU 执行，未重跑 B1/M5，未启动 B 主实验。
- [x] Agent C：修复同步、小验证及首批六档探索已完成（8c55605，详见上方当前执行入口）；后续为报告补正和现有轨迹分析。
- 未改变训练/权重/模型反馈语义，未启动服务器实验；最终指标解释与机制结论仍依实际结果讨论。

## 当前决策

- [2026-11-14 15:44][user，gpt 记录] 用户要求暂时串行协作，并继续准备 C 能力探索：s2025 已欠拟合，不安排本批评测；s42/s1234 先做，新种子完成后补入。首批聚焦不同数据条件下的节点能力和自由计算过程，尤其完成后漂移，暂不训练 C+。执行队列按既有任务→B 长轮数小试跑→C 能力探索，运行目录期间不更新；具体小规模配置由 GPT 写入下方设计单，执行/小验证交 agent。
- [2026-11-14 15:26][user，gpt 记录] **已批准先做 B 长轮数小规模探索，再决定正式 H。**下方任务交执行 agent 直接安排，实现必要汇总并做小验证，无需为这批已批准的小试跑再次请求启动确认。此前“暂不启动实验”对本项试跑解除；全量实验、局部自洽和 switch 扩展仍不在本项范围内。GPT 负责后续讨论曲线和选择正式预算。
- [2026-11-14 15:21][user，gpt 记录] 已认可四项评估选择：全窗口逐轮为主并留首轮/曾受损摘要；末尾连续 2 轮正确为稳定恢复主判据、曾恢复为辅；逐图等权为主、节点加权为辅；VALUE 与前四属性同时正确两项节点指标并列。定义与边界已写入 `consensus.md`，取代此前“四项全部未定”的状态。总观察轮数 H 和第一批运行配置仍待讨论。请执行 agent 依据共识实现汇总并做必要小验证；GPT 继续负责设计讨论，不新增防御性测试。
- [2026-11-14 15:02][user，gpt 记录] 后续 GPT 负责与用户商量和设计，其他 agent 负责执行和小验证；GPT 不需要过度防御性验证，具体疑点与验证点交执行 agent 处理。此长期分工已按用户明确要求写入协作协议与 consensus。
- [2026-11-14 15:00][user，gpt 记录] 后续统一在 **`reproduction`** 协作。GPT 已将修复 `c91b6fe` 及交付记录 `dc61d12` 快进并入本地 reproduction，本次通过 reproduction 提交推送。此前审阅分支交付方式由本条取代；指标/评估仍未定稿，全量实验状态不变。
- [2026-11-14 14:01][claude] `consensus.md` 先留空；具体口径、结果和当前状态放在本看板，待用户确认后再上升为共识。
- [2026-11-14 14:25][claude] GPT review 已完成：R1/R2/R3 是确定的实现缺陷，暂不进入正式 B 实验；先修实现，再讨论指标和跑 Phase 0。
- [2026-11-14 14:41][claude] R1/R3/R2/R7/R8 代码修复和 4 图 pipeline smoke 完成；未启动 B 全量实验，R4–R6/R9 指标仍待讨论。

## 两位执行 agent 的领取区（17:46 更新，优先看这里）

用户要求解释分工并列出待办，由用户分别交给两位 agent。本轮不由 GPT 启动/派生 agent。
**两条线互不依赖：C 做能力探索；B 整理试跑并准备机制主实验。**
原有“C 排在 B 长轮数试跑之后”的前提已经满足，不要求等 B 机制主实验全部结束。
B 试跑作为预算/现象参考保留，不能并入独立主实验冒充新增样本。

### 协作与资源：分工可同时推进，共享目录仍串行

- 两位 agent 分别使用独立 clone/固定代码目录和独立产物目录；同一服务器工作树不能同时编辑或 pull。
- 仍统一通过 `reproduction` 交付，不新建协作分支。各自实现/运行时固定提交；提交推送轮流进行，推送前 fetch/整合，不覆盖对方提交。
- 领取时在日志写负责人、工作目录、代码提交、GPU、输出目录和负责文件范围。C agent 新增 C runner/报告；B agent 负责 B 汇总/报告；共享 `trajectory_io.py` 有必要修改时先协调，由一方提交后另一方接收。
- B 当前以已有数据的 CPU 整理为主，可与 C 评估同时进行。后续两线都需 GPU 时，只有各占空闲 GPU、CPU 预算允许才同时运行；条件不满足时 GPU 任务排队，不阻止另一方整理/设计。
- 每条线内只运行一个任务，默认4线程，不恢复六进程并行；不占用/打断现有 s2026 训练。不在运行目录中切版本。board/log 也轮流更新，保留双方记录。

### Agent C：直接执行已批准的 C0 能力与漂移探索

- [2026-11-14 19:44][claude] **领取 C 线并启动正式小跑**：工作目录本仓库，代码基线 `9df92a8` + 本地未提交的 C 入口路径/测试修复；产物 `tmp/analysis/c0-capability-20261114/`；数据构建用 CPU，runner 用空闲 GPU6（seed2026 在 GPU0 不动）；负责 `tools/c_capability/` 与 C 报告。先构建六档数据，再按 s42→s1234 串行 64 图/档 × 64 轮，阶段结果写回 board/log。
- [2026-11-14 19:58][claude] **C 首批六档完成**：s42_e143 95.4 s、s1234_e143 99.7 s，各 384 图 × 64 轮，无 Traceback/OOM；GPU6 已释放，seed2026 未受影响。主要现象：VALUE 完成率 s42 59–64/64、s1234 32–64/64（缺口在 N128）；H=64 整图 VALUE 全对率大幅下降（s42 N32_shallow 47/64 → N128_deep 1/64；s1234 N64/N128 只剩 0–7/64）。first-4 的下降几乎全来自 VARIABLE（SYNTAX/OPERATION 在 H=64 仍 1.000；VARIABLE 约 0.08–0.11），本任务的 first-4 实际是 `VALUE ∧ VARIABLE`；H=36 时已无图 VARIABLE 全对。VALUE 漂移深层优先，N32 多为 1–2 个错节点，N64/N128 多为 >5 个错；s42 完成时几乎严格按 depth+1，s1234 大图有明显长尾/未完成（N128_deep 32/64 完成）。报告：`docs/records/2026-11-14/c0-capability-report.md`；汇总：`tmp/analysis/c0-capability-20261114/runs/summary/`。提交 `8c55605`（已推送 origin/reproduction）。

- [x] 领取并核实可用的 s42/s1234 权重实际路径/哈希，沿用现有 e143；s2025不测，s2026完成后同协议补入，不等它才开工。
- [x] 按 [C首批设计单](../docs/records/2026-11-14/c0-capability-exploration-plan.md) 固定六档数据：N32/64/128 × 实际深度≤6 / 7–12，每档约64图；交实际深度/图数摘要，缺档用既有生成器补齐或说明，不按模型答对筛题。
- [x] 实现必要记录与汇总，做一次简短检查后运行。连续hidden自由反馈64轮；两项节点指标并列，按深度/内部/结果端分解；记录 tV/t4、完全算对后的保持/漂移/恢复。
- [x] 交付 `docs/records/2026-11-14/c0-capability-report.md`：能力表、过程曲线、典型错误、对C+的启示。产物 `tmp/analysis/c0-capability-20261114/`。本任务不训练C+，不扩展人工注错矩阵。

### Agent B：先整理试跑，同时准备主实验；分两次交付

- [2026-11-14 17:51][claude] **本 agent 领取 B 线**：负责 B1 已有轨迹 CPU 整理与报告口径补正，以及 B2 主实验执行单和选样材料；不领取 C 线。B1 不用 GPU；B2 等用户定稿后再串行运行（4 线程、空闲 GPU、单进程），不与 C agent 同时编辑共享文件。代码基线 `reproduction@a4ce43a`，产物 `tmp/analysis/b-main-20261114/`。

**B1：已完成（2026-11-14 18:12），已有数据整理，无需再跑六项试跑。**

- [x] 保留当前pilot原始数据/manifest/报告；在报告追加补充或更正，不删除历史。明确27张有链图、42目标、单seed的范围。
- [x] 各H并列 observed / damaged / stable；直接点单列“实际注错数/首轮修回数”，避免把0输出损伤当成0恢复率；宏平均与原始计数分列。
- [x] 从完整轨迹统计“回正后再错”，包括最后又回正的情况；第一次回正与持续正确起点分开。不要用target类别是否出现recover_then_error替代逐节点全轨迹统计。
- [x] 整理r→u→q1→q2实际输入/输出、错值代入是否匹配、恢复沿依赖的顺序；核实A′ R-only中3/42例外的节点类型。交几条代表轨迹，不预先认定成因。
- [x] 更新 `docs/records/2026-11-14/b-horizon-pilot.md`，保留这是选H和观察现象的参考实验。原始数据已足够完成以上CPU分析。

**B1 交付（18:12）**：报告 §9；脚本 `tools/b_mechanism/b_pilot_analysis.py`；产物 `tmp/analysis/b-main-20261114/b1/`。要点：B direct 首轮修回 42/42（各条件），A′ U 0/42、R 3/42；B complete_R 20/360、complete_U 7/323 个节点回正→再错但最终稳定，after_u/sustained 为 0；A′ 无恢复；3 个 A′ R-only 例外均为 depth0、parents=[]，但另有 18 个 depth0 r 未修回，结构成因未定。

**B2：执行单已交（2026-11-14 18:12），主实验仍未启动，待用户定稿。**

- [x] 新建 `docs/records/2026-11-14/b-main-execution-plan.md`，把下面建议整理成确切条件清单/命令/预期记录数/数据身份及合格数；缺少实现的项目明确列出。必要的固定候选图clean选样可做，不按干预结果换图。

**B2 交付（18:12）**：执行单按用户固定选择整理：graph_id 32–287 共 256 张、不补样、每图最多 2 条主链；teacher 固定 32–95；long32 固定 32–63；主 H=16、报告 H=8；sustained 4+16（总 horizon 20）；五模块为 M0 选样、M1 early、M2 complete、M3 sustained、M4 D1/D2/D3+审计。已核实 B pilot 用 0–31、B smoke 用 0–3，本批与 pilot 不重叠；不称独立 heldout（与 C 共享同一 heldout 池）。尚缺固定区间 selection、主汇总、D1/D2/D3 新旧真值汇总、审计分图/轮次聚合；M0 selection 也未启动。
- **推荐规模（待定稿）：**A′ s42 e86 / B-M6 s42 e92；固定256张N32候选图，与本次pilot图隔离；每图最多2条主链。候选数固定，分别报告before/after/complete合格数，不不断补样凑满256合格图。
- **推荐预算（待定稿）：**主H16并截取H8；按graph_id预先固定小部分样本做64轮。持续错父仍先保持4轮，释放后H16；不要拿N32的H直接保证N128。

| 主实验模块 | 做什么 | 要回答什么 |
|---|---|---|
| 早期三种输入关系 | before_u / after_u，各clean、U-only、R-only、U+R；own主分析，teacher作状态对照 | EMPTY与已有值是否不同；自身值与父值冲突时采用谁 |
| 整图完成后目标类型 | 内部节点U/R、叶变量值、sink、无关节点；每种配clean | 已有答案是否被覆盖；传播是否沿依赖；不同节点是否能恢复 |
| 持续错父 | 固定错父4轮后释放 | 能否抵抗持续坏输入，释放后怎样恢复 |
| 改题面D1/D2/D3 | 改常数后从旧状态续算 / 从空解新题 / 每轮保持新常数 | 回到旧答案和适应新题是否不同；新真值、旧真值分别记 |
| 训练腐蚀审计 | 分图/轮次统计实际被改位置及改前EMPTY/数字 | 测试中的4→5是否在训练腐蚀中出现 |

- [ ] 执行单以 [B完整设计](../docs/records/2026-11-14/b-mechanism-design.md) 为依据，补齐D1/D2/D3新旧真值专用汇总，不直接用pilot只支持的五条件汇总脚本覆盖所有模块。
- **交付边界：**用户本轮要求先解释和列待办；H16、256图和完整B运行预算仍是建议。B agent先交B1结果和B2可执行清单，再据用户定稿启动主实验。无需为上述已交办的整理/准备反复确认。
- 跨seed/N128作为后续验证；局部自洽仍暂缓，switch暂不纳入本轮必跑。C不必等B2定稿。

## 需要 review（GPT）

- [2026-11-14 18:57][gpt] 已审阅 `a98a290` 的B/C新增执行代码和相关原实现；未代跑测试或模型。详情 [B/C执行代码审阅](handoff/2026-11-14-1857-bc-execution-review.md)。用户提醒已加入README：后续必须检查执行代码，不能仅看汇报。
- [ ] **[C agent，先处理]** 生成meta的generation_config误存含Path的STRATA配置，N64浅档会导致JSON保存失败；改为真实生成配置的可序列化副本。父位置解析漏info末列第三父，修后核对sink分类。做一次针对性保存/重载/短轨迹验证，再按已有授权推进C评估。
- [ ] **[C agent，汇总补齐]** type/depth目前只有节点加权，补图等权与支持图数；other_only不等于已正确后漂移，区分错误残留/退化；补完成后各相对轮次覆盖、曾恢复/最终恢复/反复。第64轮刚完成的图不记为已观察稳定。C暂无新模型结果，不把草稿称为已执行。
- [ ] **[B agent，主实验边界]** selection仍先以共同整图完成筛early，未独立记录early资格（本批256图全部complete，当前M0不作废）；修正或明确当前限定范围。unrelated汇总须补for_u原目标分支，不能只观察被注错无关节点自己的后代。D123汇总仍缺，补旧/新真值专用表并明确未支持条件；不据主脚本成功退出称五模块都已汇总。
- [2026-11-14 18:57][gpt] B1新增反复统计与代码逻辑一致：complete_R 20/360、complete_U 7/323为节点干预事件，最终稳定但并非全程无反复。B M0/M5已跑、M1未启动；C仅代码草稿。旧“代码待提交”等状态以a98a290实际已提交为准。

- [ ] **B1 补充 + B2 执行单（待 GPT review）**
  - [2026-11-14 18:12][claude] B1 报告 `docs/records/2026-11-14/b-horizon-pilot.md` §9；B2 执行单 `docs/records/2026-11-14/b-main-execution-plan.md`。
  - 重点：256 张 32–287 是否足以作为固定候选；teacher 32–95、long32 32–63 的截取是否合理；每图 k=2 同时限制 sink/leaf 是否可接受；sustained 总 horizon 20（4+16）与长观察 68 的口径；M4 审计 256 batch/100 trials 的覆盖；§7 尚缺实现是否还有遗漏。只审文档/计划，不启动 B 主实验。

- [x] **B 长轮数 runner 性能修改（GPT 已实现；执行 agent 已验证并完成六项重跑）**
  - [2026-11-14 17:03][gpt] 用户授权后基于最新 `968129a` 完成 CPU 批量记录/按图真值缓存、默认4线程、逐条 JSONL/进度、串行启动器；19项 CPU 检查和语法检查通过。服务器模型等价、加速倍数、训练资源影响未测；不启动任务。交接见 [性能优化交付](handoff/2026-11-14-1703-b-runner-performance-fix.md)。
  - **执行交接**：执行 agent 按原交接做必要小验证，然后用同 manifest/参数串行重跑六项，输出新目录。训练所在工作树不 pull，可用 reproduction 此提交的独立目录快照。下方“最多2项”由本轮“串行一项”取代；5×仍是待实测目标。
  - **先读**：`handoff/2026-11-14-1645-b-horizon-performance.md` 和 `log.md` 的 16:30 条目
  - **问题**：`node_state`/`factor_strings` 对每个 position、每个 factor 从 GPU 逐标量 `int()`，每轮约 580 次同步；126 条 × 64 轮约 470 万次/任务。runner 未限制线程，六进程各约 224 线程，与 seed2026 共用 84 核 cgroup，训练 epoch 从 4:23 拖到 34:07；输出只在最后一次性写盘。
  - **要求**：每轮/每图只做一次 `.cpu()`，CPU 侧构造记录并缓存 label 真值；加 `--num_threads`（默认 4–8）；进度日志 + JSONL 增量落盘；启动端最多 2 项并行。保持 manifest/资格/条件/轮数/JSON schema 不变。
  - **代码位置**：`tools/b_mechanism/b_mechanism_common.py:354,366`；`b_mechanism_early.py:26,145`；`b_mechanism_late.py:39,140,154`；`b_mechanism_controls.py:31,35`。
  - **验收**：CPU 等价测试、4 图 horizon 2/8 GPU 等价、现有 15 项 CPU 检查通过、4 图 horizon 64 至少 5× 加速且不拖慢训练。完成后在 board/log 记提交；**不要直接启动六项重跑**，由执行 agent 做。
  - **兼顾 C**：C 能力探索排在 B 之后，走 nointerm hidden 回灌，6 档 × 64 图 × 2 seed × 64 轮，逐轮全节点读出；请把 CPU 侧/批量读出做成可复用 helper，不要只为 B 三个 runner 打补丁。详见交接单末节。

- [x] **B 机制代码 v2 + 指标实现**（GPT 已完成；发现 P1 缺陷，见“Review 结论”）
  - 审查对象：`origin/reproduction` 的 B 包提交 `ab80594`；review 修复提交 `66440b8` + `c91b6fe`（分支最新提交另含本协作目录）
  - **先读日志**：`log.md` 的 2026-11-14 11:35 / 13:44 条目，了解为什么改、改了哪些
  - 代码：`tools/b_mechanism/`
  - 设计与边界：`docs/records/2026-11-14/b-mechanism-design.md`
  - 变更/修复清单：`docs/records/2026-11-14/b-mechanism-changelog.md`
  - pipeline smoke（非结果）：`docs/records/2026-11-14/b-mechanism-smoke.md`
  - 重点：A′/B 共同资格、single-parent + q2、early 2×2、late D1/D2/D3、sustained/switch、audit `S[:-1]`、summary N0/N1/N2、manifest 哈希验证
  - 回复：在下方“Review 结论”追加，并在 `log.md` 记一行；不要直接改 `consensus.md`

## 进行中

- [2026-11-14 18:45][claude] **B2 M0 选样与 M5 审计完成；M1 注错未启动。** M0：32–287 共 256 张，256/256 common-complete、0 不合格；218 图有链、365 chains、512 sinks/leaves、365 unrelated；teacher64 为 64 图/92 chains，long32 为 32 图/49 chains；manifest 与子集已落盘。M5：256 batch×100 trials，65,536 candidates 与 161,725 次实际替换的 before 全部 EMPTY，before_numeric=0，after numeric 148,820、NA 6,359、EMPTY 6,546，4→5=0。代码已实现固定区间 selection、subset helper、审计聚合、`b_main_summary.py`；待提交与主数据汇总。等待用户对 M1 正式注错启动点头。
- [2026-11-14 18:12][claude] **B1 完成，B2 执行单已交；B 主实验未启动、selection 未跑。** B1 报告 §9 已补 observed/damaged/stable、direct 首轮修回、逐节点回正→再错、r→u→q1→q2 时序和 A′ R-only 例外；B2 固定 256 张 graph_id 32–287、teacher 32–95、long32 32–63、主 H16/H8、sustained 4+16，命令与记录数见 `docs/records/2026-11-14/b-main-execution-plan.md`。等待用户定稿及 §7 缺失实现（固定区间 selection、主汇总、D1/D2/D3 汇总、审计聚合）。
- [2026-11-14 17:31][claude] B 长轮数小试跑 v2 已完成：GPT 性能版 `24e586c`，六项串行约 7–8 分钟，GPU4/5，训练 GPU0 未动；产物 `tmp/analysis/b-horizon-pilot-v2-20261114/`，汇总同目录 `summary/`，报告 `docs/records/2026-11-14/b-horizon-pilot.md`。结果待用户/GPT 讨论正式 H。
- [2026-11-14 10:12][claude] 旧 C0 seed2026 150 epoch，GPU0；输出 `tmp/runs/oldC0-newseed-150ep/`；14:42 检查 epoch 61/150，约 5.9 it/s，val 1.540。不要停止/重启。
- [2026-11-14 13:44][claude] B review 包：commit `ab80594`；GPT review 完成，R1/R2/R3/R7/R8 及后续 F1/F2 已修，当前不启动实验。

## 待办 / 下一步

- [2026-11-14 17:40][gpt] 已读取 c386f8d 的完整B试跑报告与新增汇总改动。下一阶段建议见 [试跑后规划](../docs/records/2026-11-14/next-stage-after-horizon-pilot.md)。B主H16及N32正式扩展仍为建议，未写入共识/未启动；C首批已获批准，B试跑结束后可按原设计单开始串行实现和运行，无需等待B正式扩样或s2026。
- [x] [执行agent，B1 已完成] 按上述规划区分observed/damaged/stable分母；直接注入点另报首轮修回数；检查完整轨迹的“回正→再错”，不要以target分类不存在recover_then_error证明全程无反复；给r/u/q1/q2轨迹及A′ R-only 3/42例外的节点类型。保留原结果，更正解释口径。结果见 `b-horizon-pilot.md` §9。

- [x] **[执行 agent] B 长轮数窗口探索（六项 v2 已完成，结果见报告）**
  - 结果：B-M6 在 after_u/complete/sustained 的受损节点到 H=8 稳定恢复约 99.5–100%，H=16 全部 100%；A′ 所有条件 never。报告建议主 H=16，另设 64 轮长期稳定性检查；最终 H 由用户决定。详见 `docs/records/2026-11-14/b-horizon-pilot.md`。
  - **目的：**观察恢复需要多久，以及长循环会否再次出错；根据曲线决定正式 H，不按哪个 H 最能拉开 A′/B 差距来选。
  - **模型/样本：**A′ s42 e86 与 B-M6 s42 e92，沿用已核对的权重和 clean 回灌；固定约 32 张 heldout N32 图，每图 1–2 个共同合格目标，优先 own-state。记录图 ID、目标、深度、扫描/合格数量及权重身份；本批为预算探索样本，后续独立验证另留样本。合格不足如实报告，不按注错结果重选。
  - **起点：**沿用已有 before/after/complete 的资格和预算规则；本项只跑 after_u 与 complete。64 轮仅用于干预后的观察，不用它扩大起点搜索或选样预算。早晚直接比较标明对应共同事件交集及实际 n。
  - **单次干预：**after_u 的 U-only、R-only，以及 complete 的 U-only、R-only；沿用现有 U +1 / R +3（模运算）设置并记录。每条注错后续算 64 轮；配同模型、同起点、同长度的 clean。保存每轮实际输入/输出，包含 r/u/q1/q2、全图数值节点及其前四字段，以便观察更远下游与无关分支。
  - **持续错父：**从 complete 起点固定错父 4 轮，随后释放并观察 64 轮，总计 68 轮；配同起点 68 轮 clean。分开保存强制输入与模型输出，恢复时间从释放后开始计；固定期间的状态另列。
  - **H 对比：**从同一条长轨迹截取 H=2/4/8/16/32/64，无需分别重跑；持续干预的 H 指释放后的观察轮数。各 H 使用相同固定样本并报告覆盖 n，H 内受损分母的变化要展示。采用 consensus 中四项选择，不能直接把旧 post1 N1/any-later N2 改名为新口径。
  - **交付①逐轮曲线：**直接注错点、直接/更远下游的受损与恢复，以及节点 VALUE / 前四属性全对率；并列 clean，识别干预影响和长循环自身漂移。
  - **交付②H 对照表：**曾恢复、窗口末尾连续 2 轮正确、仅末轮回正/稳定性未确认；直接注错点与受损下游分开。逐图等权为主、节点加权为辅，分母为零单列，不赋 0%/100%。
  - **交付③案例与建议：**给出少量快速稳定、延迟恢复、恢复后再错、未恢复的轨迹（未出现的类型说明即可）；说明建议的主观察 H 及是否需另设长期稳定性观察。N128 是否另补小批、64 轮后是否延长，看到本批结果再讨论，不自动扩样。
  - **实现/资源/记录：**由执行 agent 实现必要汇总并做简短小验证，沿用已通过的代码检查，不重复扩展防御性验收；使用可用资源，不打断当前 C0 训练。建议产物 `tmp/analysis/b-horizon-pilot-20261114/`，简报 `docs/records/2026-11-14/b-horizon-pilot.md`；领取、启动、完成及关键结果写回 board/log，并在 reproduction 提交实现与记录。

- [ ] **[user/gpt] 根据 B 长轮数报告确定正式主 H**（执行 agent 建议 H=16，H=8 作敏感性，另设 64 轮长期稳定性检查）；确定后再讨论正式 B 运行预算/范围。
- [x] [gpt] 按上节 review B 代码和文档，给出可执行结论。
- [ ] **[执行 agent，排在 B 长轮数试跑之后] C0 能力与自由运行过程探索**：见 [首批设计单](../docs/records/2026-11-14/c0-capability-exploration-plan.md)。s42/s1234 固定现有权重；N32/64/128 × 浅层≤6/较深7–12 六档，每档约64图，缺档如实报告或用既有生成器补齐。每图自由续算64轮，报告 VALUE/四属性节点指标、内部/逐深度/结果端分解、逐轮能力及 VALUE/四属性分别完成后的漂移。保留全部样本，不按答对筛图；s2025不跑，新种子完成后补入。执行 agent 做必要实现和简短小验证后串行运行，回写报告与关键现象；不扩大为 C+ 训练或人工注错矩阵。
- [x] [claude] 先修 R1（early 资格键统一）和 R3（selection 搜索/保存预算一致），再生成 manifest；验收含“一边合格一边不合格”和换 batch_size 的用例。
- [x] [claude] 修 R2（summary 保留运行身份、完整配对键和分组键），覆盖 A/B 同图、own/teacher、before/after 和输入文件顺序交换。
- [x] [claude] 处理 R7（controls condition/clean/汇总）和 R8（D1/D2/D3 新真值参照与汇总）接口缺口；仅 pipeline smoke。
- [x] [user/gpt] R4–R6/R9 四项原则已于 15:21 获用户认可，见 consensus；H 的选择进入上方已批准的小试跑，具体汇总由执行 agent 实现。
- [ ] [user/claude] 决定是否/何时启动 B Phase 0；当前只做了 4 图 smoke，未跑全量。
- [ ] [claude] 实现缺陷修复且用户批准后先跑 B Phase 0（A′ s42 + B-M6 s42 的 common-complete manifest），报告合格 n；保留完整原始轨迹。
- [ ] [claude] Phase 0 合格数足够后再跑 Phase 1（early 2×2、late R-only/U-only、late 四格、unrelated），单卡串行。
- [ ] [claude] Phase 2 sustained/switch；Phase 3 训练腐蚀审计；局部自洽仍不跑。
- [x] [claude/gpt] seed2026/e143 六档能力数据已于b1370b7/a62732c补入并在2026-11-15核对；原训练中状态为历史快照。仍不新增s2025评测。
- [ ] [user/claude] 决定是否/何时重新设计局部自洽错误控制。

## 阻塞 / 等待

- B 全量实验：B1 完成；B2 的 M0 选样和 M5 审计完成；主汇总 `b_main_summary.py` 已实现、D1/D2/D3 汇总和主报告仍待补。等待用户对 M1 正式注错启动点头；C 已批准的能力探索无需等待此项。
- 局部自洽控制：构造可能不自洽，DEFERRED / NOT RUN。

## 最近完成

- [2026-11-14 18:45][claude] B2 M0 选样 + M5 审计完成（均无注错）：256/256 common-complete、365 chains；teacher64 64 图/92 chains，long32 32 图/49 chains。审计 65,536 candidates / 161,725 actual 全部 before=EMPTY，before_numeric=0，4→5=0。`b_main_summary.py` 已实现并与 pilot 五条件逐值一致；正式注错未启动。
- [2026-11-14 18:12][claude] B1 完成：`b-horizon-pilot.md` §9 + `tools/b_mechanism/b_pilot_analysis.py`；B2 执行单 `b-main-execution-plan.md`。B1 关键：B direct 首轮修回 42/42（各条件），A′ U 0/42、R 3/42；B complete_R 20/360、complete_U 7/323 个节点回正→再错但最终稳定；A′ 无恢复。B 主实验未启动。
- [2026-11-14 17:31][claude] B 长轮数六项 v2 完成：性能版 `24e586c`，串行 7–8 分钟；B 到 H=8 稳定恢复约 99.5–100%、H=16 全部 100%，A′ 全部 never；报告 `docs/records/2026-11-14/b-horizon-pilot.md`，产物 `tmp/analysis/b-horizon-pilot-v2-20261114/`。
- [2026-11-14 17:12][claude] 验证 GPT 性能版：19 项 CPU 测试通过；early A′ 126 条×8 轮、late B 37 条（含 leaf_constant 新真值）、controls B 84 条×4 轮与旧版逐值 0 diff；early 单项 126 条×64 轮约 75 s，训练保持约 6 it/s。
- [2026-11-14 15:05][claude] 从 `origin/reproduction` 更新到 `23553a2`，核对 `c91b6fe`：8 项 CPU 回归通过；GPU 小样本确认 switch 实际注入第二错值一次、D1/D2/D3 常数新真值生效；未启动全量实验。
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

- [2026-11-14 18:45][claude] **B2 M0/M5（非注错结果）**：固定 32–287 的 256 张候选全部 common-complete，365 条 primary chain；teacher64/long32 子集分别为 92/49 chains。训练腐蚀审计 256 batch（2048 图，4.1%）×100 trials：65,536 candidates 与 161,725 次实际替换的 before 全部 EMPTY，before_numeric=0；after numeric 148,820、NA 6,359、EMPTY 6,546，after_is_true 6,392，4→5=0。该样本中 B 的训练 `last` 腐蚀只覆盖 EMPTY→值，不支持把测试 4→5 解释为训练内已填值腐蚀；单 seed/4.1% 样本，不称全训练。
- [2026-11-14 18:12][claude] **B1 pilot 补充（单 seed/42 targets，非最终）**：每个条件 42 个 direct 实际注错；B 首轮全部修回，A′ U 0/42、R 3/42。逐节点全轨迹：B complete_R 20/360、complete_U 7/323 个节点出现回正→再错，最终稳定；after_u/sustained 为 0；A′ 无恢复。A′ R-only 3 个例外均 depth0、parents=[]，但另有 18 个 depth0 r 未修回，结构成因未定。详见报告 §9。
- [2026-11-14 17:31][claude] **B 长轮数 v2（pilot，非最终）**：B-M6 在 after_u U/R、complete U/R、sustained_R 的 all_affected 稳定恢复 H=8 约 0.995–1.000、H=16 全部 1.000，受损节点 first recovery 均在 2–8 轮；A′ 全部 never（direct U 42/42、R 39/42 等持续错）。clean 分支 64 轮无漂移。建议主 H=16 + 64 轮长期稳定性检查；最终 H 待用户/GPT。
- [2026-11-14 16:46][claude] **仅 early 的优化前预览，不是最终结果**：B-M6 after_u U/R 的受损节点在 H=8 时稳定恢复已约 100%；A′ 同期全部 never（direct U 42/42、R 39/42 持续错）。late/sustained 未跑，H 未定；性能修复后六项重跑再正式分析。
- [2026-11-14 14:41][claude] Post-review fix smoke（4 图）：selection 9 chains / 16 sinks / 16 leaves；batch 4 与 8 的 `(T, n_loops)` 一致；early 36 条、late 66 条、controls 9+9 clean；summary 7 行，A′/B 来源分离、输入顺序不变、`duplicate_cleans=0`。**仅 pipeline smoke**。
- [2026-11-14 19:58][claude] **C 首批六档（探索，非最终）**：s42 的 VALUE 完成率 59–64/64，s1234 32–64/64（N128 缺口）；H=64 整图 VALUE 全对率 s42 N32_shallow 47/64 → N128_deep 1/64，s1234 N64/N128 0–7/64。first-4 下降几乎全来自 VARIABLE（H=64 SYNTAX/OPERATION 1.000、VARIABLE 0.08–0.11），本任务 first-4 ≈ `VALUE ∧ VARIABLE`；H=36 已无图 VARIABLE 全对。VALUE 漂移深层优先；s42 完成几乎按 depth+1，s1234 大图长尾/未完成明显。报告 `docs/records/2026-11-14/c0-capability-report.md`。
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
- [2026-11-14 15:05][claude] 服务器核对 `c91b6fe`：`python -I tools/b_mechanism/tests/test_review_regressions.py` 8/8 通过。GPU 小样本（4 图）确认 switch 的 r 输入序列为“第一错值×2 → 第二错值×1 → 释放”；D1/D2/D3 的 `const_pos` 新真值为新常数、旧真值字段保持旧值；summary 对 early/controls 正常配对、`duplicate_cleans=0`。仍只做小样本检查，未启动全量实验。
