# 工作看板

> 最近更新：2026-11-14 18:45 CST · [claude]
> 活看板：任务可以移动；状态变化在 `log.md` 记一行。条目格式 `[YYYY-MM-DD HH:MM][author]`。

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

- [ ] 领取并核实可用的 s42/s1234 权重实际路径/哈希，沿用现有 e143；s2025不测，s2026完成后同协议补入，不等它才开工。
- [ ] 按 [C首批设计单](../docs/records/2026-11-14/c0-capability-exploration-plan.md) 固定六档数据：N32/64/128 × 实际深度≤6 / 7–12，每档约64图；交实际深度/图数摘要，缺档用既有生成器补齐或说明，不按模型答对筛题。
- [ ] 实现必要记录与汇总，做一次简短检查后运行。连续hidden自由反馈64轮；两项节点指标并列，按深度/内部/结果端分解；记录 tV/t4、完全算对后的保持/漂移/恢复。
- [ ] 交付 `docs/records/2026-11-14/c0-capability-report.md`：能力表、过程曲线、典型错误、对C+的启示。产物 `tmp/analysis/c0-capability-20261114/`。本任务不训练C+，不扩展人工注错矩阵。

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
- [ ] [claude] seed2026 完成后补入上方 C 能力探索，使用与 s42/s1234 相同的数据和评估规则；旧 N32/N128 可留历史对照。按用户最新决定，不再新增 s2025 评测。
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
