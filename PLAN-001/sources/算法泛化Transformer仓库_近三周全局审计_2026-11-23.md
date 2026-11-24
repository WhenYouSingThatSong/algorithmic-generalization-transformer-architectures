# 算法泛化 Transformer · 近三周实验台账

**时间：2026 年 11 月 3–23 日。版本：`reproduction@6dbd2c7`。** 本地已与远端同步。按现有记录，11 月 6 日起可以逐项追溯实验。下文按问题和操作列项，同一训练批次的不同测试分别写明；编号用于定位，不能当作独立训练次数。[固定版本](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee)

**模型称呼。** 早期“两层 A”是小模型；后期 A′（也写 A）是与 B 主体架构匹配的离散反馈对照。B 训练时会污染部分数值输入。原始 C 传递连续隐藏状态，有常规求解监督；“后训练 C”另注明训练配方。训练周期指权重训练了多久，推理轮次指拿同一份权重续算了几步。[model-names.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/model-names.md)

**怎样读实验操作。** 题目是一组有先后依赖的模 23 加法。若内部节点 `u` 的正确值是 4，直接子节点计算 `q = (u + 9) mod 23`，则原题里 `q` 应为 13。只把模型当前的 `u` 改成 5 后，下一轮 `q` 可能继续答 13、按这个错值答 14、给出其他数，或暂时留空。记录的是这些具体区别。若改的是**题面常数**并使新题的 `u` 理应为 5，则 14 反而是新题的正确答案。

下文的“快照”指某一轮各位置的模型状态；“从同一快照分叉”指复制该状态，一份保持原样，另一份只做指定操作，再让两份模型各自续算。“供体”指提供替换向量的那次运行。“读出”指用输出头把当前位置的状态解码成字段；连续模型下一轮接收的是隐藏向量。标准求解轮数记为 D，本批通常取图的最大依赖深度加一。

**时间线索引**

- [11 月 6–10 日 · 论文复现与离散模型对照](#stage-1)（1–10）
- [11 月 9–13 日 · 连续基线、训练尝试与父子干预](#stage-2)（11–18）
- [11 月 14–16 日 · 离散状态的干预与连续模型的基础能力](#stage-3)（19–29）
- [11 月 15–16 日 · 第一轮连续模型后训练和压力测试](#stage-4)（30–35）
- [11 月 18–19 日 · 改题、复用旧状态与跨阶段程序](#stage-5)（36–43）
- [11 月 19–21 日 · C 后训练配方与正式评测](#stage-6)（44–55）
- [11 月 21–22 日 · 六项更新与读取实验](#stage-7)（56–61）
- [11 月 22 日 · 同题干预、架构原型与原始 C 基线](#stage-8)（62–70）
- [11 月 23 日 · 第一批同题重放、自然错误和更新标记](#stage-9)（71–75）

<a id="stage-1"></a>

## 11 月 6–10 日 · 论文复现与离散模型对照

**01　六种方法的最小运行验收**

- **实验做法**　给六种方法分别生成能在本地运行的小图，调用各自的模型和损失分支，做一次前向计算、反向传播和参数更新。随后用更新后的权重做有限轮推理，把权重保存并重新载入，检查载入前后的参数是否一致。显式思维链沿用自己的数据接口；这次验收走的是诊断脚本。
- **大致结果**　六种方法的核心模型与损失分支各完成一次参数更新，权重可以回载。
- **当时的分析**　当时用它排查环境和接口。六种方法的论文准确率需要另行评估。
- **记录与路径**　[six-methods-core-01.json](../../projects/codex/agta%20repro/repro_runs/reports/six-methods-core-01.json) · [check_six_methods.py](tmp/env_setup/repro_setup/scripts/check_six_methods.py)

**02　本地显存与连续更新验收**

- **实验做法**　在同一张 5060 Ti 上依次运行六种微型配置，每种连续做两次参数更新。脚本一边记录 PyTorch 在训练期间分配和预留的显存峰值，一边约每 200 毫秒采样整张卡的显存占用。前者可对应具体方法，后者还包含驱动和其他程序，报告因此分别保留两套读数。
- **大致结果**　留下六方法运行记录和独立显存采样，可查每种微型配置实际跑到了哪一步。
- **当时的分析**　当时用来划分本地能验证的流程和需要服务器的正式训练；这批显存读数只对应诊断配置。
- **记录与路径**　[six-methods-memory-20261106T082947_499701Z.json](../../projects/codex/agta%20repro/repro_runs/reports/six-methods-memory-20261106T082947_499701Z.json) · [audit_gpu_memory.py](tmp/env_setup/repro_setup/scripts/audit_gpu_memory.py)

**03　两层离散 A 的训练与规模评测**

- **实验做法**　用固定 32 个节点、最大依赖深度 6 的模加法图训练两层无纠错离散模型，保留不同随机种子和保存点。评估时从题面给定的常数出发，每轮把模型自己的预测送回下一轮，依次测 8 至 128 个节点的规模档；另用一批深度接近训练集的 32 节点题做参照。每道题检查所有非填充位置的四个计分字段。
- **大致结果**　整题成绩随评估设置和种子变化。
- **当时的分析**　当时先固定权重和题目排查评估路径；整题按四字段同时正确计分，比只看数值严格。
- **记录与路径**　[递归潜空间推理论文复现_阶段实验报告_修订版.md](递归潜空间推理论文复现_阶段实验报告_修订版.md) · [A-M5-Nodes32-ADD-L2H16D256-DeBERTa](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/experiments/configs/A-M5-Nodes32-ADD-L2H16D256-DeBERTa)

**04　离散辅助字段回灌对照**

- **实验做法**　同一道题、同一份权重从相同输入跑两遍。两遍均把模型预测的语法、变量名、运算和数值送回下一轮；区别在于是否把另两个辅助字段 REGRET、MODIFICATION 的预测也送回。历史 clean 每轮先将这两项置为 FALSE，raw 保留预测。两遍用相同标准给整题计分。
- **大致结果**　重置后部分两层 A 权重的成绩大幅回升，各权重敏感性不同。
- **当时的分析**　11 月 16 日回查原始代码，确认模型内续算与独立评估的处理确有差别。论文图表实际走哪条路径仍缺运行证据。
- **记录与路径**　[递归潜空间推理论文复现_阶段实验报告_修订版.md](递归潜空间推理论文复现_阶段实验报告_修订版.md) · [raw-aux-bug-review.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-16/raw-aux-bug-review.md)

**05　规模与深度压力测试**

- **实验做法**　固定每题 128 个节点，另造普通生成器较少产生的依赖形状。比如长链题让一个中间结果接着供下一个算式使用，浅宽题让许多算式在相近深度引用已有结果。先用小池确认构造，再在扩大后的题池评测已有权重；记录实际深度、层宽和整题是否完成，以便和常规 128 节点档对照。
- **大致结果**　部分在标准集接近满分的权重在压力池失分，受测的一份 B 仍能完成这批题。
- **当时的分析**　当时注意到常规扫档会同时改变规模和允许深度，才另造压力池并记录实际图深度。
- **记录与路径**　[递归潜空间推理论文复现_阶段实验报告_修订版.md](递归潜空间推理论文复现_阶段实验报告_修订版.md) · [递归潜空间推理论文复现_数据与结论.md](递归潜空间推理论文复现_数据与结论.md)

**06　B 的保存点筛查与外部复现权重对照**

- **实验做法**　先取本组 B 训练过程中不同时间保存的权重，在固定的 128 节点标准档上比较，再标出哪些权重用于后续压力测试。另核对外部复现权重的训练配置和保存周期，把它们放入相同的 raw、clean 评估。看表时保留权重来源和选模经过，避免把挑选权重的成绩再当成一次新测试。
- **大致结果**　同类配方的权重在 raw 下表现不同，选中的本组权重在标准集上很强。
- **当时的分析**　当时逐份核对训练进度和选模来源；参与选权重的 128 节点标准档不能再充当独立验证。
- **记录与路径**　[递归潜空间推理论文复现_数据与结论.md](递归潜空间推理论文复现_数据与结论.md) · [c0-mixedn-full-context.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L81)

**07　四层模型、快速训练路径与输出分数间隔**

- **实验做法**　加入四层、无专门数值污染训练、使用快速训练路径的离散模型，与两层 A 和带污染训练的 B 对照。在固定 128 节点题上，除了看整题分数，还逐位置查辅助字段是否翻转、数值头第一名比第二名高多少，以及回灌方式改变后第一名有没有换。这样能看见分数已受影响、最终取值却相同的位置。
- **大致结果**　无专门纠错训练的权重也能在 raw 下取得高分；部分分数发生变化，第一名并未换位。
- **当时的分析**　当时曾用“模型自行纠错”解释 raw 满分；消融和分数间隔对照削弱了这一解释，后续改为直接注错。
- **记录与路径**　[递归潜空间推理论文复现_阶段实验报告_修订版.md](递归潜空间推理论文复现_阶段实验报告_修订版.md) · [c0-mixedn-full-context.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L167)

**08　快慢训练路径的单批梯度核对**

- **实验做法**　在单个批次上关闭 dropout，让快、慢训练路径面对相同的教师输入，再比较它们产生的梯度。慢路径按轮计算并累加损失，快路径把不同轮次合进一个批次。回查脚本时还逐项核对了首轮数值损失是否加倍、跨轮取和还是取平均，因为这两处都会改变被比较的梯度尺度。
- **大致结果**　两条路径用了不同的求和或平均方式，旧梯度差值混入尺度差异。
- **当时的分析**　当时的复查指出应先对齐归一化；该单批检查无法估计完整训练后准确率的差距。
- **记录与路径**　[递归潜空间推理论文复现_阶段实验报告_修订版.md](递归潜空间推理论文复现_阶段实验报告_修订版.md) · [model.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/experiments/model.py)

**09　同架构无纠错 A′ 的三种子训练与评测**

- **实验做法**　按 B 的四层、16 头、隐藏宽度 384 和快速教师输入路径训练三份无纠错权重，分别保存并执行标准节点数扫档和额外的深图测试。这样得到的 A′ 与 B 主体架构一致，可先核对它能否从空白解题，再用同一位置的注错测试它是否愿意改写已有数值。
- **大致结果**　几份 A′ 在干净评估下已能解好大部分受测题。
- **当时的分析**　当时把 A′ 留作直接注错的对照，避免用较小的两层 A 和 B 比较时把结构差异算作训练效果。
- **记录与路径**　[all-experiments-synthesis.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-10/all-experiments-synthesis.md) · [A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/experiments/configs/A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L133)

**10　离散单点注错与扩样**

- **实验做法**　在一个已算对且有直接子节点的内部位置，只改一次数值字段，其余字段和题面照旧。小试限定真值 4、写入 5；扩样改为把实际真值加一后取模 23。分别从教师给定的正确中间状态、模型自己算到的合格状态起跑，保留不改值的同题分支，在注错后第 1、2、4 轮检查目标和直接子节点。
- **大致结果**　受测内部目标上，A′ 常保留错值，B 较常改回；子节点可以晚于目标回正。
- **当时的分析**　当时提出“检测到了错误”和“按父值重新计算”两种解释。11 月 13 日修正教师状态的轮次索引，旧首轮统计已被重跑替代。
- **记录与路径**　[all-experiments-synthesis.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-10/all-experiments-synthesis.md) · [code-review-fixes-20261113.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-13/code-review-fixes-20261113.md) · [inject_value_discrete_v2.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/scripts/inject_value_discrete_v2.py) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L194)


<a id="stage-2"></a>

## 11 月 9–13 日 · 连续基线、训练尝试与父子干预

**11　固定规模原始 C 的训练与检查点评测**

- **实验做法**　原始 C 在 32 节点图上学习逐轮求解，下一轮接收上一轮的隐藏向量，关闭专门的数值污染训练。先比较不同种子与训练保存点，再让各权重分别做常规规模档、独立留出题、深度接近训练集的 32 节点题及深图压力题。这样可以追查同一权重的失分究竟出现在哪类输入上。
- **大致结果**　早期不同种子、保存点和题档的严格整图成绩差异很大。
- **当时的分析**　当时分别查基础拟合、题目分布和评分时刻。后续主评测选定具体权重；早期种子 2025 与后补的 2026 是不同运行。
- **记录与路径**　[c0-mixedn-full-context.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L314) · [project-status-overview.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-16/project-status-overview.md)

**12　原始 C 自然续算与失分分解**

- **实验做法**　取旧 32 节点集中的 256 张图，无人工注错，也不喂教师中间答案，让每份 C 自由运行 10–14 轮。对同一个节点按时间排列读数，区分一直正确、先错后对、先对后错；再把末轮整图判错拆回各字段，并核对子节点答错时，那个数字能否由父节点之前的错数算出。
- **大致结果**　有的权重先出现变量名读数偏移，有的出现数值由对转错并伴随后续算式变化。
- **当时的分析**　当时按种子拆解整图指标，分别提出数值保持和变量身份问题；变量名读错还需进一步验证是否影响下游取值。
- **记录与路径**　[all-experiments-synthesis.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-10/all-experiments-synthesis.md) · [c0-mixedn-full-context.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L437)

**13　沿数值输出头方向改连续向量**

- **实验做法**　先选原始 C 将内部节点读成 4 的时刻，再沿数值头中“5 的权重减去 4 的权重”的方向改动该位置的隐藏向量。立即用原输出头检查是否读成 5，再让模型自由运行，与未改向量的同题轨迹比较。观察点有两个：目标下一轮还读不读成 5，直接子节点有没有算出把 5 代入其算式的答案。
- **大致结果**　注入时通常能翻转读数，下轮容易回到原值，子节点未给出预定的错值结果。
- **当时的分析**　当时怀疑这种人工方向主要改变了读出，或产生了模型平时不会遇到的状态，遂改用另一道题的真实向量。
- **记录与路径**　[c0-mixedn-full-context.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L292) · [day-summary.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/day-summary.md)

**14　反事实题的连续向量互换（早期 C4）**

- **实验做法**　从同一张图派生两道题，只改某个叶子常数，使中间目标在一题应为 4、另一题应为 5。两题各自运行到相同轮次，交换目标位置的整行隐藏向量，其余位置留在各自原题。除双向互换，还保留不换、随机打乱向量分量和加噪声的分支，比较下一轮直接子节点具体给出了哪个数。
- **大致结果**　两份旧 C 的子节点在许多合格题上给出了换入值对应的答案，随机对照较少重现这个方向。
- **当时的分析**　当时把它作为“该向量确能影响后续算术”的检查；后来补同值供体，因为被交换的是整个向量。
- **记录与路径**　[c0-mixedn-full-context.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L497) · [c-source-control.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-source-control.md)

**15　父节点与目标节点四条件干预**

- **实验做法**　沿同一条依赖链固定父节点和目标，取目标尚待计算、目标已经算对两种起点。每种起点各做四次：保持原样、只改目标、只改父值、父与目标都改。继续自由运行时，对照原题答案、人工写入的错数以及按当前错父值重新计算会得到的数，逐轮判断目标和子节点分别落在哪一种情况。
- **大致结果**　父错、目标原本正确时，B 也会写出与错父相符的数；A′ 对待填位置和已有值的处理不同。
- **当时的分析**　按父值重算成为候选解释。13 日发现旧脚本清辅助字段时误写两行，父节点实验随后重跑。
- **记录与路径**　[parent-2x2-finding.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-10/parent-2x2-finding.md) · [code-review-fixes-20261113.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-13/code-review-fixes-20261113.md) · [parent_corruption_2x2.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/scripts/parent_corruption_2x2.py)

**16　限定错值与时机的复验（P1）**

- **实验做法**　限定原题目标的正确答案为 4。在一种起点让模型先写出 4 再改为 5；另一种在同一类位置原本还空着时提前填 5。结合只改父值、只改目标等分支重放，在旧题集和独立留出题上分别筛选合格目标，比较第一步及随后几轮的读数。轮次、题集和起点均可从修复后的重跑记录追溯。
- **大致结果**　B 在不少条件下能改回，部分权重仍有残留；两个题集入选的目标不同。
- **当时的分析**　当时查的是先前结果是否依赖特定错数、干预时机或题集，修订记录分别保留了两个题集的结果。
- **记录与路径**　[p1-pilot-20261113.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-13/p1-pilot-20261113.md) · [code-review-fixes-20261113.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-13/code-review-fixes-20261113.md)

**17　混合节点数训练的首版、重建与续训**

- **实验做法**　首版将不同节点数的图混在一起，启动后检查实际深度直方图，发现部分目标深度几乎没有题，便在首个训练周期前停下。第二版先生成恰好达到指定深度的链，再补齐其余节点，使每个可行的“节点数 × 深度”组合都有样本。用重建数据训练多个种子，续训时核对实际保存点，并追踪逐周期损失。
- **大致结果**　首版未完成一个周期。后续运行的验证损失反复跳变；后来查出一批数据的变量词表过小，相关权重作废。
- **当时的分析**　当时并行查数据生成、断点和优化波动，训练波动的原因仍有多种解释。
- **记录与路径**　[c0-mixedn-full-context.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-11/c0-mixedn-full-context.md#L574) · [project-status-overview.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-16/project-status-overview.md)

**18　修正词表、双验证集的混合规模 C**

- **实验做法**　单独重建变量名词表，允许小图使用更大范围的变量名，避免“训练图只有 32 个节点”同时变成“模型只见过 32 个名字”。在混合节点数和深度的新数据上重新训练，一套验证采用混合规模数据，另一套沿用原来的 128 节点验证题，分别记录种子、检查点和验证表现；中期小测还检查评估器能否正确加载权重。此项与前一项废弃的词表运行分别登记。
- **大致结果**　这批保留了训练记录，但阶段清点将其列为基础拟合不足，后续主实验使用固定规模的原始 C。
- **当时的分析**　当时区分了“错误词表导致作废”和“配置正确但拟合不足而冷存”，两批权重不能合称。
- **记录与路径**　[p1-pilot-20261113.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tmp/records/2026-11-13/p1-pilot-20261113.md) · [project-status-overview.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-16/project-status-overview.md) · [C0-M4-MixedN8-32-D12-Vocab128-2Val-L4H16D384-DeBERTa-100ep](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/experiments/configs/C0-M4-MixedN8-32-D12-Vocab128-2Val-L4H16D384-DeBERTa-100ep) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/experiments/configs/C0-M4-MixedN8-32-D12-Vocab128-2Val-L4H16D384-DeBERTa-100ep/data_config.yaml#L40)


<a id="stage-3"></a>

## 11 月 14–16 日 · 离散状态的干预与连续模型的基础能力

**19　A′/B 长观察窗口预试**

- **实验做法**　在 A′、B 都能完成的一小批 32 节点图中，找出“父节点—目标—直接子节点”的依赖链。从模型自己的状态分叉，在目标待算、刚算完或整图完成后注错；另有分支连续数轮维持父值错误再释放。与不注错的同题轨迹逐轮比较，最长看到 64 轮，以判断正式实验需要跟踪多久。
- **大致结果**　B 的许多恢复出现在较早几轮，个别位置首次回正后还会再错。
- **当时的分析**　当时据此选正式观察窗口，并把“第一次回正”和“到窗口结束持续正确”拆成两项。持续错父只在预试范围内有结果。
- **记录与路径**　[b-horizon-pilot.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-14/b-horizon-pilot.md) · [b_long_window.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/b_mechanism/b_long_window.py)

**20　早期注错正式批（B 线 M1）**

- **实验做法**　先固定 256 张候选图，从有合格依赖链的图中取目标。在目标应当首次求值之前和已经算对之后，分别运行不改、只改目标、只改父值、两处都改四个条件，错值用原数加一取模。主批使用模型自己的起点；固定子集改用教师起点，另一子集延长至 64 轮。比较时对齐同图、同节点和同一干预时机。
- **大致结果**　B 有目标改回、下游先受影响再回正的轨迹；不同起点需先确认模型都已算到要求状态。
- **当时的分析**　当时分别核对直接被改的位置、下游、四字段与单独数值，并按图和按节点重算，避免把多个下游当作独立题。
- **记录与路径**　[b-main-m1-report.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-14/b-main-m1-report.md) · [b-main-m1-teacher-vs-own.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-14/b-main-m1-teacher-vs-own.csv)

**21　整图完成后注错正式批（B 线 M2）**

- **实验做法**　沿用 M1 的固定题号和模型，等两者都达到整图完成条件再分叉。这次分别改直接父值、内部目标、无下游的末端、叶节点的值或另一分支。每个受扰分支都与同起点的不改分支并排看 16 轮，固定子集再延长。检查目标、共有后代和分支独有节点，才能判断一个错误到底越过了哪段依赖路径。
- **大致结果**　有些条件完全没有产生可观测的输出损伤；也发现目标已回正、个别下游仍错的案例。
- **当时的分析**　当时回到具体依赖图，区分两个分支共有的后代与另一分支独有的位置；没有受损样本的条件不计算恢复率。
- **记录与路径**　[b-main-m2-report.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-14/b-main-m2-report.md) · [b-review-evidence-audit.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-14/b-review-evidence-audit.md)

**22　B 的训练污染位置抽查**

- **实验做法**　取训练批次，反复执行 B 实际使用的污染采样规则，分别记录掩码选中了哪里、是否真的替换，以及替换前该位置在教师输入中的数值。用“本轮即将首次可算”标记结构位置，再查那一格原来是空还是已有数字。这样可以直接比较训练时模型见过的污染，与评测时故意改写一个已有答案的操作。
- **大致结果**　抽到的实际替换发生在原本为空的待填位置。
- **当时的分析**　这使团队把训练中遇到的情况与评估时“篡改已填答案”区分开。抽样没有逐步回放每一次历史训练。
- **记录与路径**　[project-status-overview.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-16/project-status-overview.md) · [b_corruption_audit.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/b_mechanism/b_corruption_audit.py)

**23　原始 C 六档能力与长程保持**

- **实验做法**　把图按节点数 32、64、128 及浅、深两档交叉分组。每份原始 C 从题面开始自由运行到第 64 轮，逐图保存每个位置的字段读数。先找整图第一次所有数值都对的轮次，再沿同一轨迹检查是否保持，并分别统计各轮数值正确和四字段同时正确的情况；种子 2026 按同一协议补交。
- **大致结果**　许多图在窗口内曾算全对，继续运行后又失分；变量名、数值和整图成绩下降的时刻不一致。
- **当时的分析**　当时改查首次完成、完成后保持和固定轮次三种结果。后补的 2026 记录沿用同一评测，未另选最优保存点。
- **记录与路径**　[c0-capability-report.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-14/c0-capability-report.md) · [c0-capability-s2026.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c0-capability-s2026.md) · [artifacts](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/ai-collab/artifacts)

**24　A′/B 已填位置的改数与清空**

- **实验做法**　在两模型均可比较的图上选有直接下游的目标。一次从目标刚算对时开始，一次等整图完成；从每个快照分别不动、写回原数、写入原数加一或加三的错数、清成空位。只操作一次目标的数值字段，随后自由运行 16 轮，分别核对目标能否重新填对、原本正确的下游是否被带错。
- **大致结果**　A′ 在所测目标常留下已有错数，清空后却能填对；B 在两种操作下都能重新给出正确值。
- **当时的分析**　当时把问题收窄为“已有答案如何更新”，并发现首批选样偏向单父节点，遂安排多父节点补测。
- **记录与路径**　[b-value-update.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/b-value-update.md) · [run_b.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/repair_study/run_b.py)

**25　多父节点、多错值与残留链回查**

- **实验做法**　把目标按有一个、两个或三个数值父节点分层，在目标刚算对和整图已完成时分别运行。对同一目标换入不同的模加偏移，另查清空与正确写回对照，比较目标和全部后代。对窗口结束仍有下游错误的五个案例，回到逐轮记录，核对直接子节点出错时各父输入当时实际读出的数。
- **大致结果**　A′/B 在目标上的差别延续到多父节点题；B 的少数案例中，目标已改回而直接子节点继续出错。
- **当时的分析**　16 日的逐链复核发现，部分残留出现时父输入的可读数值已经正确；只说“纠正还没传播到”解释不了这些记录。
- **记录与路径**　[b-multiparent.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/b-multiparent.md) · [b-multiparent-residual-summary.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/b-multiparent-residual-summary.csv)

**26　原始 C 自然错误现场的单处替换**

- **实验做法**　先扫描两份原始 C 的六档自由续算轨迹，选出数值错误但父输入正确、数值错误且父输入也错、仅变量名错等事件，并给错误现场配健康对照。复现当时的状态后，一次只恢复目标向量，或恢复直接父节点向量，另保留原样续算。把替换当下、下一次模型计算后、16 轮窗口末的读数分开记录。
- **大致结果**　换回目标后可立即读对，续算后仍可能回到旧错；只恢复父状态的效果也不一致。
- **当时的分析**　当时把“刚写入后读对”“自主运行一步后读对”和“末段保持”分开，按自然错误类别报告。
- **记录与路径**　[c-local-repair.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-local-repair.md) · [run_c.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/repair_study/run_c.py)

**27　原始 C 同值与异值来源对照**

- **实验做法**　给同一目标构造多份来源题，改动叶子常数，使目标分别读出原数、原数加一或加三。将来源题在同位置、对应轮次的整行隐藏向量移入原题，其余位置保持原样；另设不换和原样写回对照。根据原题中子节点的其余父值，单独算出预期受扰答案，再与子节点下一步实际输出比较。
- **大致结果**　同值来源大多保住原答案；异值来源更常把子节点推向按该值计算的结果。
- **当时的分析**　此轮补的是早期向量互换的缺口：把“换过向量”和“换入了不同数值”分开。不同来源还可能携带其他差异。
- **记录与路径**　[c-source-control.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-source-control.md) · [run.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/counterfactual_control/run.py)

**28　原始 C 在不同求解阶段换入同一类状态**

- **实验做法**　把第 27 项的同值与异值替换移到三个起点：依赖深度预定的早期、这张图首次全部数值正确时、以及完成后再运行 16 轮。每次先检查接收目标、供体和直接子节点是否合格。分别看每个时点自己的合格题，再取三个时点都合格的子集，比较换入同一类来源后下游的第一步和后续变化。
- **大致结果**　异值来源在早期更常改变直接子节点；较晚时，下游有时保留旧答案，干净对照也会漂移。
- **当时的分析**　当时按阶段和实际入选资格重算。预定早期不一定是目标第一次算对，晚期也不能默认整图仍然正确。
- **记录与路径**　[c-stages.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-stages.md) · [stages.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/counterfactual_control/stages.py)

**29　目标与全部直接父节点联合替换**

- **实验做法**　沿用第 26 项已经选好的事件、供体和健康对照。在同一自然错误现场同时换回目标以及所有直接父节点的正确来源向量，随后完全释放，让模型自由续算 16 轮。把它与只换目标、只换父节点、保持原状的结果并排，并做整组向量原样写回检查，以排除替换动作本身造成的差异。
- **大致结果**　联合替换能改变下一步读数，窗口末段的总体改善没有延续；健康对照也有读数变化。
- **当时的分析**　当时将它与单处替换并排，判断扩大替换范围能否带来持续收益，按错误现场和健康现场分别看。
- **记录与路径**　[c-joint-repair.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-joint-repair.md) · [run_c_joint.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/repair_study/run_c_joint.py)


<a id="stage-4"></a>

## 11 月 15–16 日 · 第一轮连续模型后训练和压力测试

**30　延长求解监督的后训练**

- **实验做法**　从同一份原始 C（2026，第 143 周期）出发，用一批新生成的 32 节点图做三组后训练。若正常求解需要 D 轮，三组分别训练 D、2D、3D 轮；后两组在额外轮次继续监督已完成的正确状态。一套比较固定每张图训练了几遍，另一套调整遍数以对齐累计循环计算量，再在相同验证图上观察完成后的保持时间。
- **大致结果**　部分延长监督版本在干净题上更久保持数值正确，变量名仍可漂移。
- **当时的分析**　当时把这作为“多监督几轮能否改善保持”的试验；后续污染训练需与仅做保持训练的版本比较。
- **记录与路径**　[c-long-pilot.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-long-pilot.md) · [c-long-pilot-budget.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-long-pilot-budget.csv)

**31　仅保持、随机噪声、语义污染的后训练**

- **实验做法**　三组都从原始 C 2026 出发，在同一批图上训练相同轮数与遍数，按同一计划选择有父、有子的位置。保持组不改向量；噪声组加随机方向扰动；语义污染组移入配对改题产生的错值向量。三组仍以原题正确状态为监督。训练和评估时还核对向量改了多少，以及目标的数值读出有没有真的被改错。
- **大致结果**　语义污染版本在部分一次性干预中降低了子节点即时损伤，但未显示一致的持续恢复增益。
- **当时的分析**　当时同时看实际注入强度、目标和子节点；这些是同一基础种子的配方对照，不当作多种子排名。
- **记录与路径**　[c-corruption-pilot.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-corruption-pilot.md) · [c-corruption-training-dose.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-corruption-training-dose.csv)

**32　后训练权重的干净保持评测**

- **实验做法**　固定原始 C 及仅保持、噪声、语义污染三个后训练版本，让四者在相同六档题上从空白求解，全程不施加人工干预。对每张图先找实际首次完成的时刻，再跟踪标准求解后额外续算的读数。按节点值、四字段和整图分别保存结果，为之后每种权重自己的受扰轨迹提供干净参照。
- **大致结果**　标准求解和额外续算给出的比较不同，训练版本之间的差异在长窗口内更明显。
- **当时的分析**　当时先建立各权重自己的干净轨迹，给后续计算“注错额外造成的损伤”做参照。
- **记录与路径**　[c-basic-evaluation.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-basic-evaluation.md) · [run_clean.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/c_evaluation_suite/run_clean.py)

**33　父错、子错、父子同错的配对评测**

- **实验做法**　在每张验证图上固定一条“父节点 u → 直接子节点 q”的边，从同一快照分别只换错 u、只换错 q、两者都换错，另设同值替换和不换等对照。自由运行 16 轮。只改父时看原本正确的子答案是否保住；直接改子时看它何时重新算对；同时改两处则检查父也错时子节点怎样恢复。
- **大致结果**　语义后训练在部分条件下更能保住原本正确的子节点；直接改错子节点后，首轮恢复未呈现同样优势。
- **当时的分析**　当时把“抵御上游扰动”和“重新获得已经丢失的答案”分开，另按同题联合事件复核。
- **记录与路径**　[c-basic-evaluation.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-basic-evaluation.md) · [run_pair.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/c_evaluation_suite/run_pair.py)

**34　持续强制错父值及释放**

- **实验做法**　同一父位置连续两轮或四轮被强制写入错态，每一轮都取该绝对时点对应的供体向量。部分分支在第一轮也改错子节点一次，之后不再写子节点。约束解除后整张图自由续算 16 轮；分别记录强制期间子节点是否采用错值、释放后哪一轮回正，并与自由运行和同值供体分支对照。
- **大致结果**　一次注错时部分保住的子节点，在反复强制下仍会出错；释放后语义训练组也未持续领先。
- **当时的分析**　当时用它检验一次性干预的优势能维持多久，并逐轮核实错误确实写进了模型输入。
- **记录与路径**　[c-stress.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-stress.md) · [c-stress-sustained-events.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-stress-sustained-events.csv)

**35　随机噪声剂量校准**

- **实验做法**　在固定的验证图和目标位置上，先量出语义替换的向量距离，再按这个距离的 0.5 至 6 倍设置六档噪声。每图取两个固定随机方向，沿同一方向逐档改变幅度，同时保留同值及语义错值替换参照。分别查注入当下目标是否由对变错、子节点首轮有没有受损，以及后代随后几轮累计受了多少影响。
- **大致结果**　增大噪声能造成延迟和持续损伤，向量距离相近的两种扰动也会造成不同下游后果。
- **当时的分析**　当时认为只匹配向量距离不足以定义“同等难度”，因而并列实际读数改变和任务损伤。
- **记录与路径**　[c-stress.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-stress.md) · [c-stress-noise-dose.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-15/c-stress-noise-dose.csv)


<a id="stage-5"></a>

## 11 月 18–19 日 · 改题、复用旧状态与跨阶段程序

**36　C 父子状态的输入轮次对齐（已有轨迹重分析）**

- **实验做法**　取已有的单次父子注错、持续错父和噪声实验记录，按题号、依赖边和推理轮次重新整理。对每次子节点算错，先查本轮计算开始前父和子各读出什么，再看本轮结束后父和子各输出什么。进一步筛选其他父值也正确的事件，逐边统计仍有多少次子输出错。整个分析复用旧轨迹，没有重新运行模型。
- **大致结果**　许多看似“父已对、子仍错”的记录中，父的输入当时仍错；严格筛完仍有少量残留。
- **当时的分析**　当时意识到同轮的两个输出不能直接解释谁读了谁，遂补看子节点先前状态与其他父值。
- **记录与路径**　[priority-function-audit.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-18/priority-function-audit.md) · [function-audit-v2](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-18/function-audit-v2)

**37　A/B 合法重新赋值**

- **实验做法**　选 A、B 都已算完的题，每题挑一个有下游的叶子，改写题面给这个叶子赋的常数，按新常数重算全图真值。一条分支保留旧题所有已填答案继续运行；另一条拿改好的新题从空白求解；还有每轮把新常数固定住的分支。分别看叶子、答案应随之改变的下游和真值不变的节点，并和只损坏内部状态的分支比较。
- **大致结果**　两者从空白大多能解新题；保留旧状态时 A 常保留旧答案，B 较常更新。
- **当时的分析**　此前是把内部数值改错，这次是题面真的变了。当时用从空白解新题的分支排查基础求解能力。
- **记录与路径**　[priority-reassignment.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-18/priority-reassignment.md) · [run_reassignment.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/priority_pilots/run_reassignment.py)

**38　A/B 连续多次改同一输入**

- **实验做法**　从已完成的旧题出发，按两种顺序连续改题：一种多次改同一个输入，另一种交替改两个输入，最后都回到原题。保留状态的分支每次从上一阶段的实际末态继续，哪怕上一步做错；清除结果、固定常数和从空白重算另作对照。每个阶段重新计算本题真值，分开记录仍旧答对、接受了新值和额外受损的位置。
- **大致结果**　B 多次接受新目标，个别下游仍留下损伤；A 回到原题时可能因一直保留旧答案而答对。
- **当时的分析**　当时分别按每个阶段的新真值计分，专门区分“接受了新值”和“旧答案碰巧又正确”。
- **记录与路径**　[repeat-delivery-report.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/ab-state-reuse/repeat-delivery-report.md) · [ab-state-reuse.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/ab-state-reuse.md)

**39　A/B 复用两个已释放的结果位置**

- **实验做法**　找出原题中两个已经没有旧消费者的末端位置 a、b。先把 a 的位置交给一道新算式，再把 b 的位置交给第二道新算式，且第二道必须用到 a 刚得到的新结果。比较保留旧位置内容、先清再写、整道新题从空白重算等起点。这样能分别检查第一个旧槽能否覆盖，以及它写入的新结果能否被第二步使用。
- **大致结果**　带旧状态时，A 常留着旧槽值，B 较常写入新答案；清空会改变两者的差距。
- **当时的分析**　当时查增益来自“能写入旧位置”还是“能算出新题”，并保留旧题预计算的成本。
- **记录与路径**　[reuse-delivery-report.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/ab-state-reuse/reuse-delivery-report.md) · [ab-state-reuse.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/ab-state-reuse.md)

**40　C 修回的状态还能否供下游使用**

- **实验做法**　固定父节点 u、子节点 q 及出现故障的时刻，交叉改变两件事。整图起点有三种：原故障状态、仅把 u 换回干净向量、整图取同轮干净状态。在每种起点又分别保留 q、换入 q 还为空时的向量、换入 q 已经算出时的向量，另做原样写回对照。随后比较 q 的数值及四字段，看差别来自接收环境还是 q 的起始内容。
- **大致结果**　同一上游读数可以伴随不同下游结果；在这批条件中，较强噪声留下了更持久的数值损伤。
- **当时的分析**　当时把上游提供什么与下游原有状态分开看，并指出强噪声和语义扰动的实际剂量不同。
- **记录与路径**　[c-functional-reuse.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/c-state-reuse/c-functional-reuse.md) · [run_c_function.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/state_reuse/run_c_function.py)

**41　A/B/C 共用的多阶段赋值任务**

- **实验做法**　让 A、B 和三份原始 C 做同一批四阶段赋值题，逐次改变输入，最后回到原题。每阶段分别沿用上一状态、重置受影响的结果位置、或从本阶段输入空白求解；另设不改题而续算同样久的分支。离散模型重置数值字段，C 重置整行向量。先核对各模型起始四字段是否合格，再分别汇总全部候选和共同合格题。
- **大致结果**　B 在这批携带旧状态的任务上较常跟上新值；C 即使不改题也会退化，初始共同合格题少于候选题。
- **当时的分析**　当时将长时保持与改题造成的困难拆开，并注明离散模型清数值、连续模型重置整行隐藏状态，动作不同。
- **记录与路径**　[abc-workspace.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/abc-workspace.md) · [abc-coverage.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/abc-workspace/abc-coverage.csv)

**42　固定工作空间的长程序**

- **实验做法**　把一段长程序拆成 20 个顺序窗口，每个窗口只摆下 32 个位置。一个窗口算完，取模型实际提交的数作为下个窗口的输入；即使某分支选择在新窗口内从空白解，也要接收本分支上一窗口交来的数。逐窗记录是否正确、何时首次交错、非数值提交是否阻断后续，同时按原程序真值与实际输入应得结果分别评分。
- **大致结果**　在受测的保留状态条件下，B 较常完成全程序，A 和几份原始 C 多在中途失败；有些 C 提交了非数值。
- **当时的分析**　当时分别按原程序真值和实际继承输入计分，并追查首次失败。C 跨窗口传递的是已提交数值，窗口内才是连续反馈。
- **记录与路径**　[abc-workspace.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/abc-workspace.md) · [abc-programs.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/abc-workspace/abc-programs.csv)

**43　同一窗口边界的重置诊断**

- **实验做法**　从保留旧状态的轨迹中截取一个窗口边界，复制完全相同的现场和已提交输入。给各副本同样多的额外轮数，分别刷新题面、只重置结果、两者同时做，或拿该实际输入从空白求解。逐节点对照原分支，数清原先答错而修回的，以及原先答对却改坏的，借此排除各分支先前拿到的输入不同。
- **大致结果**　A 对清旧结果较敏感；部分 C 在同时刷新题面与重置结果时修回较多，单独处理也可能损伤健康位置。
- **当时的分析**　当时用固定起点排除“前一窗口已提交不同输入”的混杂，并把修回和改坏分别计数。
- **记录与路径**　[abc-workspace.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/abc-workspace.md) · [abc-probe-pairs.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-19/abc-workspace/abc-probe-pairs.csv)


<a id="stage-6"></a>

## 11 月 19–21 日 · C 后训练配方与正式评测

**44　只在新可算位置污染的普通纠错后训练**

- **实验做法**　从 42、1234、2026 各自的原始 C 权重出发，在原来 5 万张 32 节点训练图上分别再训练 50 个周期。按图结构选本轮刚可计算的位置，在抽中的批次和位置上，移入当前模型从配对改题得到的连续向量，标签仍指向原题的正确状态。记录污染是否实际改变读数，分别保存三个终点，供后续同源权重对照。
- **大致结果**　三份训练均有终点交付；开发评测中部分权重在额外续算后仍能保持较好的数值表现。
- **当时的分析**　当时先冻结保存点再交给独立评测；是否改善纠错，要在同一来源种子、同一题和相同时机上比较。
- **记录与路径**　[c-update-training-handoff.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-20/c-update-training-handoff.md) · [c-update-training-handoff.json](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-20/c-update-training-handoff.json) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/c_update/README.md#L25)

**45　前沿污染加第二层更新监督**

- **实验做法**　同样从原始 C 2026 出发，用新可算位置的污染训练，但在第二层加一个预测“这个位置本轮应当更新”的读出头，并训练末层计算和计划与实际改动的对齐。推理时所有位置照常计算。训练后用固定开发图检查第二层标记、末层数值和额外续算时的字段；另把向量送给冻结的原始 C，检查其后续输出。
- **大致结果**　正常求解窗口仍有较高数值正确率，延长续算后变量名更容易漂移。
- **当时的分析**　当时把中间层“标记应更新”和末层实际改变分开；也因此提出扩大训练时点，单独测试已填位置。
- **记录与路径**　[c-update-small-test.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-20/c-update-small-test.md) · [c-update-training-handoff.json](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-20/c-update-training-handoff.json) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/c_update/README.md#L54)

**46　多时点纠错加第二层监督，延长窗口版本**

- **实验做法**　以原始 C 2026 为起点，把污染时机从新可算位置扩到求解中途及较晚的预定轮次，部分训练批次延长自由续算。监督第二层标出应更新位置，并让末层给出相应结果。记录写入前接收位置究竟为空、正确还是已错；较晚时点按预算安排，未要求整张图届时已正确。沿训练周期在固定开发题上比较干净求解与额外续算。
- **大致结果**　训练结束，但终点在标准轮的干净整图表现明显低于早期保存点。
- **当时的分析**　当时先记录基础求解退化。该批同时改动时点、监督和轮数，单凭这次运行难分出哪项造成变化。
- **记录与路径**　[c-multitime-final.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-multitime-final.md) · [c-multitime-final.json](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-multitime-final.json) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-20/c-identity-and-multitime-design.md#L43)

**47　多时点纠错加第二层监督，正常窗口三种子**

- **实验做法**　用同一份训练图和位置计划，从三个来源种子的原始 C 分别启动，重新初始化各自的第二层更新预测头，每份训练 50 个周期。保留新可算、求解中和完成后注错，但用正常求解预算；要进入“完成后”条件，必须先实际观察到当时整图的数值全对。保留未满足条件的机会，并用固定开发图逐周期记录三份权重的基础拟合。
- **大致结果**　三份都有训练终点；42、1234 的终点基础拟合较好，2026 较弱，42 的中途表现也曾下降。
- **当时的分析**　当时用固定开发图追踪拟合，再选 42、1234 做正式评测。三份终点与两份正式结果是不同范围。
- **记录与路径**　[c-local-update-finals.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-local-update-finals.md) · [fit-history](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/fit-history)

**48　正式六档评测①：未注错求解与长程保持**

- **实验做法**　固定种子 42、1234 正常窗口后训练第 50 周期的权重。将 32、64、128 节点各分为浅图和深图，让两份模型做相同未注错的题；先在按每图最大依赖深度给定的标准轮数评分，再从标准轮数起点额外运行 16 轮。逐图核对数值、变量名和四字段，区分原先没算完、完成后又失分，以及只有身份字段变化。
- **大致结果**　两份权重在标准轮大多能完成；额外续算时，大而深的图更易失分。
- **当时的分析**　这批用开发集预先确定受测权重，2026 只进入训练拟合记录。以下五项共用这次评测，无新增训练。
- **记录与路径**　[c-selected-eval-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-eval-results.md) · [01-基础拟合与干净轨迹.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/selected-tables/01-基础拟合与干净轨迹.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L26)

**49　正式评测②：第二层是否标出应更新的位置**

- **实验做法**　从四种时刻取状态：结构上刚可算、求解中途、首次实际整图完成，以及标准轮数的完成检查。对每个受测位置先查当前父值是否齐、局部算式是否冲突，再看第二层有没有标出它。用已填但与父值冲突、局部一致、父值未齐等位置作为不同组，分别统计应标者中标了多少、已标者中多少确有目标情况。
- **大致结果**　有实际冲突而未被标记的记录；部分条件没有可用于该指标的合格位置。
- **当时的分析**　当时以干预前的状态定义资格，分别看漏标与误标，空分母单独保留。
- **记录与路径**　[c-selected-eval-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-eval-results.md) · [02-第二层更新定位.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/selected-tables/02-第二层更新定位.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L27)

**50　正式评测③：末层实际执行了什么**

- **实验做法**　沿用同一模型、题目、节点和轮次，将第二层标记与末层真正写出的数配起来。为每次写回计算两个参照答案：按这一刻模型可读的父值算应得什么，以及按原题常数和依赖图应得什么。父值被改错时两者可能不同。逐位置记录有无标记、数值动没动、改完分别符合哪个参照。
- **大致结果**　标记与实际改动有不一致的记录；按当前父值算对和符合原题是两项独立可查的结果。
- **当时的分析**　当时用这些列拆分三个问题：该不该改、改动有没有发生、改完是否有利于整题。
- **记录与路径**　[c-selected-eval-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-eval-results.md) · [03-末层执行与任务后果.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/selected-tables/03-末层执行与任务后果.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L28)

**51　正式评测④：注错后的目标与下游**

- **实验做法**　从每个已核实实际写入的受扰快照和同一份干净快照并行续算。逐轮跟踪被改目标、直接子节点、更远后代和无关位置。先看目标下一轮是否回正，再对确实受损的位置追踪首次回正、回正后再错以及第 15、16 轮是否连续正确；同轮干净分支本已出错的位置不能直接算成新增损伤。
- **大致结果**　同一次干预可以同时留下首轮修回和下游额外错误；部分位置还会回正后再错。
- **当时的分析**　当时按实际写入、确有损伤和是否仍有足够后续轮次分别统计，避免把未受损位置算作恢复。
- **记录与路径**　[c-selected-eval-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-eval-results.md) · [04-恢复与传播.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/selected-tables/04-恢复与传播.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L29)

**52　正式评测⑤：同图同时有空位、错值与健康位置**

- **实验做法**　等到整图实际算对，在同一张图的不同位置安排两种操作：一处换回该位置较早的空状态，另一处写入错数。从同一快照跑四条分支：都不动、只清空、只写错、两者一起做。只在四条都有合格起点的同题上检查模型能否在同一轮填空、修错并保住其余健康位置，分别给三类位置和整图计分。
- **大致结果**　混合条件中有局部位置处理正确、整图却没有同时满足要求的题。
- **当时的分析**　当时专门设这一项，检验单一位置上有效的做法能否在一张图中同时成立。
- **记录与路径**　[c-selected-eval-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-eval-results.md) · [05-同图混合.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/selected-tables/05-同图混合.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L30)

**53　正式评测⑥：修回后的新计算与冻结接收模型**

- **实验做法**　做两种功能检查。第一种在原有状态后追加一个引用目标的新算式，改变追加前额外等待的轮数，分开记录目标读数与新下游末两轮的答案。第二种把供体向量移入冻结的原始 C，先核对接收位置是否仍读成预定字段，再看直接子节点有没有算出该值对应的结果。两种检查分别检验本模型后续使用与跨权重传递。
- **大致结果**　有接收位置读到预期字段、下游却没有给出对应结果的组合；部分条件覆盖有限。
- **当时的分析**　当时把向量在原模型中读对、移入另一权重后仍能读对、足以支持新计算分作三步查验。
- **记录与路径**　[c-selected-eval-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-eval-results.md) · [06-修回后的使用.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/selected-tables/06-修回后的使用.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-selected-evaluation.md#L31)

**54　难图：形状、宽深与重复命名**

- **实验做法**　用模加法生成由多条分支组成的程序，成对控制分支宽度、最大深度及工作量；有的分支长短悬殊，有的较均衡。把同一套算式分别写成每次使用新名字、每隔四次或两次定义循环用名的版本，右侧都应读取最近有效定义。在相同题号和结果位置比较标准轮与额外续算；超出 128 次定义的附加档单列。
- **大致结果**　独立命名条件下两份权重标准轮都能完成；频繁复用名字时，节点和整图正确率下降，长短分支不均时差距可更大。
- **当时的分析**　当时要求按同一图形配对，不把重复命名、定义距离与依赖深度一起变化的差异单独归给其中一项。
- **记录与路径**　[c-hard-graphs-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-hard-graphs-results.md) · [paired-naming-summary.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/hard-graphs/paired-naming-summary.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-hard-graphs.md#L12)

**55　难图：上游替换与早期读数**

- **实验做法**　改动同题的一个叶常数，让指定上游及直接下游的正确数都产生可预测的模偏移。用改题同轮的上游向量替换原题同位置，一次只动该处；先核对原位置四字段正确、供体确实给出预定错值且其余字段合格。原题干净分支同步运行，查看上游何时回正、子节点何时首次命中那一特定错数，以及末段两处是否都正确。
- **大致结果**　部分轨迹中，上游下一步已回正，下游仍短暂给出错值对应的答案，后来两处又回正。
- **当时的分析**　当时将“受理了注错”“观察到传播”“传播后有足够轮次看恢复”分别计数，并保留供体不合格的题。
- **记录与路径**　[c-hard-graphs-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-hard-graphs-results.md) · [propagation-summary.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/hard-graphs/propagation-summary.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-hard-graphs.md#L23)


<a id="stage-7"></a>

## 11 月 21–22 日 · 六项更新与读取实验

**56　相邻两轮的状态转移（已有难图轨迹重分析）**

- **实验做法**　重新读取难图实验已保存的每一轮输出，把同一图、同一位置的相邻两轮配成一对。先区分上一轮是真值、错数、空位还是其他非数值，再统计下一轮去了哪一类。若下一轮算对，还向前查这个位置是否曾经正确，拆开“第一次算出来”和“错后回正”。标准求解期和之后 16 轮分别汇总。
- **大致结果**　同一份权重既有错数回正也有正确数变错；还有数值连续正确但变量名改变的记录。
- **当时的分析**　当时用相邻轮区分“首次求解”和“修回旧错”，并把标准求解窗口与额外轮数分开。
- **记录与路径**　[c-research-sequence.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-research-sequence.md#L10) · [contrasts.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-research-sequence/transitions/contrasts.csv)

**57　置信程度、连续相同读数与更新标记**

- **实验做法**　对每个位置记录上一轮数值头排名第一与第二的分数差，以及连续两轮是否读出同一个值（候选包括空值）；再记录下一轮运行到第二层时，模型有没有标记该位置应更新。在相同置信区间、读数稳定性和原先对错状态内，比较标记组与未标记组下一轮是否正确。这里让模型照常算完整轮，仅检查信号的预测作用。
- **大致结果**　可比较分组中，标记组与未标记组的下一轮正确率差值方向不一，部分组合缺少可比样本。
- **当时的分析**　当时把这些量当作预测线索，能否据此安全停止计算还需要真正干预停止策略。
- **记录与路径**　[c-research-sequence.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-research-sequence.md#L11) · [contrasts.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-research-sequence/signals/contrasts.csv)

**58　变量身份替换与两个新读者**

- **实验做法**　在同一道题中选两个变量，让它们分别取相同或不同数值。从交换了两个名字的配对题取得供体，只替换原题中一个定义位置的向量。随后追加两个新算式，各引用其中一个名字并加相同常数；并行测试原样、交换无关名字、只改数值等对照。两变量数值不同时，下游给出的具体数才有助区分它取了哪个变量。
- **大致结果**　某些目标改名条件破坏了原先都正确的两个读者，无关位置改名的对应对照可以保持正确。
- **当时的分析**　当时以实际下游算出的数检查身份变化的后果，同时区分两个变量本来同值和异值的题。
- **记录与路径**　[c-research-sequence.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-research-sequence.md#L12) · [contrasts.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-research-sequence/identity/contrasts.csv)

**59　固定汇合结构，改变受影响分支数**

- **实验做法**　构造两个上游、四个直接子节点、两个汇合点及一个终点。保持节点数和算式形式，改变四个子节点中有几个引用被注错的上游，取一个、两个或四个。一次只在该上游写入具有已知数值偏移的向量，随后跟踪八轮。按每个汇合点实际接入的受扰路径推算特定错数，并与同起点干净轨迹和未暴露分支比较。
- **大致结果**　定向错误既可出现在直接子节点，也可到达终点；受影响的节点轮数和终点曾错的题数记录为不同结果。
- **当时的分析**　当时只在各拓扑共同受理的题上比较，另查没有接到扰动的分支，避免把路径增多直接解释成单次读取更脆弱。
- **记录与路径**　[c-research-sequence.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-research-sequence.md#L13) · [paired-fanout.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-research-sequence/fanout/paired-fanout.csv)

**60　同名变量究竟用了哪个版本**

- **实验做法**　在同一程序里对同名变量先后赋值三次，每次赋值后各留一道读取它的算式，设计成三次正确答案互不相同。复制成算术和位置完全相同、每次使用新名字的对照程序。分别改动只影响第一、第二、第三版或无关位置的输入常数，比较同一读者在改题前后的答案是否跟随某个旧版变化，并改变末次读取的等待间隔。
- **大致结果**　部分可区分新旧值的题里，同名条件的读数随旧版本变化，而独立命名的对照能答对。
- **当时的分析**　当时要求旧、新答案确实不同才判断版本归属，仅凭一次错答无法识别读了哪一版。
- **记录与路径**　[c-reader-continuation.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-reader-continuation.md#L7) · [summary.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-reader-continuation/versions/summary.csv)

**61　上游回正后，旧子节点和新子节点能否使用**

- **实验做法**　先只改上游向量，必须观察到原有子节点给出特定错数，再等到干净、受扰两条轨迹中的上游都重新读对。就在这一步给两条轨迹各追加一道使用同一上游、同一加数的新算式，结果写入新变量。保留不追加的副本，随后逐轮看旧位置能否改正、新位置能否首次算对，并单列没有发生传播或还没回正的题。
- **大致结果**　配对表中既有仅旧子节点正确，也有仅新子节点正确；部分新增子节点起初还没有给出数值。
- **当时的分析**　当时把两者的计算历史、位置和实际可读时点列出来，避免把成功的新计算直接解释为隐藏状态已完全恢复。
- **记录与路径**　[c-reader-continuation.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-reader-continuation.md#L17) · [summary.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-reader-continuation/readers/summary.csv)


<a id="stage-8"></a>

## 11 月 22 日 · 同题干预、架构原型与原始 C 基线

**62　四种模型条件的同题注错**

- **实验做法**　在相同图和预定结构时点，对比种子 42 的四份权重：离散 A′、离散 B、原始 C、仅在新可算位置纠错后训练的 C。用只改一个输入常数的配对题产生预定错值，离散侧仅改目标数值字段，连续侧移入相应整行向量；干净副本同步续算八轮。分别看待算、刚算完和较晚起点的目标更新及直接下游是否采用该错数。
- **大致结果**　原始 C 也出现过目标回正、子节点先按错数计算、后来两处回正的轨迹；后训练改变了出现频率。
- **当时的分析**　原先的问题是纠错后训练带来了什么，因此保留原始 C 作为基线。同期复核要求并列目标与下游结果；该批自然错回放没有合格样本。
- **记录与路径**　[c-attribution-delivery.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-attribution-delivery.md) · [attribution-b-review.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution-b-review.md) · [four_conditions](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-attribution/four_conditions) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/state-repair-attribution.md#L18)

**63　父位置屏蔽**

- **实验做法**　从同一计算状态分三条路，仅在紧接的一次前向计算中改变注意力：照常读取、逐层阻断其他位置读取父节点定义位置、阻断另一分支上同深度的无关位置。分别对干净目标和已换错的目标做这三种操作。先看屏蔽是否会破坏原本能做对的正常计算，再检查目标已有值时的保持或修回有何变化。
- **大致结果**　屏蔽父位置损伤了部分原本正确的计算，匹配的无关位置屏蔽未呈现相同损伤。
- **当时的分析**　当时把它解读为该位置在这些题上有行为作用，尚未定位具体注意力头、其余路径或唯一信息来源。
- **记录与路径**　[c-attribution-delivery.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-attribution-delivery.md) · [key-comparison.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-attribution/pathway/key-comparison.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/state-repair-attribution.md#L19)

**64　目标、父值、祖先与题面来源对照**

- **实验做法**　在原题已形成中间结果后，分别替换目标、父节点、两者、祖先、源常数或另一分支；另把父节点及其整片后代一起换入，使替换边界上只留一条与当前父值冲突的算式。让模型继续算，看各位置的数是跟原题还是跟改题。另从空白独立解原题和改题，作为两套完整且自洽的参照答案。
- **大致结果**　受测权重从空白能解这批改题；从旧状态局部替换后的结果随改动位置而变。
- **当时的分析**　当时用重新求解核对改题本身可解，旧状态是否被成功改写另看干预分支。
- **记录与路径**　[state-repair-attribution.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/state-repair-attribution.md#L20) · [sources](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-attribution/sources)

**65　工作空间中的重新绑定与保存结果读取**

- **实验做法**　把包含 46 次定义的同一段算术写成两份程序：一份每次用新名字，另一份允许同名位置承担一次、两次或四次定义。配对程序保留算式、顺序与依赖，另外改变各次赋值是否交错、隔多久读取、是否须把结果留到末尾再用。分别给立即取值和延后读取的位置计分，并记录旧值实际存活及被覆盖的间隔。
- **大致结果**　反复定义时，复用条件的读数总体较差，保存后再读和立即读的结果也有差异。
- **当时的分析**　当时比较同一节点在哪一边答对，并记录实际绑定距离；多个结构因素同时变化，无法仅由总体差距指定原因。
- **记录与路径**　[c-attribution-delivery.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-attribution-delivery.md) · [graph-counts.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/c-attribution/workspace/graph-counts.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/state-repair-attribution.md#L21)

**66　循环、固定计算量、固定参数量与写出中间结果的架构原型**

- **实验做法**　用相同的小题和逐阶段监督试四种实现：反复用同一计算块、每阶段用独立但同宽的块、缩窄独立块以使参数量更接近、在共享块的每轮把四个预测字段写成后续可读的草稿。各做短训练，以另一组题走推理和同阶段供体替换，记录参数量、序列长度及计算量估算，检查四种流程是否都能跑通。
- **大致结果**　流程跑通，小样本中各组均未完成整图，也没有合格的下游干预题。
- **当时的分析**　交付时将其定为架构对照的流程验收；比较架构优劣所需的基础拟合尚未达到。
- **记录与路径**　[architecture_controls.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/c_hard_graphs/architecture_controls.py) · [architecture-controls](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/architecture-controls) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/state-repair-attribution.md#L38)

**67　跨任务评分器验收**

- **实验做法**　构造两类可由规则程序精确求解的题：十进制逐位加法和带变量赋值的小程序。先用规则生成标准答案和每一步的中间记录；再强制翻转一处进位或改变一次赋值，让规则程序从该步重算后续答案。把正常和强制两套参考输出送进评分器，逐位置核对最终答案与过程记录，验收将来接模型所需的接口。
- **大致结果**　参考答案回放通过，得到了可用于后续实验的计分文件。
- **当时的分析**　当时核验的是任务和评分器，真实模型在这些任务上的迁移表现仍待测。
- **记录与路径**　[transfer_tasks.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/c_hard_graphs/transfer_tasks.py) · [transfer-clean-check.json](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/transfer-clean-check.json) · [transfer-forced-check.json](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/attribution/transfer-forced-check.json)

**68　原始 C 标准场景与自然轨迹基线**

- **实验做法**　取原始 C 的 42、1234 两份第 143 周期权重，在三档节点数、两档深度组成的标准题上选输入节点、链中点和末端。从各模型自己的轨迹分别寻找目标四字段首次正确、目标与子节点均正确、整图数值首次全对的时刻。找到后分出不干预和替换两路，观察 16 轮；找不到保留失败原因，干净轨迹中的自然错误另记。
- **大致结果**　两份原始 C 在一些已算过的目标上可以回正，有些子节点先给出注入错数对应的答案；种子间的时序不同。
- **当时的分析**　当时首先核对用的是未做专门污染后训练的原始权重，再区分预设注错轮次和该模型实际首次算对的轮次。这些数据随后被四模型比较复用。
- **记录与路径**　[original-c-recovery.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/original-c-recovery.md) · [c-only](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/original-c-recovery/c-only) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/original-c-recovery/c-only/README.md#L8)

**69　原始 C 分层注错及第三种子补测**

- **实验做法**　固定同一批短链题，从原题及只改一个源常数的配对题分别取得同轮向量。在目标待算、刚算完及较晚的结构时点，分别改目标、直接父、较早祖先或多个位置，另设写回原向量、正确值和空值对照。对原始 C 三个种子核实改后即时读数，再与同步干净轨迹比较，区分提前给答案、给错数及随后恢复。
- **大致结果**　干预位置和时机会改变错误到达下游的顺序；目标先回正时，下游仍可能短暂跟随旧错。
- **当时的分析**　当时逐轮对照干净轨迹，区分基线漂移、额外错误和提前出数；第三种子只进入相应分层小试。
- **记录与路径**　[c-original-layered-injection.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-original-layered-injection.md) · [c-s2026-original-injection](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-s2026-original-injection)

**70　A′、B 与两份原始 C 的标准、分层及扩展场景**

- **实验做法**　把离散 A′、B 的同题轨迹补入原始 C 的对照，按三套协议组织。标准组从各模型实际轨迹找干预起点；分层组按链上位置和预定时点替换；扩展组再拆开清空内部位置、改另一分支、同时改父与子及两个父值偏差相消等条件。例如两父一增一减时，单个父值都错而正确的模加和不变。分别记录留空、给错数和首轮回正。
- **大致结果**　A′ 清空后常能填对，直接填错时常保留；B 和原始 C 都能改写部分已有值。看似相同的失败可能分别是留空、晚一轮出值或算出别的数字。
- **当时的分析**　当时回看同题逐轮记录，要求干预时点、位置和资格明确。离散侧改数值字段，连续侧替换整段向量，跨模型能比较的是行为。
- **记录与路径**　[场景对照实验报告.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/abc-input-scenarios-results/场景对照实验报告.md) · [abc-input-scenarios-results](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/tree/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/abc-input-scenarios-results) · [abc_scene_queue.py](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/tools/c_hard_graphs/abc_scene_queue.py) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/abc-input-scenarios.md#L5)


<a id="stage-9"></a>

## 11 月 23 日 · 第一批同题重放、自然错误和更新标记

**71　后训练前后的同题场景重放**

- **实验做法**　将普通纠错后训练和多时点更新监督后训练的两种连续模型，分别与同来源种子的原始 C 配对。在之前的标准、分层和扩展图上复用目标、干预方式、搜索上限与观察窗口；A′、B 和原始 C 的结果从已有轨迹读取。比较时逐步限定为共同合格、注错绝对轮次相同，标准组再要求同题干净求解合格，最后看同一道题哪一侧发生了目标回正或下游受损。
- **大致结果**　多个单点条件下后训练版本较常在下一轮修回目标；子节点采用干预错值的变化方向随时机而异，多点注错仍难。
- **当时的分析**　当时同时保留共同合格、同绝对轮次、再加干净拟合限制的比较；目标首轮恢复与健康下游受损分别回答两个问题。
- **记录与路径**　[README.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-b-trained/README.md) · [scenes-paired.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-b-trained/tables/scenes-paired.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch.md#L11)

**72　自然错态与同错数人工供体交叉回放**

- **实验做法**　沿原始 C 的自由轨迹寻找目标先正确、后变成另一个数字的时刻，并选定对应子节点。从该现场分三路：保留自然错态、换入改题在同位置同轮产生的同一错数向量、放回该位置先前正确的向量。再分别保持周围状态、恢复错误父节点或恢复其他已到计算时刻的错位，续算 16 轮。找不到匹配供体的事件单列。
- **大致结果**　在部分已有确认记录的配对边上，恢复目标能改变子节点下一轮的答案，却未带来观察窗末两轮的共同恢复。
- **当时的分析**　当时关心人工注错能否代表自然错，因而保留原上下文与恢复周围状态的分支，并登记找不到合格供体的事件。
- **记录与路径**　[README.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-c/README.md#L17) · [natural-pairs.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-c/tables/natural-pairs.csv)

**73　同一正确读数、不同历史的新读者测试**

- **实验做法**　在同题、同位置、同一绝对轮次准备三份向量：一直正确、此前持续置空而刚填对、此前读错而刚回正。核对它们当前的数值和其他三个字段后，逐一移入同一份稳定前缀，只换这一个位置，再追加一个从未见过先前错误的新算式。另写入明确的错值作阳性对照，先验证新算式在首步确实取了这个位置的值，再比较三种历史。
- **大致结果**　42 只有少数题能确认新子节点当步确实取值；1234 在对应首步几种条件下均读为空，无法比较不同来源。
- **当时的分析**　当时先问“这里有没有一次有效读取”，再问历史是否改变答案。没有读取机会的题保留为无覆盖。
- **记录与路径**　[README.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-c/README.md) · [same-value-new-reader.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-c/tables/same-value-new-reader.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch.md#L10)

**74　多子节点传播的跨模型重放**

- **实验做法**　在固定题号上让一个内部目标分别连接一、二或四个直接子节点。分别在它刚可算、刚算对而子节点仍空、子节点都已对、整图已完成时改一次目标。A、B、两份原始 C 和四份后训练 C 各留同题干净分支。逐轮记目标何时改回、几个子节点给出预期错数、后代多错或多空多少轮，以及指定输出曾否出错和末轮是否仍错。
- **大致结果**　一些后训练条件下，目标修回更快，下游也更常出现特定错数；增加子节点数时，各项传播指标未必同向变化。
- **当时的分析**　当时分开统计受损题数、错误节点持续轮数和输出末轮是否错。后续复核纠正了一处把尝试数写成合格数的标签。
- **记录与路径**　[fanout-counts.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-b-trained/tables/fanout-counts.csv) · [verification.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-c/verification.md) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch.md#L12)

**75　更新监督在训练早、中、末期的定位检查**

- **实验做法**　只取带第二层更新监督的 C，用两个来源种子各自在第 10、30、50 个后训练周期的保存点做完全相同的题。对干净、改目标、改父和同图清空加注错等输入，保存每个内部位置的标记分数、改前数值、实际输出和当前父值推出的答案。再用相同阈值逐份划分标记与未标记，分别查应填空位、已有冲突和健康位置有没有被标出、实际有没有改对。
- **大致结果**　不同种子的标记能力随训练演变不同；有目标未越过阈值却被改回，也有已改值的位置与标记不一致。
- **当时的分析**　当时用它检查过程监督头在报告什么。模型是否修改答案由实际输出判定，标记本身不能充当硬门控证据。
- **记录与路径**　[README.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-b-trained/README.md) · [mark-counts.csv](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch-b-trained/tables/mark-counts.csv) · [操作细节](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/first-batch.md#L13)

## 实施状态容易混淆的几项

- **局部自洽错态。** 11 月 14 日原设计要求整个受影响子图在错误赋值下也保持算术一致。记录明确暂缓，因为旁路父节点可能仍留旧值。后来做过父子联合替换、持续错父等实验，各自的条件见上文。[b-local-self-consistent-deferred.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-14/b-local-self-consistent-deferred.md)
- **B 的持续错父、切换与完整改题矩阵。** 长窗口预试中做过有限的持续强制；11 月 16 日清点时，完整大样本矩阵只见实现、小验证或计划。后续合法重新赋值有独立交付，不能倒填为这套旧协议已全量执行。[project-status-overview.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-16/project-status-overview.md) · [priority-reassignment.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-18/priority-reassignment.md)
- **固定题内反复赋值的单独交接。** 11 月 20 日有含长定义边界的协议与实现交接。当前能明确追溯结果的是后来的难图命名配对和工作空间复用批次；此处未给旧交接另计一次正式结果。[2026-11-20-inline-rebinding-and-c-local.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/ai-collab/handoff/2026-11-20-inline-rebinding-and-c-local.md) · [c-hard-graphs-results.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-hard-graphs-results.md)
- **逐位置停止与自适应计算。** 有逐位置停止小试方案和脚本。22 日交付说明本轮队列跳过该小试，23 日将自适应计算列为另批研究方向。本快照未见与该方案对应的正式权重实测结果。[c-stop-pilot.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-21/c-stop-pilot.md) · [c-attribution-delivery.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-22/c-attribution-delivery.md) · [research-storylines.md](https://github.com/WhenYouSingThatSong/algorithmic-generalization-transformer-architectures/blob/6dbd2c78b58688de3cb4cbdacf426c7967aee4ee/docs/records/2026-11-23/research-storylines.md)

**核查口径。** “当时的分析”依据同期记录重述；有后续修订时写出日期或修订来源。“大致结果”按可见训练回执、结果表或阶段报告概括。11 月上旬的部分服务器实验只留下报告和归档路径，本次未重放这些权重。表格重导、分母更正、看板更新和重复引用同一轨迹都没有单独列为新训练。
