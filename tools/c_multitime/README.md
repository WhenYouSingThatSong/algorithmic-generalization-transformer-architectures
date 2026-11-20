# C（多时点局部纠错＋过程监督UPDATE）

2026-11-20：用户确认不以接收模型先算对为污染前提，要求记录并实现。独立配方 `c_multitime_process_update_v1`；实现完成不等于服务器验证或训练完成。本目录不修改 `c_update/*.py`，不迁移正在跑的旧任务。

## 已确定的目标

- 从**原 C s2026/e143**重新开始本批后训练，先一个种子，沿用50 epoch上限；不是从旧UPDATE final继续训，不扩到另两个种子。
- 四层主体不变。**第二层之后监督 UPDATE**，第四层监督状态、定位与实际 VALUE 改变的对齐、UPDATE清零。没有硬门控，没有把预测mask写回hidden。
- 新可算位置与已有但局部不一致的位置都应更新。原图依赖关系仅用于生成监督，**不额外输入模型**。
- 叶结果依照原题常数；内部节点依照当前输入父值之和 mod23；父值非数字则暂保持。当父值修回后，下一轮再修下游。监督不提前使用本轮输出。
- 不要求接收节点/整图先正确，不删除难图，不用teacher答案补齐起点。自然错误同样受逐轮监督。
- VARIABLE漂移保留原前三属性监督和评估，本批不加入身份污染/身份修复定位/只读题面结构。

## 本版固定的实现参数

复用既有 `c_update` 完整 plan：同一原50,000张N32训练图、同一原C起点、同一dev/test。训练文件SHA仍由旧协议锁定，不重生成数据。按D=max depth+1分批，每epoch每图一次，50epoch=2.5M图呈现。

每个有效batch独立等概率选择以下一个模式，**选择不读模型输出**：

| 模式 | 污染候选、时点（t表示已完成的前向次数） | 总训练前向次数 |
|---|---|---|
| frontier | 每个结果节点在t=depth(u)有一次候选机会；包括叶和sink | D+1 |
| middle | t=max(1,D//2)，候选为depth(u)<t的结果节点；它们按图深度应已可算，但实际可以是EMPTY、错误或正确 | t+D+1 |
| late_budget | t=D+4，所有结果节点候选；**不保证整图已经完成** | t+D+1 |

每个有效batch仍5%概率启用人工污染；启用后每个结构候选独立50%抽签；每个节点在一条轨迹最多污染一次。三个模式的候选量和时点不同，不宣称实际剂量与B相等；按模式报告候选/尝试/源通过/实际VALUE改变。模式抽签、污染抽签不随microbatch重复。

所有模式包括未启用污染的batch，都按表中固定轮数监督。这样具有对应的无人工污染续算训练，不通过是否算对决定计算预算。**这也增加了长程训练量**：D=7时三模式为8/11/19轮，等概率平均12.67轮，旧版为8轮。新旧结果不能只归因于“污染时点”或“中间信号”；目前没有新开消融训练。

保持全程连续反馈、完整BPTT，不截断前缀。每图loss对本batch轮数取平均，保持各图权重；更长轨迹的早期轮相对权重因此不同，属于新配方的一部分。

## 连续污染

与旧版相同：对选中的u选非零delta，改一个叶常数，让反事实图u的真值变成b=(原真值+delta) mod23；当前权重eval/no_grad计算源向量，替换u完整hidden行并detach源。

- frontier源年龄为depth(u)+1；middle/late源年龄等于接收端t。年龄固定，不等待源算对。
- 源头读出须为原前三属性+b。源构造不可逆或读出不合格则该次不替换，**这是源向量质量检查，不是接收模型正确性筛选**；失败保留分母，接收图继续训练。
- 接收端改前EMPTY、非数字、正确、错误都允许替换。恰好已读为b也不重抽；记录实际VALUE是否改变与向量幅度。
- 同时记录注入前节点/整图VALUE及first4是否正确、正确直接子节点数。后者不证明内部真的曾使用该节点。多个位置同时污染时，用同一份注入前状态记录这些字段。
- 注入并不修改原图标签或父子关系。每轮局部目标重新由**实际输入值**生成。

## 损失与优化

`L_state + 2 L_locate + 5 L_align + 0.1 L_clear` 完全沿用旧UPDATE数学定义；state对前三属性及VALUE监督，locate在第二层，align惩罚定位/执行不一致的错误一侧，末层UPDATE清零。节点组先图内均值再图等权；PAD处理与旧UPDATE相同。

- 有效batch32，默认**microbatch32**，每步一次完整batch反向。允许显存证据支持的16/8回退，需新smoke目录并在运行身份中记录；不自动静默回退。
- 新AdamW，主模型1e-4、UPDATE头5e-4，50epoch cosine到0.1；float32/原dropout，不改变混合精度或编译策略。
- `tensor_targets.py` 一次性预处理紧凑节点/父索引；每batch搬入设备，每轮在GPU生成同步局部标签。另提供向量化teacher标签用于旧规则一致性验证。
- 标签热路径没有GPU→CPU→Python→GPU；日志张量每有效batch一次批量搬出。供体按年龄合批，默认32；不跨权重更新缓存供体。
- 分别记录训练step耗时/峰值显存、checkpoint耗时和dev耗时。训练计时每步边界同步一次，不在每个loop强制标量同步。阶段评估沿用旧协议以可比，本轮未重写评估性能。
- 完整BPTT最长19轮可能增加显存；“H100应该够”不是实测结果。先验收32，有OOM才回退，报告实际卡型/总显存/其他占用。

## 执行与验收

先完成旧任务的串行交接；运行中的服务器共享树不pull、不改代码。使用已有plan，不运行旧prepare，不覆盖旧run目录。

```bash
python tools/c_multitime/execute.py --phase smoke --plan tmp/analysis/c-update-20261119/data/plan.json --out-dir tmp/analysis/c-multitime-20261120 --device cuda --micro-batch 32 --donor-batch 32
python tools/c_multitime/execute.py --phase train --plan tmp/analysis/c-update-20261119/data/plan.json --out-dir tmp/analysis/c-multitime-20261120 --device cuda --micro-batch 32 --donor-batch 32
python tools/c_multitime/execute.py --phase evaluate --plan tmp/analysis/c-update-20261119/data/plan.json --out-dir tmp/analysis/c-multitime-20261120 --device cuda
python tools/c_multitime/compare_new.py --old-eval-dir tmp/analysis/c-update-20261119/run/eval --new-eval-dir tmp/analysis/c-multitime-20261120/eval --output tmp/analysis/c-multitime-20261120/comparison.json
```

Agent C负责smoke/训练，Agent B负责后续正式评估/对比，GPT审阅。与上一批协调排队，不因新任务取消上一批正式评估。

smoke只做6个最大深度、有效batch32训练步骤：三模式各clean/强制污染一次，不作为效果估计。另验证：

1. CPU/GPU向量化局部标签与旧标量定义完全一致（包含EMPTY/N/A、重复父节点、变量名漂移）；teacher标签一致；loss/梯度和micro归一化一致。
2. 原forward未改变、第二层读出存在、两轮梯度贯通、UPDATE头有梯度。
3. 无接收端正确性过滤；候选只来自时点和结构；源失败可追溯。实际被替换的位置中改前已有错误/EMPTY的情况如实报告，不能因没有足够样本暗改概率。
4. final重新加载，dev2图的正常/注错/同值/原样及冻结原C功能probe能跑通。不把技术小验证当总体效果。
5. 六步显存/速度与实际剂量按模式提交；首步含预热，不能据此声称固定倍数加速。

通过后Agent C可推进既有单seed50epoch范围，差表现不提前停止；代码/NaN/OOM故障停止记录。若显存回退，用另一个输出根目录重做smoke，main仍从原C起点开始。队列不支持把旧任务中途改为此配方。resume严格匹配本目录配方、原plan、源码、micro/donor配置；保留optimizer/RNG；数据无改动。

每epoch记录训练段；每100step及epoch末保存last；e1/3/5/10/15/20/25/30/35/40/45/50自动做旧128dev评估及原C功能probe，保存epoch/best/final。开发评估记录完成回执，恢复时不把半写JSON当完整。主结果固定e50，best仅辅助。

## 评估重点

正式评估复用原N32/64/128六档、单次/联合/持续/噪声/后续使用协议，不重新挑题。同seed与原C、C（前沿污染纠错）、C（前沿污染＋过程监督UPDATE）比较；旧三seed结果仍要补齐，新版不自动扩种子。

- 节点VALUE和first4并列，VARIABLE漂移单列。
- UPDATE在clean和受扰条件分别报tp/fp/fn、precision/recall、实际改变和执行正确，不能用大量不更新位置的一致率替代。
- 目标与下游分开看首次恢复、末两轮恢复、再错、额外损伤和错误父值传播；分母包含源失败/no_value，不把它们藏掉。
- 同时报告全固定候选覆盖、两侧源通过、共同输入资格；共同资格只用于评估解释，不是训练过滤。
- late_budget按起点实际状态再分组，不能统一写“整图完成后”。多seed/更细机制结论留后续，不因一个新种子成功就宣称普遍机制。

当前旧评估summary已经有clean/pair的UPDATE统计；更细“旧节点/首次计算/完成后”归因可由原始输入输出记录做CPU补表，不必先重跑GPU。
