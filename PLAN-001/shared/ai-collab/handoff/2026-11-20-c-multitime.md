# C多时点局部纠错＋过程监督UPDATE：执行交接

2026-11-20。代码入口：[tools/c_multitime](../../tools/c_multitime/README.md)。设计和变量身份分析：[记录](../../docs/records/2026-11-20/c-identity-and-multitime-design.md)。

## 用户已确认的关键更正

不先筛选模型已算对的节点/整图。不同轮次按概率污染，原先EMPTY/错误/正确都可接收；状态用于记录，不是注入资格。源向量是否能被读成预期错值仍有质量检查，两者不可混淆。局部父值不一致先修，下游随后修，不读取本轮修正后的父输出来生成子标签。

名称中必须明确“过程监督UPDATE”：第二层中间监督＋末层计算/对齐，没有硬gate。新模型从原s2026/e143开始，不能把旧UPDATE final当默认起点。旧机器ID/目录不改，新身份recipe_id=c_multitime_process_update_v1。

## 实施范围与调度

Agent C：旧任务保留，串行交接后新smoke/单seed50epoch。Agent B：先交旧批dev干预表/正式统一评估，后接收新final做同plan比较。共享树不在运行时pull，不抢占/kill任务；本轮未启动任何服务器进程。

新代码没有修改c_update/*.py，因此旧运行源码hash集合不因本实现变化；但仍不得违反共享树串行约定。旧c_update正在使用的plan保持原样，不补写micro字段；新入口在独立meta中记录micro32覆盖旧plan的历史micro8字段，仅复用其数据/权重/划分身份。

参数详见README：三模式等概率，5%有效batch启用、50%结构候选，frontier D+1轮/middle t+D+1轮/late 2D+5轮。全BPTT，节点同步局部标签GPU生成；新增长程预算是配方变化，不把效果全归因于UPDATE。

## 必交验收

- 本地只做stdlib/语法检查，torch测试留服务器；不要把跳过记通过。
- 六步最大深度batch32 clean/强制污染覆盖三模式；报告显存、耗时、候选/源通过/实际VALUE变化，first step预热影响说明。
- 新GPU局部标签对照旧local_step，包含未知父值、重复父节点、变量名漂移；新loss及梯度对照旧UPDATE公式；有效batch归一化一致。
- forward不变、两轮BPTT、第二层梯度、smoke权重加载和dev2/冻结原C功能检查通过后写acceptance。
- 显存不足：新输出目录用micro16或8重新验证，不绕过身份检查；donor-batch单独记录。不改原始50epoch上限、数据、概率或损失来凑通过。
- 源拒绝/低剂量是需报告现象，不靠挑接收正确样本解决。若某时点几乎无有效污染，先提交计数和原因供设计讨论。
- 原始权重/轨迹留tmp，smoke receipt、training-audit、dev汇总、阶段计时的小表和来源hash提交docs/records；看板/log更新实际执行状态。

## 新结果如何评估

不要只报clean或99%的plan_execution_agree。clean和受扰分别报定位precision/recall、执行正确，u/q/全下游的损伤与恢复、保护正确节点、VARIABLE漂移；D+4只是固定预算，不自动称全图完成。

evaluate_new复用旧runner和全部固定测试。compare_new给新旧同seed三对、两侧源通过及共同输入资格两个视图；完整候选覆盖分列，不因某模型失败多就移除。旧批尚无完整评估时，先补旧批，不修改清单或换更容易的图。
