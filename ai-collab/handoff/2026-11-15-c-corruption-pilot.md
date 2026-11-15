# C 腐蚀后训练三组小比较：执行交接

日期：2026-11-15 18:26 CST；GPT设计/核心实现，执行Agent C小验证及运行。
用户授权本批实现和执行；B既有任务继续，不启动新B实验。统一reproduction、服务器共享树串行交接，不打断已有任务。

## 1. B 实际腐蚀作为参照，不复制时机/位置

依据：`experiments/configs/B-M6-Nodes32-ADD-L4H16D384-DeBERTa/train_config.yaml`、`experiments/model.py::compute_batch_loss_fast/randomly_corrupt_values`、[已交付256 batch审计](../../docs/records/2026-11-14/b-main-execution-plan.md#36-m5训练腐蚀审计可与-m1-并行cpugpu-均可)。本轮读配置/代码和记录，没有重跑B审计。

| B的设置/实测 | 含义 |
|---|---|
| batch级corruption_prob=0.05 | 每批进入一次腐蚀分支；不是每图独立5%，也不是每一循环独立5% |
| tokenwise_corruption_prob=0.5，last | 分支内对深度展开后的新可算候选掩码独立抽样；不是全序列一半token |
| VALUE词表25类 | 0–22、N/A、EMPTY均可能抽中；可能恰好是正确值或保持EMPTY |
| 256×8=2048图、65536候选、100重复采样 | 每图平均32个跨轮次候选；仅4.1%原训练集，非全训练历史 |
| 实际161725替换，before全EMPTY | 平均每图每次0.7897个替换；没有已填数值→错值 |
| after数值148820、N/A6359、EMPTY6546；after_is_true6392 | 错误数字142428，平均每图每次0.6954个；不能把161725全算作错误数字 |

理论上每个候选被替换概率0.05×0.5=0.025；32候选平均0.8个替换。条件于某批启用腐蚀，则每图平均16个，呈批级聚集。不能据此说B每图恰好改0.8个节点。

## 2. 本次问题与固定设计

问题：在已有有效的+D保持配方上，偶尔注错是否带来额外恢复收益？随机方向训练是否足够，或VALUE语义错误训练更有用？不是复现B完全相同的腐蚀。

| 项目 | 本批固定设置 |
|---|---|
| 起点 | 原始C0 s2026/e143，SHA `40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780`，不是已经+D的权重 |
| 三组 | hold / noise / semantic；都总2D轮，3遍，batch8，lr1e-4，各自fresh AdamW |
| 数据 | 复用上一批1024 train/128 val（新生成N32/depth≤6）；不扩N，不换权重，不用val选模。该val已查看过，是研究验证而非新独立测试 |
| loss | 求解段0.5×均值＋保持段0.5×均值，前四字段权重1，aux0；保持原PAD/优化器recipe |
| 计划 | 每图每遍独立哈希50%尝试；每次至多一个位置，不成批多点破坏。无候选则保留干净训练并记录，不补样 |
| 位置 | 有父、有直接下游、有可构造匹配反事实的内部数值节点；排除叶、sink、常数、PAD。结构分层抽一个目标，不按模型行为挑 |
| 时机 | 在状态轮D到D+floor(D/2)-1内固定选择；下一forward前注入，至少留下2轮更新。不是早期EMPTY新槽位，也不按首次整图全对改变时刻 |
| 频率理由 | 名义约0.5次/图/遍，比B审计的0.79次替换略少，保留约一半干净图。不是匹配B受损强度；实际数由结构资格与固定计划决定 |
| 错值 | 每图固定一个delta∈1…22，跨图哈希覆盖所有非零模23偏移；不限定4→5，不只测+1 |
| 释放 | 注入一次后自由续算，不反复钳制，不注入多个节点 |

“D后”是预定求解预算后，不保证模型已实际算对。训练不因模型是否正确而改变注错计划；日志记录注入前是否已正确，分别汇报“破坏已有正确值”与“作用于未正确状态”。这是保证三组尝试时机/位置一致所作的选择。

### 在线反事实与噪声

复用已验证的两叶常数匹配构造：同一图、变量位置/结构不变，在源题改两处叶常数，使u分别保持a或变成a+delta。原题真值不改；注入只替换u一个hidden行，不换整图。

- 每个训练batch先由**当前模型**对异值源题自由续算到相同绝对轮次，eval模式关闭dropout、no_grad，取u向量。不是用老C0的固定向量；也不是从监督标签直接编码向量。
- 三组都做完全相同计划的源题forward（hold也做），参考计算开销单列并配对；但随着训练权重不同，源向量本身当然会不同。
- semantic直接替换一行；被覆盖位置的原hidden没有通过这次覆盖的梯度，其余位置/前轮与后续参数正常BPTT，源向量不反传。
- noise以随机方向加性扰动，L2等于当前源向量与当前原向量的距离；幅度/方向detach，原向量梯度保留。这个梯度差别是两种实际干预的实现差别，不能声称只改变错误语义。
- 噪声使用独立Generator，不消耗训练dropout随机序列；三组同批次顺序/训练种子/优化步数/接收端及参考端forward代理量。
- **不按源题读出成功与否动态丢样。** 记录源读出是否为指定错值、非VALUE字段是否匹配、原状态是否正确、注入后实际读出及下一轮。计划一致不等于实际错误量相等。若语义成功率差，先报告实现/构造局限，不能叫有效语义纠错训练。
- first4读出匹配也不能证明向量仅含VALUE变化；同值异源对照仍必需。

## 3. 评估与交付

1. 全128验证图无干预到D+64，保存每个数值节点前四字段完整轨迹；核心为节点VALUE/四属性，同时输出各字段曲线。正常D、D+16/32/64、绝对轮数并列，保留未全对图。
2. 固定结构目标，在训练时机和**D+16**两时点，各比较：unchanged、same同值异源、semantic、noise匹配幅度、noise3三倍幅度。所有三个训练组和initial都跑同一组图/目标。后续H=16，不把高强度噪声加入本批训练。
3. 五分支共享原始起点；正确参考为原题。同值与异值源改同两处常数；直接子节点错误预期用“只改u”的局部代入，不使用整张源题的q真值。
4. 同时保留all_fixed、所有四模型共同VALUE资格、共同严格first4资格。资格要求原u正确、异值源读出指定错值、同值源仍读出原值、子节点各父值正确且unchanged下一轮子节点正确。严格first4额外检查原u/同值源/异值源的四属性，不要求整图四属性全对。不能只报告有利交集；列各模型和共同覆盖。
5. target/直接子节点/全部后代/无关节点/全节点分开。post1/2/4/8/16报正确、no_value(这里定义为非数字读出，字段名nonnumeric)、配对改善/损伤、局部预期错值。H内记录第一次损伤、第一次恢复、回正后再错、最后连续正确段、末两轮正确、额外错误轮数。末两轮不代表整个窗口持续正确或永久稳定。
6. 分母拆开：scheduled、实际输入改动、输入读出错误、输出曾受损、曾恢复、末两轮正确。恢复条件分母为同轮clean正确而干预错误的节点；不同模型受损集不同，跨模型主比较仍看共同图上的准确率/额外错误轮数。给raw counts与图等权，空分母记NA。
7. 保留三组训练dose表、逐步loss/梯度、源题成功率、delta与深度/父数分布、norm分布、预算/配置/数据/源码/权重哈希。三个最终epoch3固定checkpoint，不能挑best。

这批不运行新的N128注错、混合训练、从头训练、VARIABLE身份错误训练、多点同时注错或更长H搜索。后续根据本轮讨论。

## 4. 顺序和命令

先确认B/C共享树空闲并领取看板，git同步固定提交，记录SHA；运行中不pull/修改配置。单GPU串行；不触碰其他训练进程。`GPU_ID`由执行agent选择空闲卡，显式设置可见设备。以下均仓库根目录。

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$GPU_ID"
OUT=tmp/analysis/c-corruption-pilot-20261115
DATA=tmp/analysis/c-long-pilot-20261115/data
python -m unittest discover -s tools/c_corruption_pilot/tests -v
python tools/c_corruption_pilot/prepare.py --data-dir "$DATA" --out-dir "$OUT/plan"
for ARM in hold noise semantic; do
  python tools/c_corruption_pilot/train.py --arm "$ARM" --out-dir "$OUT/smoke-$ARM" --smoke-batches 4 --num_threads 4
  python tools/c_corruption_pilot/evaluate.py --out-dir "$OUT/eval-smoke-$ARM" --smoke-ckpt "$OUT/smoke-$ARM/smoke.ckpt" --limit-graphs 8 --num_threads 4
  python tools/c_corruption_pilot/summarize.py --eval-dir "$OUT/eval-smoke-$ARM" --out-dir "$OUT/summary-smoke-$ARM"
done
# 小验证通过、记录后再执行；不从smoke权重接着训。
for ARM in hold noise semantic; do
  python tools/c_corruption_pilot/train.py --arm "$ARM" --out-dir "$OUT/$ARM" --num_threads 4
done
python tools/c_corruption_pilot/evaluate.py --out-dir "$OUT/evaluation" --num_threads 4
python tools/c_corruption_pilot/summarize.py --eval-dir "$OUT/evaluation" --out-dir "$OUT/summary"
```

首次正式执行已由用户本轮授权，小验证正常后继续，不重复请求启动。循环须使用fail-fast shell（例如`set -euo pipefail`），任一步失败不继续下一个训练。路径适配可用`--ckpt`（仍校验固定SHA）；改变模型含义/参数前回报。

### 小验证只核对必要项

- 6项stdlib＋3项torch测试；torch检查噪声范数/局部梯度与无注错+D loss等价。
- smoke三组预算和计划一致、全部final loss/梯度有限；actual目标不是PAD/常数，D后至少2轮；源题两处改动/标签不回灌。
- hold的实际范数为0；noise范数匹配对应假设语义替换；noise3范数3倍。semantic替换后读出与源行相同。
- 查看smoke训练与8图评估的原u正确率、源VALUE指定值成功率、共同资格。若没有有效语义注错/没有共同资格或明显不是预期语义，不开始主训练；提交读出与例子讨论，不能偷偷降门槛/改delta挑成功样本。少量不合格照协议保留。
- 标准模型加载smoke.ckpt成功（本evaluate使用标准loader）；source SHA、数据、spec无变化。没有额外防御性大测试。

## 5. 完成后提交

仓库`docs/records/2026-11-15/`提交短报告与小表（clean_summary、intervention_summary、event_summary、training_dose、coverage摘要、预算和哈希摘要），选2–3个包含改善/失败的完整例子。训练日志/原始JSON/NPZ/权重留tmp，不入git。

报告先回答：干净保持是否受损、两类注错各造出了什么错误、是否降低H16错误持续时间及改善下游恢复、另一类扰动/更晚时点是否仍受益。不要把单seed、反复查看的128验证图称最终泛化结论。

已知边界：原reduce_batch不能裁掉factor PAD，本批继续一致沿用，不顺手修原训练recipe。小试比较同时包含不同扰动/梯度路径，不能仅凭胜负认定内部检测机制。高强噪声在本批只是压力评估，若需要作为训练组另行讨论。
