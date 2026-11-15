# C 长程小比较：s2026/e143，额外D与2D

2026-11-15。用户要求先核对s2026数据，再实现此前认可的小规模长程比较。当前授权限于下述普通续训/+D/+2D及评测；人工注错、自然错误重放、混合训练和从头训练尚未启动。

## 1. s2026核对与起点

远端fe647bd新增s2026报告、四CSV和原始包。GPT核对归档SHA256及run_meta身份，检查CSV节点计数/比例：384图中383图曾整图VALUE全对，N32_deep为63/64，其余五档64/64。H16六档整图VALUE为63或64/64；H64的N128两档整图为0，但节点VALUE分别76.61%/69.36%，节点四属性2.65%/2.14%。这是基本求解能力与长程失稳分离，不应拿H64整图0称模型不会算。

本轮固定e143，不改用best-val或final，不按小试结果换起点。原始权重SHA256：`40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780`。路径由入库原始包run_meta读取（相对当前仓库），可用`--ckpt`适配服务器路径，但hash不能变。s2026是原C0窄分布配方，不是冷存MixedN的C0′。

## 2. 三组与轮次

每张图D=实际最大节点深度+1。图按同D分桶，固定batch成员，不能用batch偶然混入深图造成的轮数代替图本身D。

| arm | 每条轨迹 | 标签 |
|---|---|---|
| base | D轮 | 原逐深度teacher标签 |
| plus_d | 2D轮 | 前D轮原标签，随后D轮始终最终标签 |
| plus_2d | 3D轮 | 前D轮原标签，随后2D轮始终最终标签 |

例如D=7：base7轮、plus_d14轮、plus_2d21轮。连续hidden始终回灌，不重置题目、不写入标签、不解码回灌、不detach；完整BPTT一次backward。**按D延长，不按模型实际是否算完筛选。** 因而后半段同时可能帮助未完成图继续求解，不能预先把所有增益解释为完成后保持。

新runner独立于原model.py，不改原训练任务。共同AdamW新优化器（不恢复旧momentum），lr=1e-4，原betas/eps/weight_decay、原0.1 dropout，float32/matmul medium，无scheduler/有限梯度裁剪。前四字段权重保持[1,1,1,1]，aux为0，批内剩余PAD仍参与CE。

损失：base为D轮CE的均值；长程为0.5×求解阶段均值+0.5×延长阶段均值。这样两档长程的求解/保持权重相同，不把2D保持损失权重加倍。与原loss按loop求和不同，三组属于明确归一化的新后训练配方；不声称原训练完整resume。

## 3. 数据与预算

官方合成器、原N32/depth≤6/6叶/ADD mod23/父数1–3、128变量名词表。固定新生成1024训练图＋128验证图，seed20261115/20261116。与原训练/验证及六档诊断数据按最终token序列（去尾PAD）做精确排重，并检查两个新split互斥；不作图同构排重。数据有重复或结构不符时反馈，不临时按模型表现换seed/补样。

固定batch_size=8（OOM时先反馈；若必须改三组应统一重定本批，不能只改一个组）。batch成员按D固定，每epoch只打乱batch顺序。保留不足8的尾批，所有图都参与。

两套端点都必须报告：

1. **同曝光量/步数**：三组都完成2遍新训练图时保存same_exposure。相同图数、batch计划和优化步数，循环计算量不同。
2. **同循环计算代理量**：base6遍、plus_d3遍、plus_2d2遍，保存equal_compute。每组总sample×loop相同；固定batch成员/长度还保证token×loop、attention长度平方代理量相同。优化步数、数据曝光次数、反向内存/墙钟时间不同，不能称严格同FLOPs或所有预算均相同。

plus_2d的两个端点是同一权重，不重复模型评估。只用固定端点，不按loss挑epoch。重复使用1024小样本属于配方pilot，验证集用于检验过拟合；没有整套正式训练有效性结论。

## 4. Agent C执行（串行）

先确认服务器无其他agent正在运行共享worktree；空闲后同步本提交并固定版本。既有任务不打断，运行中不pull、不改数据/代码；所有输出进新目录。B只继续之前CPU补交，不争用本实验队列。

```sh
python -m unittest discover -s tools/c_long_pilot/tests -v
# 本地review主机6通过、torch1跳过；服务器应实际运行全部7项。
python tools/c_long_pilot/build_data.py

# 各组2个batch的小验证；新目录，不混入正式meta。
python tools/c_long_pilot/train.py --arm base --smoke-batches 2 --out-dir tmp/analysis/c-long-pilot-20261115/smoke/base
python tools/c_long_pilot/train.py --arm plus_d --smoke-batches 2 --out-dir tmp/analysis/c-long-pilot-20261115/smoke/plus_d
python tools/c_long_pilot/train.py --arm plus_2d --smoke-batches 2 --out-dir tmp/analysis/c-long-pilot-20261115/smoke/plus_2d

# 用已有runner验证新ckpt能被标准连续评估入口strict load。
python tools/c_capability/run_capability.py --ckpt tmp/analysis/c-long-pilot-20261115/smoke/plus_2d/smoke.ckpt --weight-name long_smoke --out-dir tmp/analysis/c-long-pilot-20261115/smoke_eval --strata N32_shallow --limit-graphs 2 --rounds 4 --batch-size 2

# 小验证正常后，已授权本小试，三组依次执行，不再次请求启动。
python tools/c_long_pilot/train.py --arm base --out-dir tmp/analysis/c-long-pilot-20261115/base
python tools/c_long_pilot/train.py --arm plus_d --out-dir tmp/analysis/c-long-pilot-20261115/plus_d
python tools/c_long_pilot/train.py --arm plus_2d --out-dir tmp/analysis/c-long-pilot-20261115/plus_2d
python tools/c_long_pilot/evaluate.py --out-dir tmp/analysis/c-long-pilot-20261115/validation
```

由agent指定空闲GPU/CUDA_VISIBLE_DEVICES（单进程，默认4线程）。smoke核对输入来自新train、真实D/轮数、无NaN/OOM、optimizer更新、checkpoint load；代码还检查base loss=原nointerm loss/D。无需扩展整套防御性验证。元数据不记录为complete就不能汇总。

若默认数据排重参考路径不在服务器，先定位原文件，显式`--exclude`列完整参考路径并记录；不能把列表清空跳过排重。保存源权重hash、新data manifest、训练steps.jsonl、配置hash、每端点预算。没有resume功能；失败保留日志反馈，不在同目录覆盖或拿不完整训练当正式组。

## 5. 评估与交付

新验证128图，初始权重和各固定端点，从题目重新自由运行至每图D+64。主指标节点VALUE、节点四属性同时正确；D/2D/3D/D+16/32/64与固定绝对8/16/32/64分别报告。保存逐图全轨迹NPZ用于后续机制分析。

整图首次完成、首次完成后出错、恢复/复错是附加指标，分别给completed/post_observed分母。固定D切片覆盖全部验证图，不因失败或未完成剔除；不得把“没完成所以没漂移”算稳定。对实际完成后事件的观察窗口不等长，不能单靠事件率选长程轮数。

validation产物：summary.csv、events.csv、event_summary.csv、graph_curves.csv、逐图NPZ、budget.csv和端点meta；paired_cases/paired_summary已经实现相同graph_id/节点相对初始权重及同预算普通续训的改善/损伤计数。优先回答：正常D附近能力是否下降，额外监督区间及更远处的VALUE/四属性是否改善，改善是更少首次错误还是更快恢复。没有人工注错，不能声称错误检测或纠错机制已确定。

新验证完成后，三个equal_compute端点再用旧六档384图×64轮作固定泛化诊断（不用于选checkpoint或调整参数）：

```sh
for arm in base plus_d plus_2d; do
  python tools/c_capability/run_capability.py --ckpt "tmp/analysis/c-long-pilot-20261115/$arm/equal_compute.ckpt" --weight-name "long_$arm" --out-dir tmp/analysis/c-long-pilot-20261115/six_strata --rounds 64 --batch-size 4
done
```

与已有s2026/e143同数据/同指标并排汇总。训练、选参数只用新train/val；旧六档是已多次查看的研究诊断集，不宣称最终独立测试集。raw/NPZ/权重留tmp，提交短报告、节点VALUE/四属性表、预算表及少量典型轨迹；不提交新权重。若需修改LR/延长训练/加其它组，先报告这轮结果再讨论。

GPT本地完成stdlib6项、语法与差异检查；PyTorch1项和实际训练由agent验证。当前只有实现和s2026既有结果核对，没有本轮训练结果。
