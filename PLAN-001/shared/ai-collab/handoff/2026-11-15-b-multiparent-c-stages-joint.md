# 下一批：B多父节点、C不同时点与联合局部修复

2026-11-15。用户已授权具体方案、实现、推送和执行；C+训练暂缓。本文件同时是方案与两个执行agent的交接单。仍用reproduction，固定代码后串行运行，不改既有训练和已保存结果。

## 1. B：父数与错值是否改变更新行为

权重沿用manifest里的A′ s42/e86、B s42/e92；数据沿用N32 main256，固定扫描graph32–95，不补样。这是已使用的探索集，不是新的最终测试集。

**重新选点，不复用旧单父节点chains。** 对每张图，分别枚举1/2/3数值父节点、至少有一个子节点、depth≥1的内部目标，每组按固定seed=20261115的哈希选一个。缺组记录，不用其他父数代替。原值不筛选，报告实际深度与原值覆盖。

两个时点：

- just_solved：原manifest预算内，该模型首次实际输出目标正确值；要求该时刻目标及所有父VALUE正确。不满足就不合格，不能寻找更晚时点补上。
- whole_graph_complete：原manifest记录的两模型首次共同整图VALUE完成状态，重新检查本模型该状态整图正确。

各时点都从同一状态分出：不改、写回正确值、清空、正确值加1/3/7/11（模23）。只改一次目标VALUE，之后自由续算16轮，父值不钳制。固定graph32–39的选中目标进一步枚举全部22种错误值，替代四错值分支而不重复计数。枚举子集按图号预定，不按异常结果选样；报告与普通四错值组分开。

**原异常另列。** graph142/pos128分别从整图完成后正常运行2轮、从完成后把pos26改错再运行2轮出发，形成相同经过轮数的正常/受扰上下文。两个起点都要求目标父值正确，允许目标本身错误；分别枚举全部错值、清空、正确值和不改。完整六字段起跑状态另存，帮助检查当前上下文差异。这是当前状态比较，不解释成跨轮隐藏记忆。原案例不并入主表。

输出两套主汇总：A′/B各时点共同合格；再限制为两个时点、两个模型都合格的同一图同一目标交集。父数、枚举/非枚举、时点均分开。目标恢复、下游传播和父值变化都保留；首轮正确与窗口末两轮正确分开。

## 2. C：同一种替换放在不同阶段

权重仍是旧C0 s42/e143、s1234/e143，六档已有数据。本批固定每档前8图，每图至多2目标；是上一批前16图的固定子集，结构选点与同值/异值来源构造不变，不因某个阶段结果不好而换目标或叶改动。

原图和同值/+1/+3来源各自自由运行，所有来源在相同绝对轮次取hidden：

| 阶段 | 定义 |
|---|---|
| early | d(u)+1个forward后的预定早期状态，保持上一批时点定义；不冒称实际第一次猜对 |
| complete | 原图第一次整图VALUE全对的输出状态，最多寻找64轮；不是四属性全对，不要求来源整图也完成 |
| late | complete之后固定再自由运行16轮；若超出64轮预算，记不可用，不缩短间隔 |

early与complete有时是同一轮，记录实际轮次，不将其当两个独立时间点的证据。complete寻找的是原图轨迹，不能按干预效果选停止点；找不到则保留失败记录。late不是要求整图仍然全对：主问题允许周围出现漂移，但目标及可比q父值必须满足下述资格。

**统一数值资格、额外提供严格子集。** 三个阶段都要求原图目标读出正确VALUE，同值与异值来源目标分别读出预定a/a+1/a+3。q必须在结构上可算、其原图父VALUE正确、不改分支下一轮q正确，所有干预分支共用q集合。对每个来源同时记录四属性是否正确、前三字段是否匹配接收目标；阶段实验不因VARIABLE漂移自动删除所有VALUE仍正确的晚期状态。

这与旧首批默认的四属性资格有区别，必须标记新协议，不直接拼接旧百分比。提交：

1. 每阶段全部数值合格样本的结果和资格损失；
2. 三阶段都合格的**同一目标、同一q交集**结果（不能只有目标相同而q不同）；
3. 三阶段来源四属性都正确的严格交集，若为空如实报告；
4. 原图在每个时点的全节点VALUE/四属性正确数，说明周围状态是否已经漂移。

操作仍是一次性替换u向量、不改/原样写回/同值/+1/+3五分支，随后自由续算16轮；完整来源题续算只是参考，不能当局部q预期。VALUE-only组可能同时存在身份读出变化，只支持该整体状态条件下的结果；要讨论排除身份因素，应看严格交集或已记录匹配字段，不把两个组混为纯数值效应。

## 3. C：自然错误现场联合修复

直接使用上一批`c-local-repair-20261115/s42`、`s1234`的57事件、114条自然/健康状态记录及保存的hidden快照。不重新挑错误，不换来源时点，不改间隔，不增加更早的全图回退。

已有四分支：不改、目标原样写回、只换目标、只换父节点。本批增加：

- **restore_joint**：把目标和全部直接父节点同时替换为原先那一轮的向量；
- **sham_joint**：相同位置原样写回。

各位置使用同一个已保存来源轮，VALUE都应读对；其它属性可能漂移，按原实验如实保留。改一次后自由续算16轮。自然错误现场和相同来源间隔的健康对照都做，观察联合修复是否改善目标及下游，以及是否损伤健康节点。

新runner从保存的base_hidden、donor_vectors直接开始，核对同一权重/数据身份，不改动续算必须逐轮重现原记录；无需重新跑64轮采集。单独修目标/父节点的分支复用原记录，联合分支新跑，预算一致。相同起点与来源保证这次主要改变的是修复范围，不是向量年龄或案例组成。失败报告，不跳过难复现案例。

先不增加祖父节点、整图回退、持续钳制或新训练；这批结果出来再决定是否扩大修复区域。

## 4. 指标与报告

- 节点VALUE和四属性同时正确均保留；各时点/父数/来源/自然或健康组分别报告。
- 逐轮看1/2/4/8/16轮；新增`endpoint_cases.csv`和`endpoint_summary.csv`并列末两轮正确、相对不改动分支的改善/损伤、曾出现输出错误及错误后末两轮正确。后者不是“永久恢复”；始终正确的节点不算恢复。
- 各分支共享样本/观察节点再作配对；原始计数及逐图等权都给，不能因某分支未输出数字而更换分母。端点表仍以同一观察集合为分母，不偷换成各模型不同的受损集合。
- 同一图多目标/多个错值不当独立图。枚举错值子集、原异常、跨阶段交集分开；不得把逐轮切片称为全程持续恢复。
- 典型案例交付实际父/目标/q数值和完整表达式，区分位置与变量名。优先提交数据自动导出，避免手抄遗漏叶贡献。

## 5. 领取、命令与顺序

**Agent B先执行B，Agent C先做已要求的CPU补交，GPU排B之后：C时点→C联合。** 不在运行目录pull/改代码；开始前同步空闲工作树并记录commit。已有训练不动。普通路径适配可做并记日志，改变选点/资格/时点须反馈。用户已授权本批，小验证正常即可启动，不重复问批准。

### Agent B

```sh
python -m unittest discover -s tools/repair_study/tests -v
python tools/repair_study/run_b.py --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json --model model_a --multi_parent --include_anchor --candidate_count 2 --all_values_first_graphs 1 --horizon 4 --out_dir tmp/analysis/b-multiparent-20261115/smoke_A
# 同参数model_b，out_dir smoke_B

python tools/repair_study/run_b.py --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json --model model_a --multi_parent --include_anchor --out_dir tmp/analysis/b-multiparent-20261115/main_A
python tools/repair_study/run_b.py --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json --model model_b --multi_parent --include_anchor --out_dir tmp/analysis/b-multiparent-20261115/main_B
python tools/repair_study/summarize.py --inputs tmp/analysis/b-multiparent-20261115/main_A/records.jsonl tmp/analysis/b-multiparent-20261115/main_B/records.jsonl --out_dir tmp/analysis/b-multiparent-20261115/summary
python tools/repair_study/summarize.py --inputs tmp/analysis/b-multiparent-20261115/main_A/records.jsonl tmp/analysis/b-multiparent-20261115/main_B/records.jsonl --require_timings just_solved whole_graph_complete --out_dir tmp/analysis/b-multiparent-20261115/summary_matched_timings
```

小验证检查：新选点确实包含多父；全部错误值无重复/不含真值；只改变目标VALUE；两种anchor上下文经过相同轮数、父值正确且记录实际起点，不强行把异常目标设成8。

### Agent C：阶段对照

```sh
python -m unittest discover -s tools/counterfactual_control/tests -v
python tools/counterfactual_control/run.py --weight s42_e143 --stage_study --strata N32_shallow N128_deep --n_graphs 2 --horizon 4 --out_dir tmp/analysis/c-stages-20261115/smoke_s42

python tools/counterfactual_control/run.py --weight s42_e143 --stage_study --n_graphs 8 --out_dir tmp/analysis/c-stages-20261115/s42
python tools/counterfactual_control/run.py --weight s1234_e143 --stage_study --n_graphs 8 --out_dir tmp/analysis/c-stages-20261115/s1234
python tools/counterfactual_control/summarize.py --inputs tmp/analysis/c-stages-20261115/s42/records.jsonl tmp/analysis/c-stages-20261115/s1234/records.jsonl --out_dir tmp/analysis/c-stages-20261115/summary_all
python tools/counterfactual_control/summarize.py --inputs tmp/analysis/c-stages-20261115/s42/records.jsonl tmp/analysis/c-stages-20261115/s1234/records.jsonl --common_stages early complete late --out_dir tmp/analysis/c-stages-20261115/summary_matched
python tools/counterfactual_control/summarize.py --inputs tmp/analysis/c-stages-20261115/s42/records.jsonl tmp/analysis/c-stages-20261115/s1234/records.jsonl --common_stages early complete late --strict_sources --out_dir tmp/analysis/c-stages-20261115/summary_strict_matched
```

小验证核对绝对轮次、late=complete+16、三个阶段的来源取同轮、同样两处叶改动、不改/sham一致、跨阶段q确实取交集。找不到完成/晚期数值资格的图照常记录，不临时缩短late_gap或放宽条件。若严格交集为空，这本身是覆盖结果，不强行补样。

### Agent C：联合修复

```sh
python tools/repair_study/run_c_joint.py --source_dir tmp/analysis/c-local-repair-20261115/s42 --limit_events 2 --out_dir tmp/analysis/c-joint-20261115/smoke_s42
# s1234同样做小验证；沿用源记录16轮，不另外改观察窗口
python tools/repair_study/run_c_joint.py --source_dir tmp/analysis/c-local-repair-20261115/s42 --out_dir tmp/analysis/c-joint-20261115/s42
python tools/repair_study/run_c_joint.py --source_dir tmp/analysis/c-local-repair-20261115/s1234 --out_dir tmp/analysis/c-joint-20261115/s1234
python tools/repair_study/summarize.py --inputs tmp/analysis/c-joint-20261115/s42/records.jsonl tmp/analysis/c-joint-20261115/s1234/records.jsonl --out_dir tmp/analysis/c-joint-20261115/summary
```

检查snapshot身份和位置集合，联合初始VALUE确实正确、其它位置未改、一次替换后释放；不改动与源记录一致、联合sham与不改动一致。不要把原目标单点恢复表和新不同案例混在一起比较。

## 6. 交付与边界

各任务各一份短报告，提交coverage/主汇总/端点及少量原始案例小表，记录实际样本数、缺组原因、命令和运行commit。完整records、snapshot留tmp，输出新目录，不覆盖已有数据。服务器小验证及正式状态写board/log，不把本地代码测试当模型验证。

GPT本地只做相关stdlib测试、语法和差异检查，无torch/权重运行；模型端小验证由执行agent负责。无C+训练、无新模型架构、无N256、无局部自洽或其它机制矩阵授权。
