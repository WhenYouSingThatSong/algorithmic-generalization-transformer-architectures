# B / C 执行代码审阅

2026-11-14 18:57 CST，GPT；审阅基线 `reproduction@a98a290`。
用户要求两条线都看，并提醒后续必须检查执行代码。此次读取新增执行代码、
报告及相关原始模型/生成器实现，未运行模型、未重复测试、未接触服务器进程。
发现交对应执行 agent 修复与做针对性小验证；不由 GPT 包揽执行。

## 状态与新增证据

- B1 CPU整理完成。报告补充 complete_R 20/360、complete_U 7/323 个
  受损节点事件出现“回正后再错”，最终稳定；不是此前所说的全程无反复。
  `b_pilot_analysis.recurrence` 确实从首次损伤后首次回正继续搜索错误。
- B M0已跑：固定256候选均共同完成，218图有链、365条链；teacher64/long32
  子集已生成。M5审计已跑：65536候选、161725次随机替换，改前均EMPTY。
  这是当前配置/采样下的审计，非历史全部训练过程的逐步日志。M1正式注错未启动。
- C提交了代码草稿及设计说明，未生成六档数据、未跑模型，暂无新能力结果。
  `run_capability.py:253-259` 的 embedding→hidden连续反馈、只解码观察，
  与 `model.py:889-895` 对齐；没有误用B离散回灌。

## C：先修的数据/指标问题

### C1 [缺陷，运行前修] 生成元信息含 Path，且没有记录真实生成配置

位置：`tools/c_capability/build_data.py:55-70,127,242`。
`make_meta` 将 STRATA 的 cfg 直接写为 generation_config。N64_shallow 的 cfg
含 pathlib.Path 类型 source；新生成后该对象会进入 manifest，最终标准
json.dumps 无法序列化。保存的pt也夹带该类型，而DAGDataModule用
weights_only=True加载。元信息同时漏了真正的 dag_config（叶数、入度、样本数等）。

处理：记录实际 dag_config 的可JSON序列化副本，源路径统一字符串；实际vocab
从tokenizer取得。执行agent只需核对一次缺档生成分支的保存/重载链路。

### C2 [缺陷，结果分类前修] 第三个父节点被遗漏

位置：`tools/c_capability/c_common.py:194`。
info为六列 `[depth,p1,op1,p2,op2,p3]`，当前 `range(1,len(row)-1,2)`
只读1、3，漏掉5。原生成器及B的parse_expr使用全部交替父位置。
例如 `[2,10,0,20,0,30]` 应有三个父节点，当前只得10、20。

影响：children缺边，某些非sink被误标sink，结果端指标和机制元数据失真。
不改变模型forward或全节点VALUE原始预测。改成覆盖最后一列，并用一个
三父表达式小例核对。现有测试只覆盖到两父，不能发现此缺陷。

### C3 [实现不足，汇总前补] 只有节点加权，没有约定的逐图等权

位置：`tools/c_capability/summarize.py:113-169,210-282`。
目前累加正确节点数/总节点数，type/depth也一样。相同N的全节点结果可能与
图等权相同，但不同图的sink数量、某深度节点数不同，分解结果会有偏重。
补每图先算比例再平均、节点加权保留；类型/深度缺失的图另列支持数。
`is_internal=(depth>=1)` 包含sink；可保留为“非叶节点”，不要描述成互斥中间节点。

### C4 [判读缺陷+缺少统计，漂移报告前修]

位置：`c_common.py:243-302`；`summarize.py:289-329`。

- `other_only = post_value & ~post_first4` 不要求此前其他字段正确。
  VALUE一直正确、四属性从未全对，也会叫other-only退化。应分别报告
  “VALUE完成后仍存在其他字段错误”与“已正确字段后来变错”，不可混称漂移。
- completion_drift的recovered实际上是末轮是否正确；曾恢复后又错会被写False。
  请明确 final_correct、ever_recovered、稳定恢复/反复，避免一个名字多种含义。
- 在第64轮才首次完成的图没有后续观察，却与有后续观察的完成图进入同一聚合。
  保留完成数，但增加有后续观察数和完成后每个相对轮次的可观察图数；
  不将无观察记作“稳定未漂移”。补原设计要求的完成后保持/退化曲线和反复统计。

这些汇总可用原始npz重算，不要求反复跑GPU。

### C：可接受选择与低优先事项

- 缺档用官方生成器全新生成、固定e143、每档64图、逐图npz均可接受，不必改成
  大JSON。当前抽样按深浅区间随机取，未匹配跨N深度直方图；报告实际分布，
  不将来源/深度混合效应直接归因于N。需要严格规模归因时再匹配。
- `--overwrite-ok` 通过存在检查后仍调用 `jsonl_writer(...open('x'))`，实际会报错；
  建议删掉误导选项并用新目录，或明确定义重跑行为，不能当作已有resume支持。
- C运行前先修C1/C2，做一次针对性生成/加载/短轨迹验证即可；C3/C4补汇总。
  C原有执行授权不变，无需因草稿进入review重新要求用户批准同一评估任务。

## B：现有结果与主实验边界

### B1 [选择协议缺陷；当前256图未受筛除影响]

位置：`tools/b_mechanism/b_mechanism_select.py:200-212`。
T=None时仍直接continue，并把early资格写0；只有共同整图完成后才挑链和
判early。因此没有实现early按自身条件、complete按整图条件的独立资格统计。
本批报告256/256均complete，所以这一分支没有删掉本批候选，不据此作废M0。
后续有未完成图时会漏掉本来early合格的样本。选择器/manifest/runner需共同
处理early可用、late不可用，而不是仅更改计数字段；否则明确限制当前协议范围。

### B2 [对照汇总不足，unrelated模块解释前补]

位置：`b_main_summary.py:90-119`；select.py:276。
unrelated记录有 `for_u`，但center_for只取被注错的target.pos；结果汇总的是
这个无关节点及其后代，没有单独报告本来要保护的原目标u及其下游。
请同时统计被注错分支和原目标分支，保留对应链身份；用实际共同后代/交叉依赖
标记限定解释。全图正确率不能替代“原目标分支有没有受影响”的阴性对照。

### B3 [已知未完成接口，不是已有pilot算错]

`b_main_summary.condition_label` 不支持 leaf_constant*，会跳过D1/D2/D3。
执行单§7已如实列出。先补旧/新真值的专用表，并显式报告不支持条件，不能
把主汇总脚本的成功退出当作五模块均已统计。D2不能机械要求旧题clean配对。
不阻止已支持的early/complete/sustained分阶段推进。

### B：其他记录边界

- pilot的节点数是“节点×干预事件”数，不当成独立图样本量。
- 叶变量的node_truth.parents已被common过滤到数值变量，故parents=[]
  不表示没有题面常数依据；3/42例外尚未证明统一结构规律。
- 子集manifest的counts保留父manifest计数又新增subset_*，可追溯但易误读；
  区分parent_counts与subset_counts，尤其别用256当teacher64覆盖数。
- 本轮只检查逻辑，没有独立复算服务器数值。主实验H/规模的最终授权状态
  以用户最新指示为准，报告中不要把GPT建议擅写为用户已确认。

## 后续分工

C agent修生成/解析及统计，再按原计划小验证和评估；B agent补独立资格、
unrelated观察对象及D123汇总，现有B1、M0/M5记录保留并注明边界。
GPT之后继续检查这些执行代码的修订差异及报告是否一致，不重复代跑实验。
