# C：纠错后训练与中间 UPDATE 监督

状态：GPT 实现，等待执行 agent 的真实权重小验证。用户已批准本批训练；性能一般不构成提前停止条件。

本目录独立于原训练入口，不修改 `experiments/model.py`、旧实验和既有运行。统一分支 `reproduction`，共享服务器工作树串行交接。

## 1. 本批四次训练

| 显示名 | 入口 arm | 起点 | 上限 |
|---|---|---|---|
| C（纠错） | correction | 原 C s2026、s42、s1234，各 e143 | 各 50 epoch |
| C（UPDATE） | update | 原 C s2026 e143 | 50 epoch |

权重来自已核对的 `tmp/analysis/abc-workspace-20261119/data/models.json`，不使用欠拟合 s2025，不再训练 A/B，不加普通续训对照。UPDATE 不自动扩到另两个种子。旧 C hold/noise/semantic 可在 prepare 时用 `--extra-models` 加入评估注册表；不新增训练。

固定原始训练文件 `experiments/data/Tr32Test128-ADD/train_data.pt`：50,000 张 N32、depth≤6、ADD mod23、6 个叶节点、fanin1–3；SHA256 `cee36b84b01b7c1fdbd2e32619b402e571275c534a4d685684db07a6e11de1ce`。不是另生成50k训练图。词表 `[7,129,2,25,2,2]`，128个变量名。

- 每 epoch 每图恰好一次，按 D 分组后打乱，不丢尾批；D=max depth+1，两路都训练 D+1 次连续前向。
- 有效 batch32，micro8；0.05 的抽签在有效 batch 层，不能每个 micro 重抽。
- 新 AdamW：主模型 lr=1e-4；UPDATE 读出头 lr=5e-4；betas .9/.999、eps1e-8、weight_decay .01；50 epoch cosine 衰减至初始的0.1。不恢复旧 optimizer。
- float32，保留原 dropout；不截断 BPTT，不引入 gradient clip（只检查非有限梯度）。同深度组损失除以 D+1。
- 保存 last（每100 step及epoch末）、epoch1/3/5/10/15/20/25/30/35/40/45/50、best-dev、final。主结果固定 final=e50；best-dev 仅补充。
- 小评测一般或暂时下降仍跑到50；NaN、OOM、代码/数据错误应停下报告，不带错硬跑。

## 2. 连续污染及与 B 的对应

B 当前配置核对为 batch32、`last`、5% batch ×50%候选位置。其候选由 teacher 新可算位置给出，输入槽通常是 EMPTY；随机替换从全部25个 VALUE 类别采样，包括 N/A/EMPTY，不能把它描述为“必定不同的数值”。

本批对齐**抽签层级、概率、新可算位置**。对每个结果节点（包括叶节点、sink），只在 forward index=depth(u) 有一次候选机会；D+1的最后保持轮不注错。

1. 为抽中的 u 选非零 delta∈1…22，目标 b=(原真值+delta) mod23。
2. 根据图中叶常数对 u 的模23线性系数，改一个可逆叶常数，使同图反事实版本中的 u 真值正好为 b。
3. 用**当前权重**、eval/no_grad 运行源图 depth(u)+1 次，取 u 的完整 hidden 行。源向量比接收端前向前状态多算一次，这是有意模拟“新槽位先填入一个错数”。不等待源模型后来算对，不反复挑 donor。
4. 源向量前四字段必须是原 SYNTAX/VARIABLE/OPERATION + b，否则跳过并计数；没有可逆叶系数也跳过。
5. 仅替换接收图 u 的 hidden 行，donor detach，其余 hidden/BPTT 保留。不会把全图解码再嵌入。

这不等同于离散精确改 VALUE：完整 hidden 行可能携带其他信息。日志同时报候选数、尝试数、源通过数、实际注入率、改前 EMPTY/numeric/N/A、位置/轮次、源和接收端前向成本。训练标签仍对所有轮监督；correction 的前四因子 CE 延续原 teacher 进度，PAD口径保留；原 aux loss 权重为0。

### 用户补充：读错还必须检查能否用错

`functional_probe.py` 单独用**同 seed、未做本批纠错后训练的原 C**接收向量，检查直接子节点 q。固定64张dev图，early 与 D+4 两个时点；在训练前跑一次，每个小评测检查点重做。

- `wrong_value`：与训练完全相同的一叶常数修改法。
- `matched_wrong_value` / `same_value`：存在两叶系数时，改相同两个常数，分别改变/保持 u 的值。无法构造时明确标记 fallback，不能称作同值不同来源证据。
- `source_original`：不改源题，只替换源模型原图的 u；`receiver_exact`：接收模型原向量写回。
- 原 C 生成向量→原 C 接收：检验同一表示空间内污染是否会让 q 用错。
- 当前训练权重生成向量→原 C 接收：检验本批实际向量的影响，**跨权重表示变化是混杂因素**，必须看同值/原图来源对照。
- 分别保留源模型的读出、冻结接收模型对注入后向量的读出、下游实际计算值。源读成5而冻结接收头读成别的值的样本单列计数，不能称为两模型都识别成同一种错。
- 源端读出通过后，再报固定资格分母中 q 首轮错误数、恰好等于错父值代入结果的数量和随后4轮值。其他父节点要原本正确，clean q 首轮能算对；其余样本保留但不计入该条件比例。
- early 源 age=depth(u)+1，与训练 donor 一样；接收端取 u 刚计算后的状态，让下游有可读的其他父值。这是功能检验，不伪称完全复刻训练的 EMPTY 起点。

不根据传播成功与否反向筛训练图、不提高噪声直到“成功”。没有传播既可能是向量未被如此使用，也可能是接收端恢复快；不能仅凭这个给机制下结论。

## 3. UPDATE 的单次前向与标签

原4层结构保持；layer_1（第二层）输出经过现有 REGRET 读出头得到 UPDATE。该头在此路重初始化，最后一层也用它读出清零目标。**没有 gate、没有冻结不更新位置、没有把中间 mask 写回向量**。MODIFICATION 不启用。

每轮输入来自本模型自由续算；第0轮用原始输入字段定义状态，后续以输入 hidden 的读出定义状态。参考依赖关系来自原题 metadata，而非模型漂移后的变量名。

| 输入情况 | 第二层 UPDATE 标签 | 最后一层 VALUE 标签 |
|---|---|---|
| 叶结果与原题常数不一致 | TRUE | 原题常数 |
| 所有父值为数字，u≠父值之和 mod23 | TRUE | **输入中的**父值之和 |
| 已与输入父值一致 | FALSE | 保留当前值 |
| 有父值尚不是数字 | FALSE | 保留当前值，等待 |

最后一层的前三字段仍监督原题正确身份/语法/运算，节点 UPDATE 输出监督 FALSE。这样不会出现同一轮既要求 q 跟输入错父值计算，又要求 q 达到父节点本轮修正后的最终真值的矛盾。全图正确性另用原题真值评估。

例：a本应3，却输入5；b=4，u=a+b原本7。该轮监督 a→3，同时 u→9。下一轮 a已经3，再要求 u→7。这是同步局部计算规则，不提前把修正后的3泄漏给同一轮 u 的标签。

```
L = L_state + 2 L_locate + 5 L_align + 0.1 L_clear
```

- state：前三字段有效非PAD CE；节点 VALUE 按应更新/应保持两组平衡；非节点 VALUE 单列有效位置CE。
- locate：第二层 UPDATE CE，TRUE/FALSE两组平衡。
- align：中间预测更新 m 与末层实际 VALUE 改变 d 不一致时，参考正确标签 s 判罚。d=s 则加罚中间定位CE；m=s 则加罚末层 VALUE CE。两个都错但彼此一致，仍由 state/locate罚。argmax判罚掩码 detach，不能把错误的一侧当教师。
- clear：末层节点 UPDATE FALSE CE。
- 各组先逐图均值，再逐图等权；不存在的组不计入组数。micro-batch改变不改变这套归一化定义。

对齐指标是“VALUE读出是否改变”，不是声称定位头已因果控制内部计算。正是因此没有硬门控，并同时报告定位、执行和正确性。

## 4. 固定评估

`prepare.py` 生成并冻结dev与test；排除原训练/旧val及本地存在的旧能力/长程pilot数据的**同token图**，检查新划分互斥；不声称排除了所有图同构。生成池固定，不因成功少补样。各模型用同一plan和target，失败不删除。

| 模块 | 固定规模、行为 |
|---|---|
| 小评测 | dev128 N32，clean到D+16；前64图 early/D+4，错u、同值来源、原样写回，各16轮；另原C功能探针64图 |
| 基础能力 | N32/64/128 × shallow(depth≤6)/deep(7–12)，每档128图；到D+64 |
| 单次/联合污染 | N32、N128各128图（各深度档前64）；u刚计算后、D+4、D+16；改u、改q、同时改u/q、同值、原样；各16轮 |
| 持续污染 | N32、N128各64图（各档前32）；D+4起对u固定写2或4轮，再释放16轮 |
| 无语义噪声 | 同64图；u行高斯方向，范数=hidden范数×1/2/3/4/6，两固定方向；A/B离散不适用，明确记NA |
| 后续使用 | N32各档前8图；u受扰后0/1/2/4轮，追加一条真实新方程 `u ADD leaf = 新变量`，新位置用embedding，旧非PAD hidden保持；与未扰动追加对照，16轮。不是换整张输入，也不宣称突破128变量限制 |
| 冻结原C功能检验 | 正式评估每个N32/N128深度档前32图；同上原C接收诊断 |

D+4/D+16是固定预算名，不保证整图完成；记录起点是否全对，不能把所有这些样本称作“已完成后”。early允许q尚EMPTY；源q在同轮还不可得的联合替换记录 source_rejected，不借未来向量补齐。联合反事实碰巧不改变q值时单列，不算两节点错。

核心指标：**节点 VALUE 正确率、节点前四属性全对率**，逐图等权；整图全对仅辅助。按深度、父节点数、叶/内部/sink分层，四因子分列。干预逐轮记录输入/输出，分开选中、源通过、起点资格、实际注入、实际损伤、恢复分母；no_value不丢弃。末两轮正确称“窗口末稳定”，不称永久稳定。保留恢复后再错、首次回正、最后正确段起点、r1/2/4/8/16。原始轨迹支持执行agent补共同资格表，不把各模型不同受损集合的条件恢复率直接作胜负结论。

原 C/A/B 在新固定题上重评，不直接抄不同旧测试表。新两路都是一个训练配方整体；未加入普通续训对照，故不能把收益唯一归因于污染或定位头，也不能据此声称 loop 相对普通 Transformer 优势已被证明。

## 5. 服务器串行执行

先确认共享树无人运行，再同步 `reproduction`。服务器选择空闲GPU；脚本不选卡、不kill、不pull、不修改别人的工作树。以下命令在仓库根目录、已有torch环境下运行：

```bash
python tools/c_update/prepare.py --out-dir tmp/analysis/c-update-20261119/data
python tools/c_update/execute.py --phase smoke --plan tmp/analysis/c-update-20261119/data/plan.json --out-dir tmp/analysis/c-update-20261119 --device cuda --num_threads 4
python tools/c_update/execute.py --phase train --plan tmp/analysis/c-update-20261119/data/plan.json --out-dir tmp/analysis/c-update-20261119 --device cuda --num_threads 4
python tools/c_update/execute.py --phase evaluate --plan tmp/analysis/c-update-20261119/data/plan.json --out-dir tmp/analysis/c-update-20261119 --device cuda --num_threads 4
```

smoke和main目录分开；main从原权重开始，不从smoke后权重开始。训练队列为C纠错s2026→UPDATE s2026→C纠错s42→s1234。`train` 可重启：完整运行跳过，未完成运行从last和optimizer/RNG恢复，丢弃last之后日志。原计划/代码/hash/micro不匹配会报错，不静默混合。中断的评估目录应保留并换新输出目录后重跑，不能覆盖成“完整”。

`execute.py` 默认正式评估5个原权重+4个final；best-dev若需补充，直接用evaluate的`--checkpoint .../best-dev.ckpt --arm ...`另目录，不能代替final。旧后训练可通过prepare的extra-models冻结进本批新数据评估，来源seed必须明确。

## 6. 验收与交付

1. 本地stdlib检查负责同步标签、源算术、frontier/有效batch抽签、分组不丢图、对齐方向、恢复分母。torch检查留服务器。
2. `verify_server.py`核对原forward完全一致、teacher标签与原函数一致、新方程不覆盖旧hidden、两轮BPTT和UPDATE梯度；同值/错值冻结原C探针先交原始计数。
3. 两个单batch smoke检查新梯度/断点保存/加载/评估全链路；force污染仅smoke，不能把100%启用当正式剂量。
4. 验收定位头不是全FALSE蒙混：交 TRUE/FALSE标签数、precision/recall所需tp/fp/fn、实际VALUE改变、计划与执行一致、执行正确、末层清零。性能差如实记录，不能为通过验收修改题或标签。
5. 正式训练交 `training-audit.json`、每epochloss/有效剂量/前向成本、小评测、四份final的路径/hash；大权重、raw records留tmp。结果报告和小表进docs/records，board/log更新。
6. 原C功能探针必须分“源读出通过”和“下游特异地用错”；跨权重对照单列。低通过/低传播不是允许暗换污染方法的理由，先报告事实。
7. 运行期间不改代码；需要修复时保留原运行身份，在空闲交接时修改、记录影响。GPT审阅执行适配与结果；无需重复已通过的无关检查。
