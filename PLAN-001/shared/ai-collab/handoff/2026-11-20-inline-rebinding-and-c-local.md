# 节点再赋值、C 的正常求解窗口后训练和正式评测

[2026-11-20][user/gpt] Agent B 先做原始 A/B/C 再赋值小测，随后评测三个新 C。Agent C 冻结清单并训练。本批依次交接 A/B/C 再赋值、C 最小验证集评测、三个种子正式训练、三份 final 的全面评估。三个种子在训练阶段可并行。旧运行继续使用旧版本。服务器保存大产物，回传摘要、版本、哈希、覆盖数及失败原因。

## 1. Agent B · 一张题里允许反复赋值

每条等式先按书写顺序读取各操作数最近一次在前文的定义，再绑定等号右面的结果变量。整题一次输入，之后模型自由续算。旧等式的结果位置仍应保持当时的值。例子按数学记法写为 `a=2; k=3; y=a+k; a=a+k; a=a+y; z=a+y`，依次为 2、3、5、5、10、15。模型收到仓库的“表达式 = 变量”格式。对照题把后两次 a 分别改成两个全新名字，之后读取相应新名字，算式、位置、答案及依赖不变。

- 八类短题分别是旧值已被使用后回收名字、读自己再赋值、两路交替复用、一个值被多个算式使用、间隔若干等式后才读、混合依赖、旧消费者保留到多次覆盖后再读、同一算式读取两个已经分别重绑的名字。每类测同一主名字有 2、4、8 个版本，每格 8 图。
- 另在 16、32、64 次定义的随机依赖图上，先生成每行 1–3 个父节点的计算，再标出每个定义的最后一次被读位置。后面新定义从已用完的名字中抽取，覆盖输入叶变量和中间变量，每档 8 图。算法按实际依赖决定可回收集。
- 四路同时延伸的题以两组名字轮换，8、16、30、33 层各 4 图。分别有 39、71、127、139 次定义。前三档有唯一名称对照，139 次定义档受 128 名字限制单列。总计 228 对配对题、4 道仅复用题。这个构造延长累计计算，单层宽度固定为 4，序列仍在 1024 token 内。
- 原始 A_s42、B_s42 和 C_s42/s1234/s2026 使用同一批题。每个条件只初始化一次，观察 D、D+4、D+8（D 为最长依赖链的节点数）。主读数为配对格“唯一名对、复用错”等四格；另报每次定义、旧消费者、新消费者、末端答案、首个对照正确而复用错误的定义行。模型输出恰好等于某一个操作数的旧值代入的结果，仅称为“与读旧值相容”，分母只含模 23 下能区分的新旧结果。原始逐轮 VALUE 可复查，三个观察时点另存逐节点四字段全对和 VARIABLE 正确标志。
- 报每模型、每类、每档。两种命名都失败提示题本身可能已经太深。139 次定义是能力探查，不能单独归因于复用。隔开“名字可以回收”和“物理 token 可以释放”，这里测前者。

从仓库根目录运行（以下命令共用这三个路径）：

```bash
abc_plan=tmp/analysis/c-update-20261119/data/plan.json
local_run=tmp/analysis/c-local-update-normal-v1
local_manifest="$local_run/manifest.json"
python tools/c_local_update/rebinding.py --plan "$abc_plan" --out-dir "$local_run/rebinding"
```

## 2. Agent C · 冻结后启动三种子

各自从同 seed 的原始 C 重新初始化第二层 UPDATE 读出，保留连续反馈；同一份五万张 N32 训练数据、相同抽图与位置、有效 batch 32、50 epoch，以 epoch 50 为正式评估。复用已验证的 GPU 局部标签与 1/2/5/0.1 状态、定位、对齐、清除损失。模型读本轮输入的直接父值求本轮局部目标，间接冲突在父值变化后再出现。全部批次只走 D+1 次前向，不训练额外长程保持。独立保存旧多时点版本。

三种时点等概率，5% 有效 batch 开启、50% 结构候选抽取。前沿在对应深度注错；中途在 max(1,D//2) 抽较浅位置，实际是空、错或已对均由日志读数区分；完成后在第 D 次前向结束时检查整图 VALUE，只有当时全对的图接受该类注错，再学习一次更新。供体用原图改常数后的自然 hidden，前三字段及预期错 VALUE 必须匹配。报告抽中、源合格、接收状态合格、实际替换、VALUE 实际变动。完成后只有一轮受监督，自由续算中的多轮恢复单列评估。

```bash
python tools/c_local_update/local_protocol.py --plan "$abc_plan" --output "$local_manifest"
python tools/c_local_update/checks/minimal_dev.py --plan "$abc_plan" --manifest "$local_manifest" --model C_s2026 --output "$local_run/minimal-dev-s2026-e0.json"
python tools/c_local_update/train.py --plan "$abc_plan" --manifest "$local_manifest" --model C_s2026 --out-root "$local_run/train" --smoke
python tools/c_local_update/train.py --plan "$abc_plan" --manifest "$local_manifest" --model C_s42 --out-root "$local_run/train"
python tools/c_local_update/train.py --plan "$abc_plan" --manifest "$local_manifest" --model C_s1234 --out-root "$local_run/train"
python tools/c_local_update/train.py --plan "$abc_plan" --manifest "$local_manifest" --model C_s2026 --out-root "$local_run/train"
```

**服务器执行顺序**：B 先提交五份原权重的再赋值结果或其路径与身份，并在看板交接。C 冻结清单，用 s2026 在预定的 8 张 dev 图上运行最小评测，交 VALUE、四字段正确率及分类 UPDATE/下游结果，并保留不可注入、未完成等实际分母。此时第二层读出头刚初始化，UPDATE 分数仅用于检验评测链路。再做最大深度三模式反传 smoke。两者可正常出数后立即启动三种子；少样本低准确率或零合格的单个条件照实记录，不是停止条件。若同一清单已有产物或训练已启动，复用已有结果，训练保持运行，缺的 8 图小测单独补。

每个种子 epoch 1 会自动在 128 张 dev 图做基础评测、在其中固定 32 图做分类。此时才报告学过的 UPDATE 结果；B 可在训练继续时整理新算/直接错/健康三组和同图混合，不等 50 epoch。若需同一批 8 图的前后对照，用上述 minimal_dev 命令加 --checkpoint 对应的 epoch-01.ckpt，写独立输出。正式测试由 B 对 epoch 50 评估。

最后三条可交给三块空闲 GPU 上的独立进程，每个进程单独设 CUDA_VISIBLE_DEVICES。默认 micro/donor 32，资源不足可试16/8，以新运行记录实际设置。smoke 用最大深度三模式验证反传，记录剂量；正式训练从原 C 起跑。中断按同身份加 --resume 从已完成 epoch 继续。训练中用 dev；记录阶段指标和用时，保留实际剂量低或性能一般的种子。

## 3. Agent B · 三份后训练 C 的正式评估

清单先冻结 dev、六个 N32/64/128 × 浅深分层及每图目标。每层 128 图做干净能力评估，前 32 图做时点/UPDATE 分类；旧配对、持续污染、噪声、同值换源、新消费者和冻结原 C 功能探针沿用原来的适用范围。唯一名再赋值与此处“修复后新增消费者”分开汇报。

- 时点覆盖训练前沿、训练中途、训练终点 D 的实际完成检查，以及 D+16 内首次真正 VALUE 全对。分开报告机会数、未完成、源拒绝和实际执行。中途正式人工单点扰动单列“原本算对且整图尚未完成”。
- 按实际输入把 UPDATE 拆成父值已备但本节点空、已有数值直接违背父值、正确且局部一致、全局错误但局部仍一致、父值未备等。读第二层计划，末层检查实际算对/保留及二者一致性。同时报告 pooled counts 和图等权；分母为零保留缺失。
- 完成后同图做干净、单点值错、另一个独立节点还原为其早期可读 EMPTY、两者同时出现、父子一起写入局部一致的错值。EMPTY 的真实旧向量须匹配身份/语法/操作；没有合格向量就记录缺失。同图混合看新算、纠错、健康三种位置各自及联合是否正确。人工制造 EMPTY 与自然未算分开。
- 干净题另外记录 VALUE、VARIABLE、四字段的自然漂移和恢复；人工受损分别看目标、直接孩子及全部下游。新消费者统计限定同图 clean 可正确使用时，交叉列出父节点读对/读错 × 新消费者最后两轮算对/算错。原 C 仅作为固定供体功能检验的接收模型。长程观察只用于测试。

```bash
for model in C_s42 C_s1234 C_s2026; do
  python tools/c_local_update/assess.py --plan "$abc_plan" --manifest "$local_manifest" --checkpoint "$local_run/train/$model/final.ckpt" --out-dir "$local_run/eval/$model"
done
python tools/c_local_update/cohort.py --results "$local_run/eval/C_s42" "$local_run/eval/C_s1234" "$local_run/eval/C_s2026" --output "$local_run/cohort.json"
```

各模型可以独立评估，单个目录一个写入者。验收交三份 final/hash、六分层、各条件实得样本和三种子分别结果。正式评估仅该批后训练 C，可报告其行为及跨种子一致性；因果归因于后训练或过程监督需要相应对照。
