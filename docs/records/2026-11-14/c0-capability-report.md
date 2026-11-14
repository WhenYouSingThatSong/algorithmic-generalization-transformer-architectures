# C0 能力与自由运行漂移：六档首批探索报告

日期：2026-11-14 19:58 CST；21:00 补充现有 NPZ 的可复跑事件分析。  
状态：**首批六档正式运行完成；探索结果，不是最终评测协议，也未训练 C+**。  
设计：`docs/records/2026-11-14/c0-capability-exploration-plan.md`  
代码：`9df92a8` + 本地未提交的 C 入口路径/测试修复；GPT review 修复见 `5669677`。  
产物：`tmp/analysis/c0-capability-20261114/`（`data/`、`runs/`、`runs/summary/`、`logs/`、`analysis/`）。  
审阅修复：2026-11-14 22:57 按 `82d9412` 的 C-1—C-5 修正独立 preserve、
graph-equal 字段表、`completed_by_H` 队列、事件 positions 和测试夹具；
未重跑模型。

> 说明：以下数字是特定 checkpoint、固定数据档和 64 轮自由续算协议下的
> 描述性统计，不是机制结论、最终能力结论或 C+ 设计决定。原始逐图 NPZ
> 为准；派生表可由 `analyze_existing.py` 重算。未做 hidden probe、干预
> 或因果实验。

## 0. 观察摘要（描述性）

1. **C0 的 VALUE 计算能跨规模、跨深度泛化，但完成时刻高度依赖深度。**
   s42 在 N32/64/128、浅/深六档的 VALUE 完成率为 59–64/64；s1234 为
   32–64/64，主要缺口在 N128。大多数完成的图在 `实际最大深度 + 1` 轮首次
   整图 VALUE 全对；N128_deep 的中位 tV 为 12（s42）/13（s1234）。

2. **本批 64 轮窗口内出现完成后退化：VALUE 会漂移，规模越大越严重。**
   H=64 时整图 VALUE 全对率从 s42 N32_shallow 的 47/64 降到 N128_deep 的
   1/64；s1234 的 N64/N128 档只剩 0–7/64。节点级 VALUE 仍较高
   （internal 0.62–0.96），说明不少图只是少数节点错，但整图 conjunction
   已经把“仍算对”变成 0。这里只陈述 64 轮窗口内的观察，不外推“没有
   稳定不动点”或 64 轮之后的行为。

3. **first-4 的崩溃几乎全部来自 VARIABLE，而不是四个字段一起坏。**
   在 ADD/单一语法任务中 SYNTAX、OPERATION 真值恒定，模型在 H=64 仍为
   1.000；VARIABLE 从 H=16 的约 0.95–1.00 跌到 H=64 的约 0.08–0.11。
   因此本任务的 first-4 实际近似于 `VALUE ∧ VARIABLE`。H=36 时所有档已经
   没有一张图 VARIABLE 全对，整图 first-4 必然为 0；不能把 first-4 曲线
   读成“计算字段全面失效”。

4. **两条失败轴要分开：完成失败/延迟 vs 完成后的状态漂移。**
   s42 的完成几乎严格按深度时钟；s1234 在大图上有明显长尾（N64_shallow
   有 tV=61，N32_shallow 有 tV=32），N128_deep 32 张从未完成。完成后的
   VALUE 漂移则两个权重都存在，且在 N64/N128、深图上更强。

5. **H=64 切片中 VALUE 错误随真实深度增加，且错误值是合法余数而非
   EMPTY/NA。** s42 N128_deep 完成图中，H=64 的 VALUE 错误率从 depth 1
   的 0.2% 升到 depth 10–12 的约 50%；s1234 同档从 2.6% 升到 93%。
   N32 上不少图只有 1–2 个 VALUE 错，N64/N128 多数完成图有 >5 个错。
   这只是 H=64 时刻的切片，不能单凭它说深层节点“最先”退化。

6. **s42 与 s1234 的差异主要在 N128，不能把 e143 当作“最佳 checkpoint”。**
   s42 在 N128_deep 的 VALUE 完成 59/64、H=64 仍全对 1/64；s1234 为
   32/64、0/64。两者都是固定 e143 的可比性选择，不是按本批结果挑出的
   最优权重；C+ 比较必须保留多 seed。

7. **对 C+ 讨论的提示（假设层面，不是设计决定）：先决定“保持什么”，
   再设计干预。** 当前证据提示把 VALUE 稳定性与 VARIABLE 身份保持分开
   记录为两个观测轴；
   如果 C+ 只优化 first-4 合计，会被 VARIABLE 读出头主导；如果只优化
   VALUE，可能掩盖读出头或表示稳定性问题。后者目前只是待检验假设，
   本批没有 hidden probe，不能据此断言表示已经离开训练流形。

## 1. 数据与权重

### 1.1 六档固定数据

每档 64 张固定图，两个权重使用完全相同的图；抽样只按固定 seed 洗牌，
不使用模型是否答对。manifest 见
`tmp/analysis/c0-capability-20261114/data/manifest.json`。

| 档 | 来源 | 合格池 | 实际最大深度 min/mean/max | 非 PAD 长度 min/max |
|---|---|---|---|---|
| N32_shallow | `tmp/analysis/matched-depth/ADD/testdata_N32.pt` | 1024 | 4 / 5.50 / 6 | 159–205 |
| N32_deep | `tmp/analysis/heldout-seed42/ADD/testdata_N32.pt` | 319 | 7 / 7.56 / 11 | 159–197 |
| N64_shallow | 新生成（官方生成器） | 256 | 6 / 6.00 / 6 | 349–399 |
| N64_deep | `experiments/evaluation/val_datasets/Tr32Test128/ADD/testdata_N64.pt` | 954 | 7 / 8.84 / 12 | 347–399 |
| N128_shallow | 新生成（官方生成器） | 256 | 6 / 6.00 / 6 | 717–789 |
| N128_deep | `tmp/analysis/heldout-seed42/ADD/testdata_N128.pt` | 755 | 9 / 10.89 / 12 | 705–789 |

表中的深度/长度是**选中 64 张**的统计，不是整个合格池的范围。
生成配置：ADD、mod 23、6 叶、fan-in 1–3、128 变量名池（129 词表）、
`max_depth=6`、`num_samples=256`、生成 seed `20261114`；再从 256 张中
按固定 seed 选 64。生成档没有与现有 54 张 N64 浅图混合。

**数据边界**：N64_shallow 与 N128_shallow 全部落在实际深度 6，而
N32_shallow 是 4–6；N128_deep 只有 9–12，没有 7–8。因此“同规模浅/深”
与“同深度跨规模”都不是严格配对，跨档比较要保留这一 caveat。N32/N64/N128
的 deep 档也不是同一深度分布。

### 1.2 权重

| 名称 | seed | checkpoint | SHA256 |
|---|---:|---|---|
| `s42_e143` | 42 | `tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-14-02:35:10/epoch=143-val_loss=0.0000.ckpt` | `4a00d6a2…8903a798` |
| `s1234_e143` | 1234 | 同目录 `seed-1234 - 2026-11-14-02:38:10/epoch=143-val_loss=0.0000.ckpt` | `0bca3a2b…d7d70475` |

实际完整哈希、路径和 argparse 参数在
`runs/<weight>/run_meta.json`；`weight_sha256` 与 c_common 中登记值一致。
C0 的训练分布是 N32、depth ≤6、129 词表；N64/N128 与 deep 档都是 OOD
能力探针，不能称作同分布泛化。s2025 未测（已欠拟合）；s2026 仍在训练，
未纳入本批。C0′ 与 vocab33 不参与。

## 2. 运行与口径

- 每图从原始输入状态出发（变量 VALUE=EMPTY）；`hidden = token_embedder(input)`，
  之后每轮 `hidden, factor_preds = model(hidden, nointerm=True)`，**只回灌
  hidden，不回灌解码字段**。解码只用于记录，不改变下一步输入。
- 每图固定 64 轮；显式设置，不使用旧配置的 `max_val_loop=36`。因此 64 是
  观察上限，不是“模型应该在第 64 轮完成”的假设。
- 训练循环只到每个 batch 的实际 max depth；本批 64 轮远超过训练深度。
  **完成后的漂移是自由续算/分布外迭代的属性，不是常规短轮数 eval 的
  准确率**；它回答的是“若继续把 hidden 回灌，状态能保持多久”，不是
  “模型在训练分布内的错误率”。
- 每个 discriminator 节点（数值节点）保存 SYNTAX/VARIABLE/OPERATION/VALUE
  的贪心解码和固定真值；节点集合排除 PAD、运算符、题面常量。节点包括
  6 个叶变量（VALUE 已在输入中给出）、内部节点、结果端 sink。
- **VALUE 节点正确率**：所有数值节点中 VALUE 正确的比例；**internal**
  指非叶节点（含 sink），**sink** 单列。叶节点 VALUE 在所有轮为 1.000，
  因此总节点指标会被 6 个叶节点轻微抬高；正文以 internal 为主。
- **整图 VALUE 全对率**：整图所有数值节点 VALUE 都正确的图比例；不等于
  节点正确率，作为辅助。
- **first-4**：同一数值节点四个字段全对；本任务 SYNTAX 恒 0、OPERATION
  恒 1，所以它实际等价于 `VALUE ∧ VARIABLE`。第 3 节给出逐字段分解。
- **tV / t4**：第一次整图 VALUE 全对 / 第一次整图 first-4 全对的 1-based
  轮次；没有则为空。t4 存在时必然 ≥ tV；本批所有 t4 存在的情形都恰好
  `t4=tV`，没有观察到“VALUE 先完成、VARIABLE 后补对”的图。
- **完成后漂移**：从 tV 的下一轮开始，按相对轮次记录第一次整图 VALUE
  失败、是否恢复、末轮是否正确。`stable_recovered` 指退化后**末尾连续
  两轮都正确**（`c_common.completion_drift` 语义），不是仅末轮；
  `recovered` 只是“退化后末轮正确”的兼容别名。本报告的“恢复”不自动
  等于永久恢复。
- 相对轮次曲线每轮以 `n_observable` 为分母；本批所有完成图的 tV≤61，
  因此都有至少 1 个后续观察。若未来出现 tV=64，`post_observed` 与末轮
  标志不可用，必须把无观察图单列，不能算作“保持”。
- 每档 64 图节点数相同，因此**全数值节点**的逐图等权与节点加权在本批
  数值上相同；叶数固定时 internal 也相同。sink/深度桶的节点数随图变化，
  不能笼统宣称所有指标宏微一致；summary 仍同时保留两套值。
- 运行：GPU6，两个权重串行，各 384 图 × 64 轮。s42 耗时 95.4 s，s1234
  99.7 s；日志无 Traceback/OOM。seed2026 训练在 GPU0，未受影响。

## 3. 能力结果

### 3.1 完成与 H 切片

“completed”指 64 轮内至少有一次整图 VALUE 全对；`H16/H64 graph VALUE all`
是固定轮次切片的整图全对率，包含尚未完成的图。

| weight | stratum | VALUE completed | tV median | first-4 completed | t4 median | H16 graph VALUE all | H64 graph VALUE all | H16 internal VALUE | H64 internal VALUE | H64 sink VALUE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 64/64 | 7 | 64/64 | 7 | 1.0000 | 0.7344 | 1.0000 | 0.9609 | 0.9499 |
| s42_e143 | N32_deep | 61/64 | 8 | 61/64 | 8 | 0.9375 | 0.5781 | 0.9898 | 0.9231 | 0.8984 |
| s42_e143 | N64_shallow | 64/64 | 7 | 64/64 | 7 | 0.9844 | 0.2656 | 0.9995 | 0.8987 | 0.8534 |
| s42_e143 | N64_deep | 62/64 | 10 | 62/64 | 10 | 0.9688 | 0.3125 | 0.9987 | 0.9001 | 0.8678 |
| s42_e143 | N128_shallow | 60/64 | 7 | 60/64 | 7 | 0.9219 | 0.0469 | 0.9963 | 0.8872 | 0.8335 |
| s42_e143 | N128_deep | 59/64 | 12 | 59/64 | 12 | 0.9062 | 0.0156 | 0.9992 | 0.8105 | 0.7320 |
| s1234_e143 | N32_shallow | 64/64 | 7 | 63/64 | 7 | 0.9688 | 0.4688 | 0.9982 | 0.9171 | 0.8790 |
| s1234_e143 | N32_deep | 63/64 | 8 | 63/64 | 8 | 0.9844 | 0.5312 | 0.9994 | 0.8882 | 0.8547 |
| s1234_e143 | N64_shallow | 60/64 | 7 | 58/64 | 7 | 0.8906 | 0.0625 | 0.9970 | 0.8297 | 0.7511 |
| s1234_e143 | N64_deep | 59/64 | 10 | 56/64 | 10 | 0.8438 | 0.1094 | 0.9876 | 0.8168 | 0.7641 |
| s1234_e143 | N128_shallow | 42/64 | 8 | 40/64 | 8 | 0.6250 | 0.0000 | 0.9927 | 0.6883 | 0.5651 |
| s1234_e143 | N128_deep | 32/64 | 13 | 19/64 | 12 | 0.4375 | 0.0000 | 0.9823 | 0.6249 | 0.5053 |

读法：

- **H16 基本是“完成后不久”的能力切片**。s42 六档的整图 VALUE 全对率为
  0.906–1.000；s1234 在 N32/N64 为 0.844–0.984，但 N128 只有 0.625/0.4375。
- **H64 的整图全对率不是同一件事**。它同时包含“还没完成”和“完成后漂移”
  两类失败；N128_deep 的 0.0156（s42）与 0（s1234）主要是后者，因为
  H16 已有 0.906/0.4375。
- **internal VALUE 在 H64 仍高，sink 更低**。以 s42 N128_deep 为例，
  internal 0.8105、sink 0.7320；s1234 为 0.6249/0.5053。这只是说明
  **H=64 时**sink 正确率更低，不能据此说结果端更早失稳；深层内部节点
  同样有错。
- **first-4 completed 的缺口主要来自 s1234 大图**：s1234 N128_deep 只有
  19/64 曾有 t4；本批有 13 张图在 tV 时 VARIABLE 已不全对，因此未观察到
  t4。这里保留“本批未观察到”的事实，不外推为以后不可能恢复并达到 t4；
  其余档 first-4 完成数与 VALUE 接近。

### 3.2 逐字段分解：first-4 的下降是 VARIABLE

本任务的 SYNTAX/OPERATION 真值恒定，模型在 H=64 的这两个字段仍全对。
下表为 internal 节点的 VARIABLE 正确率；总节点（含叶）在同一轮约
0.08–0.11，和 internal 接近，说明叶节点的 VARIABLE 身份也一起丢了。

| weight | stratum | H16 VARIABLE internal | H64 VARIABLE internal | H64 SYNTAX internal | H64 OPERATION internal |
|---|---|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 0.9952 | 0.0889 | 1.0000 | 1.0000 |
| s42_e143 | N32_deep | 0.9934 | 0.0907 | 1.0000 | 1.0000 |
| s42_e143 | N64_shallow | 0.9962 | 0.0846 | 1.0000 | 1.0000 |
| s42_e143 | N64_deep | 0.9965 | 0.0819 | 1.0000 | 1.0000 |
| s42_e143 | N128_shallow | 0.9973 | 0.0854 | 1.0000 | 1.0000 |
| s42_e143 | N128_deep | 0.9985 | 0.0850 | 1.0000 | 1.0000 |
| s1234_e143 | N32_shallow | 0.9633 | 0.0998 | 1.0000 | 1.0000 |
| s1234_e143 | N32_deep | 0.9531 | 0.0956 | 1.0000 | 1.0000 |
| s1234_e143 | N64_shallow | 0.9696 | 0.1072 | 1.0000 | 1.0000 |
| s1234_e143 | N64_deep | 0.9736 | 0.1061 | 1.0000 | 1.0000 |
| s1234_e143 | N128_shallow | 0.9761 | 0.1119 | 1.0000 | 1.0000 |
| s1234_e143 | N128_deep | 0.9776 | 0.1105 | 1.0000 | 1.0000 |

`analysis/field_acc_by_H.csv` 同时输出节点加权（micro）和 graph-equal
（macro）两列；all/internal 在固定 N、固定 6 叶时相同，sink/深度桶的
节点数随图变化，可能不同。下表用 internal 的 micro，sink 的 graph-equal
见辅助表。

补充事实：

- **H=36 时所有档没有任何一张图 VARIABLE 全对**（两个权重、六个档均为
  0/64）；因此从 H=36 起整图 first-4 必然为 0。这个结论不依赖
  VALUE 是否仍正确。
- **s42 在 tV 时 VARIABLE 仍全对**（`other_error_at_value_completion=0`），
  之后才漂移；s1234 在 N128_deep 有 13/32 张图在 tV 时 VARIABLE 已不全对，
  N128_shallow 2/42、N64_deep 3/59、N64_shallow 2/60、N32_shallow 1/64。
- VARIABLE 预测不是塌缩到某一个固定类：在 N32_shallow 上检查，H=64 的预测
  仍覆盖 117–128 个变量名，但逐节点轨迹在若干轮后跳到另一个名字并保持
  一段时间（示例见 3.6）。
  这更像隐藏状态/读出头进入另一个稳定读出，而不是简单输出常数。
- **不能据此说隐藏状态已经不再编码变量身份**。本批只观察模型自己的
  VARIABLE 头；隐藏状态是否仍线性可读、是否仍保留变量信息，需要 probe
  或干预实验（例如每轮刷新叶节点 token embedding、或训练线性 probe），
  本批未做。

### 3.3 完成后漂移

下表只统计 64 轮内完成过 VALUE 的图。`ever degraded` 是完成后至少一次
整图 VALUE 失败；`never degraded` 是完成后整图 VALUE 一直正确（且有后续
观察）；`last-round correct` 是末轮整图全对。最后一列是
“末轮正确 / completed”，**不是**从完成后从未退化的保持率。相对轮次
从 tV 的下一轮算起。两个首次退化列都是**平均**相对轮次；VALUE 列的分母
是 ever degraded 图，VARIABLE 列的分母是 3.7 表中 `variable new-error`
支持数（两者分母不同，不能直接相减）。

| weight | stratum | VALUE completed | ever degraded | never degraded | last-round correct | last-round correct / completed | VALUE first deg mean rel | VARIABLE new-error mean rel |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 64 | 17 | 47 | 47 | 73.4% | 42.7 | 12.3 |
| s42_e143 | N32_deep | 61 | 24 | 37 | 37 | 60.7% | 39.2 | 10.6 |
| s42_e143 | N64_shallow | 64 | 47 | 17 | 17 | 26.6% | 35.8 | 10.8 |
| s42_e143 | N64_deep | 62 | 42 | 20 | 20 | 32.3% | 32.3 | 8.2 |
| s42_e143 | N128_shallow | 60 | 57 | 3 | 3 | 5.0% | 27.6 | 10.5 |
| s42_e143 | N128_deep | 59 | 58 | 1 | 1 | 1.7% | 20.6 | 5.6 |
| s1234_e143 | N32_shallow | 64 | 34 | 30 | 30 | 46.9% | 34.5 | 8.6 |
| s1234_e143 | N32_deep | 63 | 30 | 33 | 34 | 54.0% | 32.7 | 6.2 |
| s1234_e143 | N64_shallow | 60 | 56 | 4 | 4 | 6.7% | 26.9 | 6.9 |
| s1234_e143 | N64_deep | 59 | 54 | 5 | 7 | 11.9% | 23.0 | 4.9 |
| s1234_e143 | N128_shallow | 42 | 42 | 0 | 0 | 0.0% | 15.8 | 5.7 |
| s1234_e143 | N128_deep | 32 | 32 | 0 | 0 | 0.0% | 11.2 | 2.2 |

读法：

- **VARIABLE 的首次新错平均早于 VALUE 的首次退化**：s42 的 VARIABLE
  平均相对轮次为 5.6–12.3，VALUE 为 20.6–42.7；s1234 为 2.2–8.6 与
  11.2–34.5。注意这两个均值来自不同事件集合，不是逐图先后；同图两类
  事件都发生的比较见 3.8（493 张中 VARIABLE 先 479、同轮 1、VALUE 先 13），
  未发生者单列。first-4 在 VALUE 仍较稳定的阶段开始下降。
- **VALUE 漂移随规模/深度增强**：s42 的 N32_shallow 末轮正确/完成图比例
  为 73%，N128_deep 为 1.7%；s1234 N128 两档为 0。s1234 N32_deep 的
  54% 略高于 N32_shallow 的 47%，但 N32_deep 本身完成得更晚、完成图更少，
  不能简单解释为“更深更稳”。
- **恢复很少**：s42 六个档没有一张图达到 `stable_recovered`（末尾连续
  两轮正确）；s1234 只有 N32_deep 1 张、N64_deep 2 张。其余退化图到
  H=64 仍错。64 轮之外是否还会回正，本批没有观察。
- **本批所有完成图都有后续观察**：tV 最大 61，故 `no_post_observation=0`；
  因此上表的 never/last-round 两列不会把“没观察到”误算成“保持”。若未来
  tV=64，该图必须单列。
- **退化不是一次全局崩溃**：N32 上有相当比例的图只有 1–2 个 VALUE 错；
  这些图在节点指标上仍接近全对，但整图全对率已经掉。N64/N128 则是
  >5 个错的图占多数（见 3.4）。

### 3.4 H=64 的 VALUE 错误随深度增加；N32 是稀疏错误

H=64、只统计完成图的整图 VALUE 错误节点数。`value_error_counts.csv` /
`value_error_by_depth.csv` 的队列是 `completed_by_H`（`tV <= H`）；当前
H=64 与正式完成集相同，未来 H=16 不会悄悄沿用 64 轮完成队列。

| weight | stratum | completed | 0 wrong | 1–2 wrong | 3–5 wrong | >5 wrong | max wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 64 | 47 | 7 | 7 | 3 | 11 |
| s42_e143 | N32_deep | 61 | 37 | 14 | 3 | 7 | 13 |
| s42_e143 | N64_shallow | 64 | 17 | 13 | 14 | 20 | 33 |
| s42_e143 | N64_deep | 62 | 20 | 12 | 10 | 20 | 44 |
| s42_e143 | N128_shallow | 60 | 3 | 8 | 9 | 40 | 51 |
| s42_e143 | N128_deep | 59 | 1 | 2 | 5 | 51 | 93 |
| s1234_e143 | N32_shallow | 64 | 30 | 18 | 7 | 9 | 16 |
| s1234_e143 | N32_deep | 63 | 34 | 5 | 10 | 14 | 20 |
| s1234_e143 | N64_shallow | 60 | 4 | 7 | 15 | 34 | 47 |
| s1234_e143 | N64_deep | 59 | 7 | 8 | 9 | 35 | 35 |
| s1234_e143 | N128_shallow | 42 | 0 | 0 | 0 | 42 | 102 |
| s1234_e143 | N128_deep | 32 | 0 | 0 | 0 | 32 | 93 |

按真实依赖深度的 VALUE 错误率（H=64，完成图，每个深度桶是该轮节点计数）：

| depth | s42 N32_shallow | s42 N128_deep | s1234 N128_deep |
|---:|---:|---:|---:|
| 0 | 0/384 (0.0%) | 0/354 (0.0%) | 0/192 (0.0%) |
| 1 | 1/398 (0.3%) | 1/533 (0.2%) | 8/308 (2.6%) |
| 2 | 8/364 (2.2%) | 11/653 (1.7%) | 21/367 (5.7%) |
| 3 | 16/327 (4.9%) | 43/814 (5.3%) | 63/465 (13.5%) |
| 4 | 15/277 (5.4%) | 76/901 (8.4%) | 108/487 (22.2%) |
| 5 | 16/187 (8.6%) | 156/989 (15.8%) | 168/538 (31.2%) |
| 6 | 9/111 (8.1%) | 216/950 (22.7%) | 218/529 (41.2%) |
| 7 | – | 261/866 (30.1%) | 261/467 (55.9%) |
| 8 | – | 254/672 (37.8%) | 220/346 (63.6%) |
| 9 | – | 186/451 (41.2%) | 163/234 (69.7%) |
| 10 | – | 121/241 (50.2%) | 77/99 (77.8%) |
| 11 | – | 52/104 (50.0%) | 39/50 (78.0%) |
| 12 | – | 12/24 (50.0%) | 13/14 (92.9%) |

叶节点（depth 0）始终全对；错误从 depth 1–2 开始出现，在 H=64 这一时刻
随深度近似单调增加。s42 N32_shallow 的绝对错误率低（最高 8.6%），而
s1234 N128_deep 在 depth 8 以上已超过 60%。**这是时刻切片，不是首次
退化时刻的排序**；深层节点完成得晚，可能只是到 H=64 时更接近窗口末尾。
错误 VALUE 都是 0–22 的合法余数，没有 EMPTY/NA；s42 的模差最集中在
±7，s1234 在 ±4，提示不是均匀随机噪声，但本批不足以把模差模式归因到
具体的错误父节点/变量混淆。

### 3.5 完成时刻的长尾

`lag = tV - (实际最大深度 + 1)`；只统计完成图。lag 0 表示按深度时钟完成。

| weight | stratum | completed | lag 0 | lag 1 | lag 2–4 | lag >4 | max lag |
|---|---|---:|---:|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 64 | 64 | 0 | 0 | 0 | 0 |
| s42_e143 | N32_deep | 61 | 60 | 1 | 0 | 0 | 1 |
| s42_e143 | N64_shallow | 64 | 64 | 0 | 0 | 0 | 0 |
| s42_e143 | N64_deep | 62 | 61 | 0 | 1 | 0 | 4 |
| s42_e143 | N128_shallow | 60 | 58 | 2 | 0 | 0 | 1 |
| s42_e143 | N128_deep | 59 | 58 | 1 | 0 | 0 | 1 |
| s1234_e143 | N32_shallow | 64 | 62 | 0 | 1 | 1 | 25 |
| s1234_e143 | N32_deep | 63 | 62 | 1 | 0 | 0 | 1 |
| s1234_e143 | N64_shallow | 60 | 49 | 9 | 0 | 2 | 54 |
| s1234_e143 | N64_deep | 59 | 51 | 7 | 0 | 1 | 5 |
| s1234_e143 | N128_shallow | 42 | 13 | 16 | 12 | 1 | 7 |
| s1234_e143 | N128_deep | 32 | 13 | 10 | 5 | 4 | 16 |

s42 的 370 张完成图中有 365 张恰好 lag 0，4 张 lag 1，1 张 lag 4；
s1234 的 320 张完成图中只有 250 张 lag 0，43 张 lag 1，18 张 lag 2–4，
9 张 lag >4。s1234 在 N128_shallow 的 42 张完成图里只有 13 张按深度时钟
完成，其余 29 张延后；N128_deep 的 32 张里也只有 13 张 lag 0。

长尾例子：s1234 N32_shallow `0023`（depth 6）tV=32、本批未观察到 t4；
s1234 N64_shallow `0054`（depth 6）tV=61。它们在早期有过 1–2 个 VALUE
错，后来短暂整图全对，再漂移。**“第 depth+1 轮没完成”不等于最终失败**；
但本批也没有把这些晚完成图排除出能力表。

### 3.6 典型轨迹

以下均为逐图 npz 的直接观察，不改变 3.1–3.7 的汇总。完整例子和分类
见 `tmp/analysis/c0-capability-20261114/analysis/examples.json`。

| 图 | 深度 | tV/t4 | 现象 |
|---|---:|---|---|
| s42 N32_shallow `0000` | 6 | 7/7 | VALUE 64 轮全对；VARIABLE 在 H=32 已错 16/32、H=64 错 29/32。只有 first-4 崩，VALUE 未崩。 |
| s42 N32_shallow `0003` | 4 | 5/5 | H=48 前 VALUE 全对；绝对第 55 轮（tV 后第 50 个相对轮）首次出现 11 个 VALUE 错（depth 2–4），H=64 仍错；VARIABLE 错 31/32。说明 VALUE 可以在很长时间后突然退化。 |
| s1234 N32_shallow `0006` | 6 | 10/10 | 比 `depth+1` 晚 3 轮完成；之后 VALUE 一直正确；VARIABLE 在 tV 后第 6 个相对轮首次新错。延迟完成与字段漂移可以同时出现。 |
| s1234 N32_shallow `0023` | 6 | 32/– | H=8/16 有 2 个 VALUE 错，H=32 才整图全对；tV 时 VARIABLE 已错 18/32，本批未观察到 t4（之后恢复并达到 t4 并非不可能）；绝对第 36 轮（tV 后第 4 个相对轮）起再次漂移，H=64 有 7 个 VALUE 错。非单调、长尾完成。 |
| s1234 N32_deep `0047` | 11 | 12/12 | 绝对第 49 轮（tV 后第 37 个相对轮）出现 VALUE 错，绝对第 54 轮回正；H=64 VALUE 仍全对（本批少见的稳定恢复），但 VARIABLE 已错 28/32，first-4 仍为 0。 |
| s1234 N64_shallow `0004` | 6 | –/– | 64 轮内从未整图 VALUE 全对；H=8/16 只错 1–2 个节点，H=64 错 6 个（depth 5–6），VARIABLE 错 60/64。不是“完全不会”，而是几个深层节点持续错。 |
| s42 N128_deep `0002` | 12 | –/– | 64 轮从未完成；与 s1234 N128_deep 的 32 张未完成图同类。 |

VARIABLE 轨迹的形态：不少节点在前 20–30 轮保持正确名字，然后跳到另一个
名字并稳定一段时间；不是每轮随机抖动。以 s42 N32_shallow `0000` 的一个
depth-1 节点为例，H=1–28 的 VARIABLE 读数为真值 `111`，H=32–64 变为
`77`，期间 VALUE 始终为 `11`。这与“读出头改变读出/隐藏状态变化”的
假设一致，但本批没有 hidden probe，不能区分二者，也不能断言隐藏状态
已经删除了变量身份。

### 3.7 失败/恢复事件支持数

事件表由 `tools/c_capability/analyze_existing.py` 从现有 NPZ 生成；每个
图一行，输出在 `analysis/events_by_graph.csv`，汇总在
`analysis/event_summary.csv`。事件行同时保存首次 VALUE/VARIABLE 受损节点的
`positions`（`value_affected_positions_first` /
`variable_affected_positions_first`，reduced-sequence 位置；配合
`source_index` 可回原图追踪）。定义：

- `never`：64 轮内没有整图 VALUE 全对；
- `delayed`：tV > `实际最大深度 + 1`；
- `VALUE never degraded`：有后续观察且完成后整图 VALUE 一直正确；
- `VALUE final wrong`：VALUE 曾退化且末轮仍错；
- `VALUE stable recovered`：退化后末尾连续两轮都正确；
- `VARIABLE new-error`：tV 后出现至少一个节点的 VARIABLE 正确→错误转移；
- `VALUE+VARIABLE preserved`：**独立布尔统计**，不是 `primary_event`；
  要求有后续观察、VALUE 从未退化、tV 时 VARIABLE 全对且之后没有
  正确→错误转移；
- `VARIABLE degraded, VALUE preserved`：VALUE 从未退化，且 VARIABLE 有
  新错误；`VARIABLE already wrong at tV` 单列，不计入新错误。

`primary_event` 仍是互斥分类，只用于案例归类；上表的 preserved 列由逐图
独立布尔量汇总，因此延迟完成但之后 VALUE/VARIABLE 都保持的图不会被漏掉。
本批恰好没有这样的图，但接口已按独立定义实现，并有合成测试。

| weight | stratum | completed | never | delayed | VALUE never degraded | VALUE final wrong | VALUE stable recovered | VARIABLE new-error | VARIABLE degraded, VALUE preserved | VARIABLE already wrong at tV | VALUE+VARIABLE preserved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 64 | 0 | 0 | 47 | 17 | 0 | 64 | 47 | 0 | 0 |
| s42_e143 | N32_deep | 61 | 3 | 1 | 37 | 24 | 0 | 61 | 37 | 0 | 0 |
| s42_e143 | N64_shallow | 64 | 0 | 0 | 17 | 47 | 0 | 64 | 17 | 0 | 0 |
| s42_e143 | N64_deep | 62 | 2 | 1 | 20 | 42 | 0 | 62 | 20 | 0 | 0 |
| s42_e143 | N128_shallow | 60 | 4 | 2 | 3 | 57 | 0 | 60 | 3 | 0 | 0 |
| s42_e143 | N128_deep | 59 | 5 | 1 | 1 | 58 | 0 | 59 | 1 | 0 | 0 |
| s1234_e143 | N32_shallow | 64 | 0 | 2 | 30 | 34 | 0 | 64 | 30 | 0 | 0 |
| s1234_e143 | N32_deep | 63 | 1 | 1 | 33 | 29 | 1 | 63 | 33 | 0 | 0 |
| s1234_e143 | N64_shallow | 60 | 4 | 11 | 4 | 56 | 0 | 59 | 3 | 1 | 0 |
| s1234_e143 | N64_deep | 59 | 5 | 8 | 5 | 52 | 2 | 59 | 5 | 0 | 0 |
| s1234_e143 | N128_shallow | 42 | 22 | 29 | 0 | 42 | 0 | 42 | 0 | 0 | 0 |
| s1234_e143 | N128_deep | 32 | 32 | 19 | 0 | 32 | 0 | 32 | 0 | 0 | 0 |

支持数读法：

- 按独立定义，本批 690 张完成图中 `VALUE preserved` 为 197，
  `VARIABLE preserved` 为 0，`VALUE+VARIABLE preserved` 为 0；
  `event_summary.csv` 逐档可查。延迟完成但两者都保持的图本批没有，
  但合成测试 `test_delayed_stable_is_independently_preserved` 已覆盖
  该接口，不会因 `primary_event` 优先级漏计。
- `VALUE never degraded + VALUE final wrong + VALUE stable recovered =
  completed`：没有“末轮正确但末尾两轮不全对”的图；少数恢复都是末尾
  连续两轮正确。
- `VALUE final wrong` 只统计曾退化图，`VALUE never degraded` 只统计有
  后续观察且从未退化图，两者不是互补但本批由上一行补齐。
- s1234 N64_shallow 的 completed=60、VARIABLE new-error=59、
  already wrong=1：有一张图 tV 时 VARIABLE 已错且之后没有新的
  正确→错误转移；分类有重叠时以逐图 flags 为准，不用汇总列互相推导。
- `no_post_observation=0`（tV 最大 61）；若未来出现 tV=64，该图不进入
  “never/final correct”的支持数，而应单列。

### 3.8 同图事件先后（仅两事件均发生的图）

为了避免把两个不同事件集合的均值比较当成逐图先后，下面只统计
**VALUE 首次退化与 VARIABLE 首次新错都发生**的图。时间差定义为
`VALUE 相对轮次 - VARIABLE 相对轮次`：正数表示 VARIABLE 先，0 表示同轮，
负数表示 VALUE 先。未发生某一事件的图单列在 `variable_only` /
`value_only` / `neither`，不进入时间差。

| weight | stratum | both events | VARIABLE first | same round | VALUE first | VARIABLE only | VALUE only | neither |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s42_e143 | N32_shallow | 17 | 17 | 0 | 0 | 47 | 0 | 0 |
| s42_e143 | N32_deep | 24 | 23 | 0 | 1 | 37 | 0 | 0 |
| s42_e143 | N64_shallow | 47 | 46 | 0 | 1 | 17 | 0 | 0 |
| s42_e143 | N64_deep | 42 | 42 | 0 | 0 | 20 | 0 | 0 |
| s42_e143 | N128_shallow | 57 | 56 | 0 | 1 | 3 | 0 | 0 |
| s42_e143 | N128_deep | 58 | 57 | 0 | 1 | 1 | 0 | 0 |
| s1234_e143 | N32_shallow | 34 | 34 | 0 | 0 | 30 | 0 | 0 |
| s1234_e143 | N32_deep | 30 | 30 | 0 | 0 | 33 | 0 | 0 |
| s1234_e143 | N64_shallow | 56 | 56 | 0 | 0 | 3 | 0 | 1 |
| s1234_e143 | N64_deep | 54 | 48 | 0 | 6 | 5 | 0 | 0 |
| s1234_e143 | N128_shallow | 42 | 39 | 1 | 2 | 0 | 0 | 0 |
| s1234_e143 | N128_deep | 32 | 31 | 0 | 1 | 0 | 0 | 0 |

合计：493 张图有两类事件；其中 VARIABLE 先 479、同轮 1、VALUE 先 13；
另有 VARIABLE-only 196、VALUE-only 0、两者都未发生 1。两事件均发生的图
时间差（VALUE−VARIABLE）均值 19.8 轮、中位 18 轮（范围 −12 到 50）。
这只描述本批轨迹中的时间顺序，不说明 VARIABLE 错误导致 VALUE 错误；
机制仍需 hidden probe/干预。

## 4. 讨论材料与尚不能判断之处

以下为基于本批现象的讨论材料，**不是已批准的 C+ 设计，也不是机制结论**；
C+ 训练未启动。

### 4.1 可由本批数据直接描述的事实

1. **在本批 s42 六档与 s1234 的 N32/N64 档，首次 VALUE 完成率较高，
   但末轮全对率明显下降。** 这些档多数图在 tV 时整图 VALUE 全对，
   s42 在 N128_deep 也有 59/64 完成；但到 H=64 只剩 1/64 整图全对。
   s1234 的 N128_shallow/N128_deep 只有 42/64、32/64 完成，**完成缺口
   仍需单独报告**；是否把长轮下降称为“主要瓶颈”取决于 C+ 的目标定义，
   本批不替这个定义作决定。

2. **VALUE 与 VARIABLE 是两条不同的稳定性轴。**
   本批 493 张同时发生两类事件的图中，VARIABLE 先 479、同轮 1、VALUE
   先 13；另有 196 张只有 VARIABLE 事件、0 张只有 VALUE 事件、1 张两者
   都未发生。H=64 时所有节点（包括叶节点）的 VARIABLE 读出都大幅下降；
   VALUE 则在 H=64 切片中随深度更差。这些只是事件形态/时间顺序描述，
   不能推出 VARIABLE 错误导致 VALUE 错误；若 C+ 的目标包含“可读出的
   完整赋值”，需要显式区分并在指标上分开。

3. **整图指标比节点指标更早报警。**
   N32 上很多图只错 1–2 个节点，节点 VALUE 仍 >0.9，但整图全对率已从
   100% 掉到 73%；N128 上 >5 错的图占多数。C+ 评估应同时保留节点级和
   整图级，不能只报节点平均。

4. **seed 差异在大图上不可忽略。**
   s42 与 s1234 在 N32 接近，在 N128_deep 的完成率 59/64 vs 32/64、
   H=64 全对 1/64 vs 0/64。固定 e143 是为了可比性，不是最佳权重；C+ 的
   效果判断需要多 seed、固定协议，不能按本批结果回头挑 checkpoint。

5. **本批至少出现几种不同的轨迹形态，不能合并成单一“稳定/不稳定”
   标签。** 例如 VARIABLE-only 漂移（s42 `0000`）、VALUE 晚期突然退化
   （s42 `0003`）、VALUE 退化后短暂恢复（s1234 `0047`）。这些是事件形态
   描述，不证明它们共享或对应不同机制。

### 4.2 候选 C+ 方向（待讨论，不是执行授权）

- **固定点/循环一致性**：对同一图在完成后的额外轮次加 VALUE 一致性约束，
  或训练模型在 hidden 迭代下保持读出不变；先确认这不会把“晚完成”的图
  锁死在错误状态。
- **身份锚定**：每轮刷新叶节点/输入 token 的 embedding，或给 VARIABLE
  头单独的锚定/一致性目标；需要先做 probe 判断 VARIABLE 漂移是读出头
  问题还是 hidden 表示问题，否则可能只修表面。
- **按深度/结果端的稳定性权重**：VALUE 错误在深层和 sink 更集中，C+ 的
  训练/评估可以按真实深度分层，而不是只报全图平均。
- **读出调度**：区分“计算轮”和“报告轮”；如果 C+ 只需要最终答案，
  可以学习何时停止，而不是无限迭代到漂移。需要与“自由续算固定 64 轮”
  的探索目标分开。
- **多 seed / 多 checkpoint 的评价协议**：本批 s1234 的 N128 缺口说明
  单 seed 结论不稳；C+ 比较至少保留现有 s42/s1234，s2026 完成后按同协议
  补入，不按测试结果选 epoch。

### 4.3 尚不能判断

- **VARIABLE 漂移是 hidden 表示变化还是读出头退化？** 本批只解码模型
  自己的头；需要线性 probe、hidden 范数/余弦、或每轮刷新 token embedding
  的干预才能区分。
- **VALUE 为什么能在 VARIABLE 崩掉后保持很久？** 可能是 VALUE 被缓存在
  hidden 的稳定子空间，也可能是 VALUE 头对变量身份不敏感。需要激活干预
  或因果替换，不能从相关性下结论。
- **VALUE 漂移与 VARIABLE 漂移是否因果相关？** 当前只是同一轨迹上的
  两个观测；深层节点两者都更差，不能推出“变量错导致数值错”。
- **64 轮之后会不会恢复？** 本批只观察到 64 轮；s1234 有少数 54 轮前
  恢复的例子，但不能外推“最终会恢复”。
- **C+ 是否应把额外轮次纳入训练分布？** 当前 C0 训练只到 max depth，
  所以本批后段是 OOD 迭代。若 C+ 要优化长轮稳定性，需要明确训练时是否
  采样额外轮次/固定点目标；否则只是在评价 OOD 行为，而不是在训练目标
  内比较。
- **旧 C0 的 strict 归零是否同一机制？** 旧 eval 的数据、轮数和 checkpoint
  协议与本批不同；本批能说明“长轮数自由运行下 first-4 会被 VARIABLE
  主导”，但不能直接把旧 N128 strict 数字对应到本批某一轮。

## 5. 边界与未做

- 每档 64 图是探索预算，不是最终样本量；所有比例都应带 n 解读。
- N64/N128_shallow 是**全深度 6 的新生成池**，与 N32_shallow 的 4–6 不
  严格匹配；N128_deep 只有 9–12。跨规模/跨深度的差异不能全部归因于
  规模或深度。
- s42/s1234 固定 e143；s1234 不是最佳 checkpoint；s2025 未测；s2026 未纳入。
- 只做 64 轮自由续算；没有 64 轮之后的观察，没有 hidden 快照/probe，
  没有人工注错，没有 C+ 训练。
- 生成档使用官方生成器但仍是新池；与 B 线共享部分 heldout 来源，不能
  称为完全独立于 B 的 heldout。
- `analysis/field_acc_by_H.csv` 由 `analyze_existing.py` 从逐图 npz 生成，
  用于解释 first-4；`summarize.py` 的正式输出仍只有 first-4 合计。若后续
  要把逐字段曲线纳入正式协议，建议给 `summarize.py` 增加 per-factor 输出
  并走 review。
- 本次事件分析只读现有 npz，不重跑模型、不生成新数据；事件表和例子
  不能替代对隐藏状态的 probe/干预。

## 6. 复现与产物

实际运行参数记录在 `runs/<weight>/run_meta.json`。命令形态：

```bash
python -I -B tools/c_capability/build_data.py --allow-generate

CUDA_VISIBLE_DEVICES=6 python -I -B tools/c_capability/run_capability.py \
  --weight s42_e143 \
  --data-dir tmp/analysis/c0-capability-20261114/data \
  --out-dir tmp/analysis/c0-capability-20261114/runs \
  --strata all --rounds 64 --batch-size 4 --num_threads 4 --device cuda:0

CUDA_VISIBLE_DEVICES=6 python -I -B tools/c_capability/run_capability.py \
  --weight s1234_e143 \
  --data-dir tmp/analysis/c0-capability-20261114/data \
  --out-dir tmp/analysis/c0-capability-20261114/runs \
  --strata all --rounds 64 --batch-size 4 --num_threads 4 --device cuda:0

python -I -B tools/c_capability/summarize.py \
  --run-dir tmp/analysis/c0-capability-20261114/runs \
  --out-dir tmp/analysis/c0-capability-20261114/runs/summary
```

现有 NPZ 的 CPU 事件分析（不加载模型、不写运行目录）：

```bash
python -I -B tools/c_capability/analyze_existing.py \
  --run-dir tmp/analysis/c0-capability-20261114/runs \
  --out-dir tmp/analysis/c0-capability-20261114/analysis --overwrite-ok
```

产物：

- `data/manifest.json`：六档来源、合格池、source indices、实际深度/长度、
  生成配置与文件哈希；
- `runs/s42_e143/`、`runs/s1234_e143/`：各 384 个逐图 npz + `index.jsonl`
  + `run_meta.json`；
- `runs/summary/`：`curves.csv`、`capability_by_H.csv`、`by_node_type.csv`、
  `by_depth.csv`、`completion.csv`、`drift.csv`、`drift_by_graph.csv`、
  `post_completion_curves.csv`、`examples.json`、`report.md`；
- `analysis/`：`field_accuracy_long.csv`、`field_acc_by_H.csv`、
  `completion_lag.csv`、`completion_lag_distribution.csv`、
  `value_error_counts.csv`、`value_error_by_depth.csv`、
  `value_wrong_predictions.csv`、`events_by_graph.csv`、`event_summary.csv`、
  `examples.json`、`analysis_meta.json`（含权重/数据/index 哈希与定义）；
- `logs/s42_e143.log`、`logs/s1234_e143.log`：无 Traceback/OOM。
