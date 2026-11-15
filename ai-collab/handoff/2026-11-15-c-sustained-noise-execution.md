# C 持续错父与噪声剂量校准：执行交接

2026-11-15。用户在 2016 审阅讨论后同意实现；本批由 GPT 写核心代码，Agent C 小验证后执行。**仅评估现有权重，不启动新训练或三种子正式后训练。**

## 共同协议与范围

- 固定四权重：s2026/e143 initial，以及原 corruption pilot 的 hold/noise/semantic epoch3。沿用原 meta/plan/SHA 检查，不挑新 checkpoint。
- 基础评估继续使用已实现的 `run_clean.py/run_pair.py` 协议：D 求解、D+16/32/64 保持、节点 VALUE/四属性、clean 配对、全轨迹和末两轮；本轮不重复这些已经完成的 GPU 运行。
- 新增 `stress_core.py`、`run_stress.py`、`summarize_stress.py`。不修改原训练、腐蚀 pilot、旧数据或旧输出。
- 沿用 128 张 N32 开发验证图与固定结构 u→q；持续实验用全部 128 图，噪声校准用固定前 64 图。选点与同值/异值源构造复用上一批，非零模偏移覆盖各图取值，不限 4→5。没有候选如实记，不补样。
- 两时点仍是原保持时点 `state_round` 和 D+16，不按答案挑时间。起点整图 VALUE/first4 是否完成另记。数据已经看过，**不是独立最终测试**。正式训练方案确定后再生成独立测试图，本批不自动扩展生成器或改变测试分布。
- 四模型同一个 source batch5（原题、u_same/u_wrong、q_same/q_wrong），没有 teacher 或解码回灌；每一轮 donor 隐向量来自当前模型、同一绝对轮次的自由运行。

## 实验一：持续错父 × 子节点是否也错

问题：当 u 不能靠下一轮输入恢复成干净状态时，q 能否保持或重新获得正确值？解除强制后，恢复如何发生？

每张图、每个时点分别运行 L=2/4，之后自由续算 H=16。每个 L 的 8 分支：

| u 处理 | q 起点保持原状 | q 在第一次输入改错一次 |
|---|---|---|
| free：不额外写 u | 全不动的 clean | 仅 q 错的参照 |
| original：每个强制轮写入该时点的干净 u 源行 | 干净轨迹回放 | 干净上游持续提供时的 q 恢复 |
| same：每个强制轮写入同值不同来源的 u 行 | 重复来源更换对照 | 同值源条件下 q 恢复 |
| wrong：每个强制轮写入异值 u 源行 | q 已有正确状态的保持 | 父子都错时 q 的重建 |

- q_wrong 只发生一次；它的错误数值与错 u 代入 q 表达式的预期一致。两条向量来自分别构造的源题，不宣称整体隐藏状态自洽。
- 每轮换的是**该绝对时点的源向量**，不是把第一次源向量冻结 L 轮。元数据/records 保留每个源时点的实际解码。若后续 donor 没有保持预期错值，保留结果并降低共同资格，不把它伪称持续有效注错。
- original 是干净源轨迹的反复回放；分支历史不同后，写回 original 会改变状态，**不是全程 sham**。只有第一次同起点 original 写回与 free 对照必须完全一致。
- 两个 L 都先完整跑固定候选。共同 VALUE/严格资格取所有四模型、两个 L 的交集，避免 L=2 与 L=4 换分母。全固定样本也报告。
- 局部 VALUE 资格：初始 u/q 和它们的父读出正确，clean 首轮 u/q 正确；每个强制时点 u 原源/同值/异值分别读出预期值，第一时点 q_wrong 源读出预期。严格资格再看 u/q 及相关源行四属性。不是整图资格。

时序例（L=2）：

```text
已有状态 h_t
强制1：写 u（q_wrong 分支同时写 q） → forward1 输出
强制2：再写当时的 u 源行，q 不再写 → forward2 输出
释放1：直接使用 forward2 输出，不再写任何行 → forward3 输出
……释放16
```

保存实际输入、输出和每次替换距离，避免把“输出修回后，下轮又强制写错”当成自然再错。`step=3` 在这个例子中是 `phase=released, phase_round=1`。恢复事件的 release 窗口以最后一个强制 forward 的输出为时间0。

## 实验二：固定噪声强度校准

四权重 × 前64图 × 两时点。每次只改 u，随后自由运行16轮；15分支：

- free、u 同值源替换、u 异值源替换三个参照。
- 随机方向扰动：语义替换向量差 L2 长度的 **0.5、1、2、3、4、6 倍**，每图固定两个方向。
- 同图、同一方向在六个强度间共用；种子不依赖模型和强度。每个模型的参考半径来自它自己的源向量差，跨模型不宣称相同隐藏损伤。
- 不找最容易打错的方向、不丢掉未打错的样本、不依据结果调整强度表。每个噪声方向是同一张图上的重复测量，不算独立图或训练 seed。

必须并列报告：

1. 原本正确→注入后 VALUE 错；原本已错另列，不能把 after-wrong 全算新增错误。
2. VALUE 是否变化、非数字读出、四属性新增错误、实际 L2。
3. 首轮/后续子节点及更远下游相对 clean 的额外损伤、恢复、额外错误轮数。
4. **输入读出没变但下游出错**、**目标输出仍错但下游较少受损**的同题记录；`dose.csv` 含输入改变/正确性与 children_output_harmed，`trajectory.csv` 含 u/q 实际输入输出，完整 records 覆盖所有节点。

共同资格只看起点、原/同/异值源和 clean，不以噪声是否成功打错来筛选。本轮输出强度—损伤关系，**不自动推荐或开启某个训练强度**。首轮解码错误率匹配也不等于完整计算损伤匹配。

## 汇总和交付口径

- `accuracy.csv`：逐轮、强制/释放阶段、u/q/其他直接子/所有后代/无关节点/全节点，VALUE/first4，逐图等权及原始计数。
- `harmed/helped` 对全不动 clean；持续实验另给 `matched_q_harmed/helped`，q_wrong 分支以“只改 q、不写 u”作参照，便于分离持续写 u 的增量。
- `events.csv`：完整窗口和释放窗口分别报告 observed/damaged/曾恢复/再错/末两轮、额外错误轮数。没有受损的条件不记为100%恢复。
- `target_events.csv`：u/q 每例的首次损伤、首次恢复、最终连续正确段起点；不要用稀疏检查点代替真实恢复时刻。release 时间0是最后一个强制输出，不是最后一次被改坏的输入。
- `dose.csv/dose_summary.csv`：强制每步或单次噪声的实际剂量和注入有效性；模型输出修回率与下一次强制输入错误分开。
- `trajectory.csv`：u/q 每轮输入/输出、父值读出模和、clean 值。模和匹配只作行为观察，不证明内部实际读取解码值。
- 噪声汇总按同图两方向平均后逐图等权；raw counts 的 n 是节点×方向试次，`graphs` 与 `trials` 分开。完整记录不提交，大表留 tmp；交付小表须明确过滤规则和 source/export 各自 SHA。
- no-candidate 见原始 coverage 及 summary meta；所有不合格保留全固定样本。若某个机制趋势只在严格资格子集出现，应同时写覆盖 n。

## Agent C 执行顺序

串行领取，确认共享工作树和 GPU 队列空闲后同步本提交并记录 SHA。运行中不 pull/改配置，不打断既有任务。Agent B 继续既有 CPU 工作，不新增 GPU。

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$GPU_ID"
OUT=tmp/analysis/c-sustained-noise-20261115

python -m unittest discover -s tools/c_evaluation_suite/tests -p test_stress.py -v
python tools/c_evaluation_suite/run_stress.py --mode sustained --smoke --out-dir "$OUT/sustained-smoke" --num_threads 4
python tools/c_evaluation_suite/summarize_stress.py --run-dir "$OUT/sustained-smoke" --out-dir "$OUT/sustained-smoke-summary"
python tools/c_evaluation_suite/run_stress.py --mode noise --smoke --out-dir "$OUT/noise-smoke" --num_threads 4
python tools/c_evaluation_suite/summarize_stress.py --run-dir "$OUT/noise-smoke" --out-dir "$OUT/noise-smoke-summary"

# 两项小验证通过后执行已授权的固定规模，不重复征求启动许可。
python tools/c_evaluation_suite/run_stress.py --mode sustained --out-dir "$OUT/sustained" --num_threads 4
python tools/c_evaluation_suite/summarize_stress.py --run-dir "$OUT/sustained" --out-dir "$OUT/sustained-summary"
python tools/c_evaluation_suite/run_stress.py --mode noise --out-dir "$OUT/noise" --num_threads 4
python tools/c_evaluation_suite/summarize_stress.py --run-dir "$OUT/noise" --out-dir "$OUT/noise-summary"
```

小验证仅检查关键语义：L次写u/q只写一次/释放后无写入；当轮源行对应绝对时点；第一次original/free完全一致；噪声同方向按强度缩放、仅目标行变化；原本错误不算新增；释放时间与分母；所有记录可汇总。若源构造在4图全部失败，反馈原因，不偷换成功样本。低恢复率或高强度大量错是结果，不是改方案的理由。

输出目录不得覆盖已有结果。正常环境路径适配可做并记录，不改实验语义；本地已通过5项stdlib（包括两模式的真实汇总CLI），2项torch行操作检查交服务器。GPT未加载模型或运行GPU。

交付：简报+小表+运行meta/哈希+各1–2个真实方程轨迹，明确训练没有变化。对下一轮训练仅给建议，不自动开始新的noise/semantic训练或三个seed正式比较。
