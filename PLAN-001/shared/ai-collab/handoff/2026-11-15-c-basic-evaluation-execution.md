# C基础评估首批：执行交接

2026-11-15 19:43 CST；用户已要求实现并交agent执行。本批是评估，**不启动任何新训练**。

## 范围与研究问题

模型固定为当前s2026/e143原始C0和本批hold/noise/semantic三个epoch3权重，读取`c-corruption-pilot-20261115`的meta与SHA。不是再挑best，也不是本次就训练三个C0种子。

| 模块 | 实现/样本 | 回答什么 |
|---|---|---|
| 已有轨迹CPU分析 | 上批全部128图×两时点records | 目标读出是否变化、子节点实际受损是否一起出现；hold/semantic的优势是否发生于同一题 |
| 统一基本能力 | 四模型、现有六档各固定前64图、共384图/模型；batch4；每图到D+64 | 不同规模/深度的正常求解、计算轮数、长程保持；叶/内部/输出的差异 |
| 父子两点对照 | 四模型、上批128图、每图原u配一个结构选定直接子q；两时点；8分支；16轮 | 子节点只是保持已有正解，还是在没有现成正确状态时也能恢复 |

六档为N32/N64/N128×原浅/深档，仍是已查看过的诊断数据；此批不声称最终独立测试。不需要新生成数据，也不自动扩N256。

## 基本能力协议

- 自由连续hidden反馈；第一轮原始题面，之后不回灌解码或teacher状态。每图实际D=max_depth+1，不按batch深度定义观察终点。
- 保存所有轮每个变量节点前四字段预测、truth、节点位置、父列表与深度。排除题面常数、PAD、运算符。
- 主指标为VALUE与四属性同时正确，逐图等权；节点总计辅助。叶/内部非sink/输出sink分开，额外按节点深度和父数分层；不要把各分层当成互不相关的独立样本。
- absolute 1–64与relative D+0…64两条固定队列曲线；D/2D/3D、D+16/32/64等切片另报。绝对曲线不因每图观察终点不同而变换图集合。
- 首次90%/99%/整图全对的时间保留逐图；未完成图单列，完成时间中位数明确条件分母。完成后失稳、恢复、再次失稳以及末两轮状态单列。节点级“曾对后错”从该节点depth+1的监督可算轮开始计数，避免把之前偶然猜中的值视为完成；整图曲线仍保留所有轮。节点读出漂移不自动解释成内部运算失效。
- 四模型本批统一batch4。原报告batch1/8的数字仅参考，不混成精确逐节点配对。

## 父子两点协议

例：u=4，q=u+9=13。原u沿用上批固定结构选点，q按固定哈希选一个可构造匹配源的直接子节点，不按模型输出挑选；没有候选就记录，不补样。

| 分支 | 操作 |
|---|---|
| unchanged / sham | 不改；或u/q原向量原样写回，要求逐轮完全一致 |
| u_wrong | 只把u替换成错值源向量，例如读出5 |
| q_wrong | 只把q替换成错值源向量，例如读出14 |
| both_wrong | 同时替换u和q的上述两条向量 |
| u_same / q_same / both_same | 对应位置换入同值不同源向量，匹配来源更换与位置数量 |

- delta_u沿用每图非零模偏移；delta_q按q表达式中u出现次数×delta_u(mod23)，使数字例子中的5/14对应局部代入。不称整个状态局部自洽，更不是重启此前延期的祖先子图自洽实验。
- u源与q源分别通过匹配的两叶常数构造；源题结构/变量不变，源hidden来自当前被评估模型、相同绝对轮次。只换指定一/两行，不把整张源题状态换入；源题其它节点可能改变但不会被复制。
- 两时点固定为原保持时点D到D+floor(D/2)-1、late=D+16；不是按首次共同完成选时刻。实际起点整图VALUE/四属性是否全对另记，不能仅凭D就称整图已完成。
- source batch固定5（原题、u同值/u错值、q同值/q错值），rollout branch batch固定8；所有模型相同。与旧u-only使用不同源batch，旧百分比不直接合并。
- 全固定样本主视图保留；共同VALUE资格要求原u/q及其父读出正确、四个源行读出目标值、clean首轮u/q正确。严格资格额外检查u/q及源行四属性。全部资格与原始失败记录保留。
- 输出u/q逐轮实际输入值与输出值、clean配对、按输入父读出计算的模和。**与父读出的模和一致仅是行为一致性，不证明下一轮真的读取了解码值。**
- 目标u、选中q、其它直接子、全部后代、无关节点、全节点分开统计。post1/2/4/8/16、相对clean损伤/改善、额外错误轮数、曾恢复/再错/末两轮都报告。
- 比较q_wrong与both_wrong能看正确上游是否帮助修复；比较u_wrong与both_wrong能看“保留现成答案”与“重新获得答案”差异。组间差异不能单独证明检测机制，先按单题联合轨迹解释。

## 执行顺序（串行，运行树固定）

Agent C先领取任务，确认共享树空闲，同步本提交并记录SHA；不修改既有正在跑的任务。原CPU补交继续并可复用新`analyze_existing.py`。Agent B不新增GPU任务。

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$GPU_ID"
OUT=tmp/analysis/c-evaluation-suite-20261115
python -m unittest discover -s tools/c_evaluation_suite/tests -v
python tools/c_evaluation_suite/analyze_existing.py --out-dir "$OUT/existing-joint"

# 每模型/档4图，独立smoke目录。仅核对必要实现。
python tools/c_evaluation_suite/run_clean.py --smoke --out-dir "$OUT/clean-smoke" --num_threads 4
python tools/c_evaluation_suite/summarize_clean.py --run-dir "$OUT/clean-smoke" --out-dir "$OUT/clean-smoke-summary"
python tools/c_evaluation_suite/run_pair.py --smoke --out-dir "$OUT/pair-smoke" --num_threads 4
python tools/c_evaluation_suite/summarize_pair.py --run-dir "$OUT/pair-smoke" --out-dir "$OUT/pair-smoke-summary"

# 小验证通过后继续；不为“效果不好”改变样本或参数。
python tools/c_evaluation_suite/run_clean.py --out-dir "$OUT/clean" --num_threads 4
python tools/c_evaluation_suite/summarize_clean.py --run-dir "$OUT/clean" --out-dir "$OUT/clean-summary"
python tools/c_evaluation_suite/run_pair.py --out-dir "$OUT/pair" --num_threads 4
python tools/c_evaluation_suite/summarize_pair.py --run-dir "$OUT/pair" --out-dir "$OUT/pair-summary"
```

必要小验证：6项stdlib＋1项torch；正确节点mask/每图D+64/固定候选数；两处行替换只动u/q；sham==unchanged；同/异值源实际读出与替换行一致；轮次0为注入输入、轮次1为第一个forward，父值比较使用输入而非同轮输出。若4图均无可用源，先反馈覆盖原因，不凭空降低资格或换成功样本；实际低恢复率是结果，不是失败理由。

模型/path适配可`--source-ckpt`，仍须固定SHA；三组后训练权重由原meta读取，不改其来源。首次执行已获用户授权，小验证通过后无需重复询问启动。运行期间不pull/改配置，输出不覆盖旧目录。

## 交付

- `docs/records/2026-11-15/`简明报告：四权重身份、各档候选/完成/失稳数量、正常预算与长程两个节点指标；对子节点保留/重建的2×2结果逐题举例。报告所有不合格与no_candidate，不只报成功子集。
- 提交小表：clean accuracy/stability/主曲线、pair coverage/dose/accuracy/events、existing joint_summary/paired_models，两个完整方程+值轨迹案例。不要只给正确数量而不交实际数值。
- 完整index/NPZ/records留tmp。审计明确source文件hash与提交export hash；如过滤子集，注明过滤条件。不能用源大表hash冒充提交子表hash。
- 不训练、不做新噪声强度搜索、不跑真实改题/持续污染/层头干预；这些仍在评估框架候选列表，等本批结果后讨论。

本地6项stdlib通过，torch替换行检查留服务器；当前未跑GPU。本批不改原训练或此前腐蚀试验源码。
