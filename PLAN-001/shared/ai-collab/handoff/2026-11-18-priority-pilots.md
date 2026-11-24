# 两个优先方向：首步执行与验收

日期：2026-11-18。用户要求选两个优先方向、具体化首步、实现、明确验收并推送。
选定 **E/M：C输入对齐诊断** 与 **W：A′/B合法重新赋值**。代码见[README](../../tools/priority_pilots/README.md)。
本条仅覆盖下面的首批小验证/试跑；不扩训练、权重、图规模或其他研究方向。

## 0. 协作与任务交接

- Agent C：负责任务F，CPU处理已有记录，不重新跑模型。
- Agent B：负责任务W，先两图四轮小验证，再固定32图试跑；同一GPU串行跑A′和B。
- 先确认共享服务器工作树空闲、同步本提交并记录SHA；运行中不pull。两方向独立，不等于允许两个agent同时更新共享树。
- GPT已写核心逻辑；agent负责路径适配、GPU小验证和执行。改变选点、真值、时序、参考条件须先反馈。保持原结果目录不动，新目录不覆盖。
- 不把一次试跑成功或零错误当最终机制结论；无合格案例也要交覆盖表，不补样。

## 1. F：读出修好是否等于下游计算恢复

### 首步要回答

在已有C单次/持续/噪声干预中：
1. u的输出已正确但q仍错，到底发生在u输入仍错的那轮，还是输入读出已经正确的后续轮？
2. 当其余父节点也正确、匹配参照q正确时，这类现象还剩多少？
3. q原本正确与原本错误时，情况是否不同？有没有回正后再次变错？

这一步是证据分层与后续实验选题，不声称仅靠读出证实潜状态功能恢复。

### 数据与命令

沿用已有完整run目录；权重由源meta继承initial/hold/noise/semantic，不另选checkpoint。
若服务器路径不同，核对protocol/status/数据与权重信息后只调整路径并记录。

```bash
python -m unittest discover -s tools/priority_pilots/tests -v

python tools/priority_pilots/function_audit.py \
  --source pair=tmp/analysis/c-evaluation-suite-20261115/pair \
  --source sustained=tmp/analysis/c-sustained-noise-20261115/sustained \
  --source noise=tmp/analysis/c-sustained-noise-20261115/noise \
  --out-dir tmp/analysis/priority-pilots-20261118/function-audit
```

### 必须通过的执行验收

- 源meta均complete；每来源的模型/持续长度配对完整。汇总保留不同来源身份，不跨数据集混合分母。
- 手工核对至少一个pair、一个持续污染、一个噪声记录的post1/post2：pair第2轮输入来自第1轮输出；持续污染必须读取实际输入；释放轮从强制结束后计。
- 四格每组计数相加等于edge_n，比例相加为1；用一组每图子节点数不等的记录核对图均权不被大图权重替代。
- both_wrong与q_wrong配对，持续q_wrong与free_q_wrong配对；不能把直接改q的损伤归因u。
- 没有数值、其余父节点错误、参照自身漂移均可追溯；不能静默丢掉后再宣称恢复好。
- all_recorded与common_value分列；候选图原始coverage和未入选情况随来源摘要交付，不宣称都代表全体图。

### 交付

新增 `docs/records/2026-11-18/priority-function-audit.md`：
- 三个来源分别的候选/记录/共同资格计数；
- 输入轴与输出轴四格的小表，展示关键轮次1/2/4/8/16（完整表保留）；
- 分“其余父节点正确且参照q正确”“q输入已有答案/没有正确答案”；
- 按固定排序给每种可观察情况前两个完整案例；若不存在，明确零例，不挑更好看的替代；
- 至少说明一种可能的替代解释和下一项有必要的干预；不要把观察四格命名为已证明机制。
- 原始大CSV留tmp；小表及来源/脚本/导出hash进git。不重新跑GPU。

## 2. W：原题错误应还原，新题面变化应接受

### 首步要回答

同一叶节点数值从a变成a+delta：
- 如果题面没变，模型是否把错值还原？
- 如果题面常数合法改变，模型能否覆盖旧答案并更新下游？
- 若热启动失败，空白求新题是否成功？若固定新常数后改善，是否与题面保持有关？

这是完整槽位复用之前的必要诊断。先选离散A′/B以明确“题面只改一格”的含义；C的连续状态写入接口留后续，不偷偷用解码重嵌入替代连续反馈。

### 权重、范围与命令

权重严格沿用 `manifest_main256.json` 中model_a/model_b（A′ s42/e86、B s42/e92）。
固定N32图32–63；每图一叶、4个delta、七条件、H16。最大每模型128条case，每case含7条分支；不合格如实减少。
本批图已参与旧探索，不能称独立最终测试。开始状态采用manifest共同完成轮次，并额外核对整题前四字段正确。

小验证（共享树固定版本后，单进程依次运行）：

```bash
python tools/priority_pilots/run_reassignment.py \
  --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json \
  --model model_a --candidate_count 2 --horizon 4 --num_threads 4 \
  --out-dir tmp/analysis/priority-pilots-20261118/reassign-smoke-A

python tools/priority_pilots/run_reassignment.py \
  --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json \
  --model model_b --candidate_count 2 --horizon 4 --num_threads 4 \
  --out-dir tmp/analysis/priority-pilots-20261118/reassign-smoke-B

python tools/priority_pilots/summarize_reassignment.py \
  --run-dir tmp/analysis/priority-pilots-20261118/reassign-smoke-A \
  --run-dir tmp/analysis/priority-pilots-20261118/reassign-smoke-B \
  --out-dir tmp/analysis/priority-pilots-20261118/reassign-smoke-summary
```

小验证通过后，运行默认32图/H16（只省略candidate_count/horizon参数并换新目录）：

```bash
python tools/priority_pilots/run_reassignment.py \
  --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json \
  --model model_a --num_threads 4 \
  --out-dir tmp/analysis/priority-pilots-20261118/reassign-main-A

python tools/priority_pilots/run_reassignment.py \
  --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json \
  --model model_b --num_threads 4 \
  --out-dir tmp/analysis/priority-pilots-20261118/reassign-main-B

python tools/priority_pilots/summarize_reassignment.py \
  --run-dir tmp/analysis/priority-pilots-20261118/reassign-main-A \
  --run-dir tmp/analysis/priority-pilots-20261118/reassign-main-B \
  --out-dir tmp/analysis/priority-pilots-20261118/reassign-main-summary
```

### 小验证必须通过的验收

1. 用实际tokenizer/图核对leaf和constant位置含义、ADD表达式、模23真值。独立求值应复现所有旧节点标签；新常数标签必须更新。
2. 对一个case比较完整起跑tensor：state_error只改叶VALUE，new_warm只改常数VALUE；new_scratch基于原始未求解输入，只有常数变化。新求解答案不得写回输入。
3. clean/sham完整已记录轨迹逐项相同；两辅助字段在全部反馈中保持FALSE。strict资格失败不偷偷放宽。
4. new_warm_pinned/new_scratch_pinned每轮输入常数VALUE等于新值；自由组不重复强制。其他位置不强制真值。
5. post0只是起跑状态；post1是第一次继续前向输出。warm从T+1轮已完成状态出发，scratch从0出发，同H不意味着总算力相等。
6. 所有候选图进入selection；无共同完成、结构不合格等均明确记录；每合格图恰好4条case，每case七分支。两图smoke若完全无合格case，先报告，不能当GPU路径已验证或换图凑成功。
7. 汇总旧/新真值不混：state_error目标旧真值，合法更新组目标新真值；模抵消导致答案未变的节点不进入“需要更新”分母；空区域不填100%。
8. 两模型数据/manifest/代码/候选/预算一致；主比较只用共同case，自身合格和覆盖率附报。summary末两轮必须两轮都对，不等于仅最后一轮。

### 结果交付与科学验收

新增 `docs/records/2026-11-18/priority-reassignment.md`：
- 32图资格与两模型共同图数，固定delta分列，不把4个delta当4份独立图。
- 七条件在leaf/changed_descendants/unchanged_nodes/constant的逐轮VALUE、四属性与末两轮表。
- 改题组的旧答案保留量、新答案接受量；new_warm和new_scratch的同后续预算差异，以及warm预先求解成本。
- 自由组常数是否漂移；pinned改善与否；未改真值节点相对clean新增损伤。
- 每种实际出现的“接受新题/冻结旧答案/常数漂移/其他错误”固定排序取前两个完整方程案例，不保证每类都有。
- 总结下一步应先做输入保持、已有答案更新，还是可以进入真正的槽位复用；不能直接说已突破128节点。
- 更新看板、日志，提交小表/hash和运行版本；大记录留tmp。

## 3. 验收边界

**代码正确且结果无收益，是有效交付；靠改样本得到改善，不是有效交付。**
若发生加载/协议/时序/真值错误，停止该任务并报告具体触发；另一独立方向可继续按范围执行。
这两步完成前，不启动新的C+训练、新结构定位头或扩图复用实验。首步结果用于决定下一项，不自动扩成六方向全量工作。
