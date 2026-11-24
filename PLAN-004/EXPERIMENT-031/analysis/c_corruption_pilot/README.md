# C 单节点腐蚀后训练小比较

入口方案：[协作交接](../../ai-collab/handoff/2026-11-15-c-corruption-pilot.md)。

状态：首批 hold/noise/semantic 三组、128 图两时点评估和汇总已完成
（2026-11-15）；结果与边界见
[docs/records/2026-11-15/c-corruption-pilot.md](../../docs/records/2026-11-15/c-corruption-pilot.md)。
原始 NPZ/权重/records 留在 `tmp/analysis/c-corruption-pilot-20261115/`，不提交。

三组从固定 s2026/e143 出发，均 +D、3 epoch、lr1e-4、batch8：
`hold`（不注错）、`noise`（L2匹配随机方向）、`semantic`（当前模型的反事实VALUE向量）。
正式源码不改原 model.py；执行 agent 先小验证、再按固定版本串行执行。

- `prepare.py`：复用长程pilot的1024/128新图，固定内部目标、均匀覆盖非零模偏移、匹配同值/异值对照题、保持阶段时点。无模型行为选样。
- `train.py`：每图每遍以固定哈希50%概率尝试一次；三组相同计划、2D轮数、阶段loss和优化步数。当前模型参考向量均无梯度生成；全程其余状态保留BPTT。
- `evaluate.py`：所有128图不注错到D+64；固定保持阶段/晚期D+16，五分支（不改/同值/异值/匹配噪声/三倍噪声），续算16轮。原始JSON/NPZ留tmp。
- `summarize.py`：逐轮节点VALUE/四属性、分字段、配对损伤与恢复、实际注入读出、共同资格及全固定样本。不能把无错误当恢复，也不能把末两轮全对当永久稳定。

运行命令和小验证门槛见交接。支持 `--smoke-batches` / `--smoke-ckpt --limit-graphs`；smoke和正式目录分离，所有输出目录必须不存在。高强度噪声暂只评估，不混入训练。

本地/服务器：9项测试通过；首批训练与评估已完成，结果见上方报告。
