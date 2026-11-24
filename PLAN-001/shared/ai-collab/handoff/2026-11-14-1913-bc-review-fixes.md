# B/C 执行代码审阅修复与交接

时间：2026-11-14 19:13 CST；基线 reproduction@fc0a4a4。
用户本轮明确授权 GPT 直接修改，然后由服务器两位 agent 同步。对应审阅：`2026-11-14-1857-bc-execution-review.md`。

## 修改内容

| 线 | 修复 | 影响与边界 |
|---|---|---|
| C 数据 | 父节点解析保留最后一个（第三个）父节点；生成元数据保存实际 dag_config，不保存含 Path 的分档配置 | 避免 sink 错分和生成数据 JSON 序列化失败；生成分布未变 |
| C 汇总 | 两项节点准确率同时输出逐图等权和节点加权，类型/深度桶含零支持图数 | 报告主列使用 `*_acc_graph_equal`；原 `*_acc` 保持节点加权 |
| C 漂移 | 区分原本字段错误、对后变错、曾恢复、恢复后再错、最终正确、末尾两轮正确 | 最后一轮才完成的图不计作有后续观察；末尾两轮正确不等于永久稳定 |
| C 过程 | 新增完成后相对轮次曲线及每轮可观察分母 | `post_completion_curves.csv`，不同相对轮次的分母可能不同 |
| C runner | embedding 放入 no_grad；运行输出只允许新目录，去掉无效 overwrite 参数；导入时不配置线程 | 不改变连续 hidden 反馈、权重或训练配置 |
| B 选样 | fixed 模式中 early 资格不再依赖共同整图完成；显式记录 T=None/common_complete=False | late/controls 跳过这类图；旧 topup 仍为 complete-only；本批256图原先全完成，不因此改变现有资格 |
| B unrelated | 同时跟踪被注入分支、原目标及下游，区分共享后代和不受该注入节点可达的区域 | 新增 `protected_*` roles；原 `all_affected` 仍指注入分支，不混分母 |
| B D1/D2/D3 | 新增 `b_new_truth_summary.py`，主汇总自动分流 | 输出 old/new truth 的逐轮/按H节点正确率、完成时间与末尾连续正确；D2 无需 clean，不称为旧答案恢复 |
| B 覆盖与子集 | 输出 condition_inventory；不支持条件显式报错；subset 重算当前范围 counts、保留 parent_counts | 避免漏汇总与把父集计数当子集计数 |

D1/D2/D3 的常数位置单独报告，核心变量节点指标排除常数；另列 changed_value_variables / unchanged_value_variables / reachable_variables。所有指标仍是观察窗口内的描述，不据此直接宣称内部检测机制。

## 本地验证

- B 23 项 CPU 测试通过，新增测试覆盖 early-only manifest、unrelated 共享后代、D2 无 clean 的新旧真值汇总。
- B/C Python 语法编译和 git diff --check 通过。
- C 新增第三父节点、已有错误与漂移区分、恢复后再错/末轮完成测试。**本地无 NumPy/PyTorch，C 测试未运行，模型和生成器未运行。**
- 未操作服务器任务、训练、GPU、权重或原始产物。没有新增实验结果。

## 两位执行 agent 同步与小验证

共享目录空闲并完成交接时，用 `git pull --ff-only origin reproduction`；各自先记录当前 HEAD。运行中的工作树不要 pull；如远端有并行提交先整合，不 reset/强推。按线领取以下检查，记录代码 SHA、实际命令、结果后再续各自已授权任务。

### Agent B

1. 运行 `python -m unittest discover -s tools/b_mechanism/tests -v`。用现有少量真实轨迹跑主汇总，确认 unrelated 的 protected_target 与注入位置确为不同节点；共享后代单列。
2. D1/D2/D3 各取一个已有或小验证记录，确认 `d123_rounds.csv`、`d123_by_H.csv`、`d123_events.csv`、`d123_coverage.csv` 都包含数据；新常数/新变量真值正确，D2 不需要 clean，旧/新真值不混合。
3. fixed 模式补一个“early 合格、整图未共同完成”的小案例，确认 early 保留、late/control 跳过。现有256图全完成的 manifest 可继续使用，不为本修复重复全量选样；subset counts 如需更新只做 CPU 重建。
4. 已有原始轨迹可重汇总，不覆盖旧汇总；这次不要求重跑 B1 或 M5。不自动扩大 B 主实验授权。

### Agent C

1. 运行 `python tools/c_capability/tests/test_c_common.py`。用两张不同节点数/不同正确率的合成轨迹确认节点加权与逐图等权不同，空深度桶不当作0分；同深度桶一张图只贡献一次宏平均。
2. 每个缺档仅生成极少图，确认 JSON/PT 可写可读，meta.generation_config 是实际生成参数，不含 Path；检查包含第三父节点的 sink 判定。
3. 一个权重、少量图跑完整 runner→summary，检查 VALUE/first4 两项节点准确率、完成后可观察分母、恢复后再错事件。新输出目录；沿用已批准的64轮自由 hidden 反馈与 s42/s1234 计划。
4. 本批 C 是能力与自然漂移探索，不训练 C+，不评 s2025。小验证通过后继续原授权实验即可。

请在 board/log 追加验证结果；代码“已修”与模型端“已验证”分别标记。本交接不替代后续结果解释或最终评估协议定稿。
