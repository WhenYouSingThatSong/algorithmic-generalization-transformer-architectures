# B 机制修复复核与必要修复

时间：2026-11-14 14:53 CST。基线：`reproduction@e27cc6b73f4daf8aaf1b5e8711a9b1a6455f19ab`；服务器修复提交：`66440b8`。

用户本轮授权必要修复并要求保留记录、遵守协作协议。本地工作分支：`review/b-mechanism-e27cc6b`。未修改 consensus、指标公式、训练配置或服务器进程；没有启动 B 全量实验。

## 复核结果

| 项目 | 本轮状态 | 验证边界 |
|---|---|---|
| R1 共同资格键 | 修复成立 | 源码 + CPU 检查覆盖 A 合格/B 不合格、反向情形、缺键及共同标记不一致 |
| R3 单图轮数预算 | 修复成立 | 搜索按单图预算截断；第 11 轮才成功的样本在 8 轮预算下被拒绝，非法 manifest 被拒绝 |
| R2 汇总配对与分组 | 原先缺陷已修 | 实际 summary CLI 对合成 A/B、own/teacher、before/after 记录分别汇总；输入顺序交换不变，冲突 clean 报错 |
| R7 controls 接口 | condition/clean 接通；切换执行仍有缺陷，本轮已修 | controls 合成记录能被 summary 配对；实际 switch hook 另做时序检查 |
| R8 新真值 | 图节点的新标签已接通；题面常数漏更新，本轮已修 | 常数、叶变量、下游的新标签一并检查；原旧标签保留 |
| R4–R6/R9 | 仍为待讨论方案 | 不把首轮/全窗口、曾恢复/持续恢复、节点/图聚合当作已定标准 |

## F1：合法 switch 参数全部被 hold 分支遮蔽（该实验的高优先级缺陷）

`66440b8` 将合法范围改为 `1 <= switch_at <= hold_rounds`，但 hook 仍先执行 `if t < hold_rounds: return first_wrong`。请求切换的轮次必然落在此分支内，第二个错值没有机会写入。

对基线源代码实际提取 CLI 校验及 hook 执行，`hold_rounds=4, switch_at=4` 被接受；注入轨迹为 `[7,7,7,7,release,release,release,release]`，没有第二个错值 10。此前 smoke 中“switch 输出了记录”只能证明流程运行，不能证明发生切换。

修复采用现有文字所描述的语义：先固定第一个错值 N 轮，第 N+1 轮注入第二个错值一次，然后释放。要求 `switch_at == hold_rounds + 1 <= horizon`；非法范围或观察轮数不足显式报错。README 例子改为 `--hold_rounds 4 --switch_at 5 --horizon 10`。对照 sustained 在第 N+1 轮直接释放。

此改动不把该模式升级成“稳定错误状态切换”，仍不能单凭它区分无条件重算和检测不一致后重算；是否启动此实验继续待定。已有错误版本的 switch 轨迹不能用来分析切换，需要按修复版本补跑；sustained 不因这处缺陷作废。

## F2：D1/D2/D3 新真值遗漏被改的常数（局部记录缺陷）

`recompute_truth()` 原来仅返回 `meta['nodes']`，其中不含题面常数；但 `tracked_positions()` 会把被修改的 `const_pos` 纳入记录。于是 `truth_label()` 更新叶变量和下游，却仍把常数的新预测与旧值比较。

例：常数 4 改为 5，叶变量和下游已按新题正确输出；`*_nodes_new[const_pos]` 的真值仍是 4，错误地标记失败。修复把 `constant_override` 加入返回的真值映射，使该常数也用新标签，旧真值字段保持原义。

影响限定于旧版本新真值记录中的被修改常数；不能据此否定已正确更新的图节点标签或模型轨迹。已有原始轨迹和 injection 元数据完整时，可重新派生新真值记录，不必重跑模型。D1/D2/D3 的综合汇总仍待单独定义。

## 本地验证

- 新增 `tools/b_mechanism/tests/test_review_regressions.py`，命令：`python -I tools/b_mechanism/tests/test_review_regressions.py`。
- 8 项 unittest 全部通过，包括参数组合子用例；8 个 Python 文件语法编译通过；`git diff --check` 通过。
- 测试从源码 AST 提取纯逻辑函数和实际嵌套 hook，张量/模型由小型替身提供；summary 使用真实 CLI 子进程和合成记录。不是模型推理或 GPU smoke。
- 修复前实际复现了常数新标签错误；切换遮蔽也通过基线实际 guard/hook 的独立检查复现。上游 R1/R2/R3 的针对性检查在改动前已经通过。
- 本机原无可用 Python；独立测试运行时置于仓库外 `C:/Users/adam2/Desktop/projects/.review-tools/python-3.12.10/`，来自 [Python 官方便携包](https://www.python.org/downloads/release/python-31210/)，校验下载页校验和；未安装系统组件或模型依赖，运行时不提交 Git。

## 文档与后续

已同步 README、design 的切换时序；把“预定 T”改为实际的“首次共同完成 T”，明确 VALUE 完成不等于四字段或持续稳定。指标段落统一标为候选、未定稿。

已在看板和只追加日志中记录本次结论，保留历史记录。服务器端合入后，建议只先做小样本确认：切换实际输入序列含第二个错值且只出现一次；D1/D2/D3 的常数、叶变量、下游新标签一致。正式模型实验未在本地执行，是否全量启动仍由用户决定。
