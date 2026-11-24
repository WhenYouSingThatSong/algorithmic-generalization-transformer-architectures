# ABC共同任务：重新赋值、固定空间长计算、边界诊断

对应用户批准的前三步；**不训练、不改模型结构、不启动第四步**。
完整范围及验收：[执行交接](../../ai-collab/handoff/2026-11-19-abc-workspace.md)。

| 文件 | 作用 |
|---|---|
| protocol.py | 新任务、合法赋值、32槽窗口、跨窗口依赖、独立参考计算、实际提交接口 |
| prepare.py | 生成共享任务清单和固定权重注册表；不加载模型 |
| runtime.py | A/B离散反馈与C连续hidden反馈；局部题面写入；显式区分清空语义 |
| run.py | 两个任务及固定时点诊断；实际输入输出、接口提交、真值与来源记录 |
| summarize.py | 逐任务等权、共同起点、程序成功／失败位置、配对修回及损伤 |
| execute.py | AB/C分队列串行运行，小验证后执行原定main；不自动重试失败目录 |

## 最小执行顺序（由服务器agent执行）

```bash
# 共享清单只生成一次；另一条线必须复用完全相同的文件和hash。
python tools/abc_workspace/prepare.py --out-dir tmp/analysis/abc-workspace-20261119/data

python tools/abc_workspace/execute.py --data-dir tmp/analysis/abc-workspace-20261119/data \
  --lane AB --phase smoke --gpu 2 --out-dir tmp/analysis/abc-workspace-20261119/AB-smoke

# 完成真实检查后写checks.md；这是执行验收，不是重新请求用户授权。
python tools/abc_workspace/execute.py --data-dir tmp/analysis/abc-workspace-20261119/data \
  --lane AB --phase main --gpu 2 --smoke-dir tmp/analysis/abc-workspace-20261119/AB-smoke \
  --checks tmp/analysis/abc-workspace-20261119/AB-smoke/checks.md \
  --out-dir tmp/analysis/abc-workspace-20261119/AB-main

# Agent C将lane和目录中的AB换成C；GPU编号只是示例，选择真实空闲资源。
```

共同汇总（两条线都完成后；不能把各自common分母拼起来）：

```bash
python tools/abc_workspace/merge_lanes.py \
  --ab-dir tmp/analysis/abc-workspace-20261119/AB-main \
  --c-dir tmp/analysis/abc-workspace-20261119/C-main \
  --out-dir tmp/analysis/abc-workspace-20261119/ABC-summary
```

独立记录复核（CPU；不加载模型、不跑GPU）：

```bash
python tools/abc_workspace/checks/verify_ab_records.py \
  --bundle tmp/analysis/abc-workspace-20261119/AB-main \
  --report docs/records/2026-11-19/ab-workspace/ab-verifier.json
```

该脚本分别核对 reference 程序真值与 actual-input conditional 真值；两者不要求相等。
它同时重建 summary、coverage、program 轨迹和 probe 交并。

## 必须随结果保留的边界

- C重新赋值只在指定题面位置写入新token的embedding，其余hidden继续传递；无每轮解码回灌。
- **长程序统一使用外部数值提交接口**：上一窗口模型实际输出成为下一窗口输入。C是“连续计算＋离散边界提交”的混合协议，不是纯潜空间无限延长。未实现纯连续跨窗口提交，也不冒充它。
- C reset_results重新初始化整行hidden；A/B只清空VALUE。两者不是等价扰动。共同scratch与共同任务提供参照，不能直接把reset差异归因架构。
- 固定32变量位置、256 padded tokens；20窗口累计526逻辑节点／520算术节点，最终输出真正依赖多少历史计算另报。不能说峰值宽度突破32或原模型原生处理526节点整图。
- 程序每条分支使用自己的实际提交值；输入错误可继承。conditional指标只说明给定已提交输入后的局部求解，不能代替原程序正确率。
- 非数值提交直接计执行失败，后续阶段继续占分母，无真值/零值回填。没有过滤失败图再补样。
- 先前材料中的机制都只是候选解释。探针条件固定且成功/失败都保留，不宣称找到了统一错误位置。

## 指标

`*_final_n/all/macro`：最后一次输出，与实际提交一致。没有final后缀的`*_n/all/macro`：末两轮同时正确。两者必须分列；稳定后缀至少两轮，不能宣称数学收敛。

`summary.csv`包含节点VALUE、四属性、终端输出、题面字段保持和条件正确率；`programs.csv`包含首次终端提交失败、全阶段成功、最终成功、实际累计长度与输出祖先算术节点数；`probe_pairs.csv`是同起点、同额外预算的修回/改坏节点交集。图内比例再图等权；先按形状分层，不把更长程序当更多独立样本。

本地6项针对性检查使用真实tokenizer及纯Python：序列/独立求值、提交错误保留、真实跨帧依赖、赋值回原题、稳定与字段区分、含无效提交的汇总与配对诊断。GPU/真实checkpoint检查交执行agent。
