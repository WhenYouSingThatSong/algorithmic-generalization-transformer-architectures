# 实验工具索引

本目录是可追踪、可审阅的复现实验入口。当前授权/执行状态以[看板](../ai-collab/board.md)为准；某个旧实验目录中的“暂不训练”只约束当时那一批，不代表后续新批次的状态。

| 目录 | 主要用途 |
|---|---|
| [b_mechanism](b_mechanism/b_mechanism_README.md) | 离散A/B的早期与完成后注错、父子传播、持续污染、训练腐蚀审计 |
| [c_capability](c_capability/README.md) | C的规模/深度六档能力、逐轮表现与漂移 |
| [repair_study](repair_study/README.md) | B多父节点、C自然错误分类与联合局部替换 |
| [counterfactual_control](counterfactual_control/README.md) | 连续向量同值/异值不同来源对照、不同时点比较 |
| [c_long_pilot](c_long_pilot/README.md) | C额外D/2D轮长程保持后训练小试 |
| [c_corruption_pilot](c_corruption_pilot/README.md) | C保持/噪声/语义污染三组小规模后训练 |
| [c_evaluation_suite](c_evaluation_suite/README.md) | 上述C版本的基础能力、父子注错、持续与噪声评估 |
| [priority_pilots](priority_pilots/README.md) | C输入读出与下游计算对齐分析、A/B合法重新赋值 |
| [state_reuse](state_reuse/README.md) | A/B重复赋值与死槽位复用、C状态恢复后继续使用 |
| [abc_workspace](abc_workspace/README.md) | ABC共同赋值任务、固定32位置的20窗口长程序及边界诊断 |
| [c_update](c_update/README.md) | 最新：三个C纠错后训练＋一个第二层UPDATE监督版，各50epoch，固定评估与原C接收功能检验 |

最新`c_update`：代码和分工已交付，仓库截至2026-11-19 22:48尚无执行agent的新训练结果提交。Agent C先训练和阶段评估，交接后Agent B正式评估全部模型；使用同一清单，串行交接。不要把早期pilot已有权重算成本批50epoch已经完成。

## 文件放置

- `tools/` = tracked, reusable code and entrypoints.
- `docs/records/` = tracked design/experiment records.
- `tmp/` remains the local workspace for data, checkpoints, analysis outputs,
  archives, trash, and one-off historical scripts.
- Other `tmp/scripts` tools are intentionally left in place for now; moving
  them requires updating their `tmp/analysis`/`tmp/runs` default paths and
  launcher references.
