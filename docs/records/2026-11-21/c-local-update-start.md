# 正常窗口局部 UPDATE：8 图小测、最大深度 smoke 与三种子启动

日期：2026-11-21。状态：**8 图小测与三模式 smoke 出数；三个种子已在 GPU1/2/3 启动；不是效果结论**。  
代码：管理 worktree `8140ca8` 的 `tools/c_local_update`；主训练树 `0e65595` 未 pull，旧 multitime 运行继续。  
配方：`c_local_update_normal_window_v1`；唯一 plan `tmp/analysis/c-update-20261119/data/plan.json` SHA256 `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`；训练数据 SHA256 `cee36b84b01b7c1fdbd2e32619b402e571275c534a4d685684db07a6e11de1ce`。  
清单：`tmp/analysis/c-local-update-normal-v1/manifest.json` SHA256 `a91fd53e2e13251ad02284b8e3262869066cf471128f7f68d0744d7fdedbfb1c`。

## 1. C_s2026 8 图 minimal dev

- 输出：`tmp/analysis/c-local-update-normal-v1/minimal-dev-s2026-e0.json`（SHA256 `c220889f36c713088326aea93ea63a06c1bef70e9b53e43a3963525f703df078`）。
- 固定 dev 图 0–7：8 图、256 节点；clean VALUE **256/256 = 1.000**，四字段 **256/256 = 1.000**；epoch 0。
- 第二层 UPDATE 为**新初始化**，因此 UPDATE 计划分数只检验评测链路，不是能力读数：
  - `frontier/clean/consistent_correct`：planned 1,890，tp 0，fp 1,890；执行正确 3,673/3,673，aligned 1,783。
  - `frontier/clean/new_calculation`：reference 175，planned 73，tp 73，fp 0，fn 102；实际 VALUE 改变 175/175，四字段 jointly 73。
  - `middle/wrong/direct_conflict`：reference 92，planned 44，tp 44，fp 0，fn 48；执行正确 68，实际改变 90，jointly correct 31。
- 实际分母已保留：每个条件 8 图；`actual_complete/parent_child_consistent_wrong` 只有 4 图有合格源，其余记为 `source_not_locally_consistent`；不可注入和未完成不重抽。
- 88 条分支记录全部在原始 JSON 中；未把新读出头的高 false-positive 当作后训练失败。

## 2. 最大深度三模式反传 smoke

- 输出目录：`tmp/analysis/c-local-update-normal-v1/train/C_s2026-smoke`；meta SHA256 `ca7e86098e09be21e467f6a2cd17c7557bcbdbceb26bef52d4ce2868e11327d3`。
- 训练数据最大深度 batch（32 图，D=7，8 轮），micro/donor **32/32**；三个模式各一次开启前向/反向，第二层 UPDATE 头梯度均非零。

| 模式 | events | accepted | actual VALUE changed | head grad norm |
|---|---:|---:|---:|---:|
| frontier | 503 | 503 | 494 | 553.802 |
| middle | 291 | 291 | 291 | 574.973 |
| completed | 510 | 328 | 328 | 591.232 |

- `completed` 的 510 个候选中 328 个通过整图 D 时 VALUE 全对检查；未通过的不注入，保留在 events 分母。
- 这是链路/梯度/剂量验收，不是污染效果估计；三模式轮数相同（D+1=8），没有长程训练。

## 3. 三种子独立训练

- 启动：2026-11-21 00:29；从同 seed 原 C 重新初始化第二层 UPDATE，保留连续反馈；同一份 50,000 张 N32、相同抽图与位置；有效 batch 32、50 epoch，总步数 78,200。
- 三个独立进程、三块已确认空闲的卡；每卡显存约 19.9 GB，启动时利用率 93–96%。

| 模型 | 物理 GPU | 原 C SHA256 | 训练 seed | 输出 |
|---|---|---:|---:|---|
| C_s42 | 1 | `4a00d6a21a7ed6ec9730daccf27f6077af45d2f2d7ff7376386bf0aa8903a798` | 1,296,029,488 | `tmp/analysis/c-local-update-normal-v1/train/C_s42` |
| C_s1234 | 2 | `0bca3a2b91e82d70d5808859d3a907e0b8a32e0a852ba483c208f955d7d70475` | 2,942,770,351 | `tmp/analysis/c-local-update-normal-v1/train/C_s1234` |
| C_s2026 | 3 | `40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780` | 900,330,519 | `tmp/analysis/c-local-update-normal-v1/train/C_s2026` |

- 三个 `meta.json` 的 identity 都记录同一 manifest SHA、micro/donor 32 和各自 source SHA；启动日志在 `tmp/analysis/c-local-update-normal-v1/train-<model>.log`。
- epoch 1 自动做 128 图 dev 和 32 图分类；正式 epoch 50 由 B 评测，C 不提前报告效果。
- 旧 multitime C_s2026 仍在 GPU5 独占运行（本记录时 epoch 42/50、step 65,132）；未打断、未共享显卡、未改旧版本。

## 4. 边界

- 8 图小测是固定链路的链路检查；VALUE/first4 1.000 只说明 clean 前向和评测读取一致，不说明 UPDATE 能力。
- smoke 的新 UPDATE 头是随机初始化；梯度非零只证明反传路径和标签可用。
- 三个种子并行训练尚未完成；不能根据当前 loss 或早期 dev 推断最终效果。
- 未启动 B 的正式评估；三份 final 的路径/SHA 出来后再交 B。原始 steps、dev、权重留 tmp，本目录只提交摘要和身份。
