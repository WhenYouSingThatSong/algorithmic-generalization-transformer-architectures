# C（多时点局部纠错＋过程监督UPDATE）s2026 50 epoch 完成

日期：2026-11-21。状态：**训练正常退出（exit 0），final 已保存；dev 显示 clean 能力明显退化，不是效果结论**。  
代码：`reproduction@c09922a` 的 `tools/c_multitime`；plan `tmp/analysis/c-update-20261119/data/plan.json` SHA256 `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`。  
配方：`c_multitime_process_update_v1`，从原 C s2026/e143 起跑；micro/donor 32/32；50 epoch、78,200 步、2,500,000 presentations；seed 4216695006；GPU5 独占。

## 1. 身份与产物

- 输出根：`tmp/analysis/c-multitime-20261120-run/train`
- `final.ckpt` SHA256 **`c4c8b3c84fcf74f2990b1c1e40d51eb33176d715c04efda6bc358d96ddfa18cd`**（96,512,010 bytes，step 78,200）
- `meta.json` SHA256 `02a6cb8351cbd19790acfd641dd41a03906ae4a7a0559fc038d48caa7f6eeaf6`，`status=complete`；原 C source SHA `40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780`。
- `dev-50.json` SHA256 `fe37d8138a21417578404bac76456327ca91f38c2febdcf564a339b2571edaf2`；`dev-50.complete.json` SHA256 `e2a1192f9a360a9d33e2da8940b8bbdf3e922c59283f6366db0af197d20f2938`。
- `last.ckpt` 同 step 78,200，SHA256 `11e63ccc8a7d848f5274b20cc0d6c2a6d6882a639889dc992e04bdfd7fb678a7`；正式身份以 `final.ckpt` 为准。
- 训练日志：`tmp/analysis/c-multitime-20261120-run-train.log`；原始 steps/injections/timings 留在 run 目录。

## 2. dev clean 曲线（128 图，整图 VALUE 全对）

| epoch | D | D+4 | D+16 VALUE | D+16 first4 |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 0.625 |
| 5 | 0.305 | 0.148 | 0.133 | 0.016 |
| 10 | 0.195 | 0.180 | 0.172 | 0.000 |
| 25 | 0.570 | 0.516 | 0.500 | 0.430 |
| 30 | 0.836 | 0.766 | 0.734 | 0.008 |
| 35 | 0.781 | 0.555 | 0.375 | 0.344 |
| 40 | 0.531 | 0.414 | 0.359 | 0.258 |
| 45 | 0.562 | 0.422 | 0.406 | 0.406 |
| **50** | **0.5625** | **0.4609** | **0.4375** | **0.4062** |

- e50 节点均值：D mean_value **0.9189**（VARIABLE 1.000、OPERATION 1.000）；D+4 0.8296；D+16 mean_value 0.8074、mean_variable 0.9983。
- `best` 仍为 **epoch 1，score 1.0**；`best-dev.ckpt` 保存在 e1，未被后训练超过。
- 曲线不是稳定收敛：e5–e20 大幅下降，e25–e30 回升，e35–e40 再下降；e45–e50 在低水平略回升，但 D 整图仍只有 0.5625。
- 这是 clean 前向读数，未经过 UPDATE 头；说明该长窗口/多时点训练配方在本 seed 上明显损伤了基础整图求解能力，而不是只影响纠错分支。不能据此单独归因于多时点、长轮数或过程监督中的某一项。

## 3. 实际剂量

- 78,200 步中启用 **3,975 步（5.08%）**；结构候选尝试 1,724,888，源接受 1,549,584（89.8%），实际 VALUE 改变 1,530,149。
- 分模式：frontier 1,363 enabled / 697,818 attempts / 661,529 accepted / 645,571 changed；middle 1,321 / 366,507 / 357,880 / 357,324；late_budget 1,291 / 660,563 / 530,175 / 527,254。
- `peak_allocated_bytes` 最大 **45.83 GB**；GPU5 全程独占，未与 B 或其他任务共享。

## 4. 边界

- 以上全部是 **dev**；正式测试与同 seed 旧 C 对比由 B 按冻结方案执行，C 不提前下效果结论。
- e50 不是 best-dev；正式评估按协议用 e50，不拿 e1/e30 回代。两者都保留，便于区分“训练中最好”与“按协议 final”。
- 该配方与新的正常窗口 `c_local_update_normal_window_v1` 在时点、轮数、损失上不同，不能把两者差异单因素归因；新的三种子训练仍在进行，尚未有 final。
- 原始 dev/steps/injections/权重留在 tmp；本目录提交 final/meta/dev-50 的哈希与摘要。
