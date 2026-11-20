# C（多时点局部纠错＋过程监督UPDATE）smoke 与训练状态

日期：2026-11-20。状态：**smoke 通过；正式单 seed s2026 训练已在 GPU5 运行；不是效果结论**。  
代码：`reproduction@c09922a` 的 `tools/c_multitime`；唯一 plan `tmp/analysis/c-update-20261119/data/plan.json` SHA256 `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`。  
配方：`c_multitime_process_update_v1`，显示名 **C（多时点局部纠错＋过程监督UPDATE）**；从原 C s2026/e143 开始，50 epoch 上限，不扩到另两个种子。

## 1. 验收与身份

- `acceptance.json`（本目录 `c-multitime-smoke-acceptance.json`）：`status=passed`。
- smoke checkpoint SHA256 `e21642121e5ca72ca238b90e435de41e44b1bc220fbb6e313ae07aaddc51fef1`。
- 通过项：
  - `six_effective_batch32_steps`
  - `GPU_labels_equal`
  - `forward_unchanged`
  - `two_loop_BPTT`
  - `middle_head_gradient`
  - `reload_dev2_and_frozen_C_probe`
- micro/donor：**32/32**；没有改精度、loss、概率或数据。
- 完整 code/plan hash 在 acceptance 和 `c-multitime-smoke.json`；smoke training-audit 在 `c-multitime-smoke-audit.json`，step timings 在 `c-multitime-smoke-timings.jsonl`。

## 2. 第一次 GPU3 OOM 与 GPU4 重试

- 第一次 smoke 在 `CUDA_VISIBLE_DEVICES=3` 上 OOM：
  - 报错时外部进程 3978376 占 74.50 GiB，只剩 122.31 MiB 可用；本次尝试再分配 128 MiB 失败。
  - 这是**卡被外部任务占满**，不是 micro32/配方本身 OOM；未延长、未改概率。
  - 失败半成品保留在 `tmp/analysis/c-multitime-20261120/`（0 条 steps/injections）。
- 重试在 `CUDA_VISIBLE_DEVICES=4` 的空闲 GPU4 上通过；输出根为 `tmp/analysis/c-multitime-20261120-run/`。
- 正式训练随后在 GPU5 独占运行；GPU3 不重用、不在 GPU5 加评估或其他任务。

## 3. 六步 smoke 分模式

`candidates` 是该模式的结构候选总数；`attempts` 是实际源检查次数；`accepted` 是源读出通过并可替换的次数；`changed` 是 actual VALUE 确实改变。`before` 只统计 accepted 的注入事件；接收节点正确与否**不作为筛选**。

| 模式 | steps | candidates | attempts | accepted | source acc. | changed | before EMPTY | before numeric | before correct/wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frontier | 2 | 2,048 | 498 | 496 | 99.6% | 483 | 261 | 235 | 2 / 494 |
| middle | 2 | 1,088 | 241 | 241 | 100% | 241 | 3 | 238 | 237 / 4 |
| late_budget | 2 | 2,048 | 528 | 528 | 100% | 528 | 0 | 528 | 528 / 0 |

- `middle` 的 241 个 accepted 中，237 个注入前 VALUE 已经是正确值，仍按结构和概率注入；这验证了**无接收正确性过滤**。`late_budget` 的 smoke batch 全部已读为正确数值，不能据此说所有 late 主批都会完成。
- `frontier` 的 496 个 accepted 中，261 个改前 EMPTY，235 个 numeric；只有 2 个改前 VALUE 正确。
- smoke 的 accepted 中 `actual_value_changed`：frontier 483/496、middle 241/241、late 528/528；没有把“源读出合格”自动等同于“实际 VALUE 改变”。
- 直接子节点：frontier correct_children 0/895，middle 269/551，late 903/942；这是 smoke batch 的状态，不是总体比例。
- 源拒绝：frontier 2/498、middle 0/241、late 0/528；失败保留在日志/分母，不重抽或换规则。

## 4. smoke 耗时与显存

每模式 clean/启用各一步；时间含完整 batch 反向和 donor，不含后续 dev/checkpoint（另列）。

| mode/step | rounds | train step s | peak allocated GB |
|---|---:|---:|---:|
| frontier clean | 8 | 0.615 | 19.56 |
| frontier enabled | 8 | 0.841 | 19.66 |
| middle clean | 11 | 0.258 | 26.78 |
| middle enabled | 11 | 0.501 | 26.77 |
| late_budget clean | 19 | 0.441 | 45.74 |
| late_budget enabled | 19 | 2.015 | 45.74 |

- smoke checkpoint 总计 0.690 s；train epoch segment 5.208 s；dev 0 s（smoke 不跑 128-dev）。
- 首步含预热；不能用 0.2–0.6 s 的单步快照推断完整 50 epoch 的耗时。正式训练当前 peak_allocated 约 **45.8 GB**，nvidia-smi 当前约 **45.4 GB**；交接里用户转述的“约 73 GB”未在本次 `peak_allocated_bytes` 记录中重现，可能包含 allocator reserve/同卡其他占用，不能据此把 73 GB 当固定需求或富余。GPU5 运行期间不再加其他任务。

## 5. 正式训练状态（snapshot）

- 输出根：`tmp/analysis/c-multitime-20261120-run/train`
- snapshot：2026-11-20T19:00:47
- 当前 epoch **4**，step **5,959 / 78,200**（50 epoch，每 epoch 1,564 步）。
- 日志 max `peak_allocated_bytes` = **45.82 GB**；最近单步约 0.19–0.67 s。
- 已记录 3 个 epoch 训练段共约 1,516 s，2 次 dev 共约 186 s，checkpoint 约 5.5 s。
- 按当前实际进度约 **8–9 分钟/epoch**；剩余 46 epoch 粗略 **6–7 小时**，加上 dev/保存估计 **7–8 小时**。这是基于当前观测的 ETA，不是保证；首 epoch 后已按真实进度更新。
- 每个时点的实际候选/尝试/源通过/改变/改前状态会按 mode 汇总，不用“loss 小”替代。

## 6. 边界

- smoke 是技术验收，不是能力或污染效果估计；六步 max-depth batch 只验证链路、标签、梯度、重载和剂量记录。
- `late_budget` 固定 D+4，不代表整图完成；主训练按起点实际状态分组报告，不能把全部 late 称为“完成后注错”。
- 三模式用同一结构候选规则且等概率；新模式轮数更长，不能说效果只归因于 UPDATE/时点，也不能与旧配方做单因素消融。
- 中/晚时点若源拒绝或有效污染不足，保留计数并报告，不提高概率、不筛选容易样本。
- 原始 steps/injections/dev/权重留 tmp；本目录提交 acceptance、audit、timings 和汇总 JSON。