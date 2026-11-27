# s42 三件套续训到累计 200 周期：里程碑取点记录

- 日期：2026-11-26 18:15 起跑 – 2026-11-27 01:33 全部结束
- 目的：PLAN-009 要求把 A′、B、C 的训练预算对齐后再比较。C 原本已到 150 周期，A′/B 只到约 86–92，故续训到同为 200。
- 协议：**沿用原论文协议不变**——训练 `Tr32Test128-ADD/train_data.pt`（N32，depth≤6），训练期验证 `Tr32Test128-ADD/val_data.pt`（N128，depth 7–18，分布外），单流。未改任何配置文件，全部走命令行参数。

## 1. 续训设置

| 权重 | 起点 | 起点 epoch | 目标 | 实际新增 epoch | 输出目录 | 结束时间 |
|---|---|---|---:|---:|---|---|
| A′ s42 | `A-L4D384fast/.../last.ckpt` | 86 | 200 | 87–199（113 个） | `tmp/runs/A-L4D384fast-200ep/` | 11-27 01:33 |
| B s42 | `B-M6/.../last.ckpt` | 92 | 200 | 93–199（107 个） | `tmp/runs/B-M6-200ep/` | 11-27 01:11 |
| C s42 | `oldC0-resume-150ep/.../final.ckpt` | 149 | 200 | 150–199（50 个） | `tmp/runs/oldC0-resume-200ep/` | 11-26 22:00 |

命令（三个并行，GPU 0/1/2）：

```bash
python experiments/train.py \
  --config_dir experiments/configs/<对应配置> --seed 42 \
  --checkpoint_dir <新目录> \
  --resume_from_checkpoint <起点> \
  --max_epochs 200 --save_top_k -1
```

- `--save_top_k -1`：每轮都存，因此 **e150 / e200 的权重必然存在**，且不依赖 `last.ckpt`（该字段在本 Lightning 版本会被"last 伪影"覆盖或滞后）。
- `--max_epochs` 的语义是"跑到哪个 epoch 编号为止"，不是"再跑多少"。已用 1-epoch 试跑逐条验证过三个起点都能恢复（`Restored all states`）。
- A′/B 的 `lr_scheduler` 是 `none`，LR 恒为 0.001，所以改 `max_epochs` 不会改变学习率调度。
- 原三个权重目录**未被写入**；`rebinding-identity.json` 钉住的 A_s42 e86 / B_s42 e92 / C_s42 e143 完好。

## 2. 逐 epoch 选择值从哪里来

checkpoint 文件名是 `{epoch}-{val_loss:.4f}.ckpt`，而实际 val 都在 1e-7–1e-5 量级，**全部渲染成 `0.0000`**，文件名不含排序信息。因此逐 epoch 的 `total_loss/val` 是从训练日志的 `Epoch N: 100%` 行解析出来的（取每 epoch 最后一次进度行）。旧段（A′ 0–86、B 0–92、C 95–149）从各自原始日志解析，新段从本次日志解析。

## 3. 里程碑取点

A′/B 的 `val`、`free val`、`teacher val` 都列出；C 只有 `total_loss/val`。

### 3.1 到 150 周期为止，val 最好的三个

| 家族 | 排名 | epoch | total_loss/val | free | teacher | 权重位置 |
|---|---:|---:|---:|---:|---:|---|
| A′ s42 | 1 | **e133** | **8.58e-7** | 9.88e-7 | 7.77e-7 | `A-L4D384fast-200ep/` |
| | 2 | e149 | 1.14e-6 | 1.13e-6 | 1.16e-6 | `A-L4D384fast-200ep/` |
| | 3 | e86 | 1.40e-6 | 1.47e-6 | 1.33e-6 | `A-L4D384fast/`（原） |
| B s42 | 1 | **e111** | **3.13e-6** | 3.21e-6 | 3.04e-6 | `B-M6-200ep/` |
| | 2 | e92 | 3.25e-6 | 3.41e-6 | 3.02e-6 | `B-M6/`（原） |
| | 3 | e142 | 3.57e-6 | 3.50e-6 | 3.65e-6 | `B-M6-200ep/` |
| C s42 | 1 | **e143** | **0.0662** | — | — | `oldC0-resume-150ep/`（原） |
| | 2 | e142 | 0.126 | — | — | `oldC0-resume-150ep/` |
| | 3 | e116 | 0.136 | — | — | `oldC0-resume-150ep/` |

第 150 个 epoch 本身的权重：A′ e149 = 1.14e-6；B e149 = 1.69e-5；C e149 = 0.567。

### 3.2 到 200 周期为止，val 最好的三个

| 家族 | 排名 | epoch | total_loss/val | free | teacher | 权重位置 |
|---|---:|---:|---:|---:|---:|---|
| A′ s42 | 1 | **e188** | **4.90e-7** | 4.87e-7 | 4.94e-7 | `A-L4D384fast-200ep/` |
| | 2 | e167 | 5.85e-7 | 5.87e-7 | 5.84e-7 | `A-L4D384fast-200ep/` |
| | 3 | e170 | 7.95e-7 | 7.74e-7 | 8.34e-7 | `A-L4D384fast-200ep/` |
| B s42 | 1 | **e174** | **2.76e-6** | 2.80e-6 | 2.71e-6 | `B-M6-200ep/` |
| | 2 | e111 | 3.13e-6 | 3.21e-6 | 3.04e-6 | `B-M6-200ep/` |
| | 3 | e153 | 3.19e-6 | 3.04e-6 | 3.31e-6 | `B-M6-200ep/` |
| C s42 | 1 | **e177** | **0.0453** | — | — | `oldC0-resume-200ep/` |
| | 2 | e190 | 0.0599 | — | — | `oldC0-resume-200ep/` |
| | 3 | e143 | 0.0662 | — | — | `oldC0-resume-150ep/`（原） |

第 200 个 epoch 本身的权重：A′ e199 = 0.0186；B e199 = 8.07e-6；C e199 = 0.128。

**注意：第 200 个 epoch 本身并不是最好的。** A′ 的 e199 是 0.0186，而 e188 是 4.90e-7，相差约 4 万倍。所以"200 周期权重"与"200 周期窗口内最好的权重"是两件事，报告时必须写明用的是哪个。

文件名逐一核对（按 seed 严格过滤，避免取到其它种子的同名 epoch）：

```
tmp/runs/A-L4D384fast-200ep/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-26-18:15:40/epoch={133,149,167,170,188,199}-val_loss=0.0000.ckpt
tmp/runs/B-M6-200ep/Nodes32-ADD - L4H16D384_DeBERTa_recorrection-seed-42 - 2026-11-26-18:15:40/epoch={111,142,149,153,174,199}-val_loss=0.0000.ckpt
tmp/runs/oldC0-resume-200ep/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-26-18:15:40/epoch={150..199}-val_loss=0.0000.ckpt
```

## 4. 这个提升是真的吗（排除"抽奖次数变多"）

val 极其不稳：A′ 的 `total_loss/val` 在 200 个 epoch 上跨 **7 个数量级**（最小 4.90e-7，中位 6.5e-5，最大 9.33，最大/最小约 1900 万倍）。因此"200 轮里取最小"会天然偏向更小的值。用**等长窗口**做对照：

| 家族 | 早窗（等长） | 早窗 best | 晚窗（等长） | 晚窗 best | 晚/早 | P(val 低于阈值) |
|---|---|---:|---|---:|---:|---|
| A′ s42 | e0–112（113 个） | 1.40e-6（e86） | e87–199（113 个） | 4.90e-7（e188） | **0.35×** | 26% → **44%**（<1e-5） |
| B s42 | e0–106（107 个） | 3.25e-6（e92） | e93–199（107 个） | 2.76e-6（e174） | **0.85×** | 11% → **32%**（<1e-5） |
| C s42 | e95–149（55 个） | 0.0662（e143） | e150–199（50 个） | 0.0453（e177） | **0.68×** | 2% → **6%**（<0.1） |

**结论：提升不是纯粹的抽奖效应。** 等长窗口下，后段的"好 epoch 出现频率"明显更高（A′ 26%→44%，B 11%→32%），且最优值也更低。B 的变化尤其值得记：种子曲线文档里"末端退化"的印象来自单点 e99，而续训后 B 的好 epoch 频率翻了三倍。

## 5. 使用这些权重时必须注意

1. **这里的 val 是 N128 分布外指标**，且它是**选点依据**。`CODEBOOK.md` 明确写着 `val_data.pt`「不是独立 test；用它报最终成绩 = 验证集泄漏」。所以本表的排名**只能用于选权重**，不能当性能报告。
2. **最终评测必须用独立的 `experiments/data/N128-heldout-ADD/testdata_N128.pt`**（2000 条，depth 7–18，seed 20261212，与旧 val 0 行重叠）。
3. **C 的 `total_loss` 与 A′/B 不可横比**（聚合口径与回灌内容不同，见 `loss-paths-and-upstream-changes.md`），C 的排名只能家族内部用。
4. 选点规则已冻结：同时报告"全段 best-by-val"与"里程碑 epoch 本身的权重"，不在看到最终评测结果后改选。

## 6. 来源

- 启动器：`tmp/scripts/resume_to_200ep_s42.sh`
- 日志：`tmp/runs/{A-L4D384fast,B-M6,oldC0-resume}-200ep/logs/*.log`
- 逐 epoch val：由上述日志的 `Epoch N: 100%` 行解析；旧段来自各 run 原始日志
- 相关：`docs/records/2026-11-26/loss-paths-and-upstream-changes.md`（损失口径与验证集口径）、`tmp/records/CODEBOOK.md`（权重与数据代号权威）、`tmp/records/WEIGHTS_DATA_EXPERIMENTS.md`（协议总表）
