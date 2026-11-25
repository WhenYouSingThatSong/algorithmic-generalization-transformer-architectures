# A/B/C 主要种子的训练与验证 loss 汇总

- 日期：2026-11-25
- 范围：与主流 A/B/C 复现使用**同一训练数据**（`experiments/data/Tr32Test128-ADD`，50k，N32，depth≤6，mod 23）和**同一 L4H16D384 主体**的权重。
- 不包含：历史 A-M5（L2H16D256）、A-L4fast（L4D256）、C0-MixedN、以及后训练 C。它们的数据/架构/目标不同，不能混进这张表。
- 数据来源：Lightning 训练日志；每个 epoch 取 `Epoch N: 100%` 的最后一个进度行。
- 指标：
  - `train_epoch` = `total_loss/train_epoch`，该 epoch 的平均训练总损失。
  - `val` = `total_loss/val`，训练时验证集总损失。
  - `free val` = `total_loss_TchrPred_False_RegCorr_False/val`，A/B 的自由运行验证。
  - `teacher val` = `total_loss_TchrPred_True_RegCorr_False/val`，A/B 的 teacher-forced 验证。
  - C 日志只有 `train_epoch` / `val`。
- **重要**：A/B 的 `total_loss` 与 C 的 `total_loss` 不是同一目标/归一化。A/B 是离散反馈 + fast 路径，C 是连续 hidden 反馈 + nointerm。loss 绝对值不要跨家族比较，只看同家族内部的收敛、选中点与末端。

## 1. 共同条件

| 项目 | A′ | B | C（原始） |
|---|---|---|---|
| 主体 | L4H16D384 | L4H16D384 | L4H16D384 |
| 反馈 | 离散字段回灌 | 离散字段回灌 | 连续 hidden 回灌（nointerm） |
| 训练数据 | Tr32Test128-ADD，50k，N32，depth≤6 | 同左 | 同左 |
| 训练路径 | fast | fast | 标准 / nointerm |
| 纠错训练 | 无 | VALUE 污染（5% batch × 50% 位置） | 无 |
| epoch | 100 | 100 | 150（s42/s1234 为续训；s2026 从头） |
| 主权重 | s42 e86、s1234 e90、s2025 e94 | B-M6 s42 e92；cyhB s42 e84、s1234 e88、s2025 e74 | s42/s1234/s2026 e143 |

> cyhB 是外部迁移的同类 B 权重（DeBERTa L4H16D384，32 节点，mod 23，batch 32，AdamW lr 0.001，max train loop 8，max val loop 36，corruption 0.05，strategy `last`，tokenwise 0.5，100 epoch）。迁移说明见 `/workspace/lab-xj/users/jinxi/code/cyh/aga-migration-20261108/README_MIGRATION.md`。

## 2. A′ 三种子（`A-L4D384fast`）

### 2.1 选中点与最后 epoch

| seed | 选中 | train_epoch | val | free val | teacher val | 最后 epoch | final train_epoch | final val | final free | final teacher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | e86 | 5.75e-5 | 1.40e-6 | 1.47e-6 | 1.33e-6 | e99 | 4.02e-5 | 5.93e-3 | 8.81e-3 | 1.96e-5 |
| 1234 | e90 | 3.53e-5 | 6.14e-7 | 5.99e-7 | 6.29e-7 | e99 | 4.12e-5 | 6.02e-6 | 1.33e-6 | 9.37e-6 |
| 2025 | e94 | 4.95e-5 | 2.59e-7 | 1.97e-7 | 3.47e-7 | e99 | 5.39e-5 | 2.23e-2 | 6.34e-2 | 2.44e-5 |

选中 checkpoint：

- s42：`tmp/runs/A-L4D384fast/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-09-15:42:45/epoch=86-val_loss=0.0000.ckpt`
- s1234：`tmp/runs/A-L4D384fast/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-1234 - 2026-11-09-16:15:53/epoch=90-val_loss=0.0000.ckpt`
- s2025：`tmp/runs/A-L4D384fast/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-2025 - 2026-11-09-16:30:43/epoch=94-val_loss=0.0000.ckpt`

### 2.2 验证 loss 曲线关键点（`total_loss/val`）

| seed \ epoch | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 选中 | 最后 e99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8.160 | 5.790 | 1.9e-5 | 6.12e-6 | 3.25e-4 | 1.43e-5 | 2.46e-6 | 8.56e-6 | 7.43e-6 | 1.11e-5 | 1.40e-6 (e86) | 5.93e-3 |
| 1234 | 7.500 | 6.28e-3 | 6.57e-2 | 8.03e-3 | 1.57e-6 | 8.52e-4 | 9.77e-6 | 1.44e-4 | 2.17e-2 | 6.14e-7 | 6.14e-7 (e90) | 6.02e-6 |
| 2025 | 8.500 | 9.06e-4 | 1.82e-2 | 9.73e-6 | 1.63e-2 | 2.40e-6 | 7.03e-6 | 1.20e-3 | 3.73e-7 | 5.74e-4 | 2.59e-7 (e94) | 2.23e-2 |

观察：

- 三条曲线都很噪，前期有大幅波动；选中点的 val 都是 1e-7–1e-6 量级。
- s1234 选中点 e90 的 val 最低（6.14e-7）；s2025 e94 也很低（2.59e-7）。
- **最后 e99 的 val 不等于主权重**：s42 从 1.4e-6 升到 5.93e-3，s2025 从 2.59e-7 升到 2.23e-2；s1234 仍很低（6.02e-6）。A′ 的三个主权重都取在末端上冲之前。

## 3. B 种子（B-M6 本组 + cyhB 外部迁移）

### 3.1 选中点与最后 epoch

| 家族 | seed | 选中 | train_epoch | val | free val | teacher val | 最后 epoch | final train_epoch | final val | final free | final teacher |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B-M6 本组 | 42 | e92 | 1.87e-4 | 3.25e-6 | 3.41e-6 | 3.02e-6 | e99 | 1.70e-4 | 1.42e-1 | 2.46e-1 | 2.69e-4 |
| cyhB | 42 | e84 | 3.34e-4 | 3.14e-5 | 3.09e-5 | 3.16e-5 | e99 | 3.27e-4 | 8.34e-5 | 8.28e-5 | 8.39e-5 |
| cyhB | 1234 | e88 | 3.42e-4 | 2.67e-5 | 2.38e-5 | 2.94e-5 | e99 | 2.57e-4 | 1.29e-2 | 1.98e-2 | 3.17e-4 |
| cyhB | 2025 | e74 | 2.16e-4 | 1.09e-5 | 1.13e-5 | 1.06e-5 | e99 | 2.09e-4 | 5.51e-5 | 5.48e-5 | 5.55e-5 |

选中 checkpoint：

- B-M6 s42 e92：`tmp/runs/B-M6/checkpoints/*/epoch=92-val_loss=0.0000.ckpt`
- cyhB s42 e84：`tmp/cyh_ckpt/seed-42/Nodes32-ADD - L4H16D384_DeBERTa_recorrection-seed-42 - 2026-11-08-10:42:15/epoch=84-val_loss=0.0000.ckpt`
- cyhB s1234 e88：`tmp/cyh_ckpt/seed-1234/Nodes32-ADD - L4H16D384_DeBERTa_recorrection-seed-1234 - 2026-11-08-10:42:15/epoch=88-val_loss=0.0000.ckpt`
- cyhB s2025 e74：`tmp/cyh_ckpt/seed-2025/Nodes32-ADD - L4H16D384_DeBERTa_recorrection-seed-2025 - 2026-11-08-11:02:06/epoch=74-val_loss=0.0000.ckpt`

### 3.2 cyhB 验证 loss 关键点（`total_loss/val`）

| seed | 60 | 70 | 80 | 选中 | 90 | 99 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1.55e-4 | 1.25e-2 | 1.79e-3 | **3.14e-5 (e84)** | 2.32e-3 | 8.34e-5 |
| 1234 | — | 3.98e-5 | 2.24e-5 | **2.67e-5 (e88)** | 5.46e-5 | 1.29e-2 |
| 2025 | 3.79e-5 | 1.33e-4 | 7.02e-4 | **1.09e-5 (e74)** | 3.95e-5 | 5.51e-5 |

观察：

- cyhB 的选中点都明显低于各自 e99；e99 的 val 在 s42/s1234/s2025 分别为 8.34e-5/1.29e-2/5.51e-5。
- B-M6 本组 e92 的 val 3.25e-6，低于 cyhB 的 1e-5–3e-5 量级；但 cyhB 是外部迁移、恢复/续训历史不同，不能把这点差直接归因于模型或配方。
- B-M6 e99 的 free val（0.246）远高于 teacher val（2.69e-4），说明最后 epoch 的自由运行状态已经不稳；e92 的 free/teacher 都在 3e-6，选中点更可靠。

## 4. C 原始三种子（L4H16D384，连续 hidden 反馈）

### 4.1 选中点与最后 epoch

| seed | 运行 | 选中 | train_epoch | val | 最后 epoch | final train_epoch | final val |
|---|---|---|---:|---:|---:|---:|---:|
| 42 | `oldC0-resume-150ep` | e143 | 0.081 | 0.0662 | e149 | 0.107 | 0.567 |
| 1234 | `oldC0-resume-150ep` | e143 | 0.115 | 0.279 | e149 | 0.106 | 0.333 |
| 2026 | `oldC0-newseed-150ep` | e143 | 0.0698 | 0.0485 | e149 | 0.0647 | 0.166 |

选中 checkpoint：

- s42：`tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-14-02:35:10/epoch=143-val_loss=0.0000.ckpt`
- s1234：`tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-1234 - 2026-11-14-02:38:10/epoch=143-val_loss=0.0000.ckpt`
- s2026：`tmp/runs/oldC0-newseed-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-2026 - 2026-11-14-10:12:43/epoch=143-val_loss=0.0000.ckpt`

### 4.2 C s2026 完整曲线

| epoch | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 99 | 143 | 149 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_epoch | 7.62 | 1.22 | 1.20 | 1.01 | 0.997 | 0.242 | 0.153 | 0.0949 | 0.0922 | 0.0717 | 0.0779 | 0.0698 | 0.0647 |
| val | 5.42 | 5.76 | 4.06 | 5.93 | 3.82 | 1.79 | 1.54 | 0.774 | 0.134 | 0.172 | 0.336 | 0.0485 | 0.166 |

C s42/s1234 的日志在 resume 过程中有较多缺行；这里只列选中/末点。它们的完整曲线应以各自 `logs/seed*.log` 为准。

观察：

- C 三种子都选 e143，不选最后的 e149；e149 的 val 分别为 0.567/0.333/0.166，均高于 e143。
- s2026 e143 的 val 最低（0.0485），s42 0.0662，s1234 0.279。但 **C 的 loss 不能与 A/B 横比**；也不能用 loss 给 C 的能力排序。这里的 s2026 是**原始 C e143**，不是后来 normal-window 后训练的 s2026（后者 dev 明显退化，且被排除在正式两种子评测外）。
- C s42/s1234 是 150 epoch 续训运行；s2026 是从头 150 epoch。三者的早期曲线不能当作同一起点比较。

## 5. 总结：主权重与“最后 epoch”的关系

1. **所有主权重都不是最后 epoch。** A′ 选 e86/e90/e94（最后 e99），B-M6 选 e92（最后 e99），cyhB 选 e84/e88/e74（最后 e99），C 选 e143（最后 e149）。
2. **最后 epoch 的验证 loss 普遍更差**：A′ s42 1.4e-6 → 5.93e-3；B-M6 3.25e-6 → 0.142；cyhB s1234 2.67e-5 → 1.29e-2；C s42 0.0662 → 0.567。只有 A′ s1234 的最后 e99 仍很低（6.02e-6）。
3. **free vs teacher**：A/B 在选中点的 free/teacher val 几乎一致；到最后 e99，free-running 明显变差（B-M6 0.246 vs 2.69e-4；A′ s42 8.81e-3 vs 1.96e-5）。这支持“选较早的稳定 checkpoint”，不要用最后 epoch 评价权重。
4. **不要把 C 的 loss 与 A/B 的 loss 放同一数值轴**；C 的内部反馈/目标不同，loss 只用来判断同一 C 运行是否收敛或末端是否漂移。

## 6. 来源

- A′：`tmp/runs/A-L4D384fast/logs/seed42.log`、`resume_seed42_20261109-154238.log`；`seed1234.log`、`resume_seed1234_20261109-161549.log`；`seed2025.log`、`resume_seed2025_20261109-161551.log`、`resume_seed2025_gpu7_20261109-163028.log`。
- B-M6 本组：`tmp/runs/B-M6/logs/seed42_gpu5_20261107-174732.log`。
- cyhB：`/workspace/lab-xj/users/jinxi/code/cyh/aga-migration-20261108/logs/seed-42.log`、`seed-1234.log`、`seed-2025.log`；权重在 `tmp/cyh_ckpt/seed-*`；迁移说明 `.../README_MIGRATION.md`。
- C：`tmp/runs/oldC0-resume-150ep/logs/seed42.log`、`seed1234.log`；`tmp/runs/oldC0-newseed-150ep/logs/seed2026.log`。
- 提取方式：`tr '\r' '\n' < log | grep 'Epoch N: 100%'`，取该 epoch 最后一个 100% 进度行；A/B 的 free/teacher 分别为 `total_loss_TchrPred_False_RegCorr_False/val`、`total_loss_TchrPred_True_RegCorr_False/val`。

## 7. 未纳入本表

- **A-M5 / A-L4fast**：L2H16D256 / L4H16D256，历史架构，不与 L4H16D384 混比。
- **C0-MixedN**：混合节点数数据、100 epoch、后因欠拟合/词表问题冷存；不是同一训练分布。
- **新 normal-window 后训练 C**（`C_s42` `f14f4bbf…`、`C_s1234` `693b6338…`、`C_s2026` `53ba33c6…`）：不同后训练目标/窗口，不能与原始 A/B/C 的 `total_loss` 直接比较。其 fit-history 另见 `docs/records/2026-11-21/c-local-update-fit-history.md`（末周期加权损失 0.1941 / 0.2906 / 0.5993，末 batch 0.1252 / 0.2111 / 0.5274）。
- **C_s2026 原始 e143** 与 **normal-window C_s2026** 是两个不同权重；上表 C 指前者。
