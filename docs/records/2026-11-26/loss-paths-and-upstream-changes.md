# A/B 与 C 的损失口径差异，以及我们相对原仓库的改动

- 日期：2026-11-26
- 用途：PLAN-009 要在对齐训练量后比较 A′、B、C、D。跨家族比较前，先把三件事写清：各家族走哪条损失路径、两条路径为什么不可跨家族比较、我们相对原仓库改了哪些代码。
- 基准：原论文仓库 `upstream` = `Awni00/algorithmic-generalization-transformer-architectures`，`upstream/main` = `52dd950`（2025-10-14）。分叉点即 `52dd950`，原仓库此后未再更新；我们领先 231 个提交，落后 0。

## 1. 结论

1. **三个家族用的都是原仓库里的损失函数，我们没有改动任何损失定义。**
2. A′、B 走 `compute_batch_loss_fast`，原始 C 走 `compute_batch_loss_nointerm`。真正造成不可比的是三点：**训练损失的聚合口径**（平均 vs 累加）、**回灌内容**（教师真值 vs 自己的隐藏态）、以及 **A/B 的训练与验证用的是不同函数**。
3. 我们对 `experiments/model.py` 的改动集中在**循环上限、第二条验证流、日志频率、forward 调用路径**；对 depth≤6 的训练数据与上游行为等价。

## 2. 各家族走哪条路

由 `train_config` 的三个开关决定（`experiments/model.py` 的 `training_step`）：

| 开关 | A′ | B | 原始 C |
|---|---|---|---|
| `nointerm` | — | — | **true** |
| `use_fast_batch_forward` | **true** | **true** | false |
| `use_teacher_pred_for_next_loop` | **true** | **true** | **false** |
| 实际调用 | `compute_batch_loss_fast` | 同左 | `compute_batch_loss_nointerm` |
| 代码位置 | `model.py:481` | 同左 | `model.py:863` |

分流代码与原仓库一致（`upstream/main` 的 `training_step` 是同一套嵌套判断），我们没有改这段。

## 3. 两条损失路径的差异（经复核）

### 3.1 聚合口径不同（量纲差异，最主要）

- **A/B**：把**所有深度拼成一个 batch 一次前向**（`batch_depthwise_inputs` 重排为 `[n_depths × batch, seq, factors]`）；每个 factor 取 `factor_at_depth_loss.mean()`，即**对「深度 × 样本 × 位置」求平均**，再按 `factor_loss_scales` 加权求和。
- **C**：**逐 loop 展开模型**（`for loop in range(max_depth_data)`），每个 loop 得到一个标量损失，然后 `total_loss += depth_loss` **在 loop 上累加**，不除以轮数。

所以 C 的 `total_loss` 大致是「每轮平均损失 × 轮数」，A/B 的是「跨深度平均」。depth≤6 的图上 loop 最多约 7 轮，**两者相差一个约 7 倍量级的系统性因子**，纯粹由聚合口径造成，与模型好坏无关。

### 3.2 criterion 构造不同——但**不构成数值差异**

```python
self.criterion      = torch.nn.CrossEntropyLoss()                  # C 用；默认 reduction='mean'
self.fast_criterion = torch.nn.CrossEntropyLoss(reduction='none')  # A/B 用；随后自己 .mean()
```

`reduction='none'` 之后立刻 `.mean()`，与默认 `mean` **在数学上完全等价**。它存在的目的只是为了先把损失 reshape 成 `[n_depths, batch, seq]` 以便分层记录，以及支撑下面那条并未启用的 focus 分支。**这一条不改变损失值。**

### 3.3 VALUE 项的 focus 加权——**实际未启用**

`compute_batch_loss_fast` 里确实有一段对 VALUE 的 focus 加权：

```python
if factor == VALUE and use_pos_focus:
    focus_ratio = value_to_compute_pos.float().mean()
    factor_loss_with_focus = factor_at_depth_loss[value_to_compute_pos].mean() \
                           + (factor_at_depth_loss[~value_to_compute_pos] * focus_ratio).mean()
```

但 `use_pos_focus` 的默认值是 `False`，而全仓库**唯一**的调用点（`model.py:342`）并没有传这个参数：

```python
total_train_loss = self.compute_batch_loss_fast(batch, log_prefix='train', log=True, on_step=True, on_epoch=True)
```

因此 A/B 训练时走的是 `else` 分支（普通 `.mean()`），**这条不构成差异**。

### 3.4 训练与验证用的是不同函数（A/B 独有）

| | 训练 | 验证（`total_loss/val`，选检查点用） |
|---|---|---|
| A′/B | `compute_batch_loss_fast`：跨深度平均，一次前向 | `compute_batch_loss`（`model.py:646`）：**逐 loop 累加**，且每次验证把 `use_teacher_pred_for_next_loop` **随机取 50/50** |
| C | `compute_batch_loss_nointerm` | 同一个函数，`max_loop=max_val_loop` |

也就是说 A/B 的监控曲线 `total_loss/val` 每一步都在「教师强制」与「自由运行」两种模式之间随机跳，而 C 的监控曲线来自与训练同一个函数。**这条对"用各自曲线选检查点"影响很大。**

### 3.5 学习目标本身不同（不只是尺度）

A/B 是 `use_teacher_pred_for_next_loop: true`，下一轮输入的是**教师的中间状态**；C 是 `false`，输入的是**模型自己上一轮的预测**。也就是说 C 是在自己可能已经错的中间状态上被监督。这改变的是**问题难度和目标本身**，不只是数值尺度。

## 4. 真实图上的中间过程（可复现）

[loss-paths-demo.py](loss-paths-demo.py) 在仓库根目录执行即可复现本节（只读，需要本地 `tmp/runs` 下的权重）。真图取自 `experiments/data/Tr32Test128-ADD`，C 的权重为 e143。

### 4.1 同一张真图

32 个结果位置，depth 0..5。`depth=-1` 的位置是题面里的字面量（例如 `18 = x_104`）；`depth>=0` 的是结果位置，输入为空、标签是真值。

### 4.2 A/B：第 k 轮**输入**里的 VALUE 已经是真值

```
   node  dep   真值 |   k0   k1   k2   k3   k4   k5
  x_104    0   18 |   18   18   18   18   18   18
   x_19    0   10 |   10   10   10   10   10   10
   x_75    0   20 |   20   20   20   20   20   20
   x_51    0    0 |    0    0    0    0    0    0
  x_100    0   11 |   11   11   11   11   11   11
   x_80    0    6 |    6    6    6    6    6    6
   x_15    1   18 |   空   18   18   18   18   18
   x_10    1   18 |   空   18   18   18   18   18
   x_70    1   10 |   空   10   10   10   10   10
   x_27    1    7 |   空    7    7    7    7    7
  x_123    1    0 |   空    0    0    0    0    0
   x_50    1    0 |   空    0    0    0    0    0
   x_16    1   11 |   空   11   11   11   11   11
  x_118    1    3 |   空    3    3    3    3    3
   x_72    2   18 |   空   空   18   18   18   18
  x_110    2   18 |   空   空   18   18   18   18
   x_92    2   18 |   空   空   18   18   18   18
   x_26    2   16 |   空   空   16   16   16   16
   x_98    2    1 |   空   空    1    1    1    1
   x_54    2    7 |   空   空    7    7    7    7
   x_68    2    1 |   空   空    1    1    1    1
   x_14    2   21 |   空   空   21   21   21   21
  x_101    3   12 |   空   空   空   12   12   12
   x_95    3   12 |   空   空   空   12   12   12
   x_87    3   14 |   空   空   空   14   14   14
   x_82    3    7 |   空   空   空    7    7    7
   x_81    3    8 |   空   空   空    8    8    8
   x_20    4   15 |   空   空   空   空   15   15
   x_31    4   14 |   空   空   空   空   14   14
   x_97    4    8 |   空   空   空   空    8    8
   x_90    5   14 |   空   空   空   空   空   14
   x_77    5    8 |   空   空   空   空   空    8
```

这是 `get_teacher_batch_with_depth` 的直接产物：先算 `value_avail_pos = (depth <= k) & (depth != -1)`，再

```python
batch_ext[..., VALUE][value_avail_pos] = batch_label_ext[..., VALUE][value_avail_pos]
```

也就是说，**每个 depth 的输入都带着真值前缀**；所有 depth 拼成一个 batch 一次前向。`compute_batch_loss_fast` 的 docstring 明说：

> This method always uses the teacher predictions for the next loop for parallelization. No for loop in the forward pass.

### 4.3 C：第 k 轮的输入是它**自己上一轮的隐藏状态**

真实跑 `compute_batch_loss_nointerm` 的循环：

```
  循环前 depth_batch=(1, 256, 384)  ← token_embedder(batch)
  loop 0: 输出 hidden (1, 256, 384)，32 个位置对了 6 个
  loop 1: 输出 hidden (1, 256, 384)，32 个位置对了 14 个
  loop 2: 输出 hidden (1, 256, 384)，32 个位置对了 22 个
  loop 3: 输出 hidden (1, 256, 384)，32 个位置对了 27 个
  loop 4: 输出 hidden (1, 256, 384)，32 个位置对了 30 个
  loop 5: 输出 hidden (1, 256, 384)，32 个位置对了 32 个
```

代码上只有最开始 `depth_batch = self.model.token_embedder(batch)` 用了一次离散输入，之后每轮都是

```python
depth_batch, factor_preds = self.model(depth_batch, True)   # 喂自己的隐藏输出
```

喂回去的是**自己的隐藏输出**，不重新嵌入，也没有任何真值被注入。它每轮自己读出的 VALUE：

```
   node  dep   真值 |   L0   L1   L2   L3   L4   L5
  x_104    0   18 |   18   18   18   18   18   18
   x_19    0   10 |   10   10   10   10   10   10
   x_75    0   20 |   20   20   20   20   20   20
   x_51    0    0 |    0    0    0    0    0    0
  x_100    0   11 |   11   11   11   11   11   11
   x_80    0    6 |    6    6    6    6    6    6
   x_15    1   18 |   空   18   18   18   18   18
   x_10    1   18 |   空   18   18   18   18   18
   x_70    1   10 |   空   10   10   10   10   10
   x_27    1    7 |   空    7    7    7    7    7
  x_123    1    0 |   空    0    0    0    0    0
   x_50    1    0 |   空    0    0    0    0    0
   x_16    1   11 |   空   11   11   11   11   11
  x_118    1    3 |   空    3    3    3    3    3
   x_72    2   18 |   空   空   18   18   18   18
  x_110    2   18 |   空   空   18   18   18   18
   x_92    2   18 |   空   空   18   18   18   18
   x_26    2   16 |   空   空   16   16   16   16
   x_98    2    1 |   空   空    1    1    1    1
   x_54    2    7 |   空   空    7    7    7    7
   x_68    2    1 |   空   空    1    1    1    1
   x_14    2   21 |   空   空   21   21   21   21
  x_101    3   12 |   空   空   空   12   12   12
   x_95    3   12 |   空   空   空   12   12   12
   x_87    3   14 |   空   空   空   14   14   14
   x_82    3    7 |   空   空   空    7    7    7
   x_81    3    8 |   空   空   空    8    8    8
   x_20    4   15 |   空   空   空   空   15   15
   x_31    4   14 |   空   空   空   空   14   14
   x_97    4    8 |   空   空   空   空    8    8
   x_90    5   14 |   空   空   空   空   空   14
   x_77    5    8 |   空   空   空   空   空    8
```

未算到的位置，它自己读出 `EMPTY`。

### 4.4 两张表长得像，来源相反

两张表都是"一层一轮"的阶梯，但阶梯的位置不同：

| | 阶梯出现在 | 来源 |
|---|---|---|
| A/B | **输入** | 真值被注入（教师强制） |
| C | **输出** | 模型自己的预测，被回灌 |

这正对应两家族各自的名字：A/B 是**离散字段回灌**（每轮把离散字段重新嵌入），C 是**连续 hidden 回灌**（直接回灌隐藏向量，不经过离散化）。

## 5. 我们相对原仓库改了哪些代码

### 5.1 总量

- 全仓 `git diff upstream/main HEAD`：4327 个文件，+1375592 / −384 —— 绝大部分是实验产物与记录，**不代表代码改动量**。
- 核心训练代码：`model.py` **+29/−14**、`tokenizers.py` **+40/−1**、`train.py` **+146/−10**。
- 数据生成相关（`utils.py`、`dag_generator.py`、`generate_data.py`）：**无改动**。
- 两个损失函数本身：`compute_batch_loss_fast` **一字未改**；`compute_batch_loss_nointerm` 见 5.2。

### 5.2 `experiments/model.py`（+29/−14）逐处

1. 新增 `max_val_loop_n128`：给第二条验证流（N128）单独的循环上限。
2. `validation_step` 增加 `dataloader_idx`：支持 `val` / `val_n128` 两条流，并传入 `max_loop`。
3. `test_step` 传入 `max_loop=-1`：**不设上限**，避免 depth>36 的测试图被截断。
4. `compute_batch_loss_nointerm` 新增 `max_loop` 参数，并据此计算 `cap` / `last_loop`。
5. `self.model.TF_forward(depth_batch)` → `self.model(depth_batch, True)`：改走已编译 forward，注释声明与 `TF_forward` 逐行同义。
6. 详细指标与 ratio 日志只在最后一轮记录（两处，纯日志）。
7. 删除一句 `print(f"loop_{loop}, full_factor_correct_ratio: ...")` 调试输出。
8. **循环终止条件改写**：原仓库由 `train` 与 `val` 分支各自判断 `max_train_loop` / `max_val_loop`；我们统一为 `if loop == last_loop: break`。

第 8 条是**真正的行为改动**。但对训练图 depth≤6，`max_depth_data ≈ 7 ≤ max_train_loop = 8`，`last_loop` 恰好就是最后一轮，**不触发截断**，因此对 A′/B/C 的训练结果没有影响。它只在 depth>36 的图上会截断——这正是第 3 条给 `test_step` 关掉上限的原因。

### 5.3 `experiments/train.py`（+146/−10）

全部与**命令行、恢复、检查点**有关，与损失定义无关：

- `torch.load` 的 `weights_only` 兼容处理（torch 2.6 默认会阻断 Lightning resume 的 optimizer/hparam 反序列化）
- 新增 `--max_epochs`、`--save_top_k`、`--resume_from_checkpoint`
- `save_top_k` 透传到 `ModelCheckpoint`
- 第二条 n128 检查点流
- 强制保存 `final.ckpt`（`save_last` 在该 Lightning 版本只在上榜检查点写入时更新，末轮可能缺失）
- `checkpoint_dir` 创建

### 5.4 `experiments/tokenizers.py`（+40/−1，纯新增）

新增 `build_factored_tokenizer()` 与 `_cfg_get()`，把 `n_vars` 的取值优先级明确为：显式参数 → `data_config.tokenizer_n_vars` → 旧回退 `max(train.num_nodes, val.num_nodes)`。

`build_factored_tokenizer` 与 `tokenizer_n_vars` 在 `upstream` 中都**不存在**，是我们的新增（唯一被删的那一行是等价重写，diff 行块边界所致）。运行日志里的

```
[tokenizers] tokenizer_n_vars not set; legacy fallback n_vars=128
```

就来自这里——说明实际走的仍是旧回退，取值与上游一致。

## 6. 实践含义

1. **不能跨家族比 loss 绝对值。** 能跨家族比的是同题节点指标：数值正确率、四字段全对率及其实际分母。
2. **同家族内部**看 loss 的收敛、选中点与末端漂移是有效的。
3. **选检查点**：A′/B 与 C 各自按自己家族的 `total_loss/val` 排序，不能用同一条曲线排名；并注意 A/B 的 `total_loss/val` 本身在教师强制与自由运行之间随机跳（§3.4）。
4. 要判断"跨家族训练是否到位"，应使用**同一套评测指标**（PLAN-009 §2 的做法），而不是 loss。
5. 引用"原始实现"时，`upstream/main` 是干净基准；`git diff upstream/main HEAD` 就是"我们改了什么"的全量清单。

## 7. 来源

- 代码与配置：`experiments/model.py`、`experiments/train.py`、`experiments/tokenizers.py`、`experiments/configs/{A-L4D384fast,B-M6,C0-M4}-Nodes32-ADD-L4H16D384-DeBERTa/train_config.yaml`
- 对照基准：`git show upstream/main:experiments/model.py`
- 相关：`docs/records/2026-11-25/seed-training-curves.md`（各家族 loss 绝对值不可横比的既有说明）
- 复现脚本：[loss-paths-demo.py](loss-paths-demo.py)（第 4 节的全部输出由它产生；只读，需本地 `tmp/runs` 权重）
