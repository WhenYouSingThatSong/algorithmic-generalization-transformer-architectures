# Parent-corruption 2×2 结果（2026-11-10）

> **更正（2026-11-13）**：本文件的 parent 2×2 旧数字全部作废——脚本把
> `st[Rg]/st[Md]` 当列索引用，实际覆盖了 token 4/5 的整行，且未清零其余 token
> 的 aux。修复与重跑见 `2026-11-13/code-review-fixes-20261113.md`；
> 新结果在 `tmp/analysis/parent-2x2-fixed/`。旧表仅保留作历史记录，**不得引用**。

> 目的：区分 B-M6 的“恢复”是 **recompute（从父节点重算）** 还是
> **recorrect（检测到 u 错，改回已存值）**。
> 协议：新留出 seed42 N32，256 图，K=4；teacher-state；
> A′ 为同架构、关闭 recorrection 的对照。
> 脚本：`tmp/scripts/parent_corruption_2x2.py`
> 原始结果：`tmp/analysis/parent-2x2/D2x2_*.json`

## 1. 设计

对每个内部节点 u（有直接子节点、label 为数字）：

- 真值 u=a；注入值 u=a+1；
- 选一个直接父节点 r（depth < d，label 为数字），改成 `r_true + 3 (mod 23)`；
- 用错 r 重算 u，得到第三个值 `a_recomp = a+3 (mod 23)`；
- 四种条件：

| 条件 | u 写入 | r 写入 |
|---|---|---|
| clean | a（或 EMPTY） | 原值 |
| U only | a+1 | 原值 |
| **R only** | **a（u 本身是对的）** | **错值** |
| U+R | a+1 | 错值 |

关键判读：

- **R only 下 u 变成 a_recomp** → 模型用了被改过的父值 → **recompute**；
- **R only 下 u 保持 a** → 模型保留/信任已存 u → **不重算/recorrect**。

## 2. after_u（u 已经算好，a→a+1）

### 2.1 u 的状态（post 1 = 注错后第一个 forward）

| 条件 | B-M6 s42 | A′-42 |
|---|---|---|
| clean | true 1024/1024 | true 1024/1024 |
| U only | true 1023/1024, other 1 | **inj 1024/1024** |
| **R only** | **recomp 935/1024 (91.3%)**, no_value 87, true 1, other 1 | **true 1024/1024** |
| U+R | **recomp 899/1024 (87.8%)**, no_value 124, other 1 | **inj 1024/1024** |

结论：

- **B-M6 在 R only 下，把本来正确的 u 改成了“按错父值算出的值”**；
- A′ 在 R only 下完全不动 u；
- 所以 B 不是“检测到 u 错再改回”，而是**父节点优先、重新计算**。

### 2.2 r 自己的状态（post 1）

| 条件 | B-M6 r | A′ r |
|---|---|---|
| clean | true 966, no_value 56, other 2 | true 1019, other 5 |
| R only | true 909, no_value 59, wrong 51, other 5 | **wrong 969, true 51, other 4** |
| U+R | true 909, no_value 59, wrong 51, other 5 | **wrong 969, true 51, other 4** |

B 在同一个 forward 里通常也把 r 改回 true；但 u 这一轮仍按**输入的错 r**算成 a_recomp，
下一轮才跟着已修正的 r 回到 a。  
A′ 不修 r，r 一直是 wrong；u 也一直不动。

### 2.3 后续轮次

B-M6 R only：

| post | true | recomp | other | no_value |
|---|---:|---:|---:|---:|
| 1 | 1 | **935** | 1 | 87 |
| 2 | 847 | 106 | 64 | 4 |
| 4 | 799 | 94 | 118 | 5 |

A′ R only：post 1/2/4 都是 true 1024。

这说明 B 的“父节点优先”不是永久错：
先按错父值算，等父节点自身修正后，下一轮再回到真值。
但**第一轮确实会跟着错父值走**。

## 3. before_u（u 还没算，EMPTY→a+1）

### 3.1 u 的状态（post 1 = 本应计算 u 的那一轮）

| 条件 | B-M6 s42 | A′-42 |
|---|---|---|
| clean | true 1023, other 1 | true 970, other 32, no_value 16, inj 3, recomp 3 |
| U only | true 1023, other 1 | **inj 1024/1024** |
| R only | no_value 632, recomp 391, other 1 | **recomp 971/1024** |
| U+R | recomp 940, no_value 83, other 1 | **inj 1024/1024** |

解读：

- **A′ 在 u 已有注入值时直接复制注入值**（U only / U+R 都是 inj），
  只有在 u 是 EMPTY 时才按父节点算（R only → recomp 971）。
- **B 即使 u 已经有注入值，也会按父节点重算**（U only → true 1023；
  U+R → recomp 940）。
- B 在 R only 下出现大量 no_value（632）：父节点错时，它有时会推迟一轮再算 u；
  到 post 2 大部分回到 true（842），post 4 为 799 true / 122 other。

## 3.5 own-state replication（2026-11-10 14:58）

从模型自己自由续算到 u 的状态起跑（只保留模型自己 u 已算对/EMPTY、且父节点 r 已算对的样本），
256 图、K=4，四种条件完全相同。结果与 teacher-state **逐项相同**：

| 条件（after_u） | B-M6 own | A′ own |
|---|---|---|
| clean | u=true 1024/1024 | u=true 1024/1024 |
| U only | u=true 1023/1024 | u=inj 1024/1024 |
| R only | **u=recomp 935/1024** | u=true 1024/1024 |
| U+R | **u=recomp 899/1024** | u=inj 1024/1024 |

before_u 也相同：

- B：U only → true 1023；R only → no_value 632 + recomp 391；U+R → recomp 940。
- A′：U only → inj 1024；R only → recomp 971；U+R → inj 1024。

为什么完全相同：在符合 own-state 条件的 u 上，模型自己的 clean 轨迹已经和 teacher 状态一致
（clean 准确率接近 100%），所以两种起跑状态在数值上相同。  
这至少说明：**“父节点优先重算”不是 teacher 状态人工造出来的 artifact**；
在模型自己的状态上，行为一样。

## 3.6 条件分析：父节点没修好时，u 会怎样？

只看 R only，按 **r 自己的状态** 分组（after_u，post 1/2/4）：

| 模型 | post | r 状态 | n | u 状态 |
|---|---:|---|---:|---|
| B-M6 s42 | 1 | true | 909 | recomp 820, no_value 87, other 1, true 1 |
| B-M6 s42 | 1 | **wrong** | **51** | **recomp 51 / 51** |
| B-M6 s42 | 2 | **wrong** | **37** | **recomp 35**, no_value 1, true 1 |
| B-M6 s42 | 4 | **wrong** | **40** | **recomp 39**, true 1 |
| cyhB s1234 | 1 | **wrong** | 26 | **recomp 26 / 26** |
| cyhB s1234 | 4 | **wrong** | 15 | recomp 9, other 4, true 1, no_value 1 |
| cyhB s2025 | 1 | **wrong** | 7 | **recomp 6**, other 1 |
| cyhB s2025 | 4 | **wrong** | 17 | recomp 8, other 7, true 2 |
| cyhB s42 | 1 | **wrong** | 52 | recomp 35, other 13, true 3, no_value 1 |
| cyhB s42 | 4 | **wrong** | 54 | recomp 41, other 10, true 3 |

结论：

- **父节点 r 没修好时，u 基本一直跟着错**（B-M6 s42：51/51、35/37、39/40）；
- 父节点 r 修好后，u 才在后续轮次回到 true；
- A′ 则始终 `r=wrong, u=true`（完全忽略父节点）。

这把“B 的恢复”进一步钉在 **父节点状态驱动** 上：
**u 能不能恢复，取决于 r 能不能恢复；B 没有独立于父节点的错误检测。**

## 4. 机制结论

1. **B-M6 的行为是 recompute，不是 recorrect。**
   决定性证据是 R only：u 本身正确，父节点错，B 仍把 u 改成按错父值算出的值。
2. **A′ 的行为是 copy/freeze。**
   u 一旦有值就保留；父节点错也不影响 u。只有在 u 还是 EMPTY 时才会计算。
3. 所以 recorrection 训练学到的更准确描述是：
   **“VALUE 输入不可信，以父节点为准重算”**，
   而不是“检测哪个 VALUE 错了再修复”。
4. 代价：**父节点错时，B 会把错传下去**（至少第一轮）。
   它不是通用纠错器，而是“信任分配/上游重算”策略。

## 5. 边界

- 目前只有 **teacher-state、N32、K4、B-M6 s42 + A′ s42**；
- B 的 R only 是 91.3%，不是 100%；8.5% 是 no_value（延迟），不是“保持原值”；
- 还没有 cyhB 其他权重、own-state、多节点/多层污染；
- “每个节点每一轮都无条件重算”没有被直接证明；
  证明的是“冲突时父节点优先”。clean 状态下重算与复制结果相同，行为上不可区分。

## 6. 下一步

- **own-state 2×2**：从模型自己自由续算到 u 算对的状态起跑，重复四格；
- **cyhB 权重 replication**：s42/s1234/s2025 至少各一份；
- **多节点/多层污染**：父节点也错时，B 能否在后续轮次真正恢复；
- 把 R only 的“第一轮跟错父值”写成正式指标：
  `recomp_rate`、`follow_parent_rate`、`keep_stored_rate`。