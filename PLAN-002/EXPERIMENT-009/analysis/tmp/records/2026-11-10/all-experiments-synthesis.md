# 全部实验数据整理与解读（2026-11-10 10:35）

> 数据来源：`tmp/analysis/inject-v2/`（teacher/own 注错）、`Aprime-full/`（A′ 评测）、
> `C0-full/`、`C0-heldout-eval/`、`C0-v2-eval/`（C0 评测）、`c0-trajectories/`、
> `c0-allfactor/`、`c0-decomposition/`（C0 诊断）。
> 设计协议：`next-phase-plan.md`；问题清单：`question-experiment-checklist.md`。
>
> **更正（2026-11-13）**：§1.2 的 teacher-state 表与 §0 中引用它的
> “B 四份权重 u 恢复 97.8–100%” 已被 v2 teacher-state 轮次索引修复取代；
> §1.3 的 own-state 表不受影响。详见 `2026-11-13/code-review-fixes-20261113.md`。

## 0. 一页结论

1. **离散线：纠错训练的行为效果非常稳。** A′ 三 seed 在 teacher/own、N=32/128、K=4/K=0
   下 u 恢复都是 0%；B 四份权重 u 恢复 97.8–100%，q 在 r1–r4 逐步恢复。
   own-state 下 A′ 1024/1024 个 u 符合条件，结果与 teacher 完全一致。
2. **但机制未定**：B 的恢复可能是“每轮从父节点重算”，也可能是“检测输入错误并改回”。
   单点注错区分不了，需要 parent-corruption 2×2。
3. **训练时机也未定**：本地 `last` 掩码腐蚀的是“新可算位置（输入 EMPTY）”，
   不是“已算好 4→5”；我们的注入是后者。B 仍 100% 恢复，说明跨时机泛化，
   但必须先核对服务器训练版本，并做 `EMPTY→5` vs `4→5` 对照。
4. **A′ 本身很强**：clean fig4 0.9995–1.0、raw fig4 0.9954–0.9998、
   clean deep_v2 0.9993–1.0、raw deep_v2 0.9609–0.9983。
5. **连续 C0 是三 seed 三种故障**：
   - s42：整图失分**全是 VARIABLE 读出漂移**；VALUE-RHS 最后为 1.0；0/30 下游按错变量算；
   - s1234：**真正的 VALUE 错误**（146 个算对后变错、57 个下游语义一致），适合 C+ 数值纠错；
   - s2025：坏 seed（16 档均值 0.0009），排除。
6. **C0 在 OOD 上整体很差**：新留出 N32 final full s42 0.2715 / s1234 0.2949、
   N128 三者 0；deep_v2 四个池 s42/s1234 全 0；而 A′ clean deep_v2 ~1.0。
7. **C+ 不能一刀切**：s1234 → 数值纠错；s42 → 变量身份/读出稳定性；s2025 → 排除。

---

## 1. 离散线

### 1.1 问题与设计

- 问题：纠错训练（recorrection）是否让模型在推理时真正“改回”被注错的中间值，
  并把下游一起恢复？
- 设计：在一个已算对的节点 u（VALUE=a）上注入 a→(a+1) mod 23（主协议；另有 4→5 衔接），
  从同一 clean 状态或模型自己的状态起跑，自由续算，观察 u 和 u 的所有直接子节点 q。
- 对照：A′（无纠错，3 seed）vs B（有纠错，4 份权重）；clean 对照；teacher/own 两种起跑。
- 计分：r1/r2/r4；q 在 dq 之前是 EMPTY，不算错；按图聚合。

### 1.2 Teacher-state 结果（新留出图 seed42）

| 模型 | 规模 | K | u 恢复 | q r1 | q r2 | q r4 |
|---|---|---:|---:|---|---|---|
| A′ 三 seed | N32 | 4 | **0%** | expected_wrong 1537 | 1772+91 | 1956+197 |
| B s42 | N32 | 4 | **100%** | correct 1537 | 1863 | 2153 |
| cyh B s42 (e84) | N32 | 4 | 97.8% | correct 1470, expected 67 | 1783/75 | 2066/80 |
| cyh B s1234 (e88) | N32 | 4 | 99.8% | correct 1531, expected 6 | 1856/6 | 2143/8 |
| cyh B s2025 (e74) | N32 | 4 | 100% | correct 1537 | 1863 | 2153 |
| A′ 三 seed | N128 | 4 | **0%** | expected 1567 | 1846+58 | 2274+155 |
| B s42 | N128 | 4 | **100%** | correct 1567 | 1904 | 2429 |
| cyh B s1234 | N128 | 4 | 99.9% | correct 1564, expected 3 | 1900/4 | 2425/4 |
| cyh B s2025 | N128 | 4 | 100% | correct 1567 | 1900 | 2403 |
| A′ s42 | N32 | 0 | **0%** | expected 5767 | 6546+350 | 7159+662 |
| B s42 | N32 | 0 | **100%** | correct 5767 | 6896 | 7821 |
| cyh B s1234 | N32 | 0 | 99.87% | correct 5749, expected 18 | 6873/21 | 7792/26 |

### 1.3 Own-state 结果（同图/同节点/同一绝对轮次；只算两边 u 都算对）

| 模型 | 规模 | K | eligible u | u 恢复 | q r1 | q r4 |
|---|---|---:|---:|---:|---|---|
| A′ 三 seed | N32 | 4 | 1024/1024 | **0%** | expected 1537 | 1956+197 |
| B s42 | N32 | 4 | 1024/1024 | **100%** | no_value 610 + expected 927 | correct 2121, no_value 29 |
| cyh B s42 | N32 | 4 | 1024/1024 | 97.8% | expected 1450 + no_value 87 | correct 2039, expected 94 |
| cyh B s1234 | N32 | 4 | 1024/1024 | 99.8% | expected 1242 + no_value 295 | correct 2109, expected 18 |
| A′ 三 seed | N128 | 4 | 1024/1024 | **0%** | expected 1567 | 2274+155 |
| B s42 | N128 | 4 | 1024/1024 | **100%** | no_value 539 + expected 1028 | correct 2306, no_value 123 |
| cyh B s1234 | N128 | 4 | 1024/1024 | 99.9% | expected 1354 + no_value 213 | correct 2298, expected 11 |
| 4→5 own：A′ s42 | N32/128 | 4 | 41/53 | 0% | expected 62/79 | 76/110 |
| 4→5 own：B s42 | N32/128 | 4 | 41/53 | 100% | no_value 35/48 + expected 27/31 | correct 82/116 |

### 1.4 机制与时机（未决）

- **recompute vs recorrect**：只改 u 时两者都表现为“u 回到 a”。需要 parent-corruption
  2×2（只改父 r / 只改 u / 两者），看 u 是否跟着错父值走；同时看 r 自己的轨迹。
- **时机**：本地 `last` 掩码 `~avail[i-1] & avail[i]` 选的是 depth==i 的新可算位置
  （输入 EMPTY），不是已算好的 4→5。我们的注入是 4→5；B 仍 100% 恢复，
  但需要核对 B-42 服务器版本，并做 `EMPTY→5` vs `4→5` 对照。
- **q 的类型**：v2 的 q 是所有直接子节点，r1 的 correct/expected 可能被 identity 类
  子节点主导；细粒度 4→5 实验里同轮 q 从不在同一 forward 算对（要么 EMPTY 推迟，
  要么先错后改）。后续要按 q 的运算类型拆开。

### 1.5 解读

- B 的优势不是单 seed、不是 teacher-state、不是某个规模/配对数的 artifact；
  四份权重、两种起跑、N32/128、K4/K0 都成立。
- cyh B s42（e84）97.8% 说明 epoch/checkpoint 有一点影响，但不是质变；
  主 B-42（e92）和 cyh B s2025（e74）都是 100%。
- 现在只能说“行为上会恢复”，不能说“检测到错误并修复”；机制实验是下一道关。

---

## 2. A′ 最终评测

| seed | clean fig4 | raw fig4 | clean deep_v2 | raw deep_v2 |
|---|---:|---:|---:|---:|
| s42 | 0.9995 | 0.9954 | 0.9993 | 0.9866 |
| s1234 | 1.0000 | 0.9963 | 1.0000 | 0.9609 |
| s2025 | 1.0000 | 0.9998 | 1.0000 | 0.9983 |

A′ 是很强的离散模型；注错实验的 0% 恢复不是“不会算”，而是“会复制”。

---

## 3. 连续线（C0 / nointerm）

### 3.1 VALUE 轨迹（256 图，8192 个节点）

| seed | 始终正确 | 首次错后对 | 首次错且不对 | 算对后变错 | 持续 | 下游语义一致 | 多错 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s42 | 8165 | 22 | 0 | 5 | 5 | 2 | 0 |
| s1234 | 7460 | 210 | 376 | 146 | 121 | 57 | 20 |
| s2025 | — | — | — | 396 | 88 | 37 | 170 |

### 3.2 整图失分分解

| seed | 前四因子峰值 | 最后 VALUE-RHS | 掉出图 | 模式 | 最后 mismatches |
|---|---:|---:|---:|---|---|
| s42 | 0.906 (r8–9) | **1.0** | 122 | **全 VARIABLE-only** | VALUE 5 / VARIABLE 366 |
| s1234 | 0.488 (r8–9) | 0.55 | 52 | 32 VARIABLE-only / 12 VALUE-only / 8 multiple | VALUE 507 / VARIABLE 139 |
| s2025 | 0 | 0 | 0 | — | VALUE 5930 / VARIABLE 6 |

- s42 的 184 次 VARIABLE 漂移都在定义位置；30 个有直接子节点，但 **0/30**
  下游按“错变量”算出对应值 → **读出漂移，不是实际计算**。
- s1234 的 VARIABLE 漂移 64 次（12 个有子节点，1 个语义一致，8.3%）；
  VALUE 错误才是主要问题。

### 3.3 评测 / OOD

- 16 档 final full 均值：s42 0.2079、s1234 0.1814、s2025 0.0009。
- 新留出 N32：s42 0.2715、s1234 0.2949、s2025 0.0；N128：三者全 0。
- deep_v2 四池：s42/s1234 全 0。
- 对照 A′ clean deep_v2 ~1.0、raw 0.96–1.0。

### 3.4 解读与 C+ 决策

- **s42**：严格整图指标崩，但算术值其实一直对；丢的是变量身份/读出。
  没有现成的“数值纠错”训练目标；应做变量身份/读数稳定性，或换指标。
- **s1234**：真正的 VALUE 错误，且 57 个有下游语义；是 C+ 数值纠错的最佳来源。
  但它的 clean 整图准确率本身只有 ~0.18–0.29，C+ 要从较弱基座出发。
- **s2025**：坏 seed，排除。
- 自然向量替换（候选 1）还没做；logit-steering 已失败（47/47 下一轮回 4，q 从未 14）。

---

## 4. 总体解读

1. **离散中间状态是 OOD 泛化的关键**：A′ 离散模型 clean 在 N=8…128、deep_v2 都接近 100%；
   C0 连续模型在 N≥40、deep_v2 彻底失败。
2. **纠错训练的行为效果成立**，且跨 seed/权重/规模/起跑状态；但“自纠错”机制
   （检测改回 vs 每轮重算）尚未证明。
3. **连续模型的失败不是单一现象**：
   - s42 的失败是“变量名读出漂移”，算术值仍对；
   - s1234 的失败是真正的 VALUE 错误；
   - s2025 是训练/seed 问题。
   这要求 C+ 按 seed 选择目标，不能混训。
4. **raw/clean 与 aux 回灌仍然不能作为纠错能力的证据**；注错实验才是直接证据。

## 5. 仍缺的实验

- parent-corruption 2×2（recompute vs recorrect）；
- `EMPTY→5` vs `4→5` 时机对照 + B-42 服务器版本核对；
- q 按运算类型拆分（identity vs 算术子节点）；
- own-state K=0 全量（已停，待单线程重跑）；
- s1234 的 C+ 数值纠错续训（等方向确认）；
- 候选 1 自然向量替换。

## 6. 数据路径

- 离散注错：`tmp/analysis/inject-v2/*newN*`、`*ownN*`
- A′ 评测：`tmp/analysis/Aprime-full/`
- C0 评测：`tmp/analysis/C0-full/`、`C0-heldout-eval/`、`C0-v2-eval/`
- C0 诊断：`tmp/analysis/c0-trajectories/`、`c0-allfactor/`、`c0-decomposition/`
- 设计与问题清单：`tmp/records/2026-11-10/next-phase-plan.md`、
  `question-experiment-checklist.md`
