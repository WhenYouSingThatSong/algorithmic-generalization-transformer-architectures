# 代码审查修复：parent 2×2 aux 索引 + v2 teacher-state 轮次索引（2026-11-13）

状态：**代码已修复、smoke 与全量重跑完成（2026-11-13 22:30）**。
本记录是修改记录（findings → fix → affected results → rerun），
原始结论不删除，但在受影响记录中加“已被本修复取代”的更正标记。

## 1. Findings

### 1.1 `parent_corruption_2x2.py`：`st[Rg]` / `st[Md]` 是行索引，不是列索引

修复前（`tmp/scripts/parent_corruption_2x2.py:271`）：

```python
st[Rg] = r_false; st[Md] = m_false
```

`st` 的形状是 `[seq_len, 6]`（6 个 factor），`Rg=4`、`Md=5` 是 **factor 列号**。
`st[Rg]` / `st[Md]` 实际选中第 4、5 个 **token 行**，把这两行的全部 6 个 factor
写成 `r_false`/`m_false`（0），而不是把所有 token 的 REGRET/MODIFICATION 列清零。

后果：

- 所有条件（包括 clean 对照）的起跑状态中，token 位置 4/5 被整体破坏；
- 选中的 u/r 若在位置 4/5，或任何依赖这两个位置的 q，读数都被污染；
- 其余 token 的 REGRET/MODIFICATION 没有被显式清零，旧 rollout 继承了 teacher/
  own 状态里的 aux 值，和“统一置 FALSE”的设计意图不一致。

修复（`:271` 起）：

```python
# st is [seq_len, 6]; Rg/Md are factor columns. Use the
# column slice. st[Rg]/st[Md] would overwrite the whole
# token rows at sequence positions Rg/Md (all 6 factors).
st[:, Rg] = r_false; st[:, Md] = m_false
```

### 1.2 `inject_value_discrete_v2.py`：teacher rollout 用绝对轮次当下标

修复前（`tmp/scripts/inject_value_discrete_v2.py:194-205`）：

```python
for r in [1, 2, 4]:
    loop = u['d'] + r
    if loop >= n_loops:
        continue
    ...
    rec['u_status'] = ('recovered' if us_inj[bi][loop] == a_val else
                       'injected' if us_inj[bi][loop] == a_inj else 'other')
    ...
    qval = qs_inj[bi][loop][j]
```

teacher rollout 从 `S[u.d + 1]` 起跑，`rollout()` 的列表下标 0 就是
**绝对轮次 `d+1`（注错后的第一个 forward）**。旧代码却把绝对轮次 `d+r`
直接当列表下标，实际读到的是：

| post round | 意图轮次 | 旧代码读到的 rollout 下标 | 对应绝对轮次 |
|---:|---:|---:|---:|
| 1 | d+1 | d+1 | 2d+2 |
| 2 | d+2 | d+2 | 2d+3 |
| 4 | d+4 | d+4 | 2d+5 |

也就是说旧 teacher-state 表把**更晚的轮次**标成了 post1/2/4，通常已经越过
B 的纠错窗口，导致 `recovered` / q `correct` 被系统性高估；
深 d 的 post4 还可能因 `loop >= n_loops` 被跳过。

修复（`:193` 起）：

```python
# rollout 从 S[d+1] 开始，列表下标 0 就对应 loop d+1，因此下标是
# r-1；绝对 loop（d+r）只用于记录/可算性判断，不能当下标。
for r in [1, 2, 4]:
    idx = r - 1
    loop = u['d'] + r
    if idx >= n_loops:
        continue
    ...
    us_inj[bi][idx] ...
    qs_inj[bi][idx][j] ...
```

- own-state 路径（`run_own`）的下标本来是 `step`，起点是 `preds_all[d]`，
  与绝对轮次 `d+step` 对齐，**不受本次修复影响**；
- 旧的 47 候选脚本 `inject_value_discrete.py` 从 `t=d+1` 起跑，也是正确的，
  **不受影响**。

## 2. Smoke 验证（B-M6 s42，32 图，k=4，N32 heldout）

产物：

- `tmp/analysis/parent-2x2-fixed/smoke/smoke_fixed_BM6_s42_after.{json,log}`
- `tmp/analysis/inject-v2-fixed/smoke/smoke_fixed_BM6_s42_teacher.{json,log}`

### 2.1 parent after_u（修复后）

| 条件 | post1 u | post1 r |
|---|---|---|
| clean | true 128/128 | true 128/128 |
| U only | true 128/128 | true 128/128 |
| R only | **recomp 117/128**, no_value 11 | true 128/128 |
| U+R | **recomp 115/128**, no_value 13 | true 128/128 |

- clean 条件 u/q 全对（q post1 true 196/196），说明起跑状态不再被 token 4/5
  的整行覆盖破坏；
- R-only recomp 91.4%，与旧结论量级一致，但这是修复后的干净数；
- post2/post4 四种条件 u 全部 true（32 图 smoke 规模下）。

### 2.2 v2 teacher-state（修复后）

| 指标 | 修复后 |
|---|---|
| u post1/2/4 recovered | 1.0 / 1.0 / 1.0 |
| q post1 | no_value 84, **expected_wrong 112**, correct 0 |
| q post2 | correct 222, expected_wrong 1, no_value 13 |
| q post4 | correct 269, expected_wrong 1, no_value 3 |

关键变化：post1 第一次正确读到“注错后第一个 forward”，q 出现 112 条
`expected_wrong`（注错值传播），而不是旧 teacher 表里 post1 几乎全 `correct`。
这符合 B-M6 “第一轮按输入值/父值走，后续轮次修复”的机制预期。

### 2.3 首轮全量对照（A′ s42，256 图）

A′ 没有纠错，u 的 headline 数字新旧相同（after 四条件分别 clean true 1024、
U only inj 1024、R only true 1024、U+R inj 1024；own-state 同），
但 **q 表明显被旧 bug 污染**：

| A′ after, post1 | 新 clean q | 旧 clean q |
|---|---|---|
| n | 1537 | 1537 |
| true | **1537** | 1492 |
| other | **0** | 42 |

- 新 clean 全部匹配 true，旧表有 42 条 `other`（token 4/5 被整行覆盖）；
- v2 teacher-state 的 A′ 结果新旧完全一致（u 永远 injected，
  q 永远 expected_wrong），说明该 bug 对 A′ v2 的比例没有可见影响，
  但仍按统一协议重跑；
- B/cyh 的 post1 需要等新结果，旧 teacher 表读的是更晚轮次，预计差异更大。

## 3. 受影响的实验结果

### 3.1 parent 脚本（全部旧结果作废，需重跑）

目录 `tmp/analysis/parent-2x2/`：

- `D2x2_*_after.json` / `_before.json`（A′ s42、B-M6 s42、cyhB s42/s1234/s2025）
- `D2x2_own_*_{after,before}.json`（A′ s42、B-M6 s42）
- `P1/P1_*_4to5*.json`、`P1/P1_*_EMPTY5*.json`（B-M6 + cyh 三 seed）
- 顶层 `P1_*` 日志同源

依赖这些数的记录/结论：

- `tmp/records/2026-11-10/parent-2x2-finding.md` 的 §2/§3/§3.5/§3.6
  （91.3% R-only recomp、88.6/99.6/98.4、EMPTY→5 等）；
- `tmp/records/2026-11-13/p1-pilot-20261113.md` 的 §2；
- 引用上述数字的 day-summary / full-report / c0-mixedn 总览中的相关段落。

更正方式：保留原文，在文首加“**已被 2026-11-13 修复取代，旧数不可引用**”
横幅，重跑后在新记录中给新表。

### 3.2 v2 脚本（仅 teacher-state 结果作废）

目录 `tmp/analysis/inject-v2/`：

- `Aprime_s42_newN32_k4.json`、`Aprime_s1234_newN32_k4.json`、
  `Aprime_s2025_newN32_k4.json`
- `BM6_s42_newN32_k4.json`
- `cyhB_s42_newN32_k4.json`、`cyhB_s1234_newN32_k4.json`、
  `cyhB_s2025_newN32_k4.json`
- `*_newN32_kall.json`（A′ s42/s1234、BM6 s42、cyhB s1234）
- `*_newN128_k4.json`（A′ s42/s1234/s2025、BM6 s42、cyhB s1234/s2025）
- 各 `*_v2_smoke.json`

**不受影响、无需重跑**：

- `*_ownN32_k4.json`、`*_ownN128_k4.json`、`*_ownN32_45.json`、
  `*_ownN128_45.json`（own-state 路径下标正确）；
- 旧 47 候选 `inject_value_discrete.py` 的全部结果；
- `c0_natural_swap.py` / C4 结果；
- C0 训练与 `eval_nointerm_model.py` 评测。

## 4. 重跑计划

原则：新目录、不覆盖旧 JSON；GPU2/3/5/6/7，避开 C0 训练占用的 GPU0/1/4。

- parent：
  - teacher after `a_plus_1` 256 图 k4：A′ s42、B-M6 s42、cyhB s42/s1234/s2025；
  - teacher before `a_plus_1` 256 图 k4：A′ s42、B-M6 s42；
  - own after/before `a_plus_1` 256 图 k4：A′ s42、B-M6 s42；
  - P1 4→5 与 EMPTY→5，128 图 k4：B-M6 + cyh 三 seed。
    **数据集更正**：旧 P1 用的是 `val_datasets/Tr32Test128/ADD/testdata_N32.pt`，
    因此另跑 `P1_oldset/`（同旧数据集，82 条 u）；首批 `P1/`（heldout-seed42，
    79 条 u）保留作第二数据集复现，不作旧表对比。
- v2 teacher：
  - N32 k4（7 个权重组合）、N32 k=0（4 个）、N128 k4（6 个），256 图。

输出根目录：

- `tmp/analysis/parent-2x2-fixed/`
- `tmp/analysis/inject-v2-fixed/`

队列脚本：`tmp/scripts/rerun_fixed_20261113.sh`；汇总/新旧对照脚本：
`tmp/scripts/summarize_fixed_20261113.py`。

实际启动：**2026-11-13 21:54**，队列 PID 977906–977910（GPU2/3/5/6/7），
队列日志 `tmp/analysis/rerun-fixed-20261113/queue_gpu*.log`，
单任务日志同目录 `gpu<N>_<tag>.log`。

重跑后本记录补 §5：新结果表、与旧数的差异、结论是否变化。

## 5. 重跑结果（进行中，2026-11-13 22:00）

### 5.1 v2 teacher-state：B-M6 s42，N32，k=4，256 图

产物：`tmp/analysis/inject-v2-fixed/BM6_s42_newN32_k4.json`。

| 指标 | 新（修复后） | 旧（错误下标） |
|---|---|---|
| u r1 | recovered 100% | recovered 100% |
| u r2 | recovered 100% | recovered 100% |
| u r4 | recovered 100% | recovered 100% |
| **q r1** | **no_value 610, expected_wrong 927, correct 0** | **correct 1537** |
| q r2 | correct 1733, no_value 97, expected_wrong 33 | correct 1863 |
| q r4 | correct 2121, no_value 29, expected_wrong 3 | correct 2153 |

**这是本次修复最重要的结果变化。** 旧 teacher-state 表声称
“B 的 q 在 r1 直接 correct”；修复后第一次 forward 的 q 没有任何一个 correct：
610 个 q 还没到可算轮次（no_value），927 个按注入值得到 `expected_wrong`。
q 的正确率是 0% → 93.1% → 98.5% 的逐轮恢复轨迹，而不是旧表的
100% → 100% → 100%。

这与 own-state 的旧结果（q r1 = no_value 610 + expected_wrong 927）
完全对齐，说明修复后的 teacher-state 起点与 own-state 一致，
旧 teacher 表实际读到了更晚的轮次。**“纠错发生在轮边界、同一 forward 内
q 看不到 u 的修正”这一机制结论不但保留，而且现在有了正确的直接证据。**

其余权重和其他规模的重跑随后补入。

### 5.2 parent 2×2：B-M6 s42，teacher vs own，after_u，256 图

产物：

- `tmp/analysis/parent-2x2-fixed/D2x2_BM6_s42_N32_K4_ap1_after.json`
- `tmp/analysis/parent-2x2-fixed/D2x2_own_BM6_s42_N32_K4_ap1_after.json`

修复后 teacher 与 own **逐项相同**：

| 指标 | teacher 新 | own 新 | 旧（两者相同） |
|---|---|---|---|
| R-only post1 u recomp | 926/1024 | 926/1024 | 935/1024 |
| R-only post1 u no_value | 97 | 97 | 87 |
| R-only post2 u true | **1024/1024** | **1024/1024** | 847 + 106 recomp + 64 other |
| R-only post4 u true | **1024/1024** | **1024/1024** | 799 + 94 recomp + 118 other |
| clean post4 u true | **1024/1024** | **1024/1024** | 892 + 112 other |
| R-only post4 q true | **1935/2153** | **1935/2153** | 1360/2153 |

解读：

- “R-only 第一轮跟随错父值重算”稳住了（90.4%，旧 91.3%）；
- 旧表 post2/post4 的残留 recomp/other 和 clean 条件的 112 条 other
  是 token 4/5 整行覆盖的 artifact；修复后 B 在父节点回正后 post2
  已 100% 回到 true，post4 q 也恢复到 89.9%；
- 旧 teacher/own 的“逐项相同”是在同一 bug 下得到的，不能作为
  teacher/own 一致性的独立证据；修复后两者仍逐项相同，这次才是真的一致。

### 5.3 v2 teacher-state：cyhB s42（N32）与 s1234（N128）

| 权重/规模 | 指标 | 新（修复后） | 旧 |
|---|---|---|---|
| cyhB s42 N32 | u r1 recovered | 97.75% | 97.75% |
| | q r1 | expected 1450, no_value 87, **correct 0** | correct 1470, expected 67 |
| | q r2 | correct 1688 (90.6%), expected 143, no_value 25 | correct 1783 |
| | q r4 | correct 2039 (94.7%), expected 94, no_value 13 | correct 2066 |
| cyhB s1234 N128 | u r1 recovered | 99.90% | 99.90% |
| | q r1 | expected 1354, no_value 213, **correct 0** | correct 1564, expected 3 |
| | q r2 | correct 1781 (93.5%), expected 45, no_value 77 | correct 1900 |
| | q r4 | correct 2298 (94.6%), expected 11, no_value 119 | correct 2425 |

结论与 B-M6 一致：旧表把“更晚轮次”标成 r1；修复后 r1 没有 correct，
r2/r4 才逐步恢复。cyhB s42 在 r4 仍有 94 条 expected_wrong（4.4%），
比 B-M6 的 3 条差，和它是未训满 checkpoint 一致。

**交叉校验**：修复后的 teacher q r1 与旧 own-state q r1（own 路径本来正确）
逐项一致：

| 权重 | 新 teacher q r1 | 旧 own q r1 |
|---|---|---|
| B-M6 s42 | no_value 610 + expected 927 | no_value 610 + expected 927 |
| cyhB s42 | no_value 87 + expected 1450 | no_value 87 + expected 1450 |
| cyhB s1234 | no_value 295 + expected 1242 | no_value 295 + expected 1242 |

这进一步证明修复后的 teacher-state 起点与 own-state 对齐；旧 teacher 的
“q r1 correct”才是异常。

### 5.4 parent 2×2：cyhB s42，teacher，after_u，256 图

产物：`tmp/analysis/parent-2x2-fixed/D2x2_cyhB_s42_N32_K4_ap1_after.json`。

| 指标 | 新（修复后） | 旧 |
|---|---|---|
| R-only post1 u recomp | **1015/1024（99.1%）** | 907/1024（88.6%） |
| R-only post1 u true/other/no_value | 0 / 1 / 8 | 19 / 77 / 20 |
| clean post1 u true | **1024/1024** | 949 + 70 other |
| R-only post2 u true | **1023/1024** | 771 + 47 recomp + 162 other |
| R-only post4 u true | **1024/1024** | 566 + 49 recomp + 326 other + 73 no_value |
| R-only post4 q true | **1915/2153（89.0%）** | 863/2153（40.1%） |

**旧表把 cyhB s42 低估得最厉害。** 旧结论“cyh s42 明显更散、88.6%”
主要是 token 4/5 整行覆盖 artifact；修复后它是 99.1% 跟随错父值重算，
post2 基本完全恢复，post4 q 89.0%。旧的 cyh 权重排序（s42 弱）
不能再用，等 s1234/s2025 新结果后重排。

### 5.5 parent 2×2：A′ s42，own，before_u，256 图（干净的四格）

产物：`tmp/analysis/parent-2x2-fixed/D2x2_own_Aprime_s42_N32_K4_ap1_before.json`。

| 条件 | post1 u（新） | post1 u（旧） |
|---|---|---|
| clean | **true 1024/1024** | true 970, other 32, no_value 16, inj 3, recomp 3 |
| U only | **inj 1024/1024** | inj 1024/1024 |
| R only | **recomp 1024/1024** | recomp 971, other 33, no_value 15, true 4 |
| U+R | **inj 1024/1024** | inj 1024/1024 |

修复后的 A′ before_u 是教科书式的 copy/freeze：

- u 已有值（U only / U+R）→ 直接复制注入值，完全忽略父节点；
- u 还是 EMPTY（R only）→ 才按父节点算，1024/1024 得到 recomp；
- clean 1024/1024 true，没有任何污染。

旧表的 32–33 条 `other` 和 3–4 条 `true/recomp` 都是 token 4/5 整行覆盖
造成的 artifact；旧“R only 971/1024”低估了 A′ 的重算比例（实际 100%）。

### 5.6 parent 2×2：cyhB s1234，teacher，after_u，256 图

产物：`tmp/analysis/parent-2x2-fixed/D2x2_cyhB_s1234_N32_K4_ap1_after.json`。

| 指标 | 新 | 旧 |
|---|---|---|
| R-only post1 u recomp | **1011/1024（98.7%）** | 1020/1024（99.6%） |
| R-only post1 u no_value | 13 | 4 |
| clean post1 u true | **1024/1024** | 1024/1024 |
| R-only post2 u true | **1024/1024** | 851 + 32 recomp + 110 other + 26 no_value |
| R-only post4 u true | **1024/1024** | 633 + 16 recomp + 332 other + 25 no_value |
| R-only post4 q true | **1926/2153（89.5%）** | 1008/2153（46.8%） |

与 cyhB s42 同一模式：post1 recomp 比例基本不变（98.7% vs 99.6%），
但旧表 post2/post4 的大量 residual 是 artifact；修复后 post2 已 100% true，
q post4 从旧 46.8% 修正到 89.5%。

### 5.7 P1（4→5，旧数据集 Tr32Test128 N32，128 图，82 条 u）

旧 P1 用的是 `experiments/evaluation/val_datasets/Tr32Test128/ADD/testdata_N32.pt`，
不是 `heldout-seed42`。用新脚本在同一数据集 probe，`n_records=984`、
82 条 u、graph1=(pos110,d3)、graph3=(pos98,d2)，与旧文件逐项对上，
因此以下重跑是同一批图、82 条 u 的 apples-to-apples 对比。

| 权重 | R-only post1 recomp | R-only post2 true | R-only post4 true（新 / 旧） |
|---|---|---|---|
| B-M6 s42 | 76/82（旧 76） | **82/82** | **82/82 / 67/82** |
| cyhB s42 | 81/82（旧 74） | **82/82** | **82/82 / 31/82** |
| cyhB s1234 | 81/82（旧 81） | **82/82** | **82/82 / 48/82** |
| cyhB s2025 | 82/82（旧 80） | **82/82** | **82/82 / 54/82** |

- post1 recomp 与旧基本一致（cyh s42 从 74 修到 81，旧值含 2 true/1 inj/3 other）；
- **旧表 post2/post4 的“开始散”是 artifact**；修复后父节点回正，post2
  已 82/82 全部恢复，post4 稳定；
- U-only 条件 cyhB s42 新 80 true + 2 inj（旧 74 true + 6 inj + 2 other），
  是真实的小残留：2/82 注入值没有被覆盖。

同一批实验在 `heldout-seed42` N32 上另有 79 条 u 的第二数据集复现
（`tmp/analysis/parent-2x2-fixed/P1/`），用于检验样本依赖，不与旧表直接比。

#### P1 EMPTY→5（before_u，同一旧数据集，82 条 u）

| 权重 | U-only post1 u=true | UR post1 u=recomp | post2 全部条件 u=true |
|---|---|---|---|
| B-M6 s42 | **82/82** | 78/82 + 4 no_value | **82/82** |
| cyhB s42 | **82/82**（旧 76 + 3 inj + 3 other） | **82/82**（旧 74） | **82/82** |
| cyhB s1234 | **82/82** | 79/82 + 3 no_value | **82/82** |
| cyhB s2025 | **82/82** | **82/82** | **82/82** |

- 父值正确时，注入的 5 全部被覆盖回 true 4（旧 cyh s42 的 76/82 是 artifact）；
- 父值错误时，u 全部按错父重算（B-M6/s1234 少数延迟到下一轮）；
- 父值回正后 post2 全部 82/82 恢复；
- 旧表 post4 的 31–67/82 全部是 token 4/5 覆盖 artifact。

### 5.8 parent 2×2 全量总表（256 图，k=4，a→a+1）

`R1rec/R1nov/R1tru` = R-only post1 的 recomp/no_value/true；
`q4` = R-only post4 q=true / q 总数。

| tag | clean1 true | R1rec | R1nov | R1tru | R2tru | R4tru | q4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A′ teacher after | 1024 | 0 | 0 | 1024 | 1024 | 1024 | 1858/2153 |
| A′ own after | 1024 | 0 | 0 | 1024 | 1024 | 1024 | 1858/2153 |
| A′ teacher before | 1024 | 1024 | 0 | 0 | 0 | 0 | 0/2056 |
| A′ own before | 1024 | 1024 | 0 | 0 | 0 | 0 | 0/2056 |
| B-M6 teacher after | 1024 | 926 | 97 | 1 | 1024 | 1024 | 1935/2153 |
| B-M6 own after | 1024 | 926 | 97 | 1 | 1024 | 1024 | 1935/2153 |
| B-M6 teacher before | 1024 | 305 | 719 | 0 | 1023 | 1024 | 1892/2056 |
| B-M6 own before | 1024 | 305 | 719 | 0 | 1023 | 1024 | 1892/2056 |
| cyhB s42 after | 1024 | 1015 | 8 | 0 | 1023 | 1024 | 1915/2153 |
| cyhB s1234 after | 1024 | 1011 | 13 | 0 | 1024 | 1024 | 1926/2153 |
| cyhB s2025 after | 1024 | 1022 | 0 | 2 | 1024 | 1024 | 1950/2153 |

读法：

- A′ before 的 R2/R4 true=0 不是错误：R-only 下 u 第一次按错父算出 recomp 后，
  A′ 就冻结这个值，后续保持 recomp（q 也跟随 recomp），不会回到 true；
- B-M6/cyh 的 R-only 在 post2 已基本全部 true，post4 全 true；
- B-M6 的 R1nov 97 是“父值错误时延迟一轮再算 u”，不是永久丢失；
- B-M6 的 R1rec 90.4% 与旧 91.3% 基本一致，但旧 post2/post4 的 residual
  在修复后消失；
- cyh s42 的 R1rec 99.1%、s1234 98.7%、s2025 99.8%，旧 cyh s42 的 88.6%
  是 artifact；
- teacher/own 四对（A′/B-M6 × after/before）修复后逐项相同，且 q4 也相同。

### 5.9 v2 teacher-state 全量（N32，256 图）

`q1/q2/q4` = 对应 post 的 q 状态；`corr/no/exp/oth` =
correct / no_value / expected_wrong / other_wrong。

| 权重 | k | u r1 | q1 | q2 | q4 |
|---|---:|---|---|---|---|
| A′ 三 seed | 4 | injected 100% | corr 0/1537, exp 1537 | corr 0, exp 1772, oth 91 | corr 0, exp 1956, oth 197 |
| B-M6 s42 | 4 | recovered 100% | corr 0/1537, no 610, exp 927 | **corr 1733/1863 (93.0%)**, no 97, exp 33 | **corr 2121/2153 (98.5%)**, no 29, exp 3 |
| cyhB s42 | 4 | recovered 97.75% | corr 0/1537, no 87, exp 1450 | corr 1688/1863 (90.6%), no 25, exp 143 | corr 2039/2153 (94.7%), no 13, exp 94 |
| cyhB s1234 | 4 | recovered 99.80% | corr 0/1537, no 295, exp 1242 | corr 1709/1863 (91.7%), no 89, exp 63 | corr 2109/2153 (97.9%), no 23, exp 18 |
| cyhB s2025 | 4 | recovered 100% | corr 0/1537, no 938, exp 599 | corr 1680/1863 (90.2%), no 170, exp 13 | corr 2113/2153 (98.1%), no 40 |
| A′ s42 | 0 | injected 100% | corr 0/5767, exp 5767 | corr 0, exp 6546, oth 350 | corr 0, exp 7159, oth 662 |
| B-M6 s42 | 0 | recovered 100% | corr 0/5767, no 2119, exp 3648 | corr 6456/6896 (93.6%), no 304, exp 135 | corr 7727/7821 (98.8%), no 85, exp 9 |

关键点：

- **修复后 r1 没有任何一个 q 是 correct**：q 要么还没到可算轮次（no_value），
  要么按注入值得到 expected_wrong；
- B/cyh 的 q 在 r2 恢复 90–94%，r4 恢复 95–99%；
- 旧表 r1 的 correct 1537/5767 是读到了更晚轮次，不能再用；
- cyhB s42 的 r4 仍有 94 条 expected_wrong（4.4%），是四个权重里最差的；
- A′ 的 u 永远 injected、q 永远 expected_wrong/other_wrong，符合 copy/freeze。

N128 的 A′/cyh 结果同向（q r1 全部 expected_wrong/no_value，r2/r4 逐步恢复），
见下节。

### 5.10 v2 teacher-state：N128 与 k=0 收尾

N128，k=4，256 图：

| 权重 | u r1 | q1 | q2 | q4 |
|---|---|---|---|---|
| A′ s42/s1234/s2025 | injected 100% | corr 0/1567, exp 1567 | corr 0, exp 1846, oth 58 | corr 0, exp 2274, oth 155 |
| B-M6 s42 | recovered 100% | corr 0/1567, no 539, exp 1028 | corr 1770/1904 (93.0%), no 115, exp 19 | corr 2306/2429 (94.9%), no 123 |
| cyhB s1234 | recovered 99.90% | corr 0/1567, no 213, exp 1354 | corr 1781/1904 (93.5%), no 77, exp 45 | corr 2298/2429 (94.6%), no 119, exp 11 |
| cyhB s2025 | recovered 100% | corr 0/1567, no 843, exp 724 | corr 1595/1904 (83.8%), no 299, exp 10 | corr 2135/2429 (87.9%), no 294 |

N32，k=0（全量 u），256 图：

| 权重 | u r1 | q1 | q2 | q4 |
|---|---|---|---|---|
| A′ s42/s1234 | injected 100% | corr 0/5767, exp 5767 | corr 0, exp 6546, oth 350 | corr 0, exp 7159, oth 662 |
| B-M6 s42 | recovered 100% | corr 0/5767, no 2119, exp 3648 | corr 6456/6896 (93.6%), no 304, exp 135 | corr 7727/7821 (98.8%), no 85, exp 9 |
| cyhB s1234 | recovered 99.87% | corr 0/5767, no 920, exp 4847 | corr 6392/6896 (92.7%), no 238, exp 259 | corr 7680/7821 (98.2%), no 75, exp 61 |

共同规律：

- 任何权重、任何规模/k，**q r1 的 correct 都是 0**；
- A′ 的 q 始终停在 expected_wrong/other_wrong；
- B/cyh 的 q 在 r2 恢复 84–94%，r4 恢复 88–99%；
- cyhB s2025 N128 的 r4 只有 87.9%，主要是 294 个 q 仍 no_value（未算到），
  不是 expected_wrong；
- 旧 N128/k0 表把 r1 写成大量 correct，同样是轮次下标 artifact。

## 6. 结论更新

### 6.1 修复本身的多重一致性检验

- 所有 parent run 的 clean 条件：**1024/1024 true**；旧表有 70–112 条 other。
  clean 是四格里唯一完全不该改变的条件，这是最强的 sanity check。
- A′ before_u 四格：clean 1024 true、U-only 1024 inj、R-only 1024 recomp、
  U+R 1024 inj；旧表有 32–33 条 other，且 R-only 只有 971/1024 recomp。
- teacher/own 四对（A′/B-M6 × after/before）修复后逐项相同，q post4 也相同；
  旧表 teacher/own 的“相同”是在同一个 bug 下得到的，不构成独立证据。
- v2 teacher q r1 与**旧 own-state q r1**（own 路径本来正确）逐项一致：
  B-M6 610+927、cyh s42 87+1450、cyh s1234 295+1242；
  这是修复后 teacher 起点正确的最强外部锚点。
- P1 在旧数据集（Tr32Test128 N32）上 probe 得到 `n_records=984`、82 条 u、
  与旧文件 graph1/graph3 的 u 位置完全一致；修复后 post1 recomp 仍为
  76/82（B-M6）、81/82（s1234），说明真实效应没有被改掉。

### 6.2 结论变化

1. **B 的 R-only 恢复比旧表更干净。**
   - post1 recomp：B-M6 926/1024（90.4%，旧 91.3%）；
     cyh s42 1015/1024（99.1%，旧 88.6%）、s1234 98.7%、s2025 99.8%；
   - 父节点回正后 post2 已 1024/1024（或 82/82）true，post4 全 true；
   - 旧表 post2/post4 的 residual（尤其 cyh s42 的 31/82）是 token 4/5
     整行覆盖 artifact，不是模型真的没恢复。
2. **cyh s42 不再是“明显更散”的弱权重。**
   旧 88.6% 是 artifact；修复后它在 parent R-only 上 99.1%，与 s1234/s2025
   同档。唯一真实小残留是 P1 4→5 的 U-only：80 true + 2 inj（2/82），
   以及 v2 r4 的 94 条 expected_wrong（4.4%）。
3. **“同一 forward 内 q 看不到 u 的修正”被直接证实。**
   - v2 teacher q r1 在任何权重/规模/k 下 correct 都是 **0**；
   - q 在 r2 恢复 84–94%，r4 恢复 88–99%；
   - 旧表 q r1 的 1537/1567/5767 条 correct 是读到了更晚轮次；
   - 修复后 teacher q r1 与 own-state q r1 完全对齐，机制从“推断”变成
     “有正确轮次标签的直接证据”。
4. **A′ 的 copy/freeze 比旧表更锐利。**
   before_u：U-only/U+R 注入值 1024/1024 保留，R-only 1024/1024 按父重算；
   旧表 R-only 只有 971/1024，另有 33 条 other。
5. **不受影响、无需重跑的部分仍然有效**：
   - 旧 47 候选 `inject_value_discrete.py` 的全部结果；
   - v2 的 own-state 路径；
   - C4 `c0_natural_swap.py`；
   - C0 训练与 `eval_nointerm_model.py` 评测；
   - C0 2Val 三 seed 训练（GPU0/1/4）未受本次分析影响。

### 6.3 产物与边界

- 新结果目录：`tmp/analysis/parent-2x2-fixed/`、`tmp/analysis/inject-v2-fixed/`；
  旧目录保持不动，便于审计。
- P1 旧数据集：`tmp/analysis/parent-2x2-fixed/P1_oldset/`（82 条 u）；
  heldout-seed42 第二数据集：`tmp/analysis/parent-2x2-fixed/P1/`（79 条 u）。
- 本记录只覆盖 2026-11-13 的两个脚本 bug；peer 对 `experiments/evaluation/`
  的未提交改动不在本次修复/提交范围内。
- C0 100ep 训练仍在进行；最终 C0 结论需等训练和正式选模完成。
