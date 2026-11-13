# 代码审查修复：parent 2×2 aux 索引 + v2 teacher-state 轮次索引（2026-11-13）

状态：**代码已修复、smoke 已通过；全量重跑进行中**。
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
- v2 teacher：
  - N32 k4（7 个权重组合）、N32 k=0（4 个）、N128 k4（6 个），256 图。

输出根目录：

- `tmp/analysis/parent-2x2-fixed/`
- `tmp/analysis/inject-v2-fixed/`

重跑后本记录补 §5：新结果表、与旧数的差异、结论是否变化。
