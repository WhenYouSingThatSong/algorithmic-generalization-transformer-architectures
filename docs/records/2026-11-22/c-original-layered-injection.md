# 原始 C 的分层人工注错：目标位置、父节点、上游节点与时机

- 日期：2026-11-22。
- 状态：**补充探索；仅原始 C 权重；未训练、未改结构、未改历史 C 产物。**
- 本批只回答“注错后逐轮观察到什么”，不判断原始 C 是否具备检测/纠错能力；最终结论由用户判断。

## 1. 权重与题面

| model | 角色 | checkpoint | SHA-256 |
|---|---|---|---|
| C_s2026 | 原始 C，非 correction/update | `tmp/runs/oldC0-newseed-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-2026 - 2026-11-14-10:12:43/epoch=143-val_loss=0.0000.ckpt` | `40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780` |
| C_s42 | 原始 C，非 correction/update | `tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-14-02:35:10/epoch=143-val_loss=0.0000.ckpt` | `4a00d6a21a7ed6ec9730daccf27f6077af45d2f2d7ff7376386bf0aa8903a798` |
| C_s1234 | 原始 C，非 correction/update | `tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-1234 - 2026-11-14-02:38:10/epoch=143-val_loss=0.0000.ckpt` | `0bca3a2b91e82d70d5808859d3a907e0b8a32e0a852ba483c208f955d7d70475` |

- 题面：`exposure_case(gid, depth)`，`gid=352..367` × `depth=4,8`，每个权重 32 图；每图同时输入 base 与 donor（donor 对第一式常数加 `delta`，`delta` 由既有 seed 规则确定）。plan SHA `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`。
- 32 图的 base 起点全部合格：每个条件下的“已计算节点等于 gold、未计算节点为 EMPTY”均为 32/32，见 summary 的 `base_ok`。
- 注错使用 **donor 的连续 hidden 行**（匹配图、匹配轮次），不是 token-ID 替换；因此注入的是一个完整状态行，不是纯数值编辑。

## 2. 条件网格

每个 case（gid×depth）注入同一组条件；`due = depth(u)+1`，`later = D+4`，每条分支从指定起点再跑 **16 轮**，每轮读取全部节点。

| 组 | 条件 | 起点 | 注入位置 | donor 来源轮 |
|---|---|---|---|---|
| clean | `clean_pending/just/later` | `due-1` / `due` / `later` | 无 | — |
| 自补丁控制 | `target_just_self` 等 | `due` | u / r / ancestor | 同轮 base 自身（应为 no-op） |
| 正确值控制 | `target_pending_correct` 等 | `due-1` | u / r / ancestor | `due` 的 **base** 正确状态 |
| 空值控制 | `target_pending_empty` | `due-1` | u | `due-1` 的 donor（EMPTY） |
| 目标-注错 | `target_pending_wrong`, `target_just`, `later_target` | `due-1` / `due` / `later` | u | `due` / `due` / `later` 的 donor |
| 父节点-注错 | `parent_pending`, `parent_just`, `later_parent` | 同上 | r（直接父） | 同轮 donor |
| 上游-注错 | `ancestor_pending`, `ancestor_just`, `later_ancestor` | 同上 | ancestor | 同轮 donor |
| 组合-注错 | `all_pending_wrong`, `all_just`, `later_all` | 同上 | u+r+ancestor | 目标 r1 用 `due`，r/ancestor 用同轮 |

- “pending”指目标 u 尚未计算；“just”指 u 刚算完；“later”指整图已完成（`D+4`）。
- 所有 32 个 target 注错的实际写入值均为 wrong（`patched_wrong=32/32`）；正确值控制为 `patched_gold=32/32`；空值控制为 `patched_empty=32/32`。没有“注错未生效”的样本。

## 3. 读法

- `gold`：base 程序真值；`alt`：donor 程序真值。
- `pair_wrong`：受影响节点在注错分支中为数值错误，而**同起点 clean 对照**在该轮为 gold；这是相对 clean 的“额外错误”，排除基线漂移。
- `pair_specific`：上述错误还恰好等于 `alt`（donor 值）；只表示“与 donor 值精确相容”，不证明复制机制。
- `premature_wrong`：clean 对照在该轮仍为 EMPTY，注错分支已有数值且错误；用于标记“提前计算”造成的错误。
- `premature_correct`：同上但数值正确。
- `baseline_wrong`：clean 对照自身在该轮为数值错误；16 轮时 late 起点可出现基线漂移，必须与 `pair_wrong` 分开。
- 统一受影响集合为 6 个节点：`ancestor, r, u, q, m1, sink`；每轮每权重最多 `32×6=192` 个节点-观察。下文的 1/8/last 分别指分支第 1/8/16 轮。

## 4. 控制：补丁动作本身不产生错误

- 全部 `*_just_self`、`*_pending_correct`：三个权重的 `pair_wrong`、`premature_wrong` 在每一步均为 0；`target_pending_empty` 亦为 0。C_s1234 的 just 类控制在第 16 轮有 `baseline_wrong_last=1`，来自 clean_just 自身的基线漂移（§7），不属于补丁效应。见 `layered-original-c-comparison.csv`。
- `target_pending_correct` 只会让 q 提前算出**正确**值：`premature_correct_1` = 26/32（C_s2026）、32/32（C_s42）、32/32（C_s1234）。
- `target_pending_wrong` 在相同起点下让 q 提前算出**错误**值：`premature_wrong_1` = 25/32、32/32、32/32。
- 因此下文的目标-预填效应可归因于注入的**值**，而不是“往 pending 位置写 hidden”这个动作本身。

## 5. 目标位置注入：readout 修回不等于下游未受损

“u1”是目标 u 在分支第 1 轮为 gold 的图数；“pairW1”是目标及受影响集合相对 clean 的额外错误数；“pairS1”是其中恰好等于 alt 的数量。

| model | condition | u1 | pairW1 | pairS1 | pairW8 | pairWlast |
|---|---|---:|---:|---:|---:|---:|
| C_s2026 | target_pending_wrong | 31/32 | 1 | 0 | 0 | 0 |
| C_s2026 | target_just | 31/32 | 33 | 32 | 0 | 0 |
| C_s2026 | later_target | 28/32 | 36 | 23 | 0 | 0 |
| C_s42 | target_pending_wrong | 28/32 | 4 | 0 | 0 | 0 |
| C_s42 | target_just | 29/32 | 35 | 32 | 0 | 0 |
| C_s42 | later_target | 25/32 | 34 | 12 | 0 | 0 |
| C_s1234 | target_pending_wrong | 17/32 | 15 | 1 | 0 | 0 |
| C_s1234 | target_just | 17/32 | 47 | 33 | 0 | 0 |
| C_s1234 | later_target | 15/32 | 36 | 8 | 0 | 0 |

- u1 与既有四条件表的目标修回一致（C_s2026 31/31/28；C_s42 28/29/25；C_s1234 17/17/15），是两套实现的交叉检查。
- `target_just` 第 1 轮 u 已修回 31/29/17，但同一轮 q 已经用旧错误目标算错（q1 正确 0/32；pairW1 33/35/47）。随后错误沿 q→m1→sink 逐层出现，8 轮时相对 clean 的额外错误已为 0（C_s2026/C_s42；C_s1234 的 later 组仍有基线漂移，见 §7）。
- `target_pending_wrong` 的主要额外错误不是 u 本身（pairW1 1/4/15），而是被提前触发的 q（premature_wrong 25/32/32）；u 在下一轮已算对 31/28/17。这是“提前可用”与“值错误”的耦合，不能只当作“目标位置注错”。

## 6. 父/上游注入：错误沿依赖链一层一轮下传

### parent_pending（r 已算完，u 未算）

| model | u1 | r1 | q2 正确/错误 | m1 第3轮 正确/错误 | sink 第4轮 正确/错误 | pairW8 | pairWlast |
|---|---:|---:|---:|---:|---:|---:|---:|
| C_s2026 | 0/32 | 30/32 | 0/32 | 0/32 | 0/32 | 0 | 0 |
| C_s42 | 0/32 | 28/32 | 0/32 | 0/32 | 0/32 | 4 | 4 |
| C_s1234 | 0/32 | 21/32 | 0/32 | 0/32 | 0/32 | 0 | 0 |

- 第 1 轮 u 为错误 32/32，且 u 的错误值 32/32 恰等于 alt[u]；r 本身在第 1 轮已回正 30/28/21。
- 错误波按 u(第1轮)→q(第2轮)→m1(第3轮)→sink(第4轮) 各层一轮出现，三层权重同形。

### ancestor_pending（ancestor、r 已算完，u 未算）

| model | u1 | u2 | r1 | q3 错误 | m1 第4轮错误 | sink 第5轮错误 | pairW8 | pairWlast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C_s2026 | 32/32 | 1/32 | 1/32 | 31 | 31 | 31 | 0 | 0 |
| C_s42 | 32/32 | 6/32 | 5/32 | 27 | 27 | 26 | 1 | 0 |
| C_s1234 | 32/32 | 6/32 | 6/32 | 25 | 26 | 27 | 10 | 6 |

- 第 1 轮被注入的 ancestor 自身回正 27/27/17，但 r 已用旧 ancestor 算错 31/27/26；u 在第 1 轮仍对（32/32/32），第 2 轮才错（1/6/6 正确）。
- 错误波整体比 parent_pending 晚一层：r(第1轮)→u(第2轮)→q(第3轮)→m1(第4轮)→sink(第5轮)。
- `pair_specific_1` 只有 16/21/11，而 `pair_wrong_1` 为 36/32/41：下游错误值多数不是 donor 的精确 alt 值，符合“用混合输入重新计算”而非“复制 donor 值”的描述；不能把 `specific` 当机制证据。

### 组合条件

- `all_pending_wrong`/`all_just`/`later_all` 的 `pairW1` 为 65–109/192，明显高于单点注入；但 8 轮后 C_s2026 全部回到 0，C_s42/C_s1234 仍有少量残留（§7）。
- 在 `all_*` 中，目标预填仍单独制造 premature q 错误；其余传播与 parent/ancestor 波一致。

## 7. 8 轮与 16 轮：C_s2026 全部洗掉，s42/s1234 有少量残留

- `pair_wrong` 是相对同起点 clean 的额外错误；`baseline_wrong_last` 是 clean 自身在第 16 轮的错误，用于说明为何不能只看绝对错误数。

| model | condition | pairW8 | pairWlast | baselineWlast |
|---|---|---:|---:|---:|
| C_s2026 | clean_later | 0 | 0 | 5 |
| C_s2026 | 全部注错条件 | 0 | 0 | 0–5 |
| C_s42 | parent_pending | 4 | 4 | 0 |
| C_s42 | parent_just | 4 | 4 | 0 |
| C_s42 | all_pending_wrong | 5 | 4 | 0 |
| C_s42 | all_just | 5 | 4 | 0 |
| C_s42 | later_parent | 4 | 8 | 0 |
| C_s42 | later_ancestor | 6 | 5 | 0 |
| C_s42 | later_all | 10 | 13 | 0 |
| C_s1234 | ancestor_pending | 10 | 6 | 0 |
| C_s1234 | ancestor_just | 5 | 0 | 0 |
| C_s1234 | all_pending_wrong | 9 | 5 | 0 |
| C_s1234 | all_just | 6 | 0 | 0 |
| C_s1234 | later_parent | 6 | 6 | 5 |
| C_s1234 | later_ancestor | 7 | 5 | 5 |
| C_s1234 | later_all | 15 | 9 | 5 |

- C_s2026 的 22 个条件在第 8 轮和第 16 轮 `pair_wrong` 全为 0；C_s42/C_s1234 的其余未列条件为 0 或 ≤1。
- `later` 起点在 16 轮时 clean 自身会出现基线漂移：C_s2026 `clean_later` 有 5 个节点-观察数值错误（u、q 各 1，其余在 m1/sink），C_s1234 同样为 5；C_s42 为 0。因此 `pairWlast` 是 late 组的主读数，绝对 `all_correct_last` 不能单独使用。
- 残留是“节点-观察计数”，不是图数或纠正率；单条件每轮最多 192 个观察。

## 8. 观察清单（不作机制结论）

1. 目标 readout 的下一轮修回可以发生，但同一轮/下一轮子节点已经消费了错误值；readout 修回不等于下游未受损。
2. 父/上游注入时，被注入节点自身通常先回正，而错误波继续按依赖层一层一轮下传；这表现为“局部重算/覆盖”，本批不能区分检测、纠错与普通重算。
3. pending 目标被预填为数值时，子节点会提前计算；预填正确值只提前算出正确结果，预填错误值才提前算错。该效应对照干净。
4. 三个原始 C 权重的定性波形一致；恢复完整度不同：C_s2026 在 8/16 轮均无额外错误，C_s42/C_s1234 在少数 late/组合条件仍有少量额外错误。
5. 16 轮 late 起点存在 clean 基线漂移，所有 late 结论必须使用配对读数。

## 9. 边界

- 不是 correction/update 的因果检验：本批没有在相同协议下运行 C_s42-correction/update；不能据此判断纠错训练是否改变上述波形。
- 不是“自然错误”回放：本批是人工 hidden 注入；`natural_replay` 的零覆盖问题仍按既有 B 复核记录处理。
- donor hidden 是完整状态行，包含值以外的上下文；正确值/空值/自补丁控制只能支持“值相关”，不能证明模型内部只有值通道。
- `specific` 只表示与 donor alt 精确相等；ancestor 注入中大量错误不等于 alt，说明下游可能是混合输入的重新计算。
- 32 图/权重、16 轮、每个 case 一个 donor；没有跨 donor、跨 delta、跨图族的泛化结论。
- 不把本批任何计数写成“检测率”“纠错率”或“架构能力”；需要 correction/update 同协议对照、更多 donor 和更长窗口后再讨论。

## 10. 文件与 SHA-256

紧凑表在 `docs/records/2026-11-22/c-s2026-original-injection/`（脚本副本也在该目录，与 `tmp/analysis/c-original-injection/` 同 hash）；原始逐记录 JSON 在 `tmp/analysis/c-original-injection/`（未提交，约 23 MB 的 16 轮 canonical + 13 MB 的 8 轮归档）。

| 文件 | SHA-256 |
|---|---|
| `layered-original-c-comparison.csv` | `efb9ed6fd81e4d0a990f071d7d8e55e67d9fb9405e93bed1199af58448d03f55` |
| `C_s2026-layered-summary.csv` | `7dff6343db5e54a7c484dfb486a84278320c768ae840b25429d8786e41f46711` |
| `C_s42-layered-summary.csv` | `f76fccc724a17fe5bcedb651822ea389a399a56cd8c77945758e275567ef9f7f` |
| `C_s1234-layered-summary.csv` | `21cc51ffb381b07c67fd2cff9f32694389321d3ebaee0b5722465ba1675f6292` |
| `C_s2026-layered-steps.csv` | `89589828eef2b55266cff2c8b9caf4c8094fac4eafb284946081a6dd39b99329` |
| `C_s42-layered-steps.csv` | `388366745171e604bb276ab2c71b98bfad512e1abdedf67cdcf80be94c1b0a1d` |
| `C_s1234-layered-steps.csv` | `499d0252ab6e67ef1690d7a97ed163290af133b560d7978b2a2892fb53352b88` |
| `C_s2026-layered-per-graph.csv` | `64b2da0e8b5b62527898026554d93761143fe38fb6e41501bc6fe796044757cc` |
| `C_s42-layered-per-graph.csv` | `1874c07a9f61e8bdf81b6121349606cbd3f9cd388b5e8e199a71d64dcaa9cdc7` |
| `C_s1234-layered-per-graph.csv` | `d075b3c24851e18f844841b3167e7fa24b603223c4527c748ffdcabe5040b813` |
| `C_s2026-layered-paired-steps.csv` | `3c9e1b16da928f3079ad8937236326ff1a6e14ce9f19b9f25e533b2ef5eef6c3` |
| `C_s42-layered-paired-steps.csv` | `eb2e8c9b81df82119a0dae69852555f85d4c5ccb98949af7a12069f41cc5e2fc` |
| `C_s1234-layered-paired-steps.csv` | `3275384a7a60d28d264873e4a74db81e045b516a7e27eb641b2e5a1826269811` |
| `run_layers.py` | `15043b43ef640767786070de3bb9b4a70fd00da446a9576593abab632eebdd83` |
| `summarize_layers.py` | `3bd60c4df9a490e0d376414e502565e2d56901f87d4ddbde1cccd35fb2a1c37c` |
| `C_s2026-layered.json`（16 轮 raw） | `66abc4ebe63592c78acf5fe1b2c964dd198b985896f90ce20816e8c7e0d35ebe` |
| `C_s42-layered.json` | `7aa29ecceaa8e0a21964f16533571c86da98b8e129b4cd562f35200b33c10add` |
| `C_s1234-layered.json` | `e8dd78d562103f8d5fa540165e3870d6bc17055d054e8051f280f1f1b88229ed` |
| `C_s2026-layered-r8.json`（8 轮归档） | `c47725b30e685eb558b873f62ad0a1e008f6325feb5f3fed3b1ed32da475494d` |
| `C_s42-layered-r8.json` | `5d1baec22026530461f6abddeea06d8191ecd9e295f8897054a4a5a1a9095b42` |
| `C_s1234-layered-r8.json` | `4a1c8b9a26bab1f158e5c0ed523d0ea7cde186b9297926052e50eecbbc1fd4cd` |
| `four-conditions-original-c.csv`（旧四条件摘要） | `cc992523a05b8a0e67b85a218dbf199ea920dc3e2ad67e714eede82bfa779328` |
