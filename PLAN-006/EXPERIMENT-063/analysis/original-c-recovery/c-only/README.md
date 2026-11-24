# 原始 C 恢复：C-only 正式新图（2026-11-22）

- 状态：**仅两个原始 C 的正式新图；A_s42/B_s42 未运行（B 相关执行待用户批准）；未训练、未改结构。**
- 入口：`tools/c_hard_graphs/original_c_recovery.py`（GPT 在 `1aa6ad8` 新增）；本批用只含两个原始 C 的 scratch wrapper 调用，未改生产入口。
- 数据根：原训练工作树 `algorithmic-generalization-transformer-architectures`；权重为固定 e143 原始 source，不是 correction/update final。
- 本目录是 C-only 交付；四模型同题比较（C_s42/C_s1234/A_s42/B_s42）尚未执行，因此 `paired.csv` 中的 `all_eligible` 在这里只表示**两个 C 共同合格**，不是四模型共同合格。

## 1. 权重与题面

| model | SHA-256 | 说明 |
|---|---|---|
| C_s42 | `4a00d6a21a7ed6ec9730daccf27f6077af45d2f2d7ff7376386bf0aa8903a798` | 原始 C，e143 |
| C_s1234 | `0bca3a2b91e82d70d5808859d3a907e0b8a32e0a852ba483c208f955d7d70475` | 原始 C，e143 |

- 每模型 192 题：`size ∈ {32,64,128}` × `depth ∈ {4,10}` × `gid ∈ 0..31`。
- 题族把一条可追踪的 spine 与独立背景表达式交错；四个输入臂共享变量名/图/位置：base、same-value sham（k1 同值）、target donor（k0+a）、parent donor（k0+b）。
- 角色：`source`（spine[0]）、`internal`（spine 中点）、`sink`（spine 末）。`internal` 另有 parent/both/held_parent/held_same_parent 条件。
- 时点：`first_correct`（target 首次四字段正确）、`child_correct`（target+child 均正确）、`graph_complete`（首次整图数值正确，搜索上限 D+4）。
- 观察窗 H=16；持续错父 HOLD=4，释放后再看 16 轮；另在 D、D+4、D+16 记录 fit。
- 每次干预与同起点的未干预 twin 并排运行，所有下游判断用 twin 差异；同值 sham 的 hidden 差也逐轮记录（`patch_differences`）。

## 2. 基础拟合与长窗口

`fit-summary.csv` 给出全部模型/规模/深度。关键点：

| model | size/depth | D+4 整图 value 全对 | D+16 整图 value 全对 | D+16 节点 first4 全对 |
|---|---|---:|---:|---:|
| C_s42 | 128/4 | 29/32 | 27/32 | 3687/4096 |
| C_s42 | 128/10 | 28/32 | 20/32 | 3027/4096 |
| C_s1234 | 128/4 | 28/32 | 24/32 | 3325/4096 |
| C_s1234 | 128/10 | 21/32 | 3/32 | 2532/4096 |
| C_s42 | 64/4 | 31/32 | 31/32 | 1844/2048 |
| C_s1234 | 64/10 | 31/32 | 22/32 | 1215/2048 |

- 标准轮 `D` 对大图不足：128/4 在 D 时只有 C_s42 3/32、C_s1234 2/32 整图 value 全对，D+4 才到 29/32、28/32。因此 `graph_complete` 时点的合格数不是 192/192，本批共同合格为 342/384（两 C），同轮共同合格只有 222/384。
- D+16 的 first4 明显漂移（VARIABLE/SYNTAX 长窗口问题），但 value 仍高；late 只应看配对 twin，不能把“最后 value 对了”直接算作纠错。

## 3. 主要读数（`common` = 两 C 共同合格，internal，全部 size/depth 合并）

`key-outcomes-common.csv` 是聚合表；`specific_child` 要求子节点恰好等于“把错值代入子表达式”的期望值，是严格条件，不等于所有受损。

| 条件 | 时点 | 合格/尝试 | target 下一轮正确 | target 末轮正确 | 子节点 specific | specific 后末两轮 target+child 都对 | 第一步跟错父值 | 末轮保持注入 target |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | first | 384/384 | 384 | 383 | 0 | — | — | — |
| same_value | first | 382/384 | 382 | 381 | 0 | — | — | — |
| target | first | 362/384 | 276 | 361 | 343 | 340/343 | — | 0/362 |
| target | graph_complete | 342/384 | 231 | 336 | 118 | 117/118 | — | 0/342 |
| parent | first | 374/384 | 26 | 369 | 139 | 138/139 | 187/374 | — |
| parent | graph_complete | 342/384 | 78 | 325 | 82 | 78/82 | 96/342 | — |
| both | first | 354/384 | 16 | 349 | 337 | 330/337 | 169/354 | 1/354 |
| both | graph_complete | 342/384 | 42 | 325 | 132 | 128/132 | 106/342 | 1/342 |
| held_parent | first | 374/384 | 26 | 356 | 355 | 347/355 | 187/374 | — |
| held_parent | graph_complete | 340/384 | 78 | 299 | 292 | 279/292 | 94/340 | — |
| held_same_parent | first | 384/384 | 384 | 379 | 0 | — | — | — |
| held_same_parent | graph_complete | 340/340 | 340 | 330 | 0 | — | — | — |

- `child_correct` 时点的 target/parent/both：target 下一轮正确 257/382，specific 219/382；parent 77/382、91/382；both 42/382、229/382。完整数在 `key-outcomes-common.csv`。
- **target**：readout 下一轮回正 276/362（76%），但同轮/随后 child specific 343/362（95%）；即“修回”与“下游已消费错值”同时发生。
- **parent**：target 下一轮几乎必错（26/374，7%），说明父值是下一轮计算的直接依据；specific child 139/374（37%）。
- **both**：target 下一轮正确仅 16/354（5%）；第一步跟错父重算值 169/354（48%），末轮仍保持注入 target 仅 1/354（0.3%）。没有“在三个候选中选正确值”的行为；剩余 52% 的第一步去向需从逐题 trace 再分解。
- **held_parent**：持续 4 轮错父期间 child specific 355/374（95%），释放后 joint 347/355（98%）；这是跟踪父状态/重算，不是主动纠错。
- **same_value / held_same_parent**：hidden 实际有变化（effective_change 382/382），但 target 全对、specific child 全 0。损伤与“错值”相关，不是打补丁伪影。
- `specific` 后的 joint 比例高（95–99%）只说明“一旦传播，多数在 H=16 内回到 target+child 都对”；它不是阻止传播，也不是全事件恢复率，分母是 specific 事件。

## 4. 自然错误回放

| model | case | opportunities | natural events | child edges | replays | specific rescues |
|---|---:|---:|---:|---:|---:|---:|
| C_s42 | 192 | 337,614 | 217 | 187 | 76 | 28 |
| C_s1234 | 192 | 315,962 | 465 | 344 | 118 | 24 |
| 合计 | 384 | 653,576 | 682 | 531 | 194 | 52 |

- `events` 是去重后的自然错值 node-time 事件；`child_edges` 是带子节点的边；`replays` 只取每图前四条合格边；`specific_rescues` 要求自然错值分支的子节点等于错值期望、且回放上一轮正确状态后子节点回到 gold。
- 自然错值本身罕见（682/653,576 个节点-轮机会），可回放 194 条，specific rescue 52 条；不能用人工 32/32 估计自发恢复率。

## 5. 观察与边界

1. **readout 回正 ≠ 下游未受损**：target 注入后 target 下一轮常常对了，但 child 仍可 specific 跟随错值。
2. **父值是强驱动**：parent 注入使 target 下一轮 7% 正确；双错冲突时第一步跟错父 48%，末轮保持注入 target 0.3%。
3. **错误会传播，随后大多被冲掉**：specific 后 16 轮内 joint 回到 target+child 都对的比例高；这可以解释为逐轮重算/覆盖，不能据此证明检测或主动纠错。
4. **同值 sham 零损伤**：值相关，不是 hidden 补丁本身。
5. **长窗口有基线漂移**：D+16 first4 下降，late 只读配对 twin；`graph_complete` 只用 D+4 内首次整图完成。
6. **C-only**：A′/B 未跑；不能判断上述波形是否 C 特有，也不能把差异归因于纠错训练。四模型同题比较需要用户批准 B 相关执行。

## 6. 文件

- `tables/summary.csv`：按 model/size/depth/role/timing/condition/scope 的聚合；`scope=all/common/common_same_round`。
- `tables/paired.csv`：逐事件配对与 `all_eligible`（本批只含两个 C）。
- `tables/events.csv`：逐事件聚合结果（不含 raw trace）。
- `tables/fit.csv`、`tables/natural.csv`：基础拟合与自然事件。
- `key-outcomes-common.csv`、`fit-summary.csv`、`natural-summary.csv`：本 README 的紧凑表。
- `raw/C_s42/`、`raw/C_s1234/`：每模型 192 个 `size-depth-gid.json.gz` + `identity.json`；每个 case 含 `events` 的 r/u/q trace、clean twin、inputs、patch_differences，以及 `natural` 的 joint_followup；**不含全节点每轮读数**（原始全节点轨迹在另一个分层注错补充的归档中）。
- `raw-manifest.sha256`：384 个 raw 文件 + identity 的逐文件 SHA-256。

### SHA-256

| 文件 | SHA-256 |
|---|---|
| `tables/summary.csv` | `f75f9311bfe95a8e7cbd6cf7ed71edfa4fbf008942ed9efbc955472c9a6d93e3` |
| `tables/paired.csv` | `42c4a94891c31ac383563e0bb11f9773c5484eacf304a26e6c6e2ce90ee95271` |
| `tables/events.csv` | `c558a109dbd093b8f86cf2161f743431bfa6ec04e2b94dc6b967c0a4cab75efa` |
| `tables/fit.csv` | `983939e7446b8e6af48b78fdda58814caa8d3a6c4b9083997924dfd0a8842457` |
| `tables/natural.csv` | `31887dd6dc60e88f4361a87cc3dced0176e7803549564cd6e0f5b4c85f68552c` |
| `key-outcomes-common.csv` | `510ffffa7c99831a7948f0fe26787b71dae63ab53727debdaa883e36d6ed6cb8` |
| `fit-summary.csv` | `93433d406a8fcd292708f915137fa3bdb6f2f6e2cae81d33822be58a3e9483e9` |
| `natural-summary.csv` | `af2d1237e3565c6ea8951a86076373178f4ce40b4794549a5ab8b1449496b252` |
| `raw-manifest.sha256` | `8dd35e110ef82609d27d14126fbf14b41e7e5a2b8d19cd25ae1d7f2b56e48385` |
| `tools/c_hard_graphs/original_c_recovery.py`（identity 记录） | `da64c812b9ef7e1ef3e3bf5d325e50d4e716e0be33b0bac2597bd1e389679295` |

## 7. 复现

```bash
# 生产入口（四模型，需要 A'/B 批准）：
python tools/c_hard_graphs/original_c_recovery.py --data-root <training-worktree>
# 本批 C-only 的 scratch wrapper（不修改生产入口）：
python tmp/analysis/run_original_c_recovery_c_only.py
```

本批的 `--data-root` 是原训练工作树；输出在 `tmp/analysis/original-c-recovery-v1/formal/`，本目录是它的 C-only 拷贝；raw 用 `gzip -dc file.json.gz`/`zcat` 读取。