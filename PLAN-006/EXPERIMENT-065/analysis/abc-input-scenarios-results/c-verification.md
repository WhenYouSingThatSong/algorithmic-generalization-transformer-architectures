# C 侧复核：四模型场景报告（43a71d9）

- 复核对象：`docs/records/2026-11-22/abc-input-scenarios-results/`（报告、`scenes.csv`、`same-graph-fit.csv`、`natural.csv`、`provenance.json`），提交 `43a71d9`。
- 状态：**只复核 C 侧 raw 与报告 C 列；未重跑模型、未改 A′/B raw、未改报告数值。**
- 复核方式：重跑 `abc_scene_report.aggregate(Path(tmp/analysis/abc-input-scenarios-v1/formal))`，它读取四模型正式 raw、校验 C raw manifest 和 layered archive，返回 `counts=5,328 / fit=2,304 / natural=768`；与提交的 `scenes.csv`、`same-graph-fit.csv`、`natural.csv` 行数一致。

## 1. C raw 与来源

- C standard raw：`docs/records/2026-11-22/original-c-recovery/c-only/raw/{C_s42,C_s1234}`，386 行 manifest 全过（384 个逐题 gz + 2 个 identity）。
- C layered raw：`docs/records/2026-11-22/c-original-layered-injection/raw-layered-16round.tar.gz`；C_s42/C_s1234 checkpoint 分别 `4a00d6a2…`/`0bca3a2b…`，各 32 case、704 records（22 事件/case），报告只读这两个模型。
- C expanded raw：`docs/records/2026-11-22/abc-input-scenarios/c-only-expanded/raw/{C_s42,C_s1234}`，130 行 manifest 全过（128 个逐题 gz + 2 个 identity）；每 seed 64 图 × 42 场景 = 2,688。
- `provenance.json` 的 C standard/layered/expanded source 均指向上述 C 侧文件，未指向 A′/B 归档；报告运行命令只列 `--models A_s42 B_s42`，C 复用两批历史 raw 和 C 侧 expanded raw。

## 2. 报告 C 列抽查

由重跑 `aggregate` 与 `scenes.csv` 核对，均一致：

| 场景（common） | C_s42 | C_s1234 |
|---|---|---|
| standard first target，eligible 137：next_correct / child_damage / end_joint / specific_end | 128 / 137 / 134 / 134 | 84 / 135 / 135 / 132 |
| standard graph_complete target，eligible 79：next_correct / child_damage | 70 / 68 | 30 / 61 |
| expanded before internal，eligible 64：next_correct / child_damage | 56 / 9 | 43 / 20 |
| expanded before both_parents_cancel，eligible 64：next_correct / child_damage | 11 / 58 | 15 / 56 |
| expanded before separate_branch，eligible 64：child_damage | 0 | 0 |
| expanded complete restore_empty，eligible 64：next_correct / child_damage | 64 / 24 | 64 / 21 |
| 64/4 D+4 fit value / nodes | 2047 / 2048 | 2048 / 2048 |
| 128/4 D+4 fit value / nodes | 4091 / 4096 | 4092 / 4096 |
| natural errors / opportunities | 217 / 337,614 | 465 / 315,962 |
| natural replays / specific rescues | 76 / 28 | 118 / 24 |

## 3. 需要标注的两个口径

1. **`end_joint` 与 `specific_end` 不是同一个数**。标准组 first target 中 C_s1234 的 `end_joint/follow` 为 135/135，而 `specific_end/specific_follow` 为 132/132。交付 README §2.1 写“末两轮两处均正确 C（1234）135/135”，主报告“已算位置”写“132/132”；两者分别对应“任何受损”和“子节点确实采用特定错值”的分母，不是数据矛盾，但需要并列说明，不能混用。
2. **C_s1234 strict 4-field 异常**：26/2,688 个 `complete` 场景的写入位置 variable token id 与 donor 不一致，value 正确；报告的 expanded 路径使用 `value_eligible`，因此这 26 个进入计数。严格四字段分母是 2,662；README 已注明，保留即可。

## 4. 执行分工更正

- 看板 2026-11-22 16:16 的字面分工就是本批实际执行：**Agent B（yhy-57）跑 A_s42/B_s42 三组；Agent C（本会话）跑 C_s42/C_s1234 expanded**。
- C 侧跨会话消息从未要求对调，原话是“No reversal—your practical split is exactly the board: … You are the B-side executor, I am the C side.” B 侧 README §5 和 board/log 把消息记成“C 跑 A/B、B 跑 C expanded”，并把 C expanded raw 记成 B 交付，是误读。
- C expanded raw 由本会话在 GPU0 生成并提交 `cd89e47`；B 侧未写 C_s42/C_s1234 目录。`provenance.json` 的 `expanded_source` 也指向 C 侧 docs。报告数值不受影响，只需更正分工备注。

## 5. 边界

- 本复核只覆盖 raw 完整性、来源、C 列数值和两个口径；未重跑 A′/B，未独立重算报告全部 5,328 行。
- 报告仍是固定权重/固定题面的行为对照，不作“纠错训练导致”或“架构差异导致”的因果结论。
- C 的 hidden 注入、A′/B 的数值字段注入信息量不同；`both_parents_cancel` 等场景受 hidden 状态影响，不能单归因于某一通道。
