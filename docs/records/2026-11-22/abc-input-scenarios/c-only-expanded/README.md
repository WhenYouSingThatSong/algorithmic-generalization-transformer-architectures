# expanded 场景：C-only 中间交付（2026-11-22）

- 状态：**仅两个原始 C 的 expanded 新 64 题**；A_s42/B_s42 由 yhy-57 在 GPU2 运行，尚未齐备。最终四模型场景报告待 B 侧完成后由 `abc_scene_report.py` 生成。
- 入口：`tools/c_hard_graphs/abc_scene_queue.py --models C_s42 C_s1234 --data-root <training-worktree>`；C 的 standard/layered 复用已交付记录，本批只跑 expanded。
- 题面：gid 400–431 × 链长 4/8，共 64 题/模型；每题 42 个场景（4 个 phase × 操作集），每次写入后自由运行 16 轮。C 连续权重按完整 hidden 行写入，离散 A′/B 只改数值字段。
- 输出：`tmp/analysis/abc-input-scenarios-v1/formal/expanded/{C_s42,C_s1234}`；本目录是 C-only 拷贝。

## 1. 覆盖与写入有效性

| model | cases | scenes | stage_ok | value_eligible | full 4-field eligible | writes | write_valid | write_value_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C_s42 | 64 | 2,688 | 2,688 | 2,688 | 2,688 | 3,456 | 3,456 | 3,456 |
| C_s1234 | 64 | 2,688 | 2,688 | 2,688 | 2,662 | 3,456 | 3,430 | 3,456 |

- `value_eligible` 只核对写入值和起始数值状态；`full 4-field eligible` 还要求 VARIABLE/SYNTAX 等字段完全等于 donor 读取。C_s1234 有 **26 个 `complete` 场景**（26/2,688）出现后者不一致，原因集中在写入位置的 variable token id（例如 before `[0,57,1,10]`、actual `[0,33,1,12]`），数值仍正确。`abc_scene_report.py` 的 expanded 路径使用 `value_eligible`，所以这 26 个会进入 C 的计数；若采用严格 4 字段分母，应扣除它们。明细见 `full-eligibility-anomalies.csv`。
- C_s42 无此类不一致。`stage_ok` 和每个 write 的实际读回逐条在 raw 里保留。

## 2. 主要 C-only 读数（64 题/模型；value_eligible 分母）

| 场景 | phase | C_s42：uNext / childDamage | C_s1234：uNext / childDamage | 说明 |
|---|---|---:|---:|---|
| clean | all | 64/64 / 0/64 | 64/64 / 0/64 | 同起点干净分支 |
| fill_correct（提前填对） | before | 64/64 / 0/64 | 64/64 / 0/64 | 预填正确值不损伤 |
| direct_parent（只改直接父） | before | 0/64 / 64/64 | 0/64 / 64/64 | 目标下一轮必错 |
| older_ancestor（只改更早祖先） | before | 64/64 / 57/64 | 64/64 / 56/64 | 目标下一轮仍对，子节点随后受损（晚一层） |
| internal（改将算目标） | before | 56/64 / 9/64 | 43/64 / 20/64 | 目标 readout 多数回正，仍有子损伤 |
| internal（改刚算目标） | after | 56/64 / 64/64 | 40/64 / 64/64 | readout 回正但子节点已消费错值 |
| input_source（改输入已算值） | before | 64/64 / 35/64 | 64/64 / 28/64 | 目标保持正确，后代可受损 |
| both_parents_add（两父值都改） | before | 0/64 / 64/64 | 0/64 / 64/64 | 目标按错父值算错 |
| both_parents_cancel（两父值改动相消） | before | 11/64 / 58/64 | 15/64 / 56/64 | 数值和回到 gold，但 hidden 供体不同；多数仍错 |
| parent_and_target_conflict | after | 4/64 / 64/64 | 2/64 / 64/64 | 与 consistent 的 uNext（3/64、2/64）几乎无差别 |
| parent_and_target_consistent | after | 3/64 / 64/64 | 2/64 / 64/64 | 同上 |
| restore_empty（换回先前空状态） | complete | 64/64 / 24/64 | 64/64 / 21/64 | 目标下一轮重算对，子节点仍可能受损 |
| separate_branch（改另一分支） | all | 64/64 / 0/64 | 64/64 / 0/64 | 负对照：无子损伤 |
| sink（改终点） | complete | 64/64 / 0/64 | 64/64 / 0/64 | 终点无子节点 |

- `early` phase 的 `uNext` 均为 0，因为目标按图深度尚未到计算轮；该 phase 的 `early_num_err` 在 `internal`/`direct_parent` 为 64/64，表示子节点被提前触发并算出数值，不能当作“下一轮目标正确率”。
- `older_ancestor` 的 64/64 只说明目标下一轮仍对；子节点 damage 57/56 说明错误在下一层才被消费。
- `both_parents_cancel` 的父值数值和等于 gold，但 C 的目标下一轮只有 11/15 正确；本批不能把差异归因于单一通道，只记录“数值相消不足以保证 hidden 状态下目标正确”。
- `parent_and_target_conflict` 与 `consistent` 在目标下一轮正确率上几乎相同（C_s42 4 vs 3，C_s1234 2 vs 2）；这与“选择正确来源”的预期相反，但不能据此证明内部机制。
- `internal`/`direct_parent`/`older_ancestor` 的 childDamage 是“干净分支 q 在该轮正确、干预分支 q 不同”的严格计数；不是所有节点或所有错误。

完整 84 行在 `scenes-c-only.csv`（model × phase × condition）；每行含 attempts、full/value eligible、writes、uNext、childDamage、q_risk、follow、end_joint、early_num_err。

## 3. 边界

- C-only；A′/B 未完成，不能判断这些模式是否 C 特有，也不能把它归因于纠错训练。
- 连续 C 的注入是 donor 的完整 hidden 行，不是纯数值编辑；`both_parents_cancel` 和 conflict/consistent 的差异受 hidden 状态影响。
- expanded 的 `value_eligible` 与严格 4 字段 eligible 在 C_s1234 差 26 个 `complete` 场景；最终报告若只报 value_eligible，需要在脚注标明这一点。
- 观察窗 16 轮；长窗口 clean 漂移另见 standard 记录；本批不把“最后 value 对了”直接称为纠错。
- 本目录不是最终四模型报告；最终报告由 `abc_scene_report.py` 在四模型 raw 齐备后生成。

## 4. 文件

- `raw/C_s42/`、`raw/C_s1234/`：各 64 个 `0-{4,8}-{gid}.json.gz` + `identity.json`；每个 case 含 42 个 scene 的 writes、start_values、rows（16×全节点 value）和 eligible 标志。
- `raw-manifest.sha256`：128 个 raw + 2 个 identity 的逐文件 hash。
- `scenes-c-only.csv`：C-only 场景聚合。
- `full-eligibility-anomalies.csv`：C_s1234 的 26 个 strict-4-field 异常。

### SHA-256

| 文件 | SHA-256 |
|---|---|
| `scenes-c-only.csv` | `071af26689de31210b0098a5692b70b2b78c5e0b684bd836e67db1cf419b8ff2` |
| `full-eligibility-anomalies.csv` | `467d92ed65b6e03d4397c232d38bc55351b3a4725b9115df055ae572a06c9afc` |
| `raw-manifest.sha256` | `a01b8dbeadc440bca311710dbb0a3d76d51d2e56bdace06739d66a7c920f8749` |
