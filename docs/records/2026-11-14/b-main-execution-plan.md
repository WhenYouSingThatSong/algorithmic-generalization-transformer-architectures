# B 主实验执行单（2026-11-14，待用户定稿）

状态：**准备稿，未启动主实验；未做任何注错运行**。B1（已有 pilot 轨迹的
CPU 补充）已完成，见 `b-horizon-pilot.md` §9。本单按用户的固定选择整理
B2：**固定 256 张 N32 候选图（graph_id 32–287，不补样）、每图最多 2 条
主链、teacher 固定前 64 张候选、主 H=16、固定前 32 张做 64 轮长观察、五
模块分阶段交付**。执行前仍需用户对范围/预算做最后确认；代码中尚缺的
项目列在 §7。

## 1. 固定选择与样本边界

### 1.1 用户已定

- 候选图：**graph_id 32–287 共 256 张**，固定不变；**不因合格数不足而
  向 288 以后补样**，也不按注错结果换图。
- 每图最多 **2 条 primary chain**（单父 u、q1/q2 不直接读 r）；不足 2 条
  按实际数量，不凑数。
- teacher start：**固定前 64 张候选，即 graph_id 32–95**；不是全部 256 张。
- own-state 是主分析；teacher 只作状态对照。
- 主观察窗 **H=16**，同轨迹截取 **H=8** 报告；不另跑 H=8。
- 长观察：**固定前 32 张候选，即 graph_id 32–63**，预先固定做 64 轮；
  不按 pilot 结果挑图。
- sustained：complete 起点固定错父 **4 轮**后释放，释放后观察 **H=16**；
  长观察子集释放后 **64 轮**（总 4+64=68 轮）。
- 阶段：B1（已完成）→ 主实验第一部分（early 四条件、complete 目标类型、
  sustained）→ 第二部分（D1/D2/D3、训练腐蚀审计）→ 后续跨 seed/N128。
- 不同阶段按各自资格取舍；**不因某图在别的阶段不合格而从固定候选集中
  删除**。只有 early vs complete 直接比较时使用共同合格交集并报 n。

### 1.2 之前使用记录（核实结果）

| 用途 | 文件/记录 | 用到的 graph_id | 与本批关系 |
|---|---|---|---|
| B 长轮数 pilot（v2） | `tmp/analysis/b-horizon-pilot-20261114/manifest.json` | 0–31（scan 32，全部 common-complete） | **不重叠**；本批从 32 开始 |
| B 代码 smoke / fix-test | `tmp/analysis/b-mechanism/{smoke,fix-test}/manifest*.json` | 0–3 | 不重叠 |
| B 其他正式结果 | 未发现（当前只有 pilot/smoke） | — | — |
| C0 能力探索（C 线） | `tmp/analysis/heldout-seed42/ADD/testdata_N32.pt` 的 deep 池 | 由 C 的固定 seed 抽样；实际 ID 以 C 的产物为准（C 线独立执行） | **同一 heldout 池，但不是 B 的选样依据**；若重叠，不构成 B pilot 泄漏，也不是“独立 heldout” |

因此本批准确表述是：**“与 B pilot（0–31）及 B smoke（0–3）不重叠的固定
候选集合”**，不是“完全未被任何实验使用过的独立 heldout”。C 线若使用同
一数据池，按 C 自己的固定 seed 抽样；B 的合格性和结论不依赖 C 的结果。
C 正在运行，本单不读取/不修改 C 的目录，不与其共用 GPU。

### 1.3 固定候选的冻结方式

- 写入 `manifest` 的字段：`candidate_start=32`、`candidate_count=256`、
  `candidate_source=heldout-seed42/ADD/testdata_N32.pt`、`sampling=`
  `"graph_id_range_no_topup"`。
- 逐候选记录 `graph_id, common_complete, T, n_loops, n_chains, n_sinks,
  n_leaves, n_unrelated, early_own_before/after, early_teacher_before/after`
  以及每个模型的 `_a/_b` 资格；不合格候选也保留记录。
- teacher64 与 long32 是从这 256 个候选中按 graph_id 截取的**固定子集**，
  不重新做“选满 64/32 个合格图”的补样。
- 若某子集内合格数不足，报告实际 n，并在该模块内使用实际合格链；不向
  子集外扩。

## 2. 权重、数据与代码身份

| 对象 | 路径 | 身份 |
|---|---|---|
| A′ 权重 | `tmp/runs/A-L4D384fast/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-09-15:42:45/epoch=86-val_loss=0.0000.ckpt` | SHA256 `0b66e196990af8fb8dca3fdb75f4d211fd4a4c56a8496d2a200f765470c1106a` |
| B 权重 | `tmp/runs/B-M6/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa_recorrection-seed-42 - 2026-11-07-17:47:37/epoch=92-val_loss=0.0000.ckpt` | SHA256 `ef58ba261fd0f4677adabec3d8c978758397bb4ee536f09bb833fef8bbde0886` |
| heldout N32 | `tmp/analysis/heldout-seed42/ADD/testdata_N32.pt` | SHA256 `08f52382f6e3a2b62ec50e4603b8a10454cba48e0f5449543dd79c9925cee707`；1024×1024×6；depth 3–11；seed 42；129 vocab |
| 训练数据（审计） | `experiments/data/Tr32Test128-ADD/train_data.pt` | SHA256 `cee36b84b01b7c1fdbd2e32619b402e571275c534a4d685684db07a6e11de1ce`；50000×256×6；max_depth 6 |
| 代码基线 | `reproduction@a4ce43a` + B1 `b_pilot_analysis.py`（未跟踪） | B2 的 selection/subset/summary 代码改动提交后，把最终提交写入 manifest；运行前固定 |

配置/组/run 名沿用 pilot manifest 的 `model_a/model_b`；命令中变量见 §6。
`batch_size=8` 固定，用于 `graph_id = batch_idx*8 + bi` 的现有索引约定；
所有 runner 必须用同一 batch_size，且数据顺序不可 shuffle。

## 3. 模块、条件与资格

### 3.1 M0：固定候选 clean 选样（无注错）

- 对 32–287 的 256 张图跑 A′/B clean free rollout；找第一个双方所有
  numeric VALUE 都正确的 T；只做 common-complete 判定，不做任何注入。
- 同时记录每个 chain 的 own/teacher、before_u/after_u 共同资格。
- 输出：`manifest_main256.json`（含全部候选状态）、
  `manifest_teacher64.json`（仅 32–95）、`manifest_long32.json`（仅 32–63）。
- 本模块不产生恢复率结论，只交合格数和身份。

### 3.2 M1：early 2×2（主分析）

两个时点 × 四个条件：

| 条件 | u 输入 | r 输入 | 用途 |
|---|---|---|---|
| clean | 原值/EMPTY | 原值 | 同起点对照 |
| U-only | a+1（默认 4→5；`--value_mode four_to_five` 时只保留 a=4） | 原值 | 已有值被改坏 |
| R-only | 原值/EMPTY | r_true+3（模 23） | 子节点是否跟随错父 |
| U+R | a+1 | r_true+3 | 冲突时采用谁 |

- own：`before_u` 与 `after_u` 各一次，主分析；A′/B 分开跑。
- teacher：只在前 64 张候选（32–95）上跑 `before_u`/`after_u`，作状态对照；
  使用 `early_teacher_*` 共同资格。
- 资格：`early_own_<timing>` / `early_teacher_<timing>` 为 true 的 chain；
  报告 `_a/_b` 不一致数。某个 chain 在 before_u 合格、after_u 不合格时，
  只进入对应时点，不从另一时点删除。
- 记录：每轮实际输入/输出、r/u/q1/q2、全部后代、node_truth；H=16。
- H=8 由同一批轨迹截取，不重跑。
- 长观察：long32 子集的 own before/after 各跑 H=64；不跑 teacher 64 轮。

### 3.3 M2：complete 目标类型（主分析）

从共同 complete 起点 T 开始，按目标分别注错：

| 条件 | 目标 | 注错 | 真值 |
|---|---|---|---|
| `late_r` | chain 的 r | r_true+3 | 不变 |
| `late_u` | chain 的 u | a+1 | 不变 |
| `sink` | 最深数值 sink | VALUE+3 | 不变 |
| `leaf_value` | 叶变量 v_pos | VALUE+3 | 不变（常数不变） |
| `unrelated` | 与 u/q1/q2 不可达的匹配节点 | VALUE+3 | 不变 |

- 每种条件配同起点 clean；`late_r` 与 `late_u` 共用同一 chain 的 clean
  （按 graph+target 去重），不重复计。
- 资格：common-complete 图；runner 重新 rollout 并要求 T 时全 numeric
  VALUE 正确，否则记 `skipped_not_complete`，不换图补样。
- H=16；H=8 截取。long32 子集同样条件跑 H=64。
- single-parent、q1/q2 不直接读 r 的 primary chain 才进主表；multi-parent
  不混入。

### 3.4 M3：sustained 错父（主分析）

- complete 起点 T；r 强制为 `r_true+3` 共 **4 个 forward**；第 5 个 forward
  起释放。
- 命令的 `--horizon` 是**总轮数**，所以主实验用 `--horizon 20`（4 保持 +
  16 释放后观察），不是 16。长观察子集用 `--horizon 68`（4+64）。
- 配同起点、同总长度的 clean；恢复时间从释放后第 1 轮计；保持窗口单独
  写 `sustained_hold.csv`，不混入 H 表。
- 资格：`early_own_after` 共同 flag（与 pilot 一致）；不是所有 chain。
- H=16/8 从释放后轨迹截取；long32 子集 H=64。

### 3.5 M4：D1/D2/D3（第二部分）

只对 `late.leaves` 的叶变量/常数对：

| 条件 | 起点 | 常数 | 旧状态 | 真值 |
|---|---|---|---|---|
| D1 `leaf_constant` | complete `base` | 改为 new_const | 保留旧叶/下游 | 重算 new truth；同时记 old truth |
| D2 `leaf_constant_scratch` | 原始 token，所有 numeric 节点 EMPTY | 改为 new_const | 无 | 重算 new truth；`horizon=max(16,n_loops)` |
| D3 `leaf_constant_pinned` | complete `base` | 每轮强制 new_const | 保留旧状态 | 重算 new truth；H=16 |

- D1/D2/D3 的 `new_truth` 必须同时包含 `const_pos` 和所有可达节点；现有
  runner 已写入 `new_truth`/`output_nodes_new`，但汇总尚未实现（§7）。
- D1/D3 的 clean 与 `leaf_value` 共用；D2 无 clean（从空题面开始）。
- 报告 old truth 与 new truth **分开**：不能把“旧答案未更新”直接判成错，
  也不能把“新题不会做”与“不会从旧答案更新”混在一起。
- 资格：common-complete 图的 leaf 目标；T 重新校验；不补样。

### 3.6 M5：训练腐蚀审计（可与 M1 并行，CPU/GPU 均可）

- 数据：B 的训练数据 `experiments/data/Tr32Test128-ADD/train_data.pt`。
- 用 `last` 策略导出所有 candidate 的**腐蚀前** VALUE，并单独采样正常
  `corruption_prob=0.05`、`tokenwise=0.5` 的实际替换。
- 固定预算：`--n_batches 256 --trials 100 --seed 20261114`，即 256×8=2048
  张训练图（约 4.1% 的 50k），报告覆盖比例；不称为“全训练审计”。
- 输出按 **graph(batch,b)、row/轮次、pos** 聚合：candidate 数、before
  `EMPTY/NA/numeric/other`、after 类别、`after_is_true`、`before numeric→after
  numeric`、明确计数 `before=4,after=5`；并保留原始 row 样本。
- 目的：判断训练腐蚀是 `EMPTY→随机值` 为主，还是也有 `已填值→错误值`；
  不直接推断 4→5 的测试行为。

### 3.7 后续（不在本批执行）

- 跨 seed：cyhB s1234/s2025（按用户后续决定）；N128 先小批 32 图、64 轮
  校准 H，再定其观察窗；局部自洽仍 DEFERRED，switch 暂不纳入必跑。

## 4. 运行矩阵与预期记录数

记号：`C` = 入选 chain 数（≤2×256=512）；`S/L` = sink/leaf 目标数
（各≤512）；`U` = unrelated 目标数（≤C）；`E_b/E_a` = own before/after
共同合格 chain 数；`E_tb/E_ta` = teacher64 子集共同合格 chain 数；
`E_a^L` = long32 子集的 after 合格 chain 数。实际值由 M0 manifest 填入；
下表上限是 **k=2、所有候选都合格** 的极端值，不是预期产出。

| 运行（每个模型一条） | 记录数（轨迹） | 上限/模型 | 说明 |
|---|---:|---:|---|
| early own before_u，H=16 | 4·E_b | 2048 | clean/U/R/U+R |
| early own after_u，H=16 | 4·E_a | 2048 | 同上 |
| early teacher before_u，H=16 | 4·E_tb | 512 | 仅 32–95 |
| early teacher after_u，H=16 | 4·E_ta | 512 | 仅 32–95 |
| complete 第一部分，H=16 | 3C+2S+2L+2U | 4608 | `late_r,late_u,sink,leaf_value,unrelated` + clean |
| sustained，总 20 轮（4+16） | 2·E_a | 1024 | clean + injected |
| D1/D2/D3，H=16/`max(16,n_loops)` | 4L | 2048 | 3 个 leaf condition + clean |
| long32 early，各 timing，H=64 | 4·E_b^L / 4·E_a^L | 256 each | 固定 32–63 |
| long32 complete，H=64 | 3C^L+2S^L+2L^L+2U^L | 576 | 固定 32–63 |
| long32 sustained，总 68 | 2·E_a^L | 128 | 固定 32–63 |

A′/B 各跑一份；主表分开，不合并。M0 的 manifest 生成后，把上表的 `C,S,L,
U,E_*` 实测值写回本单的“实际合格数”表（模板见 §8），再启动注错运行。

**early vs complete 直接比较**：只用同一 `graph_id + event_id` 且 early 与
complete 都合格的 chain 交集；在汇总中输出 `n_pairs`、A′/B 各自 n 和交集
n。不能拿 early 的全集和 complete 的全集直接相减。

## 5. 资格与分母口径（主实验沿用 pilot 结论）

- 每个 H 并列 `observed / damaged / stable`；`stable` 分母是 damaged，
  不是 observed。
- 全窗口逐轮为主；稳定恢复 = 末尾连续 2 轮正确（L=2）；曾恢复、仅末轮
  回正单列。
- direct 单列“实际注错数 / 首轮修回数 / 首轮仍错数”；B 的 direct 输出
  0 损伤不等于 0 恢复率。
- VALUE 与 first4 并列；逐图等权为主，节点加权为辅；零分母单列。
- clean 分支单独报告漂移；sustained 的 H 从释放后计。
- 不把 pilot 的 42 targets 当作本批的合格数；本批按 256 候选重新统计。

## 6. 命令

以下命令在仓库根目录执行；先设固定变量，GPU 只用空闲卡（示例 GPU4），
不占 seed2026 的 GPU0。`--num_threads 4` 显式传入；每条线同一时间只跑
一个任务。输出目录必须不存在（runner 会拒绝覆盖）。

```bash
cd /workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

DS='tmp/analysis/heldout-seed42/ADD/testdata_N32.pt'
TRAIN='experiments/data/Tr32Test128-ADD/train_data.pt'
A_CFG='A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa'
A_GROUP='Nodes32-ADD - L4H16D384_DeBERTa'
A_RUN='seed-42 - 2026-11-09-15:42:45'
A_BASE='tmp/runs/A-L4D384fast/checkpoints'
A_CKPT='epoch=86-val_loss=0.0000.ckpt'
B_CFG='B-M6-Nodes32-ADD-L4H16D384-DeBERTa'
B_GROUP='Nodes32-ADD - L4H16D384_DeBERTa_recorrection'
B_RUN='seed-42 - 2026-11-07-17:47:37'
B_BASE='tmp/runs/B-M6/checkpoints'
B_CKPT='epoch=92-val_loss=0.0000.ckpt'
OUT='tmp/analysis/b-main-20261114'
```

### M0：固定候选 selection（未实现，见 §7）

```bash
python -I tools/b_mechanism/b_mechanism_select.py \
  --ds_file "$DS" --batch_size 8 --k 2 --extra_rounds 4 \
  --candidate_start 32 --candidate_count 256 --selection_mode fixed --no_topup \
  --a_config "$A_CFG" --a_group "$A_GROUP" --a_run "$A_RUN" \
  --a_base "$A_BASE" --a_ckpt "$A_CKPT" \
  --b_config "$B_CFG" --b_group "$B_GROUP" --b_run "$B_RUN" \
  --b_base "$B_BASE" --b_ckpt "$B_CKPT" \
  --out "$OUT/manifest/manifest_main256.json" \
  --num_threads 4
```

本命令只做 clean rollout，不做注错。执行单中的 `--candidate_start`、
`--candidate_count`、`--selection_mode fixed`、`--no_topup` 是待实现的接口
名；实现后以实际 `--help` 为准，其余参数不变。

### M1：early own（A′/B 各两条；以 A′ after_u 为例）

```bash
python -I tools/b_mechanism/b_mechanism_early.py \
  --manifest "$OUT/manifest/manifest_main256.json" --ds_file "$DS" \
  --config "$A_CFG" --group "$A_GROUP" --run "$A_RUN" \
  --base "$A_BASE" --ckpt "$A_CKPT" \
  --start own --timing after_u --conditions clean,u,r,ur \
  --value_mode a_plus_1 --horizon 16 --num_threads 4 \
  --out "$OUT/early/A_own_after_u_H16.json"
```

对 A′/B × `before_u/after_u` 重复；teacher 用
`--start teacher` 且 `--manifest manifest_teacher64.json`，同样四个条件、
H=16；long32 用 `--manifest manifest_long32.json`、`--horizon 64`，只跑
own 的 before_u/after_u。

### M2：complete 第一部分（A′/B 各一条）

```bash
python -I tools/b_mechanism/b_mechanism_late.py \
  --manifest "$OUT/manifest/manifest_main256.json" --ds_file "$DS" \
  --config "$B_CFG" --group "$B_GROUP" --run "$B_RUN" \
  --base "$B_BASE" --ckpt "$B_CKPT" \
  --conditions late_r,late_u,sink,leaf_value,unrelated \
  --delta 3 --u_delta 1 --value_mode a_plus_1 --horizon 16 --num_threads 4 \
  --out "$OUT/late/B_complete_part1_H16.json"
```

A′ 换对应变量和文件名。long32 子集把 `--manifest` 换成
`manifest_long32.json`、`--horizon 64`，输出到 `late/long32_*`。

### M3：sustained（A′/B 各一条）

```bash
python -I tools/b_mechanism/b_mechanism_controls.py \
  --manifest "$OUT/manifest/manifest_main256.json" --ds_file "$DS" \
  --config "$B_CFG" --group "$B_GROUP" --run "$B_RUN" \
  --base "$B_BASE" --ckpt "$B_CKPT" \
  --mode sustained_r --hold_rounds 4 --horizon 20 --delta 3 --num_threads 4 \
  --out "$OUT/controls/B_sustained_H20.json"
```

A′ 换变量/文件名。long32 子集：`--manifest manifest_long32.json
--horizon 68`。

### M4：D1/D2/D3（第二部分，A′/B 各一条）

```bash
python -I tools/b_mechanism/b_mechanism_late.py \
  --manifest "$OUT/manifest/manifest_main256.json" --ds_file "$DS" \
  --config "$B_CFG" --group "$B_GROUP" --run "$B_RUN" \
  --base "$B_BASE" --ckpt "$B_CKPT" \
  --conditions leaf_constant,leaf_constant_scratch,leaf_constant_pinned \
  --delta 3 --horizon 16 --num_threads 4 \
  --out "$OUT/late/B_D123_H16.json"
```

D2 的 `horizon` 由 runner 取 `max(16, n_loops)`；D3 用 H=16。D1/D3 的
clean 可能与 M2 的 `leaf_value` clean 重复；该文件独立汇总，不跨文件合并
clean。

### M5：训练腐蚀审计（B 权重，CPU/GPU 均可）

```bash
python -I tools/b_mechanism/b_corruption_audit.py \
  --ds_file "$TRAIN" \
  --config "$B_CFG" --group "$B_GROUP" --run "$B_RUN" \
  --base "$B_BASE" --ckpt "$B_CKPT" \
  --batch_size 8 --n_batches 256 --trials 100 --seed 20261114 \
  --out "$OUT/audit/B_train_corruption_256b.json"
```

固定 256 个 batch（2048 图，4.1% 训练集）；不称全训练。若实际耗时明显
超过预算，先缩到 64 batch 并在报告注明覆盖，不改变其他模块。

## 7. 尚缺实现（执行前必须补齐）

1. **selection 固定区间/不补样**：当前 `b_mechanism_select.py` 从 graph_id 0
   开始、选满 `n_graphs` 即停，不能从 32 开始且会把 pilot 0–31 混入。需加
   `--candidate_start`、`--candidate_count`、`--selection_mode fixed`、
   `--no_topup`；逐候选写状态（含不合格），扫描 256 后停止，不向 288 后
   补样。`graph_id = batch_idx*batch_size+bi` 的索引约定要在 batch_size=8
   下断言/校验。
2. **子集 manifest 生成器**：从主 manifest 按 graph_id 区间（32–95、32–63）
   生成 `manifest_teacher64.json`、`manifest_long32.json`；保留 chain 的
   early/teacher flag，不重新选样。建议独立小脚本，不放 runner 里。
3. **主实验汇总**：现有 `b_mechanism_horizon_summary.py` 只识别 pilot 的
   `after_u_U/R、complete_U/R、sustained_R` 五类，且 role 以 chain u 为
   中心；不能直接汇总 `before_u`、`ur`、`teacher`、`sink`、`leaf_value`、
   `unrelated`。需新增/扩展 `b_main_summary.py`：
   - early：before/after × clean/U/R/U+R，own/teacher 分开；
   - complete：internal U/R、sink、leaf_value、unrelated，各自 direct/
     downstream/无关节点；
   - sustained：释放后 H=8/16/64，保持窗口单独；
   - observed/damaged/stable、direct 实际注错/首轮修回、graph-equal/
     node-weighted、VALUE/first4、零分母、clean 漂移；
   - early vs complete 共同合格交集 `n_pairs`。
4. **D1/D2/D3 新旧真值汇总**：现有 `b_mechanism_summary.py` 明确跳过
   `leaf_constant*`；需按 `truth_ref=old` 与 `output_nodes_new` 分开统计
   old-truth/new-truth 的 correct/wrong/stable，并报告 `const_pos` 的新
   真值。D2 从空题面开始，单独报完成轮次，不与 D1/D3 混算。
5. **腐蚀审计聚合**：当前 `b_corruption_audit.py` 只输出总数
   `candidates/actual` 和前 2000 行样本；需增加按 `(batch,b,row)`（图/轮次）
   和 `pos` 的聚合、`before/after` 转移矩阵、`before=4,after=5` 精确计数、
   覆盖比例与 depth 分层。
6. **主实验 manifest 覆盖率表**：把 `scanned=256`、`common_complete`、
   `chains`、`sinks/leaves/unrelated`、`early_own_before/after`、
   `early_teacher_before/after`、`_a/_b` 不一致数、teacher64/long32 实际
   n 写入 CSV/报告；当前 selection 的 counts 只有聚合，缺逐候选表。
7. **报告模板**：`b-main-report.md` 需分阶段产出，至少包含：
   - 样本身份/合格数/与 pilot 的重叠核验；
   - 每个模块的条件表、分母表、逐轮曲线、代表性轨迹；
   - early vs complete 交集表；
   - D1/D2/D3 old/new truth 表；
   - 审计的 EMPTY vs numeric 表；
   - 明确的 NOT RUN / 待定项。

## 8. 输出布局与报告

```
tmp/analysis/b-main-20261114/
  manifest/    manifest_main256.json, manifest_teacher64.json, manifest_long32.json
  early/       A_own_before_u_H16.json, A_own_after_u_H16.json, B_..., teacher_...
  late/        A_complete_part1_H16.json, B_..., A_D123_H16.json, B_...
  controls/    A_sustained_H20.json, B_..., long32_...
  audit/       B_train_corruption_256b.json
  summary/     early/, complete/, sustained/, d123/, audit/, tables/*.csv
  logs/        one log per run
docs/records/2026-11-14/b-main-report.md
```

- 原始 JSON/JSONL 只留在 `tmp/`，git 只提交路径、命令、哈希、汇总 CSV 和
  报告；不提交权重/大文件。
- 每个 runner 输出独立文件；summary 保留 source 身份，A′/B、own/teacher、
  H、条件不互相串。
- 运行前把最终代码提交、manifest SHA256、实际合格数补进本执行单；运行后
  在 board/log 记启动、完成和关键结果。

## 9. 停跑/不启动条件

- selection 实际扫描不是 32–287、或出现向 288 后补样：停止，修代码后重跑；
- manifest 中 A′/B 数据/权重 SHA256 不一致、T 越界、early flag 缺失或
  `_a/_b` 与 common 不一致：停止；
- 256 张候选中 common-complete 过少（例如 < 128）：不自动扩样，先报告
  实际 n 并请用户决定是否接受缩小的样本；
- 任何 runner 的 `skipped_not_complete` 异常高：先查数据/索引，不换图；
- GPU 与 C 线/seed2026 冲突、或线程数超过 4：排队，不并行抢资源；
- 未完成 §7 的汇总实现前，可以跑 M0（无注错），但不把 raw JSON 当科学
  结果；不启动正式注错主实验，直到用户对执行单和预算做最后确认。

## 10. 需要用户最后确认的点

1. 256 张候选（32–287）与 teacher64（32–95）、long32（32–63）是否按本单
   固定；是否接受“不做独立 heldout、只保证与 B pilot 不重叠”的表述。
2. 每图 `k=2` 是否同时限制 sink/leaf 各 2 个；若希望 complete 目标类型
   每图更多，需要把 chain k 与 late target k 拆开（当前实现是一个 k）。
3. D1/D2/D3 是否只对 `late.leaves` 的每图至多 2 个叶目标；是否需要在
   主实验第一部分之后单独交付。
4. 审计预算 256 batch / 100 trials（4.1% 训练集）是否接受；若要求更大
   覆盖，需增加运行时间。
5. H=16 主、H=8 报告、long32 64 轮是否按本单执行；sustained 主实验总
   horizon 20（4+16）是否正确。
