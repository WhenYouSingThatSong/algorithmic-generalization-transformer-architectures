# C 自然错误局部修复：首批执行记录

日期：2026-11-15 02:56 CST。  
状态：**CPU 选样、s42/s1234×2 case×4 轮 smoke、两 seed 固定清单×16 轮 main 均已运行；不是 C+、不是 B 实验，也不是总体发生率估计**。  
代码：`reproduction@95611a1`（GPT 新增 `tools/repair_study/`，本轮未改语义）。  
上游：[行为定义](../../../tools/repair_study/README.md)、[执行交接](../../../ai-collab/handoff/2026-11-15-0135-bc-repair-study.md)。

## 1. 选样

使用旧 C0 s42/e143、s1234/e143 的现有六档 NPZ 与原始 PT，不加载模型。

| 项目 | 数值 |
|---|---:|
| 扫描图 | 768（2 seeds × 6 strata × 64） |
| 候选事件 | 48,523 |
| `post_complete` 事件 | 40,708 |
| `not_yet_complete` 事件 | 7,815 |
| 缺失 donor/source | 68（均为 `not_yet_complete`） |
| 太晚（不足 2 轮原记录可重放） | 817 |
| selectable | 40,014 |
| 选中固定 case | 57 |
| 缺少健康对照 | 0 |
| 运行 records（case × natural/control） | 114 |

每权重选中的 case：s42 27、s1234 30。分组为
`2 seeds × 6 strata × (value_parent_correct / value_parent_wrong /
variable_only) × 自然结局`；固定 seed `20261115`，每格 1 个不同图；
不足如实保留，不补样、不看干预结果重选。

**3 个自然缺组**（没有符合条件的选中事件）：

- s42 / N32_shallow / `value_parent_correct:final_two_correct`
- s42 / N32_shallow / `value_parent_wrong:final_two_correct`
- s42 / N64_shallow / `value_parent_wrong:final_two_correct`

小表：`c-local-repair-selection-events.csv`、`c-local-repair-selection-groups.csv`。
完整候选与选样计划：`tmp/analysis/c-local-repair-20261115/selection/`。

## 2. 重放与运行

运行命令（实际使用 `python -I -B`；smoke 在 GPU6/7，main 两 seed 串行）：

```bash
python -m unittest discover -s tools/repair_study/tests -v

python tools/repair_study/select_c.py \
  --run_dir tmp/analysis/c0-capability-20261114/runs \
  --data_dir tmp/analysis/c0-capability-20261114/data \
  --out_dir tmp/analysis/c-local-repair-20261115/selection

CUDA_VISIBLE_DEVICES=6 python tools/repair_study/run_c.py \
  --cases tmp/analysis/c-local-repair-20261115/selection/cases.json \
  --weight s42_e143 --limit_cases 2 --horizon 4 \
  --out_dir tmp/analysis/c-local-repair-20261115/smoke_s42 --device cuda:0
# s1234 同参数，GPU7，out_dir smoke_s1234

CUDA_VISIBLE_DEVICES=6 python tools/repair_study/run_c.py \
  --cases tmp/analysis/c-local-repair-20261115/selection/cases.json \
  --weight s42_e143 --horizon 16 \
  --out_dir tmp/analysis/c-local-repair-20261115/s42 --device cuda:0
CUDA_VISIBLE_DEVICES=7 python tools/repair_study/run_c.py \
  --cases tmp/analysis/c-local-repair-20261115/selection/cases.json \
  --weight s1234_e143 --horizon 16 \
  --out_dir tmp/analysis/c-local-repair-20261115/s1234 --device cuda:0

python tools/repair_study/summarize.py \
  --inputs tmp/analysis/c-local-repair-20261115/s42/records.jsonl \
           tmp/analysis/c-local-repair-20261115/s1234/records.jsonl \
  --out_dir tmp/analysis/c-local-repair-20261115/summary
```

结果：

- 8 项 repair-study CPU 测试通过。
- smoke_s42 / smoke_s1234：各 2 case，status `complete`，各 4 records。
- s42：27 case，status `complete`，54 records；s1234：30 case，status
  `complete`，60 records。全部 coverage `eligible`。
- 未生成 `replay_failure.json`。runner 在选中图的原批次中重放全部 64 轮，
  逐轮核对四属性与旧 NPZ；不改动续算和 sham 分支也做了逐轮一致性检查；
  失败会直接停止。hidden 快照保存在各输出目录的 `states/*.pt`（仅 tmp）。

## 3. 初步结果

### 3.1 自然错误现场（target VALUE）

下表只汇总 `natural_error_state`、target VALUE；每格是“first forward
正确 / 最后两轮均正确”的 case 数。`initial` 是替换后尚未继续 forward 的
读出，单独列出，**不当作修复成功**。

| 类别 | cases | unchanged first/final2 | restore_target first/final2 | restore_parents first/final2 | restore_target initial |
|---|---:|---:|---:|---:|---:|
| value_parent_correct | 23 | 0 / 7 | 0 / 7 | 0 / 7 | 23 / 23 |
| value_parent_wrong | 22 | 0 / 6 | 8 / 6 | 3 / 6 | 22 / 22 |
| variable_only | 12 | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |

- `restore_target` 的 initial 全部正确是“把来源目标向量写回后立即读出正确”，
  但 value_parent_correct 的第一轮 forward 仍 0/23；说明初始读出不能当作
  后续计算修复。
- 在 value_parent_wrong 中，restore_target 把第一轮 forward 从 0/22 提到
  8/22，restore_parents 提到 3/22；但最后两轮正确数仍都是 6/22，与
  unchanged 相同。**本批没有看到 target 的稳定末两轮改善**。
- variable_only 的 VALUE 四分支都保持 12/12；first4 的 restore_target
  initial 为 12/12，但第一轮 forward 0/12、最后两轮 0/12；变量名读出
  没有因一次目标向量替换而持续修复。

### 3.2 健康对照

同一批 case 的健康匹配 lag 对照（23 个 `value_parent_correct` 现场）中，
VALUE initial 四分支均为 23/23；unchanged 第一轮 12/23、restore_target
18/23、restore_parents 15/23，但最后两轮分别为 9/23、9/23、10/23。
健康对照也能出现“首轮被改变”，因此 **value_parent_wrong 的首轮提升
不能单独解释为错误特异修复**。

`node_curves.csv` 的 all-node（rounds 15–16）汇总中，restore_target 的
helped/harmed 均为 0；restore_parents 只有少量节点轮次变化（例如 s1234
value_parent_correct 3 helped / 9 harmed、value_parent_wrong 9 helped /
4 harmed；s42 value_parent_wrong 1/2）。这些是节点轮次计数，不是独立图数，
需逐案例复核。

### 3.3 两条可核对轨迹

1. `s42_e143 / N128_shallow:0019 / pos 678 / state 23 / parent_wrong`
   （case id `C:s42_e143:N128_shallow:19:678:23:value_parent_wrong`，真值 4）：
   - unchanged：initial 11，16 轮输出全为 11；
   - restore_target：initial 4、第一轮 forward 4，从第二轮起回到 11；
   - restore_parents：initial 11，16 轮全为 11。
   这是“写回后当下正确、下一轮即丢失”的典型轨迹。

2. `s1234_e143 / N32_shallow:0014 / pos 170 / state 53 / parent_wrong`
   （真值 0）：
   - unchanged：initial 4、前两轮 4，随后 0，最后两轮回到 19；
   - restore_target：initial 0，前 14 轮 0，最后两轮 19；
   - restore_parents：前两轮 4，第 3–16 轮 0（这是本批少见的 restore_parents
     末两轮正确案例）。
   这只是 57 个固定 case 中的一条；不能据此估计修复率。

### 3.4 支持数（仅诊断样本）

- `restore_target`：first-forward 改善 8 个 case，但 final2 改善 0、恶化 0。
- `restore_parents`：first-forward 改善 3 个 case，final2 改善 1、恶化 1。
- 这些是分层诊断样本的 case 计数，不是总体概率；自然结局分组本身也会
  决定 final2 的分母。

## 4. 边界

- 本次只验证“一次性替换目标/父向量后自由续算 16 轮”；没有逐轮钳制、
  没有解码字段回灌、没有 C+ 训练。
- “初始正确”不等于后续 forward 正确；“第一轮正确”也不等于持续修复。
- 替换整个向量会同时带入变量名、位置和其它因素；即使有效也不能直接
  说修好了 VALUE 或变量名编码。
- 没有做 B 结果解释；B 使用 GPU1，C 在 smoke/main 后已释放 GPU6/7。
- 原始 NPZ/PT、完整 records 和 hidden 快照留在 tmp，不进入 git。

## 5. 产物

- 选样：`tmp/analysis/c-local-repair-20261115/selection/`
- smoke：`.../smoke_s42/`、`.../smoke_s1234/`
- main：`.../s42/`（54 records）、`.../s1234/`（60 records）
- summary：`.../summary/`（`coverage.csv`、`target_cases.csv`、
  `target_summary.csv`、`node_curves.csv`）
- 小型汇总表：`docs/records/2026-11-15/c-local-repair-key-metrics.csv`、
  `c-local-repair-selection-events.csv`、`c-local-repair-selection-groups.csv`
- 本轮没有修改原训练、原 B/C runner 或原输出目录。
