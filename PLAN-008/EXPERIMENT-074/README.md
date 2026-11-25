# 整图无正确值时的恢复

第一次前向之前，整张图没有任何一个节点符合父节点计算的要求：每个结果格要么留空，要么写着一个错数。这是实验 073 的收紧版本——073 的全预填里按数值均抽会混进碰巧正确的值（约 1/23），本实验把它们全部排除。

## 初始状态怎么构造

对每个节点，取以下两种状态之一。

- **空**：结果格留空。
- **错**：写入一个数 v，满足 v ≠ 该节点真值，且当该节点的父值足以推出数值时，还要求 v ≠ 该父值推出的数。父值不足以推出数值（有父节点留空）时，只约束 v ≠ 真值。

父值推出的数定义为：若该节点全部父节点的结果格都非空，则为其父值之和模 23；否则无从推出。按此定义，凡写入错数的位置，都与父节点当前的值不一致；留空的位置则尚未表态。构造按节点的拓扑顺序进行，因此一个节点的父值在它自己被写入前已经确定。

空与错的比例按档位扫描：全错（没有空位）、四分之一空、一半空、四分之三空。全空的原题不入本批，它由实验 073 的全空记录提供，作为上界对照。

预填数字保留均抽以外的选择空间：排除真值与父值推出的数后，其余取值等概率抽取。

## 与 072、073 的关系

沿用实验 072 的底图、两种排列（并排与交错）和同一批五份权重（A′ 42、B 42，原始 C 的 42、1234、2026）。三组实验因此可在同一题、同一初值起点上直接对照。

- 实验 072 只改一个位置，测扰动如何传播。
- 实验 073 改任意位置，但允许碰巧正确。
- 本实验改任意位置且保证无一正确，测"没有任何可信起点时能否恢复"。

## 测什么

1. **整图能否恢复**，需要多少轮，有多少张图在观察窗内始终没有全对。
2. **恢复如何推进**：按节点深度分层看逐轮正确率，检查恢复是否自源节点向外逐层展开，还是各处独立发生。源节点的真值在题面中始终可读，是可用的锚点。
3. **起初空与起初错的结局是否不同**：留空的位置何时被填对；写错的位置是否原样保留、是否被改成另一个错数、是否被改成父值推出的那个数（即跟随父值）。
4. **与同图全空、以及 073 全预填的对照**：同题同位置上的得失。

逐题轨迹保留每轮全部节点的数值读出，分层计数表另含节点角色（源节点、有下游的中间节点、无下游的非源节点）与父值证据类别，可复算细分分母。

## 运行

从仓库根目录依次执行。默认 checkpoint 位于本仓库的 tmp/runs，可用 `--data-root` 指定实际仓库位置。

```bash
python PLAN-008/EXPERIMENT-074/allbad.py --check-design
python PLAN-008/EXPERIMENT-074/allbad.py --smoke
python PLAN-008/EXPERIMENT-074/allbad.py
```

输出在 tmp/analysis/plan008-exp074 的 smoke 与 formal 子目录。

## 归档结果

2026-11-26 执行正式批次（GPU 2，`CUDA_VISIBLE_DEVICES=2`），exit 0，用时 1171 秒。产出归档在 [results/formal](results/formal)，逐文件来源与校验值见[文件清单](files.tsv)；运行时该实验脚本尚未入库，其余代码为 `041ad0a` 的工作树状态，脚本内容随本次提交一并入库。

- 汇总表：[summary.csv](results/formal/summary.csv)（含按深度与起初状态的分解）
- 五份分层计数据表：`results/formal/<模型>-strata.csv`
- 五份压缩逐题轨迹：`results/formal/<模型>.jsonl.gz`

数值读数以这些结果文件为准，主文档只记录执行状态与观察。

## 数据结构（供复算）

`results/formal/<模型>.jsonl.gz` 每行一个 JSON，一个 `(size, gid, order)` 下有 13 行 = 1 行全空对照 + 4 种条件 × 3 次抽样。

- **`condition="empty"` 的行**：含 `nodes` 与 `tokens`。`nodes` 每项有 `name`、`pos`、`truth`、`parents`（节点下标）、`depth`、`slot`、`stale`、`version`。同一 `(size, gid, order)` 的其余各行共用这份 `nodes`。
- **`condition="empty0"/"empty25"/"empty50"/"empty75"` 的行**：含 `draw`、`blank_count`、`initial`、`trace`，**不含 `nodes`**；须按 `(size, gid, order)` 与上面那份 `nodes` 对齐后使用。
- `initial`：长度等于节点数的初始 VALUE 数组，`24` 表示空，`0`–`22` 为写入的数值。
- `trace`：24 轮 × 节点数 的数值读出，`trace[t-1][i]` 为第 `t` 轮节点 `i` 读出。
- 各行另有 `model`、`sha256`（对应 checkpoint）、`size`、`gid`、`order`、`draw`。

`results/formal/<模型>-strata.csv`：按 `(size, gid, order, condition, draw, depth, role, initial_class, parent_evidence)` 的原始计数。
`results/formal/summary.csv`：按 `(model, size, order, condition, initial_class, depth)` 汇总，`correct_r{1,2,3,5,9,24}` 为对应轮次的正确比例，`same_initial_wrong_r24`、`correct_last_two`、`clean_correct_r*` 等见列名。
