# C0 s2026/e143 六档能力补充记录

日期：2026-11-15 17:07 CST。  
状态：**s2026/e143 六档已有 NPZ 的补充汇总；此前只在 tmp 运行、未提交，现补入仓库**。  
权重：`seed-2026`，`epoch=143-val_loss=0.0000.ckpt`；SHA256
`40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780`。  
运行：`tmp/analysis/c0-capability-20261114/runs/s2026_e143/`，384 图 × 64 轮，
GPU6，95.2 s；无 Traceback/OOM。

## 1. 口径

- 数据、采样和指标与 `c0-capability-report.md` 的 s42/s1234 六档完全一致；
  这里只补 s2026，不改旧报告表格。
- 检查点选择沿用固定 **e143**，以便与 s42/s1234 的 C 记录按 epoch 对齐。
  s2026 的最终 checkpoint 是 e150（val 0.166），最佳 val 在 e136（0.00879）；
  **本记录不是 final/best-val 敏感性实验**，也没有按本批结果挑 checkpoint。
- 64 轮自由续算；解码仅观察，hidden 回灌；tV/t4、节点 VALUE/first4、
  graph-equal 与节点加权均在 summary CSV 中。

## 2. 完成与 H 切片

| 档 | VALUE completed | tV median | first-4 completed | H16 graph VALUE | H64 graph VALUE | H16 internal VALUE | H64 internal VALUE | H64 sink VALUE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N32_shallow | 64/64 | 7 | 64/64 | 1.0000 | 0.6406 | 1.0000 | 0.9309 | 0.9152 |
| N32_deep | 63/64 | 8 | 63/64 | 0.9844 | 0.5000 | 0.9982 | 0.9135 | 0.8828 |
| N64_shallow | 64/64 | 7 | 64/64 | 1.0000 | 0.0625 | 1.0000 | 0.8473 | 0.7732 |
| N64_deep | 64/64 | 10 | 64/64 | 1.0000 | 0.0469 | 1.0000 | 0.8214 | 0.7597 |
| N128_shallow | 64/64 | 7 | 64/64 | 1.0000 | 0.0000 | 1.0000 | 0.7546 | 0.6445 |
| N128_deep | 64/64 | 12 | 64/64 | 0.9844 | 0.0000 | 0.9996 | 0.6785 | 0.5680 |

- s2026 的首次 VALUE 完成率很强：除 N32_deep 63/64 外，其余五档 64/64；
  H16 整图 VALUE 全对率为 0.984–1.000。
- 但 H64 整图 VALUE 全对率大幅下降（N64/N128 接近 0），internal/sink
  节点 VALUE 仍高于整图率；这与 s42/s1234 的“短轮数能力 vs 超训练轮数
  稳定性”分离一致。first4 在 H64 约 0.021–0.044，仍由 VARIABLE 读出主导。
- 完成后的末轮仍全对图数：N32_shallow 41/64、N32_deep 32/63、
  N64_shallow 4/64、N64_deep 3/64、N128_shallow 0/64、N128_deep 0/64。
  详细逐图漂移见 `c0-capability-s2026-drift.csv`。

## 3. 与 s42/s1234 的并排参考

只列同一 e143、同一数据、同一 64 轮的 H16/H64 整图 VALUE：

| 档 | s2026 H16 | s42 H16 | s1234 H16 | s2026 H64 | s42 H64 | s1234 H64 |
|---|---:|---:|---:|---:|---:|---:|
| N32_shallow | 1.000 | 1.000 | 0.969 | 0.641 | 0.734 | 0.469 |
| N32_deep | 0.984 | 0.938 | 0.984 | 0.500 | 0.578 | 0.531 |
| N64_shallow | 1.000 | 0.984 | 0.891 | 0.063 | 0.266 | 0.063 |
| N64_deep | 1.000 | 0.969 | 0.844 | 0.047 | 0.313 | 0.109 |
| N128_shallow | 1.000 | 0.922 | 0.625 | 0.000 | 0.047 | 0.000 |
| N128_deep | 0.984 | 0.906 | 0.438 | 0.000 | 0.016 | 0.000 |

- s2026 在 H16 的完成能力（尤其 N128）高于 s42/s1234；H64 整图 VALUE
  并未同步改善，N32/N64 仍明显漂移。单 seed、固定 e143，不能据此做
  总体模型排序。
- s2026 的完成率与 H16 结果好，可能部分来自该 seed/该 epoch；e136
  best-val 与 e150 final 的敏感性尚未运行，不能把 e143 写成最佳 checkpoint。

## 4. 产物

- 运行 records：`tmp/analysis/c0-capability-20261114/runs/s2026_e143/`
- 三权重汇总：`.../runs/summary-all3/`（含 s2026 行）
- 本记录小表：
  - `docs/records/2026-11-15/c0-capability-s2026-completion.csv`
  - `c0-capability-s2026-capability_by_H.csv`
  - `c0-capability-s2026-by_node_type.csv`
  - `c0-capability-s2026-drift.csv`
- 原始逐图 NPZ、`run_meta.json`、`index.jsonl`、logs 和三权重 summary 已打包为
  `ai-collab/artifacts/c0-capability-s2026-e143-raw.zip`（1.4 MB，405 files），
  并附 `.sha256`；SHA256 为
  `b593537633e50c4b46dab88c834e3e23c77c807d1a530821a7d7a6a064ea4681`。
  包内目录为 `c0-capability-s2026-e143/{run,summary,docs,logs}`，与
  `README_RAW_DATA.md` 的说明一致；解包后不放回 `tmp/` 参与重算。
- 未训练 C+；没有因为补档改旧 s42/s1234 表格。
