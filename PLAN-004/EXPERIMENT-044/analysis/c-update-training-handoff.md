# C 旧批训练完成：四份 final 交接给 Agent B

日期：2026-11-20。状态：**旧 `c_update` 四次 50 epoch 训练全部完成；正式 evaluate 未由 C 运行，交由 B**。新 `c_multitime` 任务尚未开始，等待本交接。

## 身份与唯一 plan

- 代码提交：`35edb21`（训练/小测使用的代码身份；每份 `meta.json` 内有完整 code hash 列表）。
- plan：`tmp/analysis/c-update-20261119/data/plan.json`
  - SHA256 `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`
- run 根目录：`tmp/analysis/c-update-20261119/run`
- 训练/评估分工：C 已完成训练及自动阶段 dev 小评估；B 使用同一 plan 和 run 根目录执行正式 evaluate，不重新生成 plan/数据，不重跑阶段 dev。

## 四份 final

| 名称 | 路径 | final SHA256 | meta SHA256 | status |
|---|---|---|---|---|
| C_s2026 correction | `run/train/C_s2026-correction/final.ckpt` | `e78c8ac75c63aa8654e7dc3898c449f9c828f8c7386262056ec495f6f003e007` | `4c62e701eb26c93c41eb9160072d0713d1a38023d0476f22e2483180494daf04` | complete |
| C_s2026 update | `run/train/C_s2026-update/final.ckpt` | `c6ba26a0faed6185eed1685266bcbffd847c699215a44baf377eb659c0e7d107` | 见 JSON | complete |
| C_s42 correction | `run/train/C_s42-correction/final.ckpt` | `88430930dac93c70363715c4217f4d611cee3e7c1ee14a061fe55f5868641b7e` | 见 JSON | complete |
| C_s1234 correction | `run/train/C_s1234-correction/final.ckpt` | `b9ad22aa69b4ed0676dcbbbb0604f9a4c2f17b989f24afc1c28f98a4db2c459d` | 见 JSON | complete |

机器可读版本：`docs/records/2026-11-20/c-update-training-handoff.json`（含每份 final 大小、meta/dev/audit hash、完整 identity、code hashes、dev-50 clean 行和实际 dose）。

## 实际训练剂量

四份都是 50 epoch、78,200 batches、2.5M 图呈现、5% 启用批次：

| 版本 | enabled | source acceptance | accepted candidates |
|---|---:|---:|---:|
| C_s2026 correction | 4.96% | 99.9976% | 1,985,761 |
| C_s2026 update | 4.96% | 98.93% | 1,964,581 |
| C_s42 correction | 4.96% | 99.9968% | 1,985,744 |
| C_s1234 correction | 4.96% | 99.9936% | 1,985,681 |

每份 `training-audit.json` 另有改前 EMPTY/numeric、位置/轮次和 donor/recipient forward 计数；不要只看启用比例。

## dev-50 clean（自动阶段小评估，不是正式 test）

每格为 `mean_first4 / mean_value`；`graph_first4_all` 是整图四属性全对。

| 模型 | D | D+4 | D+16 | D+16 graph first4/value |
|---|---:|---:|---:|---:|
| C_s2026 correction | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| C_s42 correction | 1.000 / 1.000 | 1.000 / 1.000 | 0.998 / 1.000 | 0.945 / 1.000 |
| C_s1234 correction | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 1.000 | 0.852 / 1.000 |
| C_s2026 update | 0.998 / 0.998 | 0.9995 / 0.9995 | 0.761 / 0.998 | 0.000 / 0.977 |

update 的 D+16 first4 下降主要来自 VARIABLE 漂移，VALUE 仍约 0.998；不要把 first4 漂移写成算术大幅失败。C_s2026 update 的 dev-50 UPDATE clean 计数和原 C 功能探针小测见 `docs/records/2026-11-20/c-update-small-test.md`（提交 `96bdde6`）。

## B 的正式评估命令

```bash
python tools/c_update/execute.py --phase evaluate \
  --plan tmp/analysis/c-update-20261119/data/plan.json \
  --out-dir tmp/analysis/c-update-20261119/run \
  --device cuda --num_threads 4
```

`execute.py evaluate` 会从 `run/train/*/final.ckpt` 取四份新 final，并评估 plan 内的五个原权重（A_s42、B_s42、C_s2026、C_s42、C_s1234）；输出在 `run/eval`。不要用 `--out-dir` 不含 `run/train` 的另一目录，否则找不到四份 final。

## 交接边界

- B 不重生成 plan/数据/阶段 dev；先完成旧批五个原权重+四 final 的统一评估，再接收新 `c_multitime` final 做同 plan 新旧比较。
- 原始 records、dev JSONL、权重和训练日志留 tmp；docs 只放小表/hash/报告。
- C_s2026 update 小测是 dev/2 图 smoke，不是正式 test；正式结论以 B 的统一评估为准。