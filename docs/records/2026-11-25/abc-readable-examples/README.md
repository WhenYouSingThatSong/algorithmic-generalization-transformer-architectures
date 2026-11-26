# 三张手工图上的 ABC 小验证 · 结果包

- 日期：2026-11-25
- 状态：**已在 GPU0 执行完成（exit 0）；固定现有权重推理，未训练、未改模型。**
- 代码：`tools/c_hard_graphs/readable_examples.py`（well-ordered 分支 `d093ace`，由 `ed0e505` cherry-pick 进入）
- 说明文档：`../abc-readable-examples.md`
- 数据/权重根：`/workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures`

## 1. 本目录文件

| 文件 | 内容 |
|---|---|
| `逐题实测.md` | 三张图逐轮实测；含每轮“已具备正确父值 / 本轮正确 / 空 / 其他数 / 无对应值”表和逐位置读数 |
| `traces.json` | 原始逐轮数据；含每模型 SHA-256、每张图的 clean 读数、两个注错时点的 written/16 轮 follow-up |
| `run.log` | 本次运行的 stdout/stderr（含加载的四个 checkpoint 路径） |
| `README.md` | 本说明 |

## 2. 权重

| 模型 | 选中 checkpoint | SHA-256 前缀 |
|---|---|---:|
| A′ s42 | `tmp/runs/A-L4D384fast/.../epoch=86-val_loss=0.0000.ckpt` | `0b66e196990a…` |
| B s42 | `tmp/runs/B-M6/.../epoch=92-val_loss=0.0000.ckpt` | `ef58ba261fd0…` |
| 原始 C s42 | `tmp/runs/oldC0-resume-150ep/.../epoch=143-val_loss=0.0000.ckpt` | `4a00d6a21a7e…` |
| 原始 C s1234 | `tmp/runs/oldC0-resume-150ep/.../epoch=143-val_loss=0.0000.ckpt` | `0bca3a2b91e8…` |

完整 SHA 和 checkpoint 路径在 `traces.json` 与 `run.log`。

## 3. 运行命令

```bash
cd /workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures
CUDA_VISIBLE_DEVICES=0 python tools/c_hard_graphs/readable_examples.py \
  --data-root /workspace/lab-xj/users/jinxi/code/yhy/algorithmic-generalization-transformer-architectures \
  --out docs/records/2026-11-25/abc-readable-examples \
  --device cuda
```

运行环境：Python 3.12，torch 2.6.0+cu124；GPU0（运行前 1 MiB 占用，空闲）。

## 4. 实验设置

- 三张图共用 6 个输入（3,4,9,10,1,6）和同一条四步链：
  `3+4=7 → 7+9=16 → 16+10=3 → 3+1=4`。
- 每层并排的算式数（层宽）：1 / 4 / 24；总节点 10 / 22 / 102。
- 每个模型自由续算 24 轮；记录目标 `u` 首次算出 7、整图数值首次全对，以及各位置逐轮读数。
- 在两个时点分别注错：
  - `u` 首次读出 7（“刚算对”）；
  - 整图数值首次全对（“整图算完”）。
- 把 `u` 写成 12：
  - A′/B 只改 VALUE 字段；
  - C 从配对题（第一个输入 3→8）换入同轮同位置的整段 hidden。
- 同起点保留干净分支，续算 16 轮；记录目标 `u` 和直接子节点 `q` 的逐轮读数。
- 该 `q` 按原值 7 应为 16；按错值 12 应为 21。

## 5. 主要读数

### 5.1 三张图都能被四个模型在 24 轮内完成

四个模型在全部三张图上都达到两个注错起点：

- `u` 首次读出 7：第 2 轮；
- 整图数值首次全对：第 5 轮。

层宽从 1 增到 24（节点数 10→102）并没有推迟这条共享链的完成。**这三张图是“可读样例/sanity check”，不是能把模型区分开的宽度压力测试。**

### 5.2 “刚算对”时注错

| 模型 | 第 1 个 follow-up 轮（u，q） | 后续 |
|---|---|---|
| A′ s42 | **12，21** | 保持 12，21；不修回（三档层宽一致） |
| B s42 | **7，21** | 目标立即回 7；子节点第 2 轮回 16 |
| 原始 C s42 | **7，21** | 目标立即回 7；子节点第 2 轮回 16 |
| 原始 C s1234 | **3，21** | 目标先读 3，下一轮回 7；子节点第 2 轮回 16 |

- A′ 保留写入的 12，子节点按 12 算出 21，之后也不修回。
- B/C 都能把目标修回 7；但子节点都已经在第一个 follow-up 轮按 12 算出 21，下一轮才回到 16。**目标修回不等于下游没有先受损。**
- C_s1234 的过渡更慢/更乱：目标先读 3 再回 7（三档层宽均如此）。

### 5.3 “整图算完”时注错

| 模型 | 第 1 个 follow-up 轮（u，q） | 后续 |
|---|---|---|
| A′ s42 | **12，16** | 保持 12；子节点保持 16（已算好，不被重开） |
| B s42 | **7，21** | 目标立即回 7；子节点先 21，下一轮回 16 |
| 原始 C s42 | **7，7（层宽 1/4）或 7，21（层宽 24）** | 下一轮回到 16；层宽 1/4 的首轮 q=7 是“其他数”过渡 |
| 原始 C s1234 | **3，5 / 3，20 / 3，21（随层宽）** | 目标下一轮回 7；子节点下一轮回 16 |

- A′ 在整图完成后不重开已算好的子节点：子节点保持 16；这与其“保留已填值”的行为一致。
- B/C 会把已经完成的子节点重新打开；B 和 C_s42 在层宽 1/4 上首轮就出现 q=21，C_s42 层宽 1/4 首轮还出现 q=7 这个非 21/16 的过渡；C_s1234 的过渡值随层宽变化（5/20/21）。
- 所有 `events` 都是“已执行”，**没有起点未达到或供体不合格的零覆盖**。

## 6. 边界

- 固定权重、每模型一个 seed、每档一张手工图；这是**可读样例核查**，不是统计实验，也不是机制/因果结论。
- 层宽 1/4/24 同时改变了并排算式数和序列长度；共享链的深度固定为 4，所以三个模型都能在 5 轮内完成。不要把“都完成”读成“宽度不影响求解”。
- “q=21” 表示子节点在那一轮按注入值 12 计算；“q=16” 表示回到原值 7 的结果；“q=7/5/20/3” 等是其他过渡读数，统一保留在 `traces.json` 和 `逐题实测.md`。
- 运行日志有一条 tokenizer 的 legacy fallback 警告（`tokenizer_n_vars` 未设置，n_vars=128）；与该批训练/评估的既有配置一致，不影响本次读数。