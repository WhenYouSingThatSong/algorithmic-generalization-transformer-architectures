# C 长程后训练小比较：s2026/e143 的 base / +D / +2D

日期：2026-11-15 17:55 CST。
状态：**三组后训练、新 128 图验证和旧六档诊断均已完成；单 seed 小样本 pilot，不是正式训练配方、最终能力结论或 C+ 机制实验**。
代码：`reproduction@08e8264`（运行期间未 pull、未改运行代码/数据）。
起点：s2026/e143，SHA256
`40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780`；固定 e143，不按本轮结果换 best/final。
交接：[C 长程 pilot](../../../ai-collab/handoff/2026-11-15-c-long-pilot.md)。

## 1. 数据与训练预算

### 数据

- 官方 N32/ADD/depth≤6/6 叶/父数 1–3/128 变量名词表，固定 seed 20261115/20261116，
  新生成 1024 train + 128 val；实测最大节点深度（raw）train 3–6、val 4–6，
  对应 D=raw depth+1：train **D4–7**（657 个 D7、283 个 D6、81 个 D5、3 个 D4），
  val **D5–7**（91/31/6）。
- 与旧原训练/验证（50000/1000 图）及六档诊断（384 图）按最终 token 序列去尾 PAD
  精确排重；8 个参考文件均记录在 `data/manifest.json`，无交集。未作图同构排重。
- `data/manifest.json` status=complete；train SHA256 `1f23380e0762...`，
  val SHA256 `676aabd2173d...`（完整值在 manifest）。

### 三组

| arm | 每图轮数 | epoch（equal_compute） | optimizer steps | 图曝光 | sample×loop | token×loop |
|---|---:|---:|---:|---:|---:|---:|
| base | D | 6 | 786 | 6144 | 40284 | 10312704 |
| plus_d | 2D | 3 | 393 | 3072 | 40284 | 10312704 |
| plus_2d | 3D | 2 | 262 | 2048 | 40284 | 10312704 |

- 共同 batch_size=8，131 个固定同 D batch/epoch；尾批保留，所有图参与；
  `same_exposure` 都训练 2 遍新图（262 steps、2048 图曝光），
  `equal_compute` 使用上表 6/3/2 遍的 sample/token/attention 等循环代理量。
- 三组各自新建 AdamW，配置相同（不共享优化器状态、不恢复旧 momentum）、lr=1e-4、原 betas/eps/weight_decay、
  float32、medium matmul、无 scheduler/有限梯度裁剪；前四字段权重 [1,1,1,1]。
  loss 为阶段均值：base=求解段均值；长程=0.5×求解段+0.5×保持段。
  因此 base 也不是原训练的精确 resume，三组是同一新后训练配方。
- `plus_2d` 的 `same_exposure` 与 `equal_compute` 都是 epoch2、两文件权重逐 tensor
  相同；评估只计算一次并复用。完整预算见 `c-long-pilot-budget.csv`。
- 7 项 CPU/torch 测试通过；smoke 验证 base loss=原连续 loss/D、真实 D/轮数、
  optimizer 更新、无 NaN/OOM；plus_2d smoke checkpoint 通过标准
  `run_capability.py` strict load，2 图×4 轮正常。正式训练无 Traceback/OOM。

## 2. 新验证：节点与整图

128 个新验证图，固定 D/2D/3D 与固定绝对 8/16/32/64 并列报告；不因失败/未完成剔除。
`D` 列是每图自身 D，`D+16/32/64` 是相对该图 D 的固定切片。

### 节点 VALUE（n=128 图，4096 个节点时点）

| arm / endpoint | D | D+16 | D+32 | D+64 | 2D | 3D | abs64 | graph VALUE@D+64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 0.9998 | 0.9995 | 0.9890 | 0.9143 | 1.0000 | 0.9995 | 0.9258 | 0.4453 |
| base / same_exposure | 1.0000 | 0.9980 | 0.9875 | 0.9373 | 1.0000 | 0.9998 | 0.9521 | 0.5938 |
| base / equal_compute | 1.0000 | 0.9995 | 0.9893 | 0.8862 | 1.0000 | 0.9995 | 0.9080 | 0.3438 |
| plus_d / same_exposure | 1.0000 | 1.0000 | 0.9990 | 0.9897 | 1.0000 | 1.0000 | 0.9907 | 0.8984 |
| plus_d / equal_compute | 1.0000 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 1.0000 | 0.9993 | 0.9688 |
| plus_2d / same=equal | 1.0000 | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 1.0000 | 0.9988 | 0.9766 |

### 节点四属性同时正确（first4）

| arm / endpoint | D | D+16 | D+32 | D+64 | 2D | 3D | abs64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| initial | 0.9998 | 0.7981 | 0.1992 | 0.0293 | 0.9993 | 0.9065 | 0.0403 |
| base / same_exposure | 1.0000 | 0.7209 | 0.1780 | 0.0295 | 0.9968 | 0.8457 | 0.0374 |
| base / equal_compute | 1.0000 | 0.6882 | 0.1768 | 0.0332 | 0.9905 | 0.8167 | 0.0425 |
| plus_d / same_exposure | 1.0000 | 0.9993 | 0.7976 | 0.2258 | 1.0000 | 1.0000 | 0.2896 |
| plus_d / equal_compute | 1.0000 | 1.0000 | 0.9565 | 0.4185 | 1.0000 | 1.0000 | 0.5205 |
| plus_2d / same=equal | 1.0000 | 1.0000 | 0.9688 | 0.4663 | 1.0000 | 1.0000 | 0.5710 |

- 新验证在 D 处已接近饱和（initial VALUE/first4 0.9998，graph VALUE 0.9922）；
  因此这里更适合看 D 之后，而不是把 D 处 1.0 当成训练新增能力。
- 两套预算下，plus_d/plus_2d 在 D+16/32/64 的 VALUE 和 first4 都高于 initial 和
  base；plus_2d 在 D+64 切片的两个节点指标最好，但不是所有切片均最好，例如 abs64 VALUE 为 plus_d equal 0.9993、plus_2d 0.9988。
- 但 first4 仍会下降：plus_2d 的 first4 从 D+32 0.9688 降到 D+64 0.4663，
  graph first4@D+64 仍是 0/128；这不是“四属性已稳定”。
- base 的更多曝光（6 遍）没有呈现 plus 组的效果：D+64 VALUE 0.8862、graph 44/128，
  低于 base same_exposure 的 0.9373、76/128（75/128 是末两轮均正确）；D+16 first4 0.6882 也低于 initial
  0.7981。该差异包含曝光、优化步数和 run 间波动，不能单归因于“普通续训有害”。

### 完成后事件（128 图）

`completed` 为对应指标首次整图全对；`post_observed` 为完成后至少还有一个观察轮。

| arm / endpoint | VALUE degraded / post | VALUE recovered / reerror | VALUE final-two graph | first4 degraded / post | first4 final-two graph |
|---|---:|---:|---:|---:|---:|
| initial | 72 / 128 | 1 / 0 | 57 / 128 | 128 / 128 | 0 / 128 |
| base / same_exposure | 53 / 128 | 1 / 0 | 75 / 128 | 128 / 128 | 0 / 128 |
| base / equal_compute | 85 / 128 | 2 / 1 | 44 / 128 | 128 / 128 | 0 / 128 |
| plus_d / same_exposure | 14 / 128 | 1 / 0 | 115 / 128 | 128 / 128 | 0 / 128 |
| plus_d / equal_compute | 4 / 128 | 0 / 0 | 124 / 128 | 128 / 128 | 0 / 128 |
| plus_2d / same=equal | 3 / 128 | 0 / 0 | 125 / 128 | 128 / 128 | 0 / 128 |

- plus 组的 VALUE 完成后失稳图数明显减少；但 first4 在所有组仍是 128/128 出现
  完成后错误，末两轮没有整图 first4 全对。旧 C0 曾观察到 VARIABLE 漂移主导，
  但本轮未重新做字段归因，不能直接沿用该归因；长程 VALUE 保持不等于四属性保持。
- `ever_recovered`/`reerror` 的分母是完成后事件，不是全部图；表中只作描述，
  不能把“未恢复”解释成机制上的不可恢复。

### 配对改善/损伤（同 graph、同节点、同预算）

- plus_2d vs base（equal_compute）D+64 VALUE：4090 vs 3630 正确，helped 464 /
  harmed 4；first4：1910 vs 136，helped 1818 / harmed 44。
- plus_d vs base（equal_compute）D+64 VALUE：4088 vs 3630，helped 460 / harmed 2；
  first4：1714 vs 136，helped 1636 / harmed 58。
- base same_exposure vs initial D+64 VALUE：3839 vs 3745，helped 257 / harmed 163；
  base equal_compute vs initial：3630 vs 3745，helped 219 / harmed 334。
  这仍是同验证集的逐节点配对，不是独立确认。
- 完整分组与图等权列见 `c-long-pilot-validation-paired.csv`。

### 三个轨迹例子

只作轨迹示例，不是随机抽样；完整逐轮见 `c-long-pilot-trajectories.csv`。

| graph | D | metric/time | initial | base equal | plus_d equal | plus_2d equal |
|---|---:|---|---:|---:|---:|---:|
| `graph_0004` | 7 | VALUE D+64 | 29/32 | 30/32 | 32/32 | 32/32 |
| `graph_0068` | 7 | VALUE D+64 | 32/32 | 32/32 | 32/32 | **30/32** |
| `graph_0000` | 7 | first4 D+32 | 3/32 | 3/32 | 31/32 | 32/32 |

graph 4 是 plus 组持续正确的例子；graph 68 是 plus_2d 在 D+64 损伤 2 个节点、
而 initial/base/plus_d 仍全对的例子；graph 0 是 first4 在 D+32 被 plus 组保住、
但 D+64 所有组都下降的例子。

## 3. 旧六档诊断（equal_compute）

同一固定六档、同一 384 图×64 轮；**不用于选 checkpoint 或调参**，且该诊断集
已被多次查看，不是独立测试集。全部三个 long_* 的 384 图 status=complete，
无缺图。正常 D 附近 tV/t4 中位数保持（N32_shallow 7、N32_deep 8、N64_shallow 7、
N64_deep 10、N128_shallow 7、N128_deep 12）；plus_d/plus_2d 的 first4 完成
均 64/64（base 有 55/64、63/64 两档未完成）。

H16（整图 VALUE 63–64/64；initial 为 s2026/e143）：
- initial：node VALUE 0.9985–1.0000，node first4 0.9917–0.9962，graph first4
  0.5625–0.8281；
- long_base：node VALUE 1.0000，node first4 0.9497–0.9784，graph first4
  0.0625–0.3906；
- long_plus_d：node VALUE 0.9996–1.0000，node first4 0.9996–1.0000，
  graph first4 0.9844–1.0000；
- long_plus_2d：node VALUE/first4 1.0000，graph VALUE/first4 64/64。

H64（同一诊断集，节点/整图）：

| stratum | VALUE node: initial / base / +D / +2D | first4 node: initial / base / +D / +2D | graph VALUE: initial / base / +D / +2D |
|---|---|---|---|
| N32_shallow | 0.9438 / 0.9009 / 0.9971 / 0.9912 | 0.0439 / 0.0464 / 0.5049 / 0.5762 | 0.6406 / 0.4375 / 0.9531 / 0.9531 |
| N32_deep | 0.9297 / 0.8589 / 0.9995 / 0.9868 | 0.0347 / 0.0430 / 0.5264 / 0.5620 | 0.5000 / 0.2812 / 0.9844 / 0.9688 |
| N64_shallow | 0.8616 / 0.8171 / 0.9878 / 0.9827 | 0.0286 / 0.0378 / 0.5415 / 0.5850 | 0.0625 / 0.0469 / 0.9062 / 0.8281 |
| N64_deep | 0.8381 / 0.7854 / 0.9941 / 0.9871 | 0.0315 / 0.0405 / 0.5588 / 0.5911 | 0.0469 / 0.0312 / 0.9062 / 0.9219 |
| N128_shallow | 0.7661 / 0.7489 / 0.9811 / 0.9832 | 0.0265 / 0.0374 / 0.5529 / 0.5880 | 0.0000 / 0.0000 / 0.5781 / 0.6875 |
| N128_deep | 0.6936 / 0.6390 / 0.9834 / 0.9883 | 0.0214 / 0.0320 / 0.5575 / 0.5950 | 0.0000 / 0.0000 / 0.7031 / 0.7031 |

- 这组数字显示 plus_d/plus_2d 在旧诊断的 H64 上同时提高了节点 VALUE、节点
  first4 和整图 VALUE；long_base 没有同样改善，部分档还低于 initial。
  与上一轮 s2026 的“H64 整图 VALUE N128 两档为 0”相比，这是明显差异。
- 但 H64 first4 的整图 all 仍为 0/64，节点 first4 也只有约 0.50–0.60；
  说明长程后训练主要改善 VALUE 的持续时间，不等于四属性整图稳定。
- 旧六档不是独立测试集；可以描述为固定规模/深度分布外诊断上的长程改善，
  不能作为最终独立泛化证据或机制归因。

## 4. 口径与边界

- 单 seed；只有两套固定端点和三组训练，不按验证效果挑 checkpoint，不做早停。
- 新验证 D 处接近饱和；D+64 的差异只说明这个 128 图新验证集上的长程表现。
- plus_d/plus_2d 同时改变训练标签延续、循环数和优化步数；三组不能拆出
  “标签保持”“额外 BPTT”“额外曝光”各自的贡献。base 也不是原 recipe 的精确续训。
- first4 在旧 C0 中主要由 VARIABLE 读出漂移主导；本报告没有重新做字段归因，
  因此只说节点/整图指标，不说错误检测、纠错或机制已确定。
- 已完成长程保持后训练这一候选组件；没有人工注错、自然错误重放、混合训练、
  N64/N128 训练或从头训练，尚未验证明确的纠错训练机制。
- 旧六档是研究诊断集，不能称为最终独立测试；新验证也只有 128 图。

## 5. 产物

- 数据：`tmp/analysis/c-long-pilot-20261115/data/{train.pt,val.pt,manifest.json}`
- 三组：`.../{base,plus_d,plus_2d}/{meta.json,steps.jsonl,same_exposure.ckpt,equal_compute.ckpt}`
- 新验证：`.../validation/`（summary、event_summary、paired_summary、budget、
  graph_curves、events、逐图 NPZ、meta）
- 六档：`.../six_strata/long_{base,plus_d,plus_2d}/` 及 `six_strata/summary/`
- 提交小表：
  - `c-long-pilot-budget.csv`
  - `c-long-pilot-validation-summary.csv`
  - `c-long-pilot-validation-events.csv`
  - `c-long-pilot-validation-paired.csv`
  - `c-long-pilot-six-strata.csv`
  - `c-long-pilot-six-strata-completion.csv`
  - `c-long-pilot-trajectories.csv`
  - `c-long-pilot-audit.json`
  - `c-long-pilot-field-curves.csv`
  - `c-long-pilot-field-drift.csv`
- 原始 NPZ、权重、完整 records 留在 tmp；不提交新权重。训练和验证均由 agent C
  在 GPU2 串行执行，未与其他 agent 争用共享工作树。

## 6. 审阅补交：审计摘要与逐字段曲线（CPU）

- 来源/参数/预算审计摘要：`c-long-pilot-audit.json`。它只包含 source SHA、
  data manifest/train/val SHA、三组 args/optimizer/steps、checkpoint SHA 和
  validation model/hash，不包含权重、原始张量或训练数据。
- validation 的逐图 NPZ 保存 `predictions[round,node,4]`（SYNTAX/VARIABLE/
  OPERATION/VALUE）与 truth；因此可以补逐字段曲线和首次失错时间，不需要重跑。
- `c-long-pilot-field-curves.csv`：按 endpoint×field×time 的节点准确率和
  整图 all 率；`c-long-pilot-field-drift.csv`：只在 D 处该字段整图全对的图上，
  统计 D 后首次失错（相对 D 的轮次）和末两轮是否仍全对。

| endpoint | VALUE all@D | VALUE post-D fail | VALUE first-fail median | VARIABLE post-D fail | VARIABLE first-fail median | first4 all@D |
|---|---:|---:|---:|---:|---:|---:|
| initial | 127/128 | 71/127 | 45 | 128/128 | 11 | 127/128 |
| base / same_exposure | 128/128 | 53/128 | 45 | 128/128 | 10 | 128/128 |
| base / equal_compute | 128/128 | 85/128 | 42 | 128/128 | 8.5 | 128/128 |
| plus_d / same_exposure | 128/128 | 14/128 | 51.5 | 128/128 | 21 | 128/128 |
| plus_d / equal_compute | 128/128 | 4/128 | 55.5 | 128/128 | 29.5 | 128/128 |
| plus_2d / same=equal | 128/128 | 3/128 | 47 | 128/128 | 31 | 128/128 |

- SYNTAX 和 OPERATION 在所有 endpoint 的所有时点都是 1.0000；first4 的下降
  主要来自VARIABLE，但first4与VARIABLE准确率并不完全相等。例如+D equal在D+64
  分别为1714/4096与1719/4096，另5个节点在VARIABLE正确时VALUE错误。
  因此不能把first4损失全部归因于VARIABLE，更不能据此确定隐藏状态机制。
- VALUE 的改善是“首次失错更晚/更少”：plus_2d 只有 3/128 在 D 后发生整图
  VALUE 失错；但 first4/VARIABLE 仍在 D+32 后持续下降，最终 all 为 0。
- `initial` 的 VALUE all@D 是 127/128（1 图在 D 处未全对），所以 VALUE 的
  post-D 分母是 127；表中明确使用该条件分母，不把它当作 128 图稳定率。

> 2026-11-15 18:05 GPT 审阅更正：区分末轮 76/128 与末两轮 75/128；明确独立优化器、+2D 优势的切片范围和字段归因边界。未修改代码或重跑模型。详见 [审阅记录](../../../ai-collab/handoff/2026-11-15-1805-c-long-results-review.md)。

> 2026-11-15 18:49 执行 agent C 补：审计 JSON、逐字段曲线/首次失错表（仅读现有 NPZ）。未重跑模型。
