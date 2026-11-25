# 原始 A/B/C 的顺序再赋值小测

- 日期：2026-11-20。
- 状态：**原始权重评估；未训练、未改结构、未重跑旧 C 训练。**
- 代码/工作树：`management-20261120@8140ca8`；GPU1 串行；输出根目录 `tmp/analysis/c-local-update-normal-v1/rebinding`。
- 原 plan：`tmp/analysis/c-update-20261119/data/plan.json`，SHA `b3c4b1be9325dd37aefc2004dc19b6e5ee6d82c6ff097c25c072d3f39038bc24`。
- 题目：`cases.json` SHA `996104e673a4d27c5d91c8439881f19a09eb86a82011d250f302298b2a66b7da`；232 个 case = 228 对“唯一名/复用名”配对 + 4 个 139 定义单臂 wide-33。
- 模型：A_s42、B_s42、C_s42、C_s1234、C_s2026，均使用原权重；每份 1,144 条逐轮记录。身份与文件 hash 见 `rebinding-identity.json`。

## 1. 覆盖和读法

- 每条等式先按前文最近定义绑定操作数，再绑定等号左值；整题一次输入后自由续算。配对题的算式、位置、答案和依赖相同，只把后续左值改成新名字或复用原名。
- 观察点 `D`、`D+4`、`D+8`（D 为最长依赖链节点数）。表内 `unique1/rebound0` 表示唯一名正确、复用名错误；`both0` 表示两者都错。每个 cell 8 图，配对分母 228。
- 模型输出恰等于某个操作数旧值代入结果只记为“与读旧值相容”，不是机制判断；wide-33 单列，不能作为单纯复用能力归因。

## 2. 配对结果（228 对）

| model | D: unique1/rebound0, both0 | D+4 | D+8 | both1 at D/D+4/D+8 |
|---|---|---|---:|---:|
| A_s42 | 128, 100 | 223, 5 | 228, 0 | 0/0/0 |
| B_s42 | 118, 108 | 217, 9 | 226, 0 | 2/2/2 |
| C_s42 | 212, 16 | 204, 24 | 196, 32 | 0/0/0 |
| C_s1234 | 54, 170 | 47, 178 | 40, 185 | 4/3/3 |
| C_s2026 | 211, 17 | 225, 3 | 224, 4 | 0/0/0 |

- A/B/C_s42/C_s2026 的多数可区分配对格是“唯一名对、复用名错”；C_s1234 在两个命名臂上都大量失败，不能把它的差异单独归因于复用。
- 139 定义 wide-33 四题在五个模型的 D/D+4/D+8 端点均为 0/4 正确；该档只作能力探查，不单独解释为复用失败。
- 每个 family/档的完整 counts 在 `rebinding-pairs.csv`、`rebinding-families.csv`。

## 3. 逐节点读数

`rebinding-node-pairs.csv` 将节点分为 `uses_rebound`（至少一个父节点是复用版本）与 `other`。D+8 时：

| model | uses_rebound: unique1/rebound0 | uses_rebound: both1 | uses_rebound: both0 |
|---|---:|---:|---:|
| A_s42 | 3,332 | 357 | 0 |
| B_s42 | 3,295 | 394 | 0 |
| C_s42 | 2,941 | 456 | 277 |
| C_s1234 | 1,465 | 413 | 1,732 |
| C_s2026 | 3,229 | 434 | 25 |

- 这些是节点-观察计数，按每图固定位置相加，不是独立样本；分母不按成功过滤。
- 旧值相容的候选在 `counts` 的 `stale_eligible/matches_stale` 中保留；只表示输出与某个旧值代入结果相容，不能证明模型读取了该旧值。
- B_s42 在 D 档有约 1,002 个结果位置为空；A/B/C 的空值仍保留在分母，不能把空值当错误之外的独立类别或静默删除。

## 4. 证据与边界

- 原始 JSON：五份每模型 1,144 条 records；`rebinding-identity.json` 含 plan/cases hash、模型 SHA 和各模型文件 hash。
- 工作树 `8140ca8` 的 `rebinding.py` 只加载原始权重；本次未运行新 C 训练、未启动 pending C eval 队列。
- 顺序再赋值与“物理 token 是否释放”不同；这里只测名字复用。139 定义档只说明该构造超出当前可稳定求解范围，不能单独归因。
- 下一步按 ordered handoff：C 在冻结清单的 8 张 dev 图上做最小评测和最大深度反传 smoke，再启动三种子正常窗口训练；B 在 50 epoch 后正式评估。

## 5. 文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `rebinding-pairs.csv` | `b08781379794dbbfdac17c7959299f71855597697a39519a392e8c6988d4925b` |
| `rebinding-families.csv` | `e7b3fc6582bf817023e36d04ce4c4aaa08751f766c596f7e3a64471a556eb28f` |
| `rebinding-node-pairs.csv` | `c00c3de7c277b97094e591ebeabdf8a687f17ee82759fea9ee5da2a260f83742` |
| `rebinding-identity.json` | `ebc1515db71c7aec92b7005ea15056b24cdfe41076afc95f3aba04361b1631d1` |
