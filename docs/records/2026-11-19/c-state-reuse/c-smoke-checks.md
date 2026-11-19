# C state-reuse smoke 执行检查记录

日期：2026-11-19。执行者：Agent C。运行目录：`tmp/analysis/state-reuse-exec-20261119/C-smoke`。  
固定代码：`reproduction@6ef63e3`（`batch.py prepare --lane C --phase smoke` 的 `base_commit`）。GPU：2。  
范围：四个 C 版本各前 2 图、H4；未训练、未扩图/种子。所有检查基于本次真实 GPU records、snapshot tensor 与源码；不是模型收益验收。

## 1. 队列与覆盖

- `python -m unittest discover -s tools/state_reuse/tests -v`：12 项通过（smoke 前本地确认）。
- `batch.py run --bundle ... --gpu 2`：6 个任务全部正常退出；`C-initial/hold/noise/semantic` 各 72 条 records，`C-summary` 和 `C-delivery` 生成。
- 每个版本 2 个固定候选图（0、1），每图 36 个 case：3 timing ×4 mode ×3 lag；每 case 10 分支 ×H4。coverage 全部 `selected`，无零记录；每版本 36 个 snapshot tensor。
- 来源/权重：data `676aabd2…`、plan `1971a635…`；initial `40babb20…`、hold `24f6b071…`、noise `cdf398a7…`、semantic `559fed2b…`（完整 hash 见各 arm meta）。四版本同源 seed2026，不是四种子。
- `meta['code_hashes']` 与当前 runner/core/suite/tokenizer/model 文件逐一 sha 匹配（无 mismatch）。

## 2. 结构、donor 与时序

- 对 4×72 条 records 检查：`u,q` 均在 positions；`u` 是 `q` 的父；`q` 为内部节点；每图 36 个 (timing, mode, lag) 组恰好 1 条。
- `early` 的 `state_round == depths[q]`；`donor_pre_round == depths[q]`，`donor_solved_round == pre+1`；`absolute_round == state_round + lag`；所有 donor 轮次 ≤ probe 轮次（未使用未来向量）。
- 每版本 72/72 条：`donor_pre_empty=True`、`donor_pre_identity_correct=True`、`donor_solved_first4_correct=True`；`other_parents_value_correct=True` 72/72。
- `sham == observed_keep`：4 版本各 72/72 条完全相等（runner 另有硬校验）。
- `natural` 的 `observed_keep == clean_keep`：4 版本各 18/18 条逐项相同，没有需要解释的 argmax 差异。

## 3. Snapshot 与十路重建

对 144 个 snapshot（4 版本×36 case；`--snapshot-graphs 1` 的固定前缀）逐一重建 10 个分支状态，共 1,440 个状态：

- `sham` 与 `observed` 完全相同；`observed_pre/solved` 相对 `observed` 只改 q 行，且 q 行分别等于 `pre/solved`。
- `restore_u_keep` 只改 u 行，u 行等于 `clean` 的 u 行；`restore_u_pre/solved` 除 u/q 外逐元素等于 `observed`。
- `clean_keep` 全状态等于 `clean`；`clean_pre/solved` 相对 `clean` 只改 q 行。
- 没有任何未允许行被修改；0 个重建失败。

## 4. 扰动向量与时序

- `same`：18/18（每版本）实际距离等于其自身 donor 距离（均值约 0.42–0.49，远小于 semantic）；解码 u VALUE 均等于原题真值（18/18/18/18）。u 四属性正确分别为 15、18、18、18（版本顺序 initial/hold/noise/semantic）。
- `semantic`：18/18 实际距离等于 donor 距离；解码 u VALUE 等于 expected_u 18/18；u 四属性正确 16/16/16/16。
- `noise2`：18/18 实际 L2 恰为同一状态语义距离的 2 倍（`abs(actual - 2*radius) < 1e-4`）；无零半径 case；同 graph/timing 的 seed 在 1/2/4 lag 和各 mode 间一致（每 (graph,timing) 只有一个 seed）。
- `natural`：实际距离 0、无人工 u 注入。`same`/`semantic` 的 donor 均来自同图、同模型、同变量位置的 `sources[t0]`，不是未来轮次。
- 解码只用于记录和 branch 输入输出：`run_c_function.py:88` 只在初始 token 上调用一次 `token_embedder`；之后 `lit.model(h,True)`/`lit.model(batch,True)` 直接传 hidden（`run_c_function.py:92,112,138`），分支循环没有离散重嵌入或 teacher forcing。`lit.nointerm` 在运行前强制检查。

## 5. 手工核对十路案例

- 找到“u 读对、q 仍错”的 probe case：initial 5、hold 6、noise 6、semantic 5 条。固定取 `noise:0:early:semantic:1`（u truth=1、q truth=22；q probe 读数错，其他父正确，donor pre EMPTY、solved first4 正确）：
  - `observed_keep` 与 `sham` q 输出：16,22,22,22；
  - `observed_pre`、`observed_solved`、`restore_u_*`、`clean_*` q 输出：22,22,22,22。
  这表示在该固定接收状态上，历史 q 向量或 clean 状态替换使首轮 q 输出恢复；只能说明该状态替换的作用，不证明模型主动检测或一般纠错机制。
- `semantic:0:early:semantic:1` 呈相同模式（observed/sham 首轮 16，其余 22）。

## 6. 结论与边界

- 真实 GPU smoke 的协议/结构/donor/时序/十路重建检查通过；没有零记录、没有结构跳过、没有数值/轨迹异常。
- 该 smoke 只覆盖四版本各前 2 图/H4，不能当最终测试或机制结论；下一步按交接进入四版本前 32 图/H16 主批，仍同源 seed2026，不是四种子。
- 本记录不是用户批准或模型成功证明；accept-smoke 仅表示上述执行检查已完成。
