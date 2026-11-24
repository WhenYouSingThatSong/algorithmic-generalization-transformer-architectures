# B 剩余核查：补齐现有证据，不重跑模型

基线：`04187bd`。用户要求 GPT 先写核心逻辑、执行 agent 负责适配/小验证/运行；本次落实离线证据导出与 clean 比较。未启动 M3/M4/C+，未改训练、runner、干预或指标定义。

## 已核查到哪里

- graph142，late_u，target_index=0，注入26、观察128：clean VALUE始终2；干预第1轮2，第2–16轮8。CSV的32行是两个角色重复记录同一节点事件。该图不在long32中，H64结果不能外推到它。
- unrelated：12个节点事件分布于146/149/275/276四图，clean均正确；干预后错误最晚第6轮回正。但窄CSV只有受损节点附近的关系，尚不能独立确认这些节点不在注入点下游，也不能做完整错父代入。
- teacher/own入口跳过clean。已有12行CSV直接核对的是2模型×2时点×3干预，每组92条交集；报告含clean的368条一致尚未由该入口完整复核。不是证明clean不同。

## 本次实现

1. `b_compare_teacher_own.py` 不再跳过clean；保留原有身份键与重复/冲突计数。执行时仍仅使用既定A′/B单seed、固定权重、同数据、H16的四个teacher文件与对应own文件；不要混入H64或其它seed。新结果正常应有16行而非12行，但以实际输出为准，不把预期写成结果。
2. `b_export_review_bundle.py` 从一个late JSONL流式读取，只留指定5张图，再按公共完整target键配clean。缺记录/缺clean/冲突clean/窗口长度不同会报错，不静默跳过。完整保留每轮所有已记录节点的输入输出，包括未受损父节点。
3. 从run meta中的`ds_file`/`manifest`定位原文件，要求dataset SHA与manifest匹配。导出原始5图label tensor与dataset_info到服务器本地bundle；另独立从dataset_info提取各节点depth、所有父位置和操作ID，保留第三父及叶节点对常数的引用，用于与manifest结构交叉核对。该导出不直接宣布机制结论。
4. 固定案例清单 `review_cases_20261114.json`：142/late_u/0；146/unrelated/0；149/unrelated/0；275/unrelated/0；276/unrelated/1。不根据结果另选案例。

## 执行 agent 的工作（CPU即可）

在空闲工作树同步reproduction，从仓库根目录运行，已有训练目录不pull。先跑新增4个小测试，不重复整套GPU验证：

```sh
python -m unittest discover -s tools/b_mechanism/tests -p test_review_export.py -v
python tools/b_mechanism/b_export_review_bundle.py \
  --input tmp/analysis/b-main-20261114/late/B_complete_main256_H16.jsonl \
  --cases tools/b_mechanism/review_cases_20261114.json \
  --out_dir tmp/analysis/b-main-20261114/review-evidence-2322
```

输出目录必须新建。若实际sidecar名称与上述不同，按原运行记录修正路径并记录；若meta原路径不存在或hash不同，先反馈，不能临时换数据/manifest通过检查。

- `review_bundle.json` 含原始5图tensor，留服务器tmp，不提交git。
- `review_evidence.json` 仅完整案例轨迹、由原数据提取的图关系、manifest子集和来源hash。检查文件大小后作为小型审阅产物提交 `docs/records/2026-11-14/b-main-review-evidence.json`；若明显过大，反馈大小，不能再次裁成只剩受损节点的CSV。
- 使用原teacher/own比较命令与原输入，仅把输出换成新文件，重新运行比较。提交CSV并更新M1报告§4.1：分别列clean/r/u/ur、各组交集/缺失/重复/冲突/差异。不能混入long32/H64。若clean不同，保留实际差异，不能宣称368条全同。
- 首次导出后做一个很小的核对：共5个干预/5个匹配clean/5张原图，各窗口16轮；graph142的pos128仍为2→8…8，原父22/64/92的输入输出现在可见。原图读取需服务器PyTorch，本地未验证这一实际文件路径。
- 完成后回写board/log、命令、代码SHA、产物路径并推送。GPT拿到完整证据后核查：①原数据依赖与manifest一致吗；②12事件是否真的没有从注入点到受损节点的路径；③graph142及其它案例的父值能否解释下一轮输出。此时才讨论是否追加graph142长窗口，不提前重跑。

## 验证边界

GPT本地新增4项stdlib测试：clean纳入比较、完整pair原样保留且不混其它target、错target clean拒绝、第三父/叶常数关系保留。未读取服务器原始JSONL/PT、未执行模型，未宣称完整因果核查已完成。既有窄CSV保留，不覆盖旧报告的历史记录。
