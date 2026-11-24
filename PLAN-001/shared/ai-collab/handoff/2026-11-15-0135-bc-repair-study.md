# B问题① / C问题③：代码交接与首批执行

2026-11-15。用户已决定先做①“已有答案如何更新”、③“连续自然错误能否局部修复”，并授权GPT实现、上传，由执行agent适配和运行。问题②暂缓。完整行为定义见 [repair_study/README](../../tools/repair_study/README.md)。

**本次是新工具目录，不修改原训练、原B/C runner、原输出。** 本地已做8项针对性stdlib测试及Python语法检查；无NumPy/PyTorch，本地未做实际CPU选样、权重加载或GPU运行。这些由执行端小验证完成，不将提交状态写成实验完成。

## 领取与执行顺序

1. 两位agent先串行同步`reproduction`到空闲工作树，记录commit。共享运行目录不pull，已有训练不动。
2. B agent先做B小验证，正常后运行固定64图。C agent可先在不修改运行代码的前提下做CPU选样；GPU运行排在B任务之后。
3. C agent做2案例、4轮替换验证，再运行同一份固定选样清单、16轮。按seed42→1234串行。
4. 默认不需再次询问本批是否启动。发现改变实验含义的问题、原轨迹不能重放或明显资源冲突，先报告，不自行换权重/数据、放松对照或删除难例。

## B agent：运行与核对

主权重从既有manifest读取：A′ s42/e86、B s42/e92。主样本固定graph32–95，与原异常图142分开；仍是已有探索数据，不冒称全新独立测试集。每个目标有首次实际算对、整图首次共同算对两个时点；只在父值正确时纳入。两模型时点可不同，实际轮次记录，汇总取共同合格目标。

五分支：不改、写回正确值、加1错值、加3错值、清空。改一次后自由续算。graph142另作为诊断组恢复原注错过程，再改变位置128自身数值。

```sh
python -m unittest discover -s tools/repair_study/tests -v

python tools/repair_study/run_b.py --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json --model model_a --candidate_count 2 --horizon 4 --include_anchor --out_dir tmp/analysis/b-value-update-20261115/smoke_A
python tools/repair_study/run_b.py --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json --model model_b --candidate_count 2 --horizon 4 --include_anchor --out_dir tmp/analysis/b-value-update-20261115/smoke_B

python tools/repair_study/run_b.py --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json --model model_a --include_anchor --out_dir tmp/analysis/b-value-update-20261115/main_A
python tools/repair_study/run_b.py --manifest tmp/analysis/b-main-20261114/manifest/manifest_main256.json --model model_b --include_anchor --out_dir tmp/analysis/b-value-update-20261115/main_B
python tools/repair_study/summarize.py --inputs tmp/analysis/b-value-update-20261115/main_A/records.jsonl tmp/analysis/b-value-update-20261115/main_B/records.jsonl --out_dir tmp/analysis/b-value-update-20261115/summary
```

小验证只核对关键语义：

- `initial`只有目标VALUE发生指定变化，父值/其它字段没被误改；mod23的两错值不同且不等于正确值。
- 首次正确时点确实是模型第一次输出正确值；若父值尚不正确，该case标不合格，不能推迟起点凑资格。
- `outputs[0]`对应改后第一次forward。正确值原样写回与不改分支一致；主样本与anchor没有混表。
- graph142准备后实际父值/目标值如实记录，若不同于旧现象不要强行改出旧答案。

报告：候选图/有目标图/共同合格目标数，各时点、各分支目标逐轮正确率和末两轮正确率；下游损伤/改善、其它位置损伤；至少一条恢复和一条不恢复案例（没出现则说明），图内及逐图等权统计。不要只报告恢复率或只讲graph142。

## C agent：先选样，再替换

CPU分析固定旧C₀ s42/s1234 e143的已有六档NPZ与对应原数据。完整候选保留；本次干预优先整图数值完成后的事件。按父读出类型分组，另外保留变量名变化但数值保持的对照；自然结局分层仅用于诊断，不估总体频率。

来源向量取目标和全部数值父节点VALUE均正确的最近历史状态；VARIABLE/四属性是否正确另外记录。健康对照匹配来源间隔；找不到来源/健康对照单列，不换规则补样。默认每权重×档×类别×自然结局1个case，固定哈希选不同图，不按干预成功与否挑选。

```sh
python tools/repair_study/select_c.py --run_dir tmp/analysis/c0-capability-20261114/runs --data_dir tmp/analysis/c0-capability-20261114/data --out_dir tmp/analysis/c-local-repair-20261115/selection

python tools/repair_study/run_c.py --cases tmp/analysis/c-local-repair-20261115/selection/cases.json --weight s42_e143 --limit_cases 2 --horizon 4 --out_dir tmp/analysis/c-local-repair-20261115/smoke_s42
python tools/repair_study/run_c.py --cases tmp/analysis/c-local-repair-20261115/selection/cases.json --weight s1234_e143 --limit_cases 2 --horizon 4 --out_dir tmp/analysis/c-local-repair-20261115/smoke_s1234

python tools/repair_study/run_c.py --cases tmp/analysis/c-local-repair-20261115/selection/cases.json --weight s42_e143 --out_dir tmp/analysis/c-local-repair-20261115/s42
python tools/repair_study/run_c.py --cases tmp/analysis/c-local-repair-20261115/selection/cases.json --weight s1234_e143 --out_dir tmp/analysis/c-local-repair-20261115/s1234
python tools/repair_study/summarize.py --inputs tmp/analysis/c-local-repair-20261115/s42/records.jsonl tmp/analysis/c-local-repair-20261115/s1234/records.jsonl --out_dir tmp/analysis/c-local-repair-20261115/summary
```

选样后先回写类别数量、可用来源数量、缺健康对照数量。首批按固定清单执行，不因某组不足自动扩图。小验证检查：

- 只重放包含已选图的原始批次，保持原batch size及序列裁剪；所有已选图64轮四属性与原NPZ一致。失败会停止，不跳过该图当成功。
- 不改动续算与已有未来轨迹一致；原样写回与不改动一致。检查至少一个有健康对照case；若最前2个没有，可从同一清单选一个补小验证并记录，不改正式清单。
- 目标替换只换一个位置的hidden；父替换只换原图对应父位置的hidden；改一次后释放，没有解码字段回灌或逐轮钳制。
- 保存`initial`、后续输出及hidden快照。不能把来源向量立刻读对当作模型已经恢复；看接下来forward及下游。
- 同时核对目标四属性、下游数值、其它正确节点是否受损。健康对照与错误现场、两个seed、各自然结局分别汇总。

输出metadata含原权重/data/NPZ哈希。服务器路径若改变，可用`run_c.py --ckpt <实际路径>`，哈希仍必须等于原权重。选样清单保存实际数据绝对路径；路径适配留记录，不更换内容。

## 交付

- B/C各写一份短报告到`docs/records/2026-11-15/`，引用命令、运行提交、样本数、输出目录、主要数值及少量具体轨迹。小CSV可提交，NPZ/PT/hidden和完整大轨迹留tmp。
- 不改原结果、不新增C+训练、probe、注意力分析、问题②或其它实验矩阵。GPT后续审阅新增适配代码和结果。
- README/board/log记录领取、运行、完成与异常。保持reproduction普通推送；推前fetch整合，不覆盖其它agent未完成工作。
