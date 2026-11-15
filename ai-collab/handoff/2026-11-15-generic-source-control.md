# 执行交接：通用同值/异值来源对照

用户已授权GPT实现、推送并通知agent，且明确不限4/5，要覆盖通用场景和不同数据。本批直接按下述固定协议小验证→执行，无需重复询问启动许可。代码与定义：[counterfactual_control](../../tools/counterfactual_control/README.md)。

## 分工与顺序

- **Agent C领取本任务。** Agent B继续上一审阅单的已有数据CPU补交，本次不扩B实验。
- 两位agent串行同步reproduction，在空闲工作树固定提交后执行；不得在运行目录pull/改代码。GPU任务串行，现有训练不动；每次运行记录实际commit和GPU。不要重复上一轮多GPU同时启动的安排。
- 核心方案不是旧C4结果后补一个百分比：同值和两种异值分支在统一数据/权重/起点上重新配对运行，旧表不作分母。

## 小验证

本地已有6项针对性stdlib测试通过，未加载torch模型。服务器只做必要的模型/数据小验证：

```sh
python -m unittest discover -s tools/counterfactual_control/tests -v
python tools/counterfactual_control/run.py --weight s42_e143 --strata N32_shallow N128_deep --n_graphs 2 --horizon 4 --out_dir tmp/analysis/c-source-control-20261115/smoke_s42
```

核对实际改变的是同两个叶常数；三个来源目标真值分别a/a+1/a+3；源与接收位置/轮次一致；只替换一行hidden、sham输出相同；正确区分局部q预期与完整来源q真值。至少查看一个非4原值和一个多父目标；若固定小样本无此类型，先在选样日志说明，再在正式固定候选范围内取一个补验证，不改变正式样本。不要为满足类型而扫描范围外图或按成功率挑例。

小验证可能没有合格读出，此时保留失败并报告原因；不能放松四属性源资格、临时换epoch或扩样来使结果好看。如果发现实现错误，记录触发条件与适配修改，实验含义变更先反馈。

## 首批正式执行

沿用两个e143权重、现有六档；每档固定前16图，每图至多2个结构候选，值不筛选，父数/深度分组取样；偏移+1和+3，各16轮。按种子顺序在同一可用GPU串行运行：

```sh
python tools/counterfactual_control/run.py --weight s42_e143 --out_dir tmp/analysis/c-source-control-20261115/s42
python tools/counterfactual_control/run.py --weight s1234_e143 --out_dir tmp/analysis/c-source-control-20261115/s1234
python tools/counterfactual_control/summarize.py --inputs tmp/analysis/c-source-control-20261115/s42/records.jsonl tmp/analysis/c-source-control-20261115/s1234/records.jsonl --out_dir tmp/analysis/c-source-control-20261115/summary
```

默认数据/来源目录与最近C实验相同。若只需适配checkpoint路径，可用--ckpt；仍必须满足原run_meta的sha256。其它路径适配记录命令，不换权重/数据。新建输出目录，不覆盖旧C4、六档评估、局部修复结果。

## 交付

1. 更新board/log领取和状态；提交小型报告与coverage/summary.csv、少量具体案例轨迹，完整records留tmp。带实际代码提交、运行命令、数据/权重哈希、耗时和未通过资格数；没有新训练。
2. 明确六档、1/2/3父数、真实depth、原值及改变量的实际覆盖；结构无法构造、模型源读出不匹配、缺可比q分开报告。不能把“主表没出现”当该类没有问题。
3. 主表逐档逐seed：同值来源的q变化/损伤、异值来源的特定预期命中、两者配对差；第一轮为主，附2/4/8/16轮与target/q四属性。图等权、原始分子分母均保留，不只给成功数字输出的比例。
4. 区分same来源hidden完全相同与非零变化，报告各来源距离。完整来源参考按其新真值评估，不能作为单点替换的预期答案。
5. 至少两条不同原值/结构的例子，写出原方程、两叶修改、局部q预期、即时读出和后续轨迹。不把u下一轮修回原值自动判为注入失败。

本批不扩N256、不训练C+、不做跨图/跨位置来源、随机噪声或B机制新矩阵。需要更多样本或新的对照类型时先根据首批覆盖和结果讨论。
