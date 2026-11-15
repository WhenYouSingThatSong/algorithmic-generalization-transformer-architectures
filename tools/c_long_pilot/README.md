# C 长程训练小比较

目的：固定s2026/e143，比较普通续训、额外D轮、额外2D轮。用户2026-11-15授权本轮实现与小规模比较；没有启动人工注错/自然错误重放/混合五路，也不是从头训练。

状态：首批三组、新128图验证和旧六档诊断已完成（2026-11-15）；结果与边界见
[docs/records/2026-11-15/c-long-pilot.md](../../docs/records/2026-11-15/c-long-pilot.md)。
本README保留设计与运行入口；完整NPZ/权重在 `tmp/analysis/c-long-pilot-20261115/`，不提交。

详见 [执行交接与设计](../../ai-collab/handoff/2026-11-15-c-long-pilot.md)。

- `build_data.py`：原官方N32、depth≤6配方生成1024训练图＋128验证图，使用128变量名词表；固定新seed，不用旧诊断图训练；精确token序列排重。
- `train.py`：独立PyTorch后训练入口，不改现有model.py/Lightning训练路径。全程连续反馈、完整BPTT；前D轮使用原逐深度标签，之后始终最终标签；不重置、不离散回灌、不按是否完成筛选。
- `protocol.py`：图深度D、固定同D批次、权重与预算、状态事件定义。
- `evaluate.py`：新验证图上初始/各固定端点自由运行至D+64；节点VALUE/四属性、整图完成及完成后漂移/恢复分开；固定D时点与实际完成后条件事件分开。
- `tests/`：stdlib设计测试；服务器另运行单个torch梯度/标签测试。smoke还核对base loss等于原连续路径loss/D。

从仓库根目录运行。默认输出 `tmp/analysis/c-long-pilot-20261115/`，已有输出一律不覆盖。重跑用新study/data/out目录并完整记录命令。没有断点续训功能，失败记录后由agent反馈。

```sh
python -m unittest discover -s tools/c_long_pilot/tests -v
python tools/c_long_pilot/build_data.py
# 先smoke；详见交接。正式按以下顺序串行。
python tools/c_long_pilot/train.py --arm base --out-dir tmp/analysis/c-long-pilot-20261115/base
python tools/c_long_pilot/train.py --arm plus_d --out-dir tmp/analysis/c-long-pilot-20261115/plus_d
python tools/c_long_pilot/train.py --arm plus_2d --out-dir tmp/analysis/c-long-pilot-20261115/plus_2d
python tools/c_long_pilot/evaluate.py --out-dir tmp/analysis/c-long-pilot-20261115/validation
```

两种比较点：都训练2遍新图的`same_exposure`；base6遍、+D3遍、+2D2遍的`equal_compute`。固定批成员保证后者sample/token/attention计算代理量相同，但不声称耗时、反向内存、优化步数或图曝光数都相同。前者步数和图曝光数相同、计算量不同。报告必须并列，不能择有利口径。

三组共同：全精度float32、原dropout、AdamW重新初始化、lr=1e-4、原beta/weight_decay、无scheduler、无有限值梯度裁剪、原前四字段权重、批内剩余PAD仍参与CE。loss按阶段平均：base=solve均值；长程=0.5×solve均值+0.5×hold均值，两长程组相同阶段权重。与原训练按loop求和不同，明确属于共同归一化后的后训练小试，不冒称完全原recipe续跑。
