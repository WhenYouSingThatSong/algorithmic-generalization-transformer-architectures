# c_channel：连续模型跨轮信道的冻结消融与更新幅度

对应 [PLAN-010](../../PLAN-010/README.md)。设计、消融阶梯与缺失实验清单见
[docs/records/2026-11-28/vector-difference-and-freeze-ablation-design.md](../../docs/records/2026-11-28/vector-difference-and-freeze-ablation-design.md)；
完整现状见 [channel-investigation-status.md](../../docs/records/2026-11-28/channel-investigation-status.md)。

只用现有 C 权重、不重训。

| 文件 | 作用 |
|---|---|
| `common.py` | 权重加载、图解析（按“父值之和 mod 23”重算校验）、带冻结的前向循环、正式指标 |
| `trace_one_graph.py` | 单图逐轮逐位置追踪；`--intervention` 支持 none / random / all_wrong / correct 四种初始条件（对应 EXPERIMENT-073/074）；`--save-vectors` **默认关闭**，只有明确要求时才存原始隐藏向量 |
| `ablation.py` | 冻结阶梯（十个条件，条件名不复用字母）+ 五栏信息传递度量（含正交分解自检） |
| `freeze_sweep.py` | 提前量 g 的精度扫描 |
| `update_stats.py` | 逐轮更新幅度（方向 / 模长 / 总量），不冻结与各冻结条件 |

```bash
python tools/c_channel/ablation.py --seeds 42,1234,2026 --n-graphs 32 --chunk 8
python tools/c_channel/update_stats.py --seeds 42,1234,2026 --n-graphs 8 \
  --conditions none,freeze0,freeze1,freeze2
```

输出写入 `tmp/analysis/channel/`（历史路径，未改名以免打断已有产出）。
