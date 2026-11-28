# c_channel_probe：连续模型跨轮信道的探针与关闭信道

对应 [PLAN-010](../../PLAN-010/README.md) §1.2，设计与依据见
[docs/records/2026-11-28/probe-and-channel-shutdown-design.md](../../docs/records/2026-11-28/probe-and-channel-shutdown-design.md)。

只用现有 C 权重、不重训。回答三件事：

1. **存在性**：在节点自身轮次之前，能否用线性探针从它的隐藏态解出它的真值。
2. **载体**：把跨轮输入换回离散 token（写回或不写回）后是否退化。
3. **形成**：冻结“未到轮次”的位置、只让已达轮次的位置更新，是否退化。

另含三个控制：父值改动（信息是否跟随输入）、无关叶子（安慰剂）、位置置换（负控）。

```bash
python tools/c_channel_probe/probe.py --seeds 42,1234,2026 --n-graphs 200 \
  --out tmp/analysis/channel-probe
```

输出 `probe.json`：逐种子的基线指标、探针（overall / train_acc / shuffled / majority / 按 lead 分组）、
控制读数、两个关闭信道变体与离散写回变体的指标。

**读法要求**：探针必须同时看 `train_acc`。train 高而 val 低是过拟合（样本不足），
不能读作“没有信号”。
