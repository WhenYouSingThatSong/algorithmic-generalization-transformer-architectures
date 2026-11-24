# raw辅助字段回灌：原始代码与论文核查

日期：2026-11-16。当前仓库基线`508afdd`；原始公开代码取Git历史`52dd950dce8d32a8e72513999cac5a310e01ca53`（2025-10-14），不是仅检查我们后续修改过的代码。本次只读核查，不重跑模型、不改变评估实现。

## 结论

**原始公开仓库的离散独立评估器存在bug：把训练及模型内验证/续算明确禁用、且没有任务损失监督的REGRET和MODIFICATION预测回灌，导致推理输入与训练路径不一致。** 历史raw就是沿用此行为；clean将两字段重置FALSE，恢复训练时的处理。

这个结论不依赖“clean分数更高”来倒推；数据构造、loss、续算禁用、评估反馈和输入嵌入组成了直接证据。论文给出四字段状态定义，与此判断一致。

## 1. 论文描述的是什么

核对本地58页论文v1，标题 *Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning*，arXiv `2510.14095v1`，标题页2025-10-17。PDF SHA256：

`a8a7a6c742fdb0fd5c6bf58f7055477d8fe317544b9c9143716ecde7bef19a9d`

本次核对本地PDF哈希与来源记录一致，检索已有全文提取，并检查附录第34页渲染。

- **第11页§4机制3**：离散空间由四种属性构成——syntax、variable、operation、value。
- **第34页附录B.3公式(7)**：解码、重新嵌入的集合明确列出上述四项，输入为四种嵌入的和。
- **第34–35页附录B.4**：纠错通过污染数值成分来训练，没有定义额外REGRET/MODIFICATION状态或对应监督。
- 全文提取中没有检索到这两个名称；更关键的证据是上面的四字段定义，而非单纯关键词缺席。

这只能说明论文公开描述没有将它们作为机制组成部分；不能据此了解作者未公开的所有历史运行。[论文v1](https://arxiv.org/abs/2510.14095v1)

## 2. 原始公开实现的证据链

以下链接固定到原始提交，行号也是该版本，避免当前改动造成行号混淆。

| 环节 | 原始实现 | 判断 |
|---|---|---|
| 数据/标签初始化 | [tokenizers.py:134](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/tokenizers.py#L134)：额外加入两字段，默认FALSE；返回六字段 | 代码保留扩展槽位，但常规状态没有动态启用它们 |
| 损失权重 | [model.py:254](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/model.py#L254)：`[1,1,1,1,0,0]` | 两个辅助输出不接受该任务的监督损失；不能把其argmax当作学会的错误信号 |
| fast teacher输入 | [model.py:456](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/model.py#L456)：复制输入形成逐轮状态，只填VALUE；fast使用这些状态 | 不会把辅助预测作为下一轮输入；VALUE腐蚀也不启用辅助字段 |
| 模型内teacher/自由续算 | [model.py:786](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/model.py#L786)、[804](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/model.py#L804)：两种路径均重置FALSE；`loop_forward`也这样做 | 即使不是teacher forcing，原实现也明确禁用它们，不是普通的teacher/free差异 |
| 调用的开关 | [model.py:370](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/model.py#L370)：验证调用`use_regret_for_correction=False`，训练对应调用也为False | 预留regret分支不能作为这些公开运行实际使用字段的证据 |
| 独立评估生成下一轮状态 | [metric_utils.py:552](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/evaluation/metric_utils.py#L552)：全部输出头argmax后直接`step_stack.append(preds)` | 漏掉训练/模型内续算的辅助字段重置 |
| 下一轮读取 | [metric_utils.py:62](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/evaluation/metric_utils.py#L62)：离散路径取`step_stack[-1]` | 辅助预测确实进入下一轮，不只是保存用于展示 |
| 嵌入组成 | [model.py:182](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/model.py#L182)：对所有字段的embedding求和 | 辅助字段改变会直接改变计算输入 |
| 评估计分 | [eval_model.py:209](https://github.com/Awni00/algorithmic-generalization-transformer-architectures/blob/52dd950dce8d32a8e72513999cac5a310e01ca53/experiments/evaluation/eval_model.py#L209)：计分切片为`slice(0,4)` | 不统计辅助字段不等于不回灌；这解释了为什么四字段分数也会受损 |

原始`calc_batch_metrics`的另一分支也有相同的全部预测回灌；并非只有fast评估分支出现该行为。

## 3. 为什么它会影响真实计算，而不只是影响统计

假设某节点前四字段均正确、VALUE为4。训练中下一轮接收：

```text
[正确语法, 正确变量名, 正确运算, 4, FALSE, FALSE]
```

辅助输出没有相应任务监督，仍可能argmax为TRUE；raw随后输入可能变成：

```text
[正确语法, 正确变量名, 正确运算, 4, TRUE, FALSE]
```

虽然数值仍是4，输入向量已经增加了`E_REGRET(TRUE) - E_REGRET(FALSE)`。这不是论文规定的纠错步骤，也不是模型训练中必须处理的常规状态；后续注意力和运算会受影响。输出是确定性argmax时，也不能把它描述为每轮随机抽样噪声。

clean做的是恢复最后两项FALSE，而不是填写正确数值、替模型修答案或删除两项嵌入。两项FALSE嵌入仍然存在，保留原权重的输入结构。

## 4. 影响范围与当前状态

- 影响使用此反馈方式的**离散**A/B评估；敏感程度随权重不同。某些权重raw也接近满分，并不使bug消失。
- 当前分支`experiments/evaluation/metric_utils.py`的离散默认路径仍有全部字段回灌：当前约第66、481–482行；本轮未改它。评分修复和辅助反馈修复是不同问题。
- 当前B机制实验在`tools/b_mechanism/b_mechanism_common.py`使用`force_aux_false`，已避免此bug。历史raw保留为缺陷行为的对照，不作为干净算法能力或自纠错收益的主证据。
- C的`nointerm`在第一轮后传的是隐藏向量，辅助解码结果不作为离散状态回灌，不能直接套用这个bug解释C的状态漂移。
- 原始数据压缩包名里的`raw`通常指原始导出，不据此判定反馈方式。
- **已确认公开代码的bug；尚未取得作者Figure 4的实际运行版本/日志。** 不把两个判断合并成“已证实论文所有结果被bug造成”。

本轮是证据核查及文档更正，不涉及新实验、默认评估器改写或历史结果重跑。若后续修改默认入口，应明确标记clean/legacy_raw并保留可追溯性。
