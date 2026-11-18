# 历史名称映射（只用于追溯）

日期：2026-11-18。[当前统一命名](../../model-names.md)是后续展示规范；此表记录旧叫法，不改动历史权重/原始数据。

| 历史名称/机器ID | 当前展示名 | 必须保留的区别 |
|---|---|---|
| A′、Aprime、A-L4D384fast、当前manifest model_a | A | 指L4H16D384 fast无纠错对照，不泛指所有历史A |
| B-M6、BM6、当前manifest model_b | B（本组） | 本批seed42/e92；训练为last污染，不能改称已训练过所有旧值错误 |
| cyhB、cyh B | B（外部复现） | 来源及seed/epoch必须标明；同seed42与本组B不是同权重 |
| 旧C₀/C0、固定N32/depth6的M4/nointerm可用权重 | C（原始） | 当前42/1234/2026；C是连续反馈，不是离散无纠错A |
| C+（笼统研究方向） | C的纠错后训练方向 | 不对应唯一已有checkpoint，不作为正式表格中的权重身份 |
| initial（本轮C注错小试源） | C（原始） | seed2026/e143，SHA以源meta为准 |
| hold（本轮C注错小试源） | C（保持后训练） | 属该注错小试批次，与前一批长程小试的保持权重区分 |
| noise（本轮C注错小试源） | C（噪声后训练） | 保持设置加噪声；noise也可能是测试来源/分支名，注意字段含义 |
| semantic（本轮C注错小试源） | C（语义后训练） | 保持设置加语义扰动，不能直接称已证明更会纠错 |
| 新C₀、C₀′、C0 2Val、MixedN8–32/depth12 | 历史连续配置（归档） | 是另一训练配方，不直接并入当前C（原始）的三种子统计 |
| vocab33事故配置、已取消运行 | 无效/已取消历史运行 | 不能映射成可用C权重 |
| A-M5、A-L4、A-L4fast（D256）、L2corr等 | 历史离散配置（注明结构/训练） | 不自动改叫当前A/B；层数、宽度、fast/slow配方不同 |
| F、W（旧优先任务标签） | C：输入对齐诊断；A/B：合法重新赋值 | 属实验任务，不是新增模型 |
| C报告中的情况A/B/C；CSV字段A_outfix_inwrong/B_allu_qinwrong_ctrl/C_allreadouts_qwrong_ctrl | 现象①/②/③ | 原CSV列保留以匹配hash；新文档不用模型字母给现象编号 |

本轮重新赋值源身份：A为A-L4D384fast s42/e86，B为本组B-M6 s42/e92。完整源hash见priority-reassignment-audit.json。
本轮C输入对齐的四个展示版本均来自同一seed2026连续基线及其后训练分支，源hash见c-function-audit-audit.json。它们不是四个独立种子。
