# 本批结果审阅入口

本表为自动计数摘要，不包含机制判断。完整条件和每轮结果见源summary及records。
节点VALUE与四属性按各图比例再等权平均；末两轮要求两轮均正确。空分母不记100%。
C四版本同源，不是四种子；共同输入资格的覆盖必须与全体并列。
examples按观察到的结果类别选最小两个graph_id，仅作解释例子，不是独立测试集。

| 模型 | 条件 | 分支 | 区域/分母 | VALUE | 四属性 | 图均权VALUE |
|---|---|---|---|---:|---:|---:|
| C（原始） | ["late", "semantic", 2] | observed_keep | q; 32节点/32图 | 29 | 3 | 0.90625 |
| C（原始） | ["late", "semantic", 2] | observed_pre | q; 32节点/32图 | 29 | 29 | 0.90625 |
| C（原始） | ["late", "semantic", 2] | observed_solved | q; 32节点/32图 | 29 | 26 | 0.90625 |
| C（原始） | ["late", "semantic", 2] | restore_u_pre | q; 32节点/32图 | 29 | 29 | 0.90625 |
| C（原始） | ["late", "semantic", 2] | clean_pre | q; 32节点/32图 | 31 | 31 | 0.96875 |
| C（保持后训练） | ["late", "semantic", 2] | observed_keep | q; 32节点/32图 | 32 | 31 | 1.0 |
| C（保持后训练） | ["late", "semantic", 2] | observed_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（保持后训练） | ["late", "semantic", 2] | observed_solved | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（保持后训练） | ["late", "semantic", 2] | restore_u_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（保持后训练） | ["late", "semantic", 2] | clean_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（噪声后训练） | ["late", "semantic", 2] | observed_keep | q; 32节点/32图 | 32 | 30 | 1.0 |
| C（噪声后训练） | ["late", "semantic", 2] | observed_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（噪声后训练） | ["late", "semantic", 2] | observed_solved | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（噪声后训练） | ["late", "semantic", 2] | restore_u_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（噪声后训练） | ["late", "semantic", 2] | clean_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（语义后训练） | ["late", "semantic", 2] | observed_keep | q; 32节点/32图 | 32 | 31 | 1.0 |
| C（语义后训练） | ["late", "semantic", 2] | observed_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（语义后训练） | ["late", "semantic", 2] | observed_solved | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（语义后训练） | ["late", "semantic", 2] | restore_u_pre | q; 32节点/32图 | 32 | 32 | 1.0 |
| C（语义后训练） | ["late", "semantic", 2] | clean_pre | q; 32节点/32图 | 32 | 32 | 1.0 |

交付时补充：真实小验证检查记录、运行版本和预算、无资格/零事件、固定案例说明、支持与未区分的解释。
