# C 基础评估套件

[方案/执行交接](../../ai-collab/handoff/2026-11-15-c-basic-evaluation-execution.md)。

本批评估现有initial s2026/e143及hold/noise/semantic，不新训练。

1. `analyze_existing.py`：CPU读取上一批records，输出目标读出×子节点损伤联合表、hold与semantic逐题配对，避免拼接两组边际均值。
2. `run_clean.py` / `summarize_clean.py`：统一四权重、六档各64图、batch4、到每图D+64；逐轮与固定切片VALUE/四属性、分字段、节点角色/深度/父数、求解速度和完成后漂移。未完成图保留。
3. `run_pair.py` / `summarize_pair.py`：复用128验证图、结构选一对u→q，保持/晚期两时点，clean/原样写回/只改u/只改q/一起改，以及对应同值异源控制。单次替换后16轮自由续算；输出实际输入和输出的数值，不将解码写回。

所有原始records/NPZ留tmp。CLI输出目录必须不存在，`--smoke`使用独立目录。两个GPU入口都先加载并核对上一批三组meta/计划/固定checkpoint，四模型资格共同交集和全固定样本分开。

本地6项stdlib测试通过；1项torch行替换检查交服务器。语法检查通过。尚未运行GPU评估。
