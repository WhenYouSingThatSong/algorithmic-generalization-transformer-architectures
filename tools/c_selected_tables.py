"""CPU reporting for the two selected normal-window checkpoints."""
import csv
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from c_selected_eval import MODELS, RECEIPT, read

NAMES = ("01-基础拟合与干净轨迹", "02-第二层更新定位", "03-末层执行与任务后果",
         "04-恢复与传播", "05-同图混合", "06-修回后的使用")
ARMS = ("clean", "wrong", "empty_only", "empty_and_wrong")
HORIZONS = (0, 4, 16, 64)
LABELS = {
    "frontier": "结构上新可算时", "middle": "求解中途", "actual_complete": "首次实际完成",
    "train_end": "标准轮数完成检查", "clean": "同起点干净", "wrong": "单点错值",
    "empty_only": "独立位置清空", "empty_and_wrong": "空位与错值同时出现",
    "parent_child_consistent_wrong": "父子局部一致但全局错误",
    "new_calculation": "已就绪仍空", "direct_conflict": "已有值直接冲突",
    "consistent_correct": "正确且一致", "indirect_consistent": "局部一致但全局错误",
    "blocked": "父值未齐", "non_numeric_other": "其他非数值", "post1": "首轮",
    "all": "全部", "target": "目标位置", "children": "直接下游", "descendants": "全部下游",
    "fill": "空位", "healthy": "健康位置", "conflict": "直接冲突",
}
HEAD = ("种子", "数据档", "时点", "条件", "范围", "指标", "有分母图数或训练周期数",
        "分子合计", "分母合计", "合并比值", "逐图等权均值或周期值")


class Table:
    """One add per graph and cell. Empty denominators remain visible."""
    def __init__(self, model, split):
        self.prefix = (model, split)
        self.cells = defaultdict(lambda: [0, 0, 0, 0.0])

    def add(self, timing, arm, scope, metric, num, den):
        key = tuple(LABELS.get(v, v) for v in (timing, arm, scope, metric))
        cell = self.cells[key]
        cell[0] += num
        cell[1] += den
        if den:
            cell[2] += 1
            cell[3] += num / den

    def write(self, writer):
        for key, (num, den, graphs, summed) in sorted(self.cells.items()):
            writer.writerow((*self.prefix, *key, graphs, num, den, num / den if den else "",
                             summed / graphs if graphs else ""))


def clean_graph(table, r):
    nodes, gold = r["nodes"], [v[3] for v in r["truth"]]
    rows, d = r["rows"], r["D"]
    n = len(nodes)
    for extra in HORIZONS:
        out = rows[d + extra - 1]["output"]
        right = sum(z[0] == v for z, v in zip(out, gold))
        group = (f"标准求解后追加{extra}轮", "clean", "全部定义位置")
        for metric, count in (("节点数值正确", right), ("节点四字段同时正确", sum(z[1] for z in out)),
                              ("数值对但变量名错", sum(z[0] == v and not z[3] for z, v in zip(out, gold)))):
            table.add(*group, metric, count, n)
        table.add(*group, "整图数值全对", int(right == n), 1)
        error_counts = Counter({n - right: 1})
        for errors in range(n + 1):
            table.add(*group, f"恰有{errors}个错节点的图", error_counts[errors], 1)
    # Each transition is observed once. Initialize from the actual first input.
    for field, column in (("节点数值", 0), ("变量名", 3), ("四字段", 1)):
        prev = []
        for before, truth in zip(rows[0]["input"], r["truth"]):
            prev.append(before[3] == truth[3] if column == 0 else
                        before[1] == truth[1] if column == 3 else before == truth)
        ever = prev.copy()
        drifted = [False] * n
        for step, row in enumerate(rows, 1):
            counts = Counter()
            for i, z in enumerate(row["output"]):
                ok = z[0] == gold[i] if column == 0 else bool(z[column])
                if ok and not ever[i]:
                    counts["首次正确"] += 1
                if prev[i] and not ok:
                    counts["正确转错"] += 1
                if ok and not prev[i] and drifted[i]:
                    counts["曾对后错再恢复"] += 1
                drifted[i] |= prev[i] and not ok
                ever[i] |= ok
                prev[i] = ok
            timing = f"标准求解后追加{step-d}轮" if step >= d else f"从输入起第{step}轮（标准预算内）"
            for metric in ("首次正确", "正确转错", "曾对后错再恢复"):
                table.add(timing, "clean", field, metric, counts[metric], n)


def standard(tables, path, split):
    uses = defaultdict(dict)
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            gid = r["gid"]
            if r["split"] != split:
                raise ValueError("trajectory split mismatch")
            if r["module"] == "clean" and r["status"] == "ok":
                if gid in seen:
                    raise ValueError("duplicate clean graph")
                seen.add(gid)
                clean_graph(tables[0], r)
            elif r["module"] == "reuse" and r["status"] == "ok":
                idx = r["nodes"].index(r["new_node"])
                used = all(z["output"][idx][0] == r["truth"][idx][3] for z in r["rows"][-2:])
                parent = r["u_readout"][3] == r["truth"][r["nodes"].index(r["u"])][3]
                uses[gid, r["lag"]][r["condition"]] = (parent, used)
    if seen != set(range(128)):
        raise ValueError(f"{split}: incomplete clean graph IDs")
    if split.startswith("N32"):
        for lag in (0, 1, 2, 4):
            for gid in range(8):
                arms = uses.get((gid, lag), {})
                paired = "clean" in arms and "perturbed" in arms
                capable = paired and arms["clean"][1]
                group = (f"新算式前续算{lag}轮", "新消费者", "末两轮")
                tables[5].add(*group, "两臂均有记录／全部预选图", int(paired), 1)
                tables[5].add(*group, "干净可用／两臂齐全", int(capable), int(paired))
                for parent in (False, True):
                    for used in (False, True):
                        metric = f"目标读{'对' if parent else '错'}且新下游{'对' if used else '错'}"
                        tables[5].add(*group, metric, int(capable and arms["perturbed"] == (parent, used)), int(capable))
    else:
        tables[5].add("新消费者", "未登记该档", "末两轮", "可评图数", 0, 0)


def focused(tables, records):
    arms_by_gid = defaultdict(dict)
    keys = set()
    for r in records:
        t, arm, gid = r["timing"], r["condition"], r["gid"]
        key = (gid, t, arm)
        if key in keys:
            raise ValueError(f"duplicate focused event: {key}")
        keys.add(key)
        ok = r["status"] == "ok"
        tables[3].add(t, arm, "覆盖", "执行成功／候选", int(ok), 1)
        if not ok:
            tables[3].add(t, arm, "覆盖", r["status"] + "（看分子）", 1, 1)
            continue
        detail = r.get("selected_detail", {})
        if "first_wrong_positions" not in detail:
            raise ValueError("missing focused details; use c_selected_eval.py focused for this model")
        if t == "actual_complete" and arm in ARMS:
            arms_by_gid[gid][arm] = r
        counts = Counter()
        for change in detail["input_changes"]:
            counts["数值实际改变"] += 1
            counts["正确变错数值"] += change["before"] == change["truth"] and 0 <= change["after"] < 23
            counts["实际变为空"] += change["after"] == 24
        for name in ("数值实际改变", "正确变错数值", "实际变为空"):
            tables[3].add(t, arm, "每张执行图的位置数", name, counts[name], 1)
        for item in r["metrics"]:
            c = Counter(item["counts"])
            if not c["observed"]:
                continue
            scope = LABELS.get(item["scope"], item["scope"]) + "／" + LABELS.get(item["category"], item["category"])
            for metric, num, den in (("定位召回", c["tp"], c["reference"]),
                                     ("定位精确率", c["tp"], c["planned"]),
                                     ("误报率", c["fp"], c["observed"] - c["reference"])):
                tables[1].add(t, arm, scope, metric, num, den)
            for metric, col in (("局部执行正确", "execute_correct"), ("符合原题", "gold_correct"),
                                ("计划与改动一致", "aligned"), ("计划与局部执行均对", "jointly_correct")):
                tables[2].add(t, arm, scope, metric, c[col], c["observed"])
        # Aggregate within graph before computing macro averages.
        conflicts = defaultdict(Counter)
        for step, target, old_ok, plan, change, local, gold, count in detail["conflicts"]:
            label = "目标" if target else "其他位置"
            for scope in ("全窗", "首轮") if step == 1 else ("全窗",):
                c = conflicts[scope, label, old_ok]
                c["n"] += count
                c["局部对且原题错"] += count * (local and not gold)
                c["已标且改值且局部对但原题错"] += count * (plan and change and local and not gold)
                c["已标且改值且原题对"] += count * (plan and change and gold)
        for (scope, label, old_ok), c in conflicts.items():
            desc = f"{scope}／{label}／直接冲突且改前{'全局对' if old_ok else '全局错'}"
            for metric in ("局部对且原题错", "已标且改值且局部对但原题错", "已标且改值且原题对"):
                tables[2].add(t, arm, desc, metric, c[metric], c["n"])
        for role, raw in r["endpoints"].items():
            c = Counter(raw)
            for name, num, den in (("受损比例", "damaged", "observed"),
                                   ("输入受损首轮修回", "first_round_repaired", "input_damaged"),
                                   ("受损后曾回正", "restored", "damaged"),
                                   ("受损后末两轮正确", "stable_after_damage", "damaged"),
                                   ("已回正后再错", "reerror", "restored"),
                                   ("首次回正延迟（轮）", "restoration_delay", "restored"),
                                   ("每图新增错节点轮数", "wrong_rounds", None)):
                tables[3].add(t, arm, role, name, c[num], c[den] if den else 1)
    expected = {(gid, t, arm) for gid in range(32) for t in ("frontier","middle","train_end","actual_complete")
                for arm in (("clean","wrong","empty_only","empty_and_wrong","parent_child_consistent_wrong")
                            if t == "actual_complete" else ("clean","wrong"))}
    if keys != expected:
        raise ValueError("incomplete focused conditions or candidate IDs")
    for gid in range(32):
        arms = arms_by_gid[gid]
        valid = all(arm in arms for arm in ARMS)
        tables[4].add("actual_complete", "四臂配对", "覆盖", "四臂均成功／候选", int(valid), 1)
        if not valid:
            continue
        mix = arms["empty_and_wrong"]
        u, v = mix["target"], mix["fill_target"]
        if u == v or arms["empty_only"]["fill_target"] != v or any(
                r["target"] != u or r["start"] != mix["start"] for r in arms.values()):
            raise ValueError("mixed arms are not paired at identical positions and time")
        for arm in ARMS:
            r = arms[arm]
            d = r["selected_detail"]
            wrong = set(d["first_wrong_positions"])
            scope = "四臂共同合格图／首轮"
            tab = tables[4]
            tab.add("actual_complete", arm, scope, "指定错值位置符合原题", int(u not in wrong), 1)
            tab.add("actual_complete", arm, scope, "指定空位位置符合原题", int(v not in wrong), 1)
            others = d["node_count"] - 2
            tab.add("actual_complete", arm, scope, "其余位置符合原题", others - len(wrong - {u, v}), others)
            tab.add("actual_complete", arm, scope, "整张图数值正确", int(not wrong), 1)
            actual_mix = d["mixed_all_gold_correct"] is not None
            tab.add("actual_complete", arm, scope, "输入实际具备三类", int(actual_mix), 1)
            tab.add("actual_complete", arm, scope, "三类局部计划与执行均对",
                    int(actual_mix and r["rounds"][0]["jointly_correct"]), int(actual_mix))
            tab.add("actual_complete", arm, scope, "三类输出均符合原题",
                    int(actual_mix and d["mixed_all_gold_correct"]), int(actual_mix))
            for role, c in d["first_round_roles"].items():
                tab.add("actual_complete", arm, scope + "／" + LABELS[role], "该类符合原题",
                        c["gold_correct"], c["observed"])


def stress_summary(table, summary):
    # These frozen summaries retain sums and per-graph means separately.
    for item in summary["counts"]:
        key = item["key"]
        if len(key) != 9 or key[1] not in ("sustained", "noise", "pair"):
            continue
        _, module, t, arm, hold, scale, direction, role, field = key
        if module == "sustained":
            arm = f"{arm}／持续{hold}轮后释放"
        if module == "noise":
            arm = f"噪声强度{scale}／方向{direction}"
        c = Counter(item["counts"])
        for name, a, b in (("首轮修回", "first_round_repair", "actual_injected"),
                           ("受损后末两轮正确", "stable", "damaged"),
                           ("曾受损", "damaged", "observed")):
            # Counts pool nodes; do not claim a per-graph mean from a single pooled entry.
            yield (table.prefix + (t, arm, f"标准干预／{role}／{field}", name, "",
                                   c[a], c[b], c[a]/c[b] if c[b] else "", ""))


def natural_summary(table, outcome):
    for item in outcome["natural"]:
        c = Counter(item["counts"])
        window = ("输入起至标准轮数加一" if item["window"] == "normal"
                  else "标准轮数加一至观测末轮")
        for name, a, b in (("自然由对变错", "drifted", "ever_correct"),
                           ("漂移后曾恢复", "ever_recovered", "drifted"),
                           ("漂移后末两轮正确", "last_two_after_drift", "drifted")):
            yield table.prefix + (window, "自然运行", item["field"], name, "",
                                  c[a], c[b], c[a]/c[b] if c[b] else "", "")


def probe_records(table, probe):
    for r in probe["records"]:
        if r["status"] == "no_target":
            continue
        group = (r["timing"], r["condition"], "冻结原始C／供体" + r["origin"])
        accepted = r["status"] == "ok"
        eligible = accepted and r["eligible"]
        intended = eligible and r["receiver_reads_intended"]
        table.add(*group, "供体合格／候选", int(accepted), 1)
        table.add(*group, "接收方读出预期字段／双方合格", int(intended), int(eligible))
        table.add(*group, "首轮下游匹配预期错值／接收方读出预期字段",
                  int(intended and r["first_expected_wrong"]), int(intended))


def write_report(paths, output, supplements=None):
    receipt = read(RECEIPT)
    finals = {r["model"]: r["final_ckpt"]["sha256"] for r in receipt["runs"]}
    inputs = {}
    for path in paths:
        meta = read(path / "meta.json")
        model = meta["source"]["model"]
        if (model not in MODELS or model in inputs or meta["status"] != "complete"
                or meta["checkpoint"] != finals[model] or meta["source"]["manifest"] != receipt["manifest_sha256"]):
            raise ValueError(f"unexpected evaluation identity: {path}")
        inputs[model] = path
    if set(inputs) != set(MODELS):
        raise ValueError("both selected seeds required")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    from contextlib import ExitStack
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory) / "tables"
        stage.mkdir()
        with ExitStack() as files:
            writers = []
            for name in NAMES:
                f = files.enter_context((stage / (name + ".csv")).open("w", encoding="utf-8-sig", newline=""))
                writer = csv.writer(f)
                writer.writerow(HEAD)
                writers.append(writer)
            for model, path in sorted(inputs.items()):
                seed = model.split("s")[-1]
                history = read(RECEIPT.parent / "fit-history" / (model + ".json"))
                if history["final_sha256"] != finals[model]:
                    raise ValueError("fit history mismatch")
                fit = Table(seed, "训练及开发集")
                for e in history["epochs"]:
                    t = f"后训练周期{e['epoch']}"
                    fit.add(t, "训练", "训练图加权均值", "训练总损失", e["graph_weighted_mean_training_loss"], 1)
                    fit.add(t, "训练", "末批次", "训练总损失", e["last_batch_training_loss"], 1)
                    dev = e.get("dev_standard") or {}
                    for key, label in (("mean_value","节点数值正确"),("mean_first4","节点四字段同时正确"),
                                       ("graph_value_all","整图数值全对")):
                        if key in dev:
                            fit.add(t, "开发集", f"{dev['graphs']}图标准轮数", label, dev[key], 1)
                fit.write(writers[0])
                for size in (32, 64, 128):
                    for depth in ("shallow", "deep"):
                        split = f"N{size}_{depth}"
                        band = "≤6" if depth == "shallow" else "7–12"
                        tables = [Table(seed, f"{size}节点，最大依赖深度{band}") for _ in writers]
                        name = split + "-classes.json"
                        classes = read(path / name)
                        if supplements and (supplements / model / name).exists():
                            extra = read(supplements / model / name)
                            if extra["checkpoint"] != finals[model] or extra["manifest"] != receipt["manifest_sha256"]:
                                raise ValueError("supplement identity mismatch")
                            classes = extra
                        if any(r["split"] != split for r in classes["records"]):
                            raise ValueError("focused split mismatch")
                        focused(tables, classes["records"])
                        standard(tables, path / (split + "-standard.jsonl"), split)
                        summary = read(path / (split + "-standard-summary.json"))
                        writers[3].writerows(stress_summary(tables[3], summary))
                        writers[3].writerows(natural_summary(tables[3], read(path / (split + "-outcomes.json"))))
                        if size != 64:
                            probe_records(tables[5], read(path / (split + "-frozen-probe.json")))
                        for table, writer in zip(tables, writers):
                            table.write(writer)
        stage.rename(output)
