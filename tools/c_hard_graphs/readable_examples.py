"""Readable ABC examples: identical first lane, four addition layers, three widths."""
import argparse
import json
from pathlib import Path

from design import render

WIDTHS = (1, 4, 24)
ROOTS = (3, 4, 9, 10, 1, 6)
MODELS = ("A_s42", "B_s42", "C_s42", "C_s1234")
ROUNDS = 24
FOLLOW = 16


def graph(width, offset=0):
    roots = [f"k{i}" for i in range(6)]
    ops = list(zip(roots, ROOTS))
    ops[0] = ("k0", ROOTS[0] + offset)
    ids = list(range(6))
    for layer in range(1, 5):
        for lane in range(width):
            # Lane zero is 3+4, then +9, +10, +1 in every drawing.
            left = roots[lane % 6] if layer == 1 else f"L{layer-1}_{lane}"
            right = roots[(lane + layer + lane // 6) % 6]
            ops.append((f"L{layer}_{lane}", (left, right)))
            ids.append(6 + 24 * (layer - 1) + lane)
    # Same variable spelling for every shared equation across widths.
    return render(ops, [f"x_{(37*i+11)%128}" for i in ids], True)


def run_graph(rt, width):
    from attribution_cases import patch_state
    from research_probes import encode, decoded

    base, donor = graph(width), graph(width, 5)
    nodes = base["nodes"]
    pos = [n["pos"] for n in nodes]
    truth = [n["truth"] for n in nodes]
    u, q = 6, 6 + width
    x = encode(rt, [base, donor])
    labels = x[0, pos, :3].tolist()
    state = rt.init(x)
    clean, events = [], {}
    observed = set()
    for t in range(1, ROUNDS + 1):
        state, logits, _ = rt.forward(state)
        reads = decoded(logits, pos).tolist()
        values = [z[3] for z in reads[0]]
        clean.append(values)
        clocks = {"刚算对": values[u] == truth[u], "整图算完": values == truth}
        for stage, ready in clocks.items():
            if not ready or stage in observed:
                continue
            observed.add(stage)
            record = {"round": t, "before": values, "donor_read": reads[1][u]}
            events[stage] = record
            if (reads[1][u] != labels[u] + [donor["nodes"][u]["truth"]]
                    or reads[0][u][:3] != labels[u]):
                record["status"] = "原位置或供体读出未通过"
                continue
            pair = state[:1].repeat(2, 1, 1)
            patch_state(rt, pair[1:], state[1:], [pos[u]])
            written = rt.decode(pair)[1, pos[u], :4].tolist()
            if written != reads[1][u]:
                raise ValueError("written state disagrees with donor")
            record.update(status="已执行", written=written, steps=[])
            for _ in range(FOLLOW):
                pair, result, _ = rt.forward(pair)
                read = result[3][:, pos].argmax(-1).tolist()
                record["steps"].append({"clean": read[0], "changed": read[1]})
    for stage in ("刚算对", "整图算完"):
        events.setdefault(stage, {"status": "24 轮内未达到起点"})
    return dict(nodes=nodes, target=u, child=q, clean=clean, events=events)


def display(v):
    return {23: "无对应值", 24: "空"}.get(v, str(v))


def report(results):
    lines = ["# 三张图的逐轮实测", "", "计算取模 23。每张图有六个输入和四层加法。"
             "层宽只统计运算位置。注错表中的两个数依次为被改的位置和它的直接子节点。", ""]
    for model, data in results.items():
        lines += [f"## {model}", "", f"权重 SHA-256 为 `{data['sha256']}`。", ""]
        for width, run in data["graphs"].items():
            nodes = run["nodes"]
            lines += [f"### 每层 {width} 个算式", "", "下表只数上一轮所有直接父值都读对的运算位置。真值取原题标签。", "",
                      "| 轮次 | 已具备正确父值 | 本轮正确 | 本轮空 | 本轮其他数 | 本轮无对应值 |",
                      "|---:|---:|---:|---:|---:|---:|"]
            for t in range(2, ROUNDS + 1):
                prev, now = run["clean"][t-2:t]
                ready = [i for i,n in enumerate(nodes) if n["parents"] and
                         all(prev[j] == nodes[j]["truth"] for j in n["parents"])]
                counts = [len(ready), sum(now[i] == nodes[i]["truth"] for i in ready),
                          sum(now[i] == 24 for i in ready),
                          sum(now[i] < 23 and now[i] != nodes[i]["truth"] for i in ready),
                          sum(now[i] == 23 for i in ready)]
                lines.append("| " + " | ".join(map(str, [t, *counts])) + " |")
            lines += ["", "| 位置 | 题面 | 真值 | 第 3 轮 | 第 5 轮 | 第 9 轮 | 第 16 轮 | 第 24 轮 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
            for i, node in enumerate(nodes):
                parents = node["parents"]
                expr = " + ".join(nodes[p]["slot"] for p in parents) if parents else str(node["truth"])
                reads = [display(run["clean"][t-1][i]) for t in (3, 5, 9, 16, 24)]
                lines.append("| " + " | ".join([node["slot"], expr, str(node["truth"]), *reads]) + " |")
            u, q = run["target"], run["child"]
            for stage, event in run["events"].items():
                lines += ["", f"**{stage}**　{event['status']}（干净轮次 {event.get('round', '—')}）", ""]
                if "steps" not in event:
                    continue
                lines += ["| 注错后轮数 | 同起点干净分支 | 写错分支 |", "|---:|---|---|"]
                for step in range(FOLLOW + 1):
                    if step:
                        row = event["steps"][step-1]
                        a, b = row["clean"], row["changed"]
                    else:
                        a = event["before"]
                        b = list(a)
                        b[u] = event["written"][3]
                    lines.append(f"| {step} | {display(a[u])}，{display(a[q])} | {display(b[u])}，{display(b[q])} |")
            lines.append("")
    return "\n".join(lines)


def main():
    import torch
    from attribution_queue import model_specs
    from hard_eval import ROOT, PLAN, P, RECEIPT
    from runtime import Runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT/"tmp/analysis/abc-readable-examples")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = args.data_root / PLAN.relative_to(ROOT)
    if P.sha(plan) != P.read(RECEIPT)["plan_sha256"]:
        raise ValueError("unexpected checkpoint plan")
    specs = model_specs(P.read(plan), args.data_root)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    with torch.inference_mode():
        for model in MODELS:
            rt = Runtime(specs[model], args.device)
            graphs = {w: run_graph(rt, w) for w in WIDTHS}
            results[model] = dict(sha256=specs[model]["sha256"], graphs=graphs)
            del rt
    (args.out/"traces.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf8")
    (args.out/"逐题实测.md").write_text(report(results), encoding="utf8")


if __name__ == "__main__":
    main()
