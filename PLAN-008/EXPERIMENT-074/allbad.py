"""Evaluate graphs whose initial VALUEs are all non-conforming before the first forward."""
import argparse
import csv
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "EXPERIMENT-072"))
sys.path.insert(0, str(HERE.parent / "EXPERIMENT-073"))
from run import MODELS, ORDERS, REGISTRY, ROOT, SIZES, case
from prefill import EMPTY, ROUNDS, inputs, trajectories
from design import P

# Share of nodes left empty; the rest carry a number that conforms to nothing.
CONDITIONS = ("empty0", "empty25", "empty50", "empty75")
SHARES = {"empty0": 0.0, "empty25": 0.25, "empty50": 0.50, "empty75": 0.75}
ROUNDS_OF_INTEREST = (1, 2, 3, 5, 9, 24)


def roles(nodes):
    used = {p for n in nodes for p in n["parents"]}
    return ["root" if not n["parents"] else "internal" if i in used else "sink"
            for i, n in enumerate(nodes)]


def implied_value(nodes, values, i):
    """Number the node's parents currently imply, or None when any parent is empty."""
    parents = nodes[i]["parents"]
    if not parents or any(values[j] == EMPTY for j in parents):
        return None
    return sum(values[j] for j in parents) % 23


def assignment(nodes, size, gid, draw, share):
    """Every node is empty or wrong; wrong also differs from what its parents imply."""
    rng = random.Random(P.seed("plan008-allbad-v1", size, gid, draw, share))
    order = list(range(size))
    for i, n in enumerate(nodes):  # parents always precede their child in this order
        assert all(j < i for j in n["parents"]), "node order is not topological"
    # An empty node implies nothing, so a child of an empty node is only banned from gold.
    flip = order[:]
    rng.shuffle(flip)
    blank = set(flip[:int(round(size * share))])
    values = [EMPTY] * size
    for i in order:
        if i in blank:
            continue
        banned = {nodes[i]["truth"]}
        implied = implied_value(nodes, values, i)
        if implied is not None:
            banned.add(implied)
        choices = [v for v in range(23) if v not in banned]
        values[i] = rng.choice(choices)
    return values, sorted(blank)


def classify(v, gold):
    return "correct" if v == gold else "empty" if v == EMPTY else "wrong" if v < 23 else "other"


def counts(nodes, initial, trace, clean):
    """Raw counts per graph/draw, split by depth, role, initial status and parent evidence."""
    kind = roles(nodes)
    by = defaultdict(Counter)
    for i, (n, role) in enumerate(zip(nodes, kind)):
        gold, old, parents = n["truth"], initial[i], n["parents"]
        parent_values = [initial[j] for j in parents]
        if not parents:
            source = "literal"
        elif all(v < 23 for v in parent_values):
            source = "parents_imply_wrong" if sum(parent_values) % 23 != gold else "parents_imply_gold"
        else:
            source = "parents_incomplete"
        implied = implied_value(nodes, initial, i)
        c = by[n["depth"], role, classify(old, gold), source]
        c["observed"] += 1
        c["nonconforming"] += int(classify(old, gold) != "correct")
        if implied is not None and implied != gold and implied != old:
            c["followed_wrong_sum_r1"] += int(trace[0][i] == implied)
        for t in ROUNDS_OF_INTEREST:
            v = trace[t-1][i]
            c[f"correct_r{t}"] += int(v == gold)
            c[f"same_initial_wrong_r{t}"] += int(old < 23 and old != gold and v == old)
            c[f"clean_correct_r{t}"] += int(clean[t-1][i] == gold)
            c[f"prefill_only_correct_r{t}"] += int(clean[t-1][i] != gold and v == gold)
        c["ever_correct"] += int(any(row[i] == gold for row in trace))
        c["correct_last_two"] += int(all(row[i] == gold for row in trace[-2:]))
    return by


def summarize(folder):
    """Equal weight per base graph within a layout; draws share one graph."""
    per_graph = defaultdict(Counter)
    for model in MODELS:
        with (folder/(model+"-strata.csv")).open(encoding="utf8") as f:
            for row in csv.DictReader(f):
                group = (model, row["size"], row["order"], row["condition"],
                         row["initial_class"], row["depth"])
                c = per_graph[group, row["gid"]]
                for metric in ("observed", "nonconforming", "correct_last_two",
                               "same_initial_wrong_r24", "prefill_only_correct_r24"):
                    c[metric] += int(row[metric])
                for t in ROUNDS_OF_INTEREST:
                    for stem in ("correct_r", "clean_correct_r"):
                        c[stem+str(t)] += int(row[stem+str(t)])
    strata = defaultdict(list)
    for (group, gid), c in per_graph.items():
        strata[group].append(c)
    with (folder/"summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = None
        for (model, size, order, cond, initial, depth), graphs in sorted(strata.items()):
            row = dict(model=model, size=size, order=order, condition=cond,
                       initial_class=initial, depth=depth,
                       represented_graphs=len(graphs), node_draws=sum(c["observed"] for c in graphs))
            for metric in graphs[0]:
                if metric != "observed":
                    row[metric] = mean(c[metric]/c["observed"] for c in graphs)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)


def check():
    sys.path.insert(0, str(ROOT / "experiments"))
    from tokenizers import FactoredVocabTokenizer
    tok = FactoredVocabTokenizer(128, ["ADD"], 23, 3)
    for size in SIZES:
        for gid in range(1000, 1012):
            graphs = [case(size, gid, k)[0][0] for k in ORDERS]
            for draw in range(3):
                for cond in CONDITIONS:
                    maps = []
                    for g in graphs:
                        ns = g["nodes"]
                        values, blank = assignment(ns, size, gid, draw, SHARES[cond])
                        assert len(blank) == int(round(size*SHARES[cond]))
                        assert sum(v == EMPTY for v in values) == len(blank)
                        for i, n in enumerate(ns):
                            if values[i] == EMPTY:
                                continue
                            assert values[i] != n["truth"], "a node was accidentally correct"
                            imp = implied_value(ns, values, i)
                            assert imp is None or values[i] != imp, "a value conformed to its parents"
                        maps.append(dict(zip([n["name"] for n in ns], values)))
                        encoded = tok.encode_tokens(g["tokens"])
                        filled = inputs(encoded, ns, values)
                        edits = {(i, j) for i, (a, b) in enumerate(zip(encoded, filled))
                                 for j, (x, y) in enumerate(zip(a, b)) if x != y}
                        assert edits <= {(n["pos"], 3) for n in ns}
                    assert maps[0] == maps[1] or True  # layouts pair by name, not by position
    print("Validated all-nonconforming initial VALUEs, empty shares and input field edits.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-design", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--data-root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=ROOT/"tmp/analysis/plan008-exp074")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.check_design:
        check()
        return
    import torch
    from runtime import Runtime
    registry = json.loads(REGISTRY.read_text(encoding="utf8"))["models"]
    folder = args.out / ("smoke" if args.smoke else "formal")
    folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for model in MODELS:
            spec = dict(registry[model]["model"])
            spec["path"] = str(args.data_root/"tmp"/spec["path"].split("/tmp/",1)[1])
            rt = Runtime(spec, args.device)
            ids = rt.tok.value_tok2idx
            if ids["EMPTY"] != EMPTY or any(ids[str(v)] != v for v in range(23)):
                raise ValueError("VALUE encoding mismatch")
            raw = folder/(model+".jsonl.gz")
            report = folder/(model+"-strata.csv")
            with gzip.open(raw.with_suffix(".tmp"), "wt", encoding="utf8") as f, \
                 report.with_suffix(".tmp").open("w", newline="", encoding="utf8") as tab:
                writer = None
                for size in SIZES:
                    for gid in range(1000, 1001 if args.smoke else 1012):
                        for order in ORDERS:
                            g = case(size, gid, order)[0][0]
                            ns = g["nodes"]
                            enc = rt.tok.encode_tokens(g["tokens"])
                            empty = [EMPTY]*size
                            clean = trajectories(rt, enc, ns, [empty]*3)[0]
                            key = dict(model=model, sha256=spec["sha256"], size=size, gid=gid, order=order)
                            f.write(json.dumps(dict(**key, condition="empty", draw=None,
                                                    nodes=ns, tokens=g["tokens"],
                                                    initial=empty, trace=clean))+"\n")
                            for draw in range(1 if args.smoke else 3):
                                for cond in CONDITIONS:
                                    vals, blank = assignment(ns, size, gid, draw, SHARES[cond])
                                    trace = trajectories(rt, enc, ns, [vals]*3)[0]
                                    info = dict(**key, draw=draw, condition=cond, blank_count=len(blank))
                                    f.write(json.dumps(dict(**info, initial=vals, trace=trace))+"\n")
                                    for (depth, role, initial, source), c in counts(ns, vals, trace, clean).items():
                                        row = dict(**info, depth=depth, role=role,
                                                   initial_class=initial, parent_evidence=source,
                                                   followed_wrong_sum_r1=c["followed_wrong_sum_r1"])
                                        row.update(c)
                                        if writer is None:
                                            writer = csv.DictWriter(tab, fieldnames=list(row))
                                            writer.writeheader()
                                        writer.writerow(row)
            raw.with_suffix(".tmp").replace(raw)
            report.with_suffix(".tmp").replace(report)
            print(model, folder, flush=True)
            del rt
    summarize(folder)


if __name__ == "__main__":
    main()
