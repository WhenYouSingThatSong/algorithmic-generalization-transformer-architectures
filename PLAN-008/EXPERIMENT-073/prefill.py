"""Evaluate paired initial VALUE assignments before the first forward."""
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
from run import MODELS, ORDERS, REGISTRY, ROOT, SIZES, case
from design import P

CONDITIONS = ("quarter", "half", "three_quarters", "all", "roots", "internal", "sinks")
ROUNDS = 24
EMPTY = 24


def roles(nodes):
    used = {p for n in nodes for p in n["parents"]}
    return ["root" if not n["parents"] else "internal" if i in used else "sink"
            for i, n in enumerate(nodes)]


def assignments(nodes, size, gid, draw):
    rng = random.Random(P.seed("plan008-initial-v1", size, gid, draw))
    names = sorted(n["name"] for n in nodes)
    numbers = {name: rng.randrange(23) for name in names}
    rng.shuffle(names)
    rank = {name: i for i, name in enumerate(names)}
    kind = roles(nodes)
    masks = {k: {name for name in names if rank[name] < count}
             for k, count in zip(CONDITIONS[:4], (size//4, size//2, 3*size//4, size))}
    for cond, role in zip(CONDITIONS[4:], ("root", "internal", "sink")):
        masks[cond] = {n["name"] for n, r in zip(nodes, kind) if r == role}
    return {k: [numbers[n["name"]] if n["name"] in selected else EMPTY for n in nodes]
            for k, selected in masks.items()}


def inputs(encoded, nodes, values):
    """Materialize an independent input on CPU before embedding or inference."""
    if len(values) != len(nodes):
        raise ValueError("assignment length")
    x = [list(row) for row in encoded]
    for n, v in zip(nodes, values):
        p = n["pos"]
        if x[p][3] != EMPTY or not (0 <= v < 23 or v == EMPTY):
            raise ValueError("invalid initial VALUE")
        x[p][3] = v
    return x


def trajectories(rt, encoded, nodes, treatments):
    torch = rt.torch
    pos = [n["pos"] for n in nodes]
    batch = [inputs(encoded, nodes, vals) for vals in treatments]
    state = rt.init(torch.tensor(batch, device=rt.device))
    frames = []
    for _ in range(ROUNDS):
        state, logits, _ = rt.forward(state)
        frames.append(logits[3][:, pos].argmax(-1))
    return torch.stack(frames, dim=1).cpu().tolist()


def classify(v, gold):
    return "correct" if v == gold else "empty" if v == EMPTY else "wrong" if v < 23 else "other"


def counts(nodes, initial, trace, clean):
    """Raw counts per graph/draw, split by initial node status and source evidence."""
    by = defaultdict(Counter)
    for i, (n, role) in enumerate(zip(nodes, roles(nodes))):
        gold, old, parents = n["truth"], initial[i], n["parents"]
        parent_values = [initial[j] for j in parents]
        implied = gold if not parents else None
        source = "literal"
        if parents:
            source = "parents_incomplete"
            if all(v < 23 for v in parent_values):
                implied = sum(parent_values) % 23
                source = "parents_imply_wrong" if implied != gold else "parents_imply_gold"
        c = by[role, classify(old, gold), source]
        c["observed"] += 1
        c["numeric_conflict"] += int(implied is not None and implied != gold and implied != old)
        if implied is not None and implied != gold and implied != old:
            c["followed_wrong_sum_r1"] += int(trace[0][i] == implied)
        for t in (1, 5, 9, 24):
            v = trace[t-1][i]
            c[f"correct_r{t}"] += int(v == gold)
            c[f"same_initial_wrong_r{t}"] += int(old < 23 and old != gold and v == old)
            c[f"clean_correct_r{t}"] += int(clean[t-1][i] == gold)
            c[f"clean_only_correct_r{t}"] += int(clean[t-1][i] == gold and v != gold)
            c[f"prefill_only_correct_r{t}"] += int(clean[t-1][i] != gold and v == gold)
        c["ever_correct"] += int(any(row[i] == gold for row in trace))
        c["correct_last_two"] += int(all(row[i] == gold for row in trace[-2:]))
        c["ever_lost_initial_gold"] += int(old == gold and any(row[i] != gold for row in trace))
    return by


def summarize(folder):
    """Equal weight per base graph within a layout; draws share one graph."""
    per_graph = defaultdict(Counter)
    for model in MODELS:
        with (folder/(model+"-strata.csv")).open(encoding="utf8") as f:
            for row in csv.DictReader(f):
                group = (model, row["size"], row["order"], row["condition"], row["initial_class"])
                c = per_graph[group, row["gid"]]
                for metric in ("observed", "correct_r1", "correct_r5", "correct_r9",
                               "correct_r24", "clean_correct_r24", "correct_last_two",
                               "same_initial_wrong_r24", "ever_lost_initial_gold"):
                    c[metric] += int(row[metric])
    strata = defaultdict(list)
    for (group, gid), c in per_graph.items():
        strata[group].append(c)
    with (folder/"summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = None
        for (model, size, order, cond, initial), graphs in sorted(strata.items()):
            row = dict(model=model,size=size,order=order,condition=cond,initial_class=initial,
                       represented_graphs=len(graphs),node_draws=sum(c["observed"] for c in graphs))
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
                maps = []
                for g in graphs:
                    ns = g["nodes"]
                    vals = assignments(ns, size, gid, draw)
                    maps.append({k: dict(zip([n["name"] for n in ns], vs)) for k,vs in vals.items()})
                    assert [sum(v != EMPTY for v in vals[k]) for k in CONDITIONS[:4]] == [
                        size//4, size//2, 3*size//4, size]
                    for i in range(size):
                        assert sum(vals[k][i] != EMPTY for k in CONDITIONS[4:]) == 1
                        assert all(a[i] in (EMPTY, vals["all"][i]) for a in vals.values())
                    encoded = tok.encode_tokens(g["tokens"])
                    filled = inputs(encoded, ns, vals["all"])
                    edits = {(i,j) for i,(a,b) in enumerate(zip(encoded,filled))
                             for j,(x,y) in enumerate(zip(a,b)) if x!=y}
                    assert edits == {(n["pos"],3) for n in ns}
                assert maps[0] == maps[1]
    print("Validated graph/layout/draw pairing, masks and input field edits.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-design", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--data-root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, default=ROOT/"tmp/analysis/plan008-exp073")
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
            with gzip.open(raw.with_suffix(".tmp"), "wt", encoding="utf8") as f, report.with_suffix(".tmp").open("w",newline="",encoding="utf8") as tab:
                writer = None
                for size in SIZES:
                    for gid in range(1000, 1001 if args.smoke else 1012):
                        for order in ORDERS:
                            g = case(size,gid,order)[0][0]
                            ns = g["nodes"]
                            enc = rt.tok.encode_tokens(g["tokens"])
                            empty = [EMPTY]*size
                            # Three inputs in every forward, including the final incomplete batch.
                            clean = trajectories(rt,enc,ns,[empty]*3)[0]
                            key = dict(model=model,sha256=spec["sha256"],size=size,gid=gid,order=order)
                            f.write(json.dumps(dict(**key,condition="empty",draw=None,nodes=ns,
                                                    tokens=g["tokens"],initial=empty,trace=clean))+"\n")
                            for draw in range(1 if args.smoke else 3):
                                vals = assignments(ns,size,gid,draw)
                                for off in range(0,len(CONDITIONS),3):
                                    names = CONDITIONS[off:off+3]
                                    batch = [vals[k] for k in names]
                                    traces = trajectories(rt,enc,ns,batch+[empty]*(3-len(batch)))
                                    for cond, trace in zip(names,traces):
                                        info = dict(**key,draw=draw,condition=cond)
                                        f.write(json.dumps(dict(**info,initial=vals[cond],trace=trace))+"\n")
                                        for (role,initial,source), c in counts(ns,vals[cond],trace,clean).items():
                                            row = dict(**info,role=role,initial_class=initial,parent_evidence=source,
                                                       followed_wrong_sum_r1=0)
                                            row.update(c)
                                            if writer is None:
                                                writer = csv.DictWriter(tab,fieldnames=list(row))
                                                writer.writeheader()
                                            writer.writerow(row)
            raw.with_suffix(".tmp").replace(raw)
            report.with_suffix(".tmp").replace(report)
            print(model, folder, flush=True)
            del rt
    summarize(folder)


if __name__ == "__main__":
    main()
