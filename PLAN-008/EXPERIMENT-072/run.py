"""PLAN-008 paired graph layouts and recovery tests."""
import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/c_hard_graphs"))
from recovery_graphs import family
from design import render
from summary import summarize

SIZES = (32, 64, 128)
MODELS = ("A_s42", "B_s42", "C_s42", "C_s1234", "C_s2026")
ORDERS = ("layered", "interleaved")
REGISTRY = ROOT / "docs/records/2026-11-20/inline-rebinding/rebinding-identity.json"


def case(size, gid, order):
    branches, spine = family(gid, size, 4)
    original = branches[0]["nodes"]
    indices = list(range(size))
    if order == "layered":
        indices.sort(key=lambda j: original[j]["depth"])
    arms = []
    for graph in branches[:3]:  # base, different source / same value, wrong value
        nodes = graph["nodes"]
        ops = []
        for j in indices:
            n = nodes[j]
            expr = tuple(nodes[k]["slot"] for k in n["parents"]) or n["truth"]
            ops.append((n["slot"], expr))
        arms.append(render(ops, [nodes[j]["name"] for j in indices], True))
    return arms, [indices.index(j) for j in spine[2:]]  # target, child, terminal


def observe(rt, arms, focus, probe):
    from attribution_cases import patch_state
    from research_probes import encode, decoded

    nodes = arms[0]["nodes"]
    pos = [n["pos"] for n in nodes]
    gold = [n["truth"] for n in nodes]
    u, q, out = focus
    x = encode(rt, arms)
    fields = x[0, pos[u], :3].tolist()
    state = rt.init(x)
    clean, events = [], {}
    for t in range(1, 25):
        state, logits, _ = rt.forward(state)
        reads = decoded(logits, pos).tolist()
        values = [v[3] for v in reads[0]]
        clean.append(values)
        if not probe:
            continue
        for clock, ready in (("first", values[u] == gold[u]), ("complete", values == gold)):
            if not ready or clock in events:
                continue
            event = dict(t=t, before=[values[j] for j in focus],
                         reads=[arm[u] for arm in reads], branches={})
            events[clock] = event
            batch = state[:1].repeat(3, 1, 1)
            for i, name in enumerate(("clean", "same", "wrong")):
                intended = arms[i]["nodes"][u]["truth"]
                branch = dict(intended=intended)
                event["branches"][name] = branch
                if reads[0][u] != fields+[gold[u]] or reads[i][u] != fields+[intended]:
                    branch["status"] = "readout_ineligible"
                    continue
                if i:
                    patch_state(rt, batch[i:i+1], state[i:i+1], [pos[u]])
                written = rt.decode(batch[i:i+1])[0, pos[u], :4].tolist()
                if written != fields+[intended]:
                    raise ValueError("Unexpected written state")
                branch.update(status="ok", written=written,
                              effective=not batch[i,pos[u]].equal(state[0,pos[u]]), steps=[])
            for _ in range(16):
                batch, outputs, _ = rt.forward(batch)
                rows = outputs[3][:, [pos[j] for j in focus]].argmax(-1).tolist()
                for row, branch in zip(rows, event["branches"].values()):
                    if branch["status"] == "ok":
                        branch["steps"].append(row)
    return dict(nodes=nodes, focus=focus, clean=clean, events=events,
                expected_child=arms[2]["nodes"][q]["truth"])


def validate_graphs():
    for size in SIZES:
        for gid in range(1000, 1012):
            layouts = [case(size, gid, order) for order in ORDERS]
            signatures = []
            for arms, (u,q,out) in layouts:
                sig = []
                for graph in arms:
                    ns = graph["nodes"]
                    assert len(ns) == size and max(n["depth"] for n in ns) == 4
                    assert len(graph["tokens"]) <= 1024
                    sig.append({n["name"]: (n["truth"], tuple(ns[j]["name"] for j in n["parents"]))
                                for n in ns})
                assert u in ns[q]["parents"] and q in ns[out]["parents"]
                for j in (u,q,out):
                    truths = [a["nodes"][j]["truth"] for a in arms]
                    assert truths[0] == truths[1] != truths[2]
                signatures.append(sig)
            assert signatures[0] == signatures[1]
            assert layouts[0][0][0]["tokens"] != layouts[1][0][0]["tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-design", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT/"tmp/analysis/plan008-exp072")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_graphs()
    if args.check_design:
        print("36 graph pairs and both donor interventions verified")
        return
    import torch
    from runtime import Runtime

    registry = json.loads(REGISTRY.read_text(encoding="utf8"))["models"]
    folder = args.out / ("smoke" if args.smoke else "formal")
    folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for model in MODELS:
            spec = dict(registry[model]["model"])
            spec["path"] = str(args.data_root / "tmp" / spec["path"].split("/tmp/",1)[1])
            rt = Runtime(spec, args.device)
            rows = []
            for size in SIZES:
                for gid in range(1000, 1002 if args.smoke else 1012):
                    for i,order in enumerate(ORDERS):
                        arms, focus = case(size, gid, order)
                        probe = i == gid % 2
                        row = observe(rt, arms, focus, probe)
                        rows.append(dict(size=size, gid=gid, order=order, probe=probe, **row))
            dest = folder / (model+".json.gz")
            tmp = dest.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf8") as stream:
                json.dump(dict(model=spec, graphs=rows), stream, separators=(",",":"))
            tmp.replace(dest)
            print(model, len(rows), flush=True)
            del rt
    summarize(folder, MODELS)


if __name__ == "__main__":
    main()
