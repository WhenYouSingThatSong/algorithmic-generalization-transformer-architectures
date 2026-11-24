"""Registered evaluation of post-trained C, including simultaneous computation and repair."""
import argparse
import json
from pathlib import Path

import local_protocol as S
from scenarios import metrics, summarize

P = S.P


def focused(rt, x, y, g, split, gid, target):
    from evaluate import baseline, branch, source
    identity = dict(split=split, gid=gid)
    labels = y.tolist()
    nodes = g["nodes"]
    horizon = S.PROTOCOL["recovery_horizon"]
    middle = max(1, g["D"] // 2)
    # A predeclared structural choice, independent of weights and readout.
    candidates = [p for p in nodes if g["depths"][p]<middle and P.descendants(g,p)]
    middle_u = min(candidates, key=lambda p: P.seed("middle",split,gid,p), default=None)
    last = g["D"] + 16
    rows, _ = baseline(rt,x,g=g,y=y,rounds=last+horizon,keep=set())
    complete = next((t for t in range(1,last+1) if all(
        out[0]==labels[p][3] for out,p in zip(rows[t-1]["output"], nodes))), None)
    u = target["u"] if target else None
    starts = [("frontier",g["depths"][u] if u is not None else None,u),
              ("middle",middle,middle_u), ("actual_complete",complete,u), ("train_end",g["D"],u)]
    # Replay only to required input boundaries, avoiding a full sequence of hidden snapshots.
    keep = {t for _,t,p in starts if t is not None and p is not None}
    # Find an independent solved node with an authentic earlier EMPTY hidden.
    # Identity/operation/syntax must match; never manufacture a hidden vector.
    fill = None
    if u is not None and complete is not None:
        independent = [v for v in nodes if v!=u and v not in P.descendants(g,u)
                       and u not in P.descendants(g,v)]
        independent.sort(key=lambda v:P.seed("mixed-fill",split,gid,v))
        for age in range(1,complete):
            for v in independent:
                out = rows[age-1]["output"][nodes.index(v)]
                if out[0]==24 and all(out[2:5]):
                    fill = (age,v)
                    break
            if fill:
                keep.add(age)
                break
    snapshots = {}
    if keep:
        _, snapshots = baseline(rt,x,g=g,y=y,rounds=max(keep)+1,keep=keep)
    for timing,t,p in starts:
        conditions = ["clean","wrong"]
        if timing=="actual_complete":
            conditions.extend(("parent_child_consistent_wrong","empty_only","empty_and_wrong"))
        for cond in conditions:
            meta = dict(**identity, timing=timing, condition=cond, start=t, target=p)
            if t is None or p is None:
                reason = "not_completed_by_budget" if timing=="actual_complete" and complete is None else "no_target"
                yield dict(**meta,status=reason)
                continue
            base = snapshots[t]
            clean = rows[t:t+horizon]
            i = nodes.index(p)
            prior = clean[0]["input"][i][3]
            meta.update(prior=prior, truth=labels[p][3],
                        complete_before=all(a[3]==labels[n][3] for a,n in zip(clean[0]["input"],nodes)))
            if timing=="train_end" and cond!="clean" and not meta["complete_before"]:
                yield dict(**meta,status="unfinished_at_training_check")
                continue
            if cond in ("empty_only","empty_and_wrong") and not fill:
                yield dict(**meta,status="no_independent_identity_matched_empty")
                continue
            if cond=="clean":
                trace = clean
            else:
                if timing=="middle" and (prior!=labels[p][3] or meta["complete_before"]):
                    yield dict(**meta,status="not_correct_during_unfinished_graph")
                    continue
                pos = [] if cond=="empty_only" else [p]
                if cond == "parent_child_consistent_wrong":
                    pos.append(target["q"])
                delta = target["delta"] if target else 1+P.seed(split,gid,"delta")%22
                src, read, ok = {}, {}, True
                if pos:
                    src, read, ok = source(rt,x,y,g,pos,P.donor_edits(g,p,delta),
                                           t+1 if timing=="frontier" else t)
                if not ok or any(read[str(n)][3]==labels[n][3] for n in pos):
                    yield dict(**meta,status="source_rejected")
                    continue
                if cond=="parent_child_consistent_wrong":
                    q = target["q"]
                    parent_values = [read[str(k)][3] if k in pos else clean[0]["input"][nodes.index(k)][3]
                                     for k in g["parents"][q]]
                    if read[str(q)][3] != sum(parent_values)%23:
                        yield dict(**meta,status="source_not_locally_consistent")
                        continue
                if cond in ("empty_only","empty_and_wrong"):
                    age, v = fill
                    pos.append(v)
                    src[v] = snapshots[age][0,v]
                    meta.update(fill_target=v,fill_source_round=age)
                trace = branch(rt,base,g,y,pos,src,horizon)
                meta["actual_value_changed"] = trace[0]["input"][i][3]!=prior
            scored = metrics(g,labels,trace,clean,timing,cond,p)
            yield dict(**meta,status="ok",**{k:v for k,v in scored.items() if k not in meta})


def focused_split(rt, plan, manifest, split, out):
    from evaluate import load_split
    import torch
    x,y,graphs = load_split(rt,plan["splits"][split])
    records=[]
    with torch.inference_mode():
        for case in manifest["splits"][split]["cases"]:
            gid = case["gid"]
            records.extend(focused(rt,x[gid:gid+1].to(rt.device),y[gid].to(rt.device),graphs[gid],
                                   split,gid,case["target"]))
    P.write(out,dict(summary=summarize(records),records=records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    plan=S.validate(args.plan,args.manifest)
    manifest=P.read(args.manifest)
    from runtime import Runtime
    from evaluate import load_split,run_graph
    from functional_probe import run_probe
    from summarize import summarize_file
    import torch
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    ident=ck["identity"]
    if (ck["epoch"]!=50 or ident["version"]!=S.VERSION or ident["manifest"]!=P.sha(args.manifest)
            or ident["model"] not in S.MODELS):
        raise ValueError("formal test requires the registered epoch-50 checkpoint")
    del ck
    src=next(m for m in plan["models"] if m["id"]==ident["model"])
    rt=Runtime(dict(src,arm="update",path=str(args.checkpoint),sha256=P.sha(args.checkpoint)),args.device)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status="running",source=ident, checkpoint=P.sha(args.checkpoint))
    P.write(args.out_dir/"meta.json",meta)
    for split in P.STRATA:
        focused_split(rt,plan,manifest,split,args.out_dir/(split+"-classes.json"))
        x,y,graphs=load_split(rt,plan["splits"][split])
        path=args.out_dir/(split+"-standard.jsonl")
        with torch.inference_mode(),path.open("x",encoding="utf-8") as f:
            def emit(record):
                f.write(json.dumps(record)+"\n")
            for gid,g in enumerate(graphs):
                run_graph(rt,x[gid:gid+1].to(rt.device),y[gid].to(rt.device),g,split,gid,
                          plan["splits"][split]["targets"][gid],emit)
        summarize_file(path,args.out_dir/(split+"-standard-summary.json"))
        from outcomes import additional
        P.write(args.out_dir/(split+"-outcomes.json"),additional(path))
        if not split.startswith("N64"):
            P.write(args.out_dir/(split+"-frozen-probe.json"),run_probe(rt,plan,split,limit=32))
    meta["status"]="complete"
    P.write(args.out_dir/"meta.json",meta)


if __name__=="__main__":
    main()
