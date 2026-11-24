"""Paired recovery of original continuous states on exact-size modulo-23 graphs."""
import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from attribution_cases import patch_state
from attribution_queue import OLD, model_specs, csv_output
from design import P, TOOLS
from hard_eval import ROOT, PLAN, RECEIPT
from rebinding import render
from research_probes import encode, decoded
from runtime import Runtime

MODELS = ("C_s42", "C_s1234", "A_s42", "B_s42")
PROVENANCE = ROOT / "docs/records/2026-11-22/attribution/provenance.json"
H = 16
HOLD = 4


def historical():
    folder = PROVENANCE.parent / "c-attribution/four_conditions"
    path = folder / "graph-counts.csv"
    if P.sha(path) != P.read(folder / "delivery.json")["table_hashes"][path.name]:
        raise ValueError("Delivered table digest mismatch")
    totals = defaultdict(Counter)
    ids = defaultdict(set)
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model not in MODELS:
                continue
            if row["metric"] == "fit" and row["timing"] == "0":
                for field in ("observed", "value", "first4"):
                    totals[model][field] += int(row[field])
            if row["metric"] == "natural_replay":
                totals[model]["replays"] += int(row["observed"])
            if (row["metric"], row["condition"], row["timing"], row["block"]) != (
                    "intervention", "target", "just_solved", "none"):
                continue
            key = int(row["gid"]), int(row["cell"])
            if key in ids[model]:
                raise ValueError("Duplicate graph instance")
            ids[model].add(key)
            for field in ("eligible", "direct_faults", "direct_repaired_next", "specific",
                          "damage_with_followup", "stable_after_damage"):
                totals[model][field] += int(row[field])
    expected = {(g,d) for g in range(352,368) for d in (4,8)}
    if any(ids[m] != expected for m in MODELS):
        raise ValueError("Incomplete paired graph set")
    hashes = P.read(PROVENANCE)["models"]
    original = P.read(OLD)["finals"]
    for m in ("C_s42","C_s1234"):
        run = original[m+"-correction"]
        source = run["identity"]["source"]
        if (source["sha256"]!=hashes[m] or hashes[m]==run["final_sha256"]
                or "oldC0-resume-150ep" not in source["path"]
                or "epoch=143-" not in source["path"]):
            raise ValueError(f"Expected the original e143 source for {m}")
    return {m: dict(instances=len(ids[m]), natural_replay=totals[m]["replays"],
                    sha256=hashes[m], **totals[m]) for m in MODELS}


def family(gid, size, depth):
    """Interleave one traceable spine with independent background expressions."""
    rng = random.Random(P.seed("ordinary-c-recovery-v1", gid, size, depth))
    roots = [(f"k{i}", v) for i,v in enumerate(rng.sample(range(23), 6))]
    ops = roots.copy()
    pool = {f"k{i}": 0 for i in range(1, 6)}
    spine = [0]
    slots = set(rng.sample(range(6, size), depth))
    for i in range(6, size):
        dest = f"v{i}"
        if i in slots:
            parents = (ops[spine[-1]][0], f"k{rng.randrange(2,6)}")
            spine.append(i)
        else:
            available = [s for s,d in pool.items() if d < depth]
            parents = tuple(rng.sample(available, rng.randint(1,3)))
            pool[dest] = 1 + max(pool[p] for p in parents)
        ops.append((dest, parents))
    names = [f"x_{i}" for i in rng.sample(range(128), size)]
    base = render(ops, names, True)
    a,b = rng.sample(range(1,23), 2)
    variants = [base]
    # Same-position, same-time donors. k1 never feeds the observed spine.
    for root,delta in ((1,a),(0,a),(0,b)):
        edited = ops.copy()
        slot, val = ops[root]
        edited[root] = slot, (val+delta)%23
        variants.append(render(edited, names, True))
    return variants, spine


def outcomes(clean, changed, truth, expected_q, release):
    """All indices are outputs after an intervention; release counts clamped inputs."""
    u,q = truth
    good = lambda row: row["u"] == u and (q is None or row["q"] == q)
    harm = [i for i,(c,z) in enumerate(zip(clean, changed))
            if q is not None and c["q"] == q and z["q"] == expected_q != q]
    last_bad = max((i for i,z in enumerate(changed) if not good(z)), default=-1)
    stable = len(changed)-max(release,last_bad+1) >= 2
    return dict(target_correct_next=int(changed[0]["u"]==u),
                target_correct_end=int(changed[-1]["u"]==u),
                q_at_risk=int(q is not None and any(c["q"]==q for c in clean)),
                specific_child=int(bool(harm)),
                specific_with_followup=int(bool(harm) and harm[0]<len(clean)-2),
                joint_stable_after_specific=int(bool(harm) and stable and harm[0]<len(clean)-2),
                first_specific=harm[0]+1 if harm else None,
                stable_from=max(release,last_bad+1)+1 if stable else None)


def measure(rt, gid, size, depth):
    arms,spine = family(gid,size,depth)
    nodes = arms[0]["nodes"]
    x = encode(rt,arms)
    pos = [n["pos"] for n in nodes]
    gold = [[n["truth"] for n in a["nodes"]] for a in arms]
    labels = x[0,pos,:3].tolist()
    D = depth+1
    # Search starts through D+4. The final clean observation covers the last release+H.
    limit = D+4
    snapshots, reads = [rt.init(x)], [None]
    for t in range(1,limit+H+HOLD+1):
        h,logits,_ = rt.forward(snapshots[-1])
        snapshots.append(h)
        reads.append(decoded(logits,pos).tolist())

    def ready(t, arm, i):
        return reads[t][arm][i] == labels[i]+[gold[arm][i]]

    def focus(vals, r, u, q):
        return {k: vals[v][3] if v is not None else None
                for k,v in (("r",r),("u",u),("q",q))}

    children = [[] for _ in nodes]
    for i,n in enumerate(nodes):
        for p in n["parents"]:
            children[p].append(i)
    # Count EVERY naturally arising correct -> wrong number on a previously due node.
    # Replay the first four eligible parent-child edges per graph in fixed time/node order.
    natural = []
    replayed = 0
    opportunities = 0
    for t in range(2,len(reads)-1):
        for u,n in enumerate(nodes):
            wrong = reads[t][0][u][3]
            if t-1 < n["depth"]+1 or not ready(t-1,0,u):
                continue
            opportunities += 1
            if not 0<=wrong<23 or wrong==gold[0][u]:
                continue
            for q in children[u] or [None]:
                z = dict(t=t,u=u,q=q,wrong=wrong,identity_ok=reads[t][0][u][:3]==labels[u])
                natural.append(z)
                if q is None:
                    continue
                z["others_correct"] = all(ready(t,0,p) for p in nodes[q]["parents"] if p!=u)
                z["expected"] = (gold[0][q]+wrong-gold[0][u])%23
                z["observed_child"] = reads[t+1][0][q][3]
                if replayed>=4 or not z["identity_ok"] or not z["others_correct"]:
                    continue
                replay = snapshots[t][:1].repeat(2,1,1)
                patch_state(rt,replay[1:],snapshots[t-1][:1],[pos[u]])
                _,logits,_ = rt.forward(replay)
                z["paired_child"],z["replay_child"] = logits[3][:,pos[q]].argmax(-1).tolist()
                z["specific_rescue"] = z["observed_child"]==z["paired_child"]==z["expected"] and z["replay_child"]==gold[0][q]
                # Untouched trajectory after the natural error, uncensored count available.
                z["joint_followup"] = [[reads[j][0][i][3] for i in (u,q)] for j in range(t+1,min(t+H+1,len(reads)))]
                replayed += 1

    complete = next((t for t in range(1,limit+1) if [z[3] for z in reads[t][0]]==gold[0]),None)
    events = []
    for role,k in (("source",0),("internal",len(spine)//2),("sink",len(spine)-1)):
        u = spine[k]
        r = spine[k-1] if k else None
        q = spine[k+1] if k+1<len(spine) else None
        first = next((t for t in range(1,limit+1) if ready(t,0,u)),None)
        times = {"first_correct":first, "graph_complete":complete}
        if q is not None:
            times["child_correct"] = next((t for t in range(1,limit+1)
                                            if ready(t,0,q) and ready(t,0,u)),None)
        for timing,t in times.items():
            conditions = {"clean":[], "same_value":[(u,1)], "target":[(u,2)]}
            if role=="internal":
                conditions.update(parent=[(r,3)], both=[(u,2),(r,3)],
                                  held_parent=[(r,3)], held_same_parent=[(r,1)])
            for condition, patches in conditions.items():
                e = dict(role=role,timing=timing,condition=condition,t=t,r=r,u=u,q=q)
                events.append(e)
                if t is None or not ready(t,0,u):
                    e["skip"] = "no_qualified_start"
                    continue
                hold = HOLD if condition in ("held_parent","held_same_parent") else 1
                if not all(ready(t,0,i) and all(ready(t+j,arm,i) for j in range(hold))
                           for i,arm in patches):
                    e["skip"] = "source_or_donor_readout"
                    continue
                # Pair each intervention with an untouched twin in the same forward.
                state = snapshots[t][:1].repeat(2,1,1)
                trace, clean, inputs, patch_differences = [],[],[],[]
                for j in range(H+(hold if hold>1 else 0)):
                    if j<hold:
                        diffs = []
                        for i,arm in patches:
                            previous = state[0,pos[i]].clone()
                            patch_state(rt,state[:1],snapshots[t+j][arm:arm+1],[pos[i]])
                            diffs.append((state[0,pos[i]].float()-previous.float()).norm().item())
                        patch_differences.append(diffs)
                        actual = rt.decode(state)[0,pos,:4].tolist()
                        if any(actual[i]!=labels[i]+[gold[arm][i]] for i,arm in patches):
                            raise ValueError("Actual injected input failed readback")
                        inputs.append(focus(actual,r,u,q))
                    state,logits,_ = rt.forward(state)
                    modified,twin = decoded(logits,pos).tolist()
                    trace.append(focus(modified,r,u,q))
                    clean.append(focus(twin,r,u,q))
                # In the double intervention, parent-implied and injected target values differ.
                e.update(inputs=inputs,trace=trace,clean=clean,patch_differences=patch_differences,
                         effective_change=any(d>0 for step in patch_differences for d in step),
                         target_end_readout=trace[-1]["u"],
                         target_ends_injected=(trace[-1]["u"]==gold[2][u]) if (u,2) in patches else None,
                         first_uses_wrong_parent=(trace[0]["u"]==gold[3][u]) if (r,3) in patches else None,
                         other_parents_correct=all(ready(t,0,p) for p in nodes[u]["parents"] if p!=r),
                         parent_implied_q=gold[3][q] if q is not None else None,
                         parent_implied_u=gold[3][u] if any(a==3 for _,a in patches) else gold[0][u],
                         wrong_target=gold[2][u], gold_target=gold[0][u],
                         gold_child=gold[0][q] if q is not None else None)
                arm = 2 if condition in ("target","both") else 3
                expected = gold[arm][q] if q is not None else None
                # Score the sham against the same numerical alternative as its matched arm.
                if condition in ("same_value","clean") and q is not None:
                    expected = gold[2][q]
                e.update(expected_child=expected, **outcomes(clean,trace,(gold[0][u],e["gold_child"]),
                         expected,hold if hold>1 else 0))
    fits = {}
    for t in (D,D+4,D+16):
        fits[t] = dict(nodes=size,value=sum(z[3]==y for z,y in zip(reads[t][0],gold[0])),
                       first4=sum(ready(t,0,i) for i in range(size)))
    return dict(gid=gid,size=size,depth=depth,spine=spine,nodes=nodes,gold=gold,complete=complete,
                fit=fits,events=events,natural=natural,natural_opportunities=opportunities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--data-root",type=Path,default=ROOT)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--out",type=Path,default=ROOT/"tmp/analysis/original-c-recovery-v1")
    args = parser.parse_args()
    print(json.dumps(historical(),indent=2),flush=True)
    if args.audit_only:
        return
    plan = args.data_root/PLAN.relative_to(ROOT)
    if P.sha(plan)!=P.read(RECEIPT)["plan_sha256"]:
        raise ValueError("Plan mismatch")
    specs = model_specs(P.read(plan),args.data_root)
    root = args.out/("smoke" if args.smoke else "formal")
    root.mkdir(parents=True,exist_ok=True)
    rows,fit_rows,natural_rows = [],[],[]
    for m in MODELS:
        spec = specs[m]
        if spec["sha256"]!=P.read(PROVENANCE)["models"][m]:
            raise ValueError("Source weights differ from historical attribution")
        rt = Runtime(spec,args.device)
        folder = root/m
        folder.mkdir(exist_ok=True)
        paths = [Path(__file__),TOOLS/"c_update/runtime.py",TOOLS/"c_local_update/rebinding.py",
                 Path(__file__).with_name("attribution_cases.py"),Path(__file__).with_name("research_probes.py"),
                 ROOT/"experiments/model.py",ROOT/"Simtransformer/simtransformer/model_base.py"]
        paths.extend(sorted((ROOT/"experiments/configs"/spec["config"]).glob("*.yaml")))
        identity = dict(model=m,checkpoint=spec["sha256"],config=spec["config"],plan=P.sha(plan),
                        code={p.name:P.sha(p) for p in paths})
        if (folder/"identity.json").exists() and P.read(folder/"identity.json")!=identity:
            raise ValueError(f"Incompatible saved records at {folder}")
        P.write(folder/"identity.json",identity)
        for size in ((32,) if args.smoke else (32,64,128)):
            for depth in ((4,) if args.smoke else (4,10)):
                for gid in ((999,) if args.smoke else range(32)):
                    file = folder/f"{size}-{depth}-{gid}.json.gz"
                    if file.exists():
                        with gzip.open(file,"rt") as f:
                            case = json.load(f)
                    else:
                        with torch.inference_mode():
                            case = measure(rt,gid,size,depth)
                        temp = file.with_suffix(".tmp")
                        with gzip.open(temp,"wt",encoding="utf8") as f:
                            json.dump(case,f,separators=(",",":"))
                        temp.replace(file)
                    key = dict(model=m,size=size,depth=depth,gid=gid)
                    for t,fit in case["fit"].items():
                        fit_rows.append(dict(**key,t=int(t),**fit))
                    natural_rows.append(dict(**key,opportunities=case["natural_opportunities"],
                        events=len({(z["t"],z["u"]) for z in case["natural"]}),
                        child_edges=sum(z["q"] is not None for z in case["natural"]),
                        replays=sum("replay_child" in z for z in case["natural"]),
                        specific_rescues=sum(z.get("specific_rescue",False) for z in case["natural"])))
                    for e in case["events"]:
                        rows.append(dict(**key,eligible=int("skip" not in e),
                                         **{k:v for k,v in e.items() if k not in ("clean","trace","inputs","patch_differences")}))
                print(m,size,depth,"complete",flush=True)
        del rt
    csv_output(root/"events.csv",rows)
    csv_output(root/"fit.csv",fit_rows)
    csv_output(root/"natural.csv",natural_rows)
    groups = defaultdict(list)
    for row in rows:
        k = tuple(row[f] for f in ("size","depth","gid","role","timing","condition"))
        groups[k].append(row)
    for group in groups.values():
        if len(group)!=len(MODELS):
            raise ValueError("Missing comparison weight")
        for row in group:
            row["all_eligible"] = int(all(z["eligible"] for z in group))
            row["same_round_all_eligible"] = int(row["all_eligible"] and len({z["t"] for z in group})==1)
    csv_output(root/"paired.csv",rows)
    accum = defaultdict(Counter)
    for row in rows:
        cell = tuple(row[k] for k in ("model","size","depth","role","timing","condition"))
        for scope in ("all","common","common_same_round"):
            c = accum[cell+(scope,)]
            c["attempts"] += 1
            qualifies = {"all":row["eligible"],"common":row["all_eligible"],
                         "common_same_round":row["same_round_all_eligible"]}[scope]
            if qualifies:
                c["eligible"] += 1
                for field in ("effective_change","target_correct_next","target_correct_end","target_ends_injected",
                              "first_uses_wrong_parent","q_at_risk","specific_child","specific_with_followup",
                              "joint_stable_after_specific"):
                    if row.get(field) is not None:
                        c[field] += int(row[field])
    table = [dict(zip(("model","size","depth","role","timing","condition","scope"),key),
                  eligible=counts["eligible"],**{k:v for k,v in counts.items() if k!="eligible"})
             for key,counts in accum.items()]
    csv_output(root/"summary.csv",table)


if __name__ == "__main__":
    main()
