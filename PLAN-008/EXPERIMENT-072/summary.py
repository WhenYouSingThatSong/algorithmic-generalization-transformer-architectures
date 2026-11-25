"""CPU reports for width and local recovery. Counts retain graph identity."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def first(flags):
    return next((i for i, flag in enumerate(flags, 1) if flag), None)


def recovery(g, stage, arm):
    focus = g["focus"]
    gold = [g["nodes"][j]["truth"] for j in focus]
    e = g["events"].get(stage, {})
    b = e.get("branches", {}).get(arm, {})
    row = dict(status=b.get("status", "start_unreached"), start=e.get("t"),
               child_at_start=e.get("before", [None]*3)[1], effective=b.get("effective"),
               first_return=None, next_correct=None, stable_target=None, specific=None,
               other_wrong=None, extra_empty=None, joint_stable=None, followup=None,
               specific_recovered=None, terminal_correct=None)
    if row["status"] != "ok":
        return row
    changed, clean = b["steps"], e["branches"]["clean"]["steps"]
    hit = [a[1] == g["expected_child"] and c[1] == gold[1] for a,c in zip(changed,clean)]
    stable = all(a[:2] == gold[:2] and c[:2] == gold[:2] for a,c in zip(changed[-2:],clean[-2:]))
    follow = any(hit[:-2])
    row.update(first_return=first(v[0]==gold[0] for v in changed),
               next_correct=changed[0][0]==gold[0], stable_target=all(v[0]==gold[0] for v in changed[-2:]),
               specific=any(hit), other_wrong=any(c[1]==gold[1] and v[1]<23 and v[1] not in
               (g["expected_child"],gold[1]) for v,c in zip(changed,clean)),
               extra_empty=any(c[1]==gold[1] and v[1]>=23 for v,c in zip(changed,clean)),
               joint_stable=stable, followup=follow, specific_recovered=follow and stable,
               terminal_correct=changed[-1][2]==gold[2])
    return row


def summarize(folder, models):
    width, nodes, events, examples = [], [], [], []
    for model in models:
        with gzip.open(folder/(model+".json.gz"), "rt", encoding="utf8") as f:
            data = json.load(f)
        for g in data["graphs"]:
            key = dict(model=model, size=g["size"], gid=g["gid"], order=g["order"])
            ns, reads = g["nodes"], g["clean"]
            truth = [n["truth"] for n in ns]
            first_delays = []
            for i,n in enumerate(ns):
                if not n["parents"]:
                    continue
                ready = [all(prev[j]==truth[j] for j in n["parents"]) for prev in reads[:-1]]
                t = first(ready)
                t = t+1 if t else None
                solved = first(v[i]==truth[i] for v in reads)
                delayed = t is not None and reads[t-1][i]!=truth[i]
                if delayed:
                    first_delays.append((t,n["name"]))
                nodes.append(dict(**key, node=n["name"], depth=n["depth"], parent_ready_input_round=t,
                                  read_when_ready=reads[t-1][i] if t else None, gold=truth[i],
                                  first_correct=solved, delayed=delayed))
            earliest = min((t for t, _ in first_delays), default=None)
            width.append(dict(**key, width=max(sum(n["depth"]==d for n in ns) for d in range(1,5)),
                              complete=first(v==truth for v in reads), first_delay=earliest,
                              first_delayed_nodes=" ".join(name for t,name in first_delays if t==earliest),
                              **{f"correct_r{t}":sum(a==b for a,b in zip(reads[t-1],truth)) for t in (5,9,21,24)}))
            if not g["probe"]:
                continue
            for stage in ("first","complete"):
                for arm in ("clean","same","wrong"):
                    events.append(dict(**key, stage=stage, arm=arm, **recovery(g,stage,arm)))
                if g["gid"]==1000:
                    e=g["events"].get(stage,{})
                    branches=e.get("branches",{})
                    eq=[]
                    for j in g["focus"]:
                        n=ns[j]
                        eq.append(n["name"]+" = "+" + ".join(ns[k]["name"] for k in n["parents"])+f" ≡ {truth[j]}")
                    examples.append(f"### {model} · {g['size']} 节点 · {stage}\n\n"+ "；".join(eq))
                    for a in ("clean","same","wrong"):
                        b=branches.get(a,{})
                        seq=" → ".join(str(tuple(v)) for v in b.get("steps", []))
                        examples.append(f"- {a}：{seq or b.get('status','未达到起点')}")
    write(folder/"width-by-graph.csv",width)
    write(folder/"paired-nodes.csv",nodes)
    write(folder/"recovery-by-graph.csv",events)
    groups=defaultdict(list)
    for r in width:
        groups[r["model"],r["size"],r["order"]].append(r)
    write(folder/"width-summary.csv",[dict(model=m,size=n,order=o,graphs=len(rs),
          completed=sum(r["complete"] is not None for r in rs),
          graphs_with_delay=sum(r["first_delay"] is not None for r in rs),
          **{f"acc_r{t}":mean(r[f"correct_r{t}"]/n for r in rs) for t in (5,9,21,24)})
          for (m,n,o),rs in groups.items()])
    c_ok=defaultdict(set)
    for r in events:
        if r["model"].startswith("C_") and r["status"]=="ok":
            c_ok[r["size"],r["gid"],r["stage"],r["arm"]].add(r["model"])
    groups=defaultdict(list)
    for r in events:
        key=r["model"],r["size"],r["stage"],r["arm"]
        groups[key].append(r)
    rows=[]
    for (m,n,s,a),rs in groups.items():
        for cohort in ("individual","common_C"):
            eligible=[r for r in rs if r["status"]=="ok" and (cohort=="individual" or
                      len(c_ok[n,r["gid"],s,a])==3)]
            row=dict(model=m,size=n,stage=s,arm=a,cohort=cohort,candidates=len(rs),
                     reached=sum(r["start"] is not None for r in rs),eligible=len(eligible))
            for metric in ("next_correct","stable_target","specific","other_wrong","extra_empty",
                           "followup","specific_recovered","joint_stable","terminal_correct"):
                row[metric]=sum(r[metric] for r in eligible)
            rows.append(row)
    write(folder/"recovery-summary.csv",rows)
    (folder/"examples.md").write_text("# 固定示例 gid=1000，依次读目标、子节点、链终点\n\n"+
                                      "\n\n".join(examples),encoding="utf8")
