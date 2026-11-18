#!/usr/bin/env python3
"""W1/W2 summary: old/new truth separated, common A'/B cohort, graph-weighted."""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from io_utils import CSV, jsonl, read_json, sha, write_json
from reassignment_core import BRANCHES

def record_key(r):
    return r["graph_id"],r["leaf"],r["constant"],r["delta"]

def metrics(rows,gold,indices):
    n=len(indices)
    return dict(n=n,value_n=sum(rows[i][3]==gold[i][3] for i in indices),
                first4_n=sum(rows[i]==gold[i] for i in indices))

def run(runs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    models={}; records={}; sels={}; provenance={}
    for path in map(Path,runs):
        m=read_json(path/"meta.json"); arm=m["model_key"]
        if m["status"]!="complete" or m["protocol"]!="discrete_reassignment_v1":
            raise ValueError("incomplete/unsupported run")
        if arm in models:raise ValueError("duplicate model")
        models[arm]=m
        rs={}
        for r in jsonl(path/"records.jsonl"):
            k=record_key(r)
            if k in rs:raise ValueError("duplicate record")
            if set(r["branches"])!=set(BRANCHES):raise ValueError("missing branch")
            if any(b["truth_ref"]!=("new" if name.startswith("new_") else "old") for name,b in r["branches"].items()):
                raise ValueError("branch has incorrect truth reference")
            if any(len(b["outputs"])!=m["horizon"] or len(b["inputs"])!=m["horizon"] for b in r["branches"].values()):
                raise ValueError("incomplete horizon")
            rs[k]=r
        records[arm]=rs
        selected={}
        for s in jsonl(path/"selection.jsonl"):
            if s["graph_id"] in selected:raise ValueError("duplicate coverage")
            selected[s["graph_id"]]=s
        if set(selected)!=set(m["candidate_ids"]):raise ValueError("incomplete coverage")
        expected={(gid,s["leaf"],s["constant"],d) for gid,s in selected.items() if s["status"]=="selected" for d in m["deltas"]}
        if set(rs)!=expected:raise ValueError("selected targets missing records/deltas")
        sels[arm]=selected
        if sha(path/"records.jsonl")!=m["records_sha256"] or sha(path/"selection.jsonl")!=m["selection_sha256"]:
            raise ValueError("source checksum mismatch")
        provenance[arm]=dict(path=str(path),meta_sha256=sha(path/"meta.json"),meta=m)
    if set(models)!={"model_a","model_b"}:raise ValueError("both requested A'/B runs required")
    ma,mb=models["model_a"],models["model_b"]
    for name in ("dataset_sha256","manifest_sha256","candidate_ids","horizon","batch_size","deltas","code_hashes"):
        if ma[name]!=mb[name]:raise ValueError("unmatched protocol: "+name)
    common=set(records["model_a"])&set(records["model_b"])
    for gid in ma["candidate_ids"]:
        aa,bb=sels["model_a"][gid],sels["model_b"][gid]
        if aa["status"]==bb["status"]=="selected" and (aa["leaf"],aa["constant"])!=(bb["leaf"],bb["constant"]):
            raise ValueError("models selected different structural targets")
    for k in common:
        a,b=records["model_a"][k],records["model_b"][k]
        for field in ("positions","old_truth4","new_truth4","regions","state_round","parents"):
            if a[field]!=b[field]:raise ValueError("unmatched target/truth: "+field)
    coverage=CSV(out/"coverage.csv"); curves=CSV(out/"case_curves.csv")
    summary=CSV(out/"summary.csv"); contrasts=CSV(out/"paired_models.csv")
    budgets=CSV(out/"warm_vs_scratch.csv")
    bucket=defaultdict(list); lookup={}
    try:
        for gid in ma["candidate_ids"]:
            aa,bb=sels["model_a"][gid],sels["model_b"][gid]
            coverage.add(dict(graph_id=gid,a_status=aa["status"],b_status=bb["status"],
                common_selected=aa["status"]==bb["status"]=="selected"))
        for arm,rs in records.items():
            for k,r in rs.items():
                old,new,pos=r["old_truth4"],r["new_truth4"],r["positions"]
                for name,b in r["branches"].items():
                    goal=new if b["truth_ref"]=="new" else old
                    for region,ps in r["regions"].items():
                        ix=[pos.index(p) for p in ps]
                        for step,rows in enumerate([b["initial"]]+b["outputs"]):
                            correct=metrics(rows,goal,ix)
                            oldm,newm=metrics(rows,old,ix),metrics(rows,new,ix)
                            inp=b["initial"] if step==0 else b["inputs"][step-1]
                            clean=r["branches"]["clean"]
                            clean_rows=clean["initial"] if step==0 else clean["outputs"][step-1]
                            ci=pos.index(r["constant"])
                            unchanged=[i for i in ix if old[i][3]==goal[i][3]]
                            rr=dict(arm=arm,graph_id=r["graph_id"],leaf=r["leaf"],delta=r["delta"],
                                branch=name,truth_ref=b["truth_ref"],start_kind=b["start_kind"],region=region,
                                post_round=step,common_qualified=k in common,**correct,
                                constant_input=inp[ci][3],constant_output=rows[ci][3],
                                constant_input_matches_goal=inp[ci][3]==goal[ci][3],
                                unchanged_truth_n=len(unchanged),
                                unchanged_extra_damage_n=sum(clean_rows[i][3]==goal[i][3] and rows[i][3]!=goal[i][3] for i in unchanged),
                                old_value_n=oldm["value_n"],new_value_n=newm["value_n"])
                            curves.add(rr);lookup[(arm,k,name,region,step)]=rr
                            for cohort in ["own_qualified"]+(["common_qualified"] if k in common else []):
                                bucket[(arm,r["delta"],name,region,step,cohort)].append(rr)
                # last-two conjunction is NOT just the final endpoint.
                for name,b in r["branches"].items():
                    goal=new if b["truth_ref"]=="new" else old
                    for region,ps in r["regions"].items():
                        ix=[pos.index(p) for p in ps]
                        v=sum(all(x[i][3]==goal[i][3] for x in b["outputs"][-2:]) for i in ix)
                        f=sum(all(x[i]==goal[i] for x in b["outputs"][-2:]) for i in ix)
                        if len(b["outputs"])<2:continue
                        rr=dict(graph_id=r["graph_id"],n=len(ix),value_n=v,first4_n=f)
                        for cohort in ["own_qualified"]+(["common_qualified"] if k in common else []):
                            bucket[(arm,r["delta"],name,region,"last_two",cohort)].append(rr)
        for bk,rrs in bucket.items():
            observed=[r for r in rrs if r["n"]]
            n=sum(r["n"] for r in observed)
            # exactly one selected leaf per graph and delta; graph macro is explicit
            if len({r["graph_id"] for r in rrs})!=len(rrs):raise ValueError("unexpected repeated graph")
            summary.add(dict(zip(("arm","delta","branch","region","round_or_window","cohort"),bk),
                selected_graph_n=len(rrs),observed_graph_n=len(observed),node_n=n,
                value_n=sum(r["value_n"] for r in observed),first4_n=sum(r["first4_n"] for r in observed),
                value_micro=sum(r["value_n"] for r in observed)/n if n else None,
                value_macro=sum(r["value_n"]/r["n"] for r in observed)/len(observed) if observed else None,
                first4_macro=sum(r["first4_n"]/r["n"] for r in observed)/len(observed) if observed else None))
        for k in sorted(common):
            r=records["model_a"][k]
            for name in BRANCHES:
                for region in r["regions"]:
                    for step in range(ma["horizon"]+1):
                        a,b=[lookup[(arm,k,name,region,step)] for arm in ("model_a","model_b")]
                        contrasts.add(dict(graph_id=k[0],leaf=k[1],delta=k[3],branch=name,region=region,post_round=step,
                            node_n=a["n"],a_correct=a["value_n"],b_correct=b["value_n"],
                            b_minus_a_correct=b["value_n"]-a["value_n"]))
            for arm in models:
                for pinned in (False,True):
                    suffix="_pinned" if pinned else ""
                    for region in ("all_nodes","changed_nodes","changed_descendants"):
                        for step in range(1,ma["horizon"]+1):
                            warm=lookup[(arm,k,"new_warm"+suffix,region,step)]
                            scratch=lookup[(arm,k,"new_scratch"+suffix,region,step)]
                            budgets.add(dict(arm=arm,graph_id=k[0],delta=k[3],pinned=pinned,region=region,post_round=step,
                                node_n=warm["n"],warm_new_correct=warm["new_value_n"],scratch_new_correct=scratch["new_value_n"],
                                warm_minus_scratch=warm["new_value_n"]-scratch["new_value_n"]))
        write_json(out/"meta.json",dict(status="complete",protocol="reassignment_summary_v1",sources=provenance,
            common_cases=len(common),common_graphs=len({k[0] for k in common}),
            code_sha256=sha(__file__),note="Post-round 0 is intervention, not recovery; changed-node region distinguishes old/new truth."))
    finally:
        for f in (coverage,curves,summary,contrasts,budgets):f.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,action="append",required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.out_dir)
if __name__=="__main__":main()
