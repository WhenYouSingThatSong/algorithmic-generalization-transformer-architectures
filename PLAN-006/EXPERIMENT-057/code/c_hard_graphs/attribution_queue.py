"""Serial fixed-weight attribution and workspace experiments. No adaptive policies."""
import argparse
import gzip
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path

import torch
from attribution_cases import run_graph,workspace_program
from design import P,TOOLS
from hard_eval import PLAN,RECEIPT,ROOT,write_csv
from research_probes import encode
from runtime import Runtime
from transfer_tasks import export

OLD=ROOT/"docs/records/2026-11-20/c-update-training-handoff.json"
OUT=ROOT/"tmp/analysis/c-attribution-v1"
STAGES=("four_conditions","pathway","sources","workspace")


def model_specs(plan,root):
    registry={m["id"]:m for m in plan["models"]}
    ordinary=P.read(OLD)
    selected=P.read(RECEIPT)
    specs={}
    for ident in ("A_s42","B_s42","C_s42","C_s1234"):
        specs[ident]=dict(registry[ident])
        saved=Path(registry[ident]["path"])
        if not saved.is_absolute():
            specs[ident]["path"]=str(root/saved)
        elif not saved.is_file() and "tmp" in saved.parts:
            specs[ident]["path"]=str(root.joinpath(*saved.parts[saved.parts.index("tmp"):]))
    for seed in (42,1234):
        name=f"C_s{seed}"
        corr=ordinary["finals"][name+"-correction"]
        extra=next(r for r in selected["runs"] if r["model"]==name)
        if extra["source_sha256"]!=registry[name]["sha256"] or corr["identity"]["source"]["sha256"]!=registry[name]["sha256"]:
            raise ValueError("continuous weights do not have a common origin")
        specs[name+"-correction"]=dict(registry[name],arm="correction",path=str(root/corr["final"]),
                                       sha256=corr["final_sha256"])
        specs[name+"-update"]=dict(registry[name],arm="update",path=str(root/extra["final_ckpt"]["path"]),
                                   sha256=extra["final_ckpt"]["sha256"])
    return specs


def workspace(rt,gid,settings):
    rewrites,live,gap,spaced=settings
    unique,repeat,peak=workspace_program(gid,rewrites,live,gap,spaced)
    nodes=repeat["nodes"]
    if [n["truth"] for n in nodes]!=[n["truth"] for n in unique["nodes"]]:
        raise ValueError("unmatched program")
    pos=[n["pos"] for n in nodes]
    gold=[n["truth"] for n in nodes]
    previous={}
    distances=[]
    for i,n in enumerate(nodes):
        if n["slot"] in previous:
            distances.append(i-previous[n["slot"]])
        previous[n["slot"]]=i
    D=max(n["depth"] for n in nodes)+1
    h=rt.init(encode(rt,[unique,repeat]))
    rows=[]
    for t in range(1,D+9):
        h,logits,_=rt.forward(h)
        if t in (D,D+4,D+8):
            a,b=logits[3][:,pos].argmax(-1).tolist()
            for role in ("all","saved_reader","immediate_reader"):
                ix=[i for i,n in enumerate(nodes) if role=="all" or
                    (role=="saved_reader" and n["slot"].startswith("out")) or
                    (role=="immediate_reader" and n["slot"].startswith("s"))]
                rows.append(dict(role=role,extra=t-D,definitions_per_register=rewrites,retained=live,
                    reader_spacers=2*gap,cyclic=spaced,peak=peak,rebindings=len(distances),
                    mean_rebind_distance=sum(distances)/len(distances) if distances else None,
                    nodes=len(ix),unique_correct=sum(a[i]==gold[i] for i in ix),
                    reused_correct=sum(b[i]==gold[i] for i in ix),
                    unique_only=sum(a[i]==gold[i]!=b[i] for i in ix),
                    reused_only=sum(b[i]==gold[i]!=a[i] for i in ix)))
    return dict(rows=rows,nodes=nodes,unique_tokens=unique["tokens"],reused_tokens=repeat["tokens"])


def counts(case):
    gold,alt=case["gold"],case["alt"]
    clean=case["clean"]
    out=[]
    # Gold and natural behavior use exactly the same graph IDs as the interventions.
    for t in (case["D"],case["D"]+4,case["D"]+16):
        rows=clean[t-1]
        value=sum(y==p[3] for y,p in zip(gold,rows))
        out.append(dict(metric="fit",timing=str(t-case["D"]),condition="clean",block="none",
            observed=len(gold),value=value,graphs=1,graph_correct=int(value==len(gold)),
            first4=sum(y==p[3] and p[:3]==s for y,p,s in zip(gold,rows,case["structural_labels"]))))
    for name,section,start in (("within_standard",clean[:case["D"]],1),
                               ("additional",clean[case["D"]-1:],case["D"])):
        by_due=defaultdict(Counter)
        for t,(before,after) in enumerate(zip(section,section[1:]),start):
            for a,b,y,n in zip(before,after,gold,case["nodes"]):
                c=by_due["due_at_input" if t>=n["depth"]+1 else "before_due"]
                was=a[3]==y
                ok=b[3]==y
                c.update(observed=1,correct_inputs=was,new_errors=was and not ok,numeric_errors=a[3]<23 and not was,
                         numeric_repaired=a[3]<23 and not was and ok,non_numeric=a[3]>=23,
                         newly_filled=a[3]>=23 and ok)
        for due,c in by_due.items():
            out.append(dict(metric="natural",timing=name,condition="clean",block="none",readiness=due,**c))
    focus=case["focus"]
    u,q=focus[1:3]
    if "from_blank" in case:
        for arm,expected in enumerate((gold,alt)):
            vals=[node[3] for node in case["from_blank"][-1][arm]]
            out.append(dict(metric="fresh_solution",timing="standard_plus_8",block="none",
                condition=("original","edited")[arm],observed=len(focus),
                correct=sum(v==expected[i] for v,i in zip(vals,focus))))
    for e in case["natural_replay"]:
        plain,restored=e["child_observed_replayed"]
        out.append(dict(metric="natural_replay",timing=str(e["round"]),block="none",
            condition="previous_correct_state",observed=1,other_parents_correct=int(e["others_correct"]),
            paired_specific=int(e["others_correct"] and plain==e["expected_child"] and restored==gold[q])))
    controls={(e["timing"],e["block"]):e for e in case["events"] if e["condition"]=="clean"}
    for e in case["events"]:
        control=controls[e["timing"],e["block"]]
        paired=e["eligible"] and control["eligible"]
        row=dict(metric="intervention",timing=e["timing"],condition=e["condition"],block=e["block"],
                 attempts=1,eligible=int(paired),specific=0,damage_with_followup=0,stable_after_damage=0,target_correct_next=0,
                 direct_faults=0,direct_repaired_next=0,child_at_risk=0,child_ever_harmed=0,
                 child_damage_rounds=0,input_empty=int(e["target_input"][3]==24),
                 input_correct=int(e["target_input"][3]==gold[u]),
                 input_numeric_wrong=int(e["target_input"][3]<23 and e["target_input"][3]!=gold[u]),
                 start_eligible=int(e["start_ok"]),donor_eligible=int(e["donor_ok"]))
        if paired:
            path=e["trajectory"]
            reference=control["trajectory"]
            damage=[k for k,(a,b) in enumerate(zip(path,reference)) if
                    a[2][3]==alt[q]!=gold[q] and b[2][3]==gold[q]]
            hit=case["positions"][u] in e["patch_positions"]
            injected=hit and e["effective_input"][1][3]==alt[u]!=gold[u]
            row.update(target_correct_next=int(path[0][1][3]==gold[u]),
                       direct_faults=int(injected),direct_repaired_next=int(injected and path[0][1][3]==gold[u]),
                       child_at_risk=int(any(b[2][3]==gold[q] for b in reference)),
                       child_ever_harmed=int(any(a[2][3]!=gold[q]==b[2][3] for a,b in zip(path,reference))),
                       child_empty_next=int(path[0][2][3]==24),specific=int(bool(damage)),
                       damage_with_followup=int(bool(damage) and damage[0]<len(path)-2),
                       child_damage_rounds=sum(a[2][3]!=gold[q]==b[2][3] for a,b in zip(path,reference)),
                       stable_after_damage=int(bool(damage) and damage[0]<len(path)-2 and
                           all(z[1][3]==gold[u] and z[2][3]==gold[q] for z in path[-2:])))
        out.append(row)
        if paired:
            for step,readout in enumerate(path,1):
                out.append(dict(metric="focus_trajectory",timing=e["timing"],condition=e["condition"],
                       block=e["block"],step=step,observed=1,
                       **{role+"_"+label:int(node[3]==truth) for role,i,node in zip(
                           ("parent","target","child","sink"),focus,readout)
                           for label,truth in (("original",gold[i]),("edited",alt[i]),("empty",24))}))
    return out


def csv_output(path,rows):
    fields=dict.fromkeys(k for r in rows for k in r)
    tmp=path.with_suffix(".tmp")
    write_csv(tmp,({k:r.get(k) for k in fields} for r in rows))
    tmp.replace(path)


def four_way(rows):
    """Each row is one graph/timing on which all four primary weights were attempted."""
    arms=("A_s42","B_s42","C_s42","C_s42-correction")
    groups=defaultdict(dict)
    for r in rows:
        if r["metric"]=="intervention" and r["model"] in arms:
            groups[r["gid"],r["cell"],r["timing"],r["condition"],r["block"]][r["model"]]=r
    out=[]
    for (gid,cell,timing,condition,block),g in groups.items():
        common=all(g[a]["eligible"] for a in arms)
        row=dict(gid=gid,cell=cell,timing=timing,condition=condition,block=block,common=int(common))
        for a in arms:
            row[a+"_eligible"]=g[a]["eligible"]
            for k in ("direct_faults","direct_repaired_next","specific","damage_with_followup","stable_after_damage","child_ever_harmed"):
                row[a+"_"+k]=g[a][k] if common else None
        out.append(row)
    return out


def fit_sources(specs):
    ordinary=P.read(OLD)["finals"]
    selected={r["model"]:r for r in P.read(RECEIPT)["runs"]}
    rows={}
    for name,spec in specs.items():
        entry=dict(config=spec["config"],checkpoint=spec["sha256"],path=spec["path"])
        if name.endswith("-update"):
            base=name.split("-")[0]
            history=ROOT/"docs/records/2026-11-21/fit-history"/(base+".json")
            entry.update(validation=selected[base].get("dev_clean"),history=P.read(history))
        elif name in ordinary:
            entry.update(validation=ordinary[name].get("dev_clean"),epochs=ordinary[name]["epochs"])
        rows[name]=entry
    return rows


def pathway_contrasts(rows):
    groups=defaultdict(dict)
    for r in rows:
        if r["metric"]=="intervention":
            groups[r["model"],r["gid"],r["cell"],r["timing"],r["condition"]][r["block"]]=r
    out=[]
    for (model,gid,cell,timing,cond),g in groups.items():
        eligible=all(g[b]["eligible"] for b in ("none","parent","unrelated"))
        clean=g["none"]["target_correct_next"]
        out.append(dict(model=model,gid=gid,cell=cell,timing=timing,condition=cond,
                        eligible=int(eligible),unmasked_correct=int(eligible and clean),
                        parent_only_impairs=int(eligible and clean and g["unrelated"]["target_correct_next"]
                                                and not g["parent"]["target_correct_next"]),
                        unrelated_only_impairs=int(eligible and clean and g["parent"]["target_correct_next"]
                                                and not g["unrelated"]["target_correct_next"])))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--data-root",type=Path,default=ROOT,help="original server worktree containing tmp/analysis")
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--device",default="cuda")
    p.add_argument("--smoke-only",action="store_true")
    p.add_argument("--include-prior-mechanisms",action="store_true",
                   help="finish the six prior observational stages without scheduling stop policies")
    args=p.parse_args()
    if P.sha(args.plan)!=P.read(RECEIPT)["plan_sha256"]:
        raise ValueError("plan mismatch")
    specs=model_specs(P.read(args.plan),args.data_root)
    P.write(args.out/"fit-sources.json",fit_sources(specs))
    if args.include_prior_mechanisms:
        import reader_queue
        import research_queue
        from types import SimpleNamespace
        receipt,plan=P.read(RECEIPT),P.read(args.plan)
        for module,stages,runner in ((research_queue,research_queue.STAGES,research_queue.execute_stage),
                                     (reader_queue,reader_queue.SETTINGS,reader_queue.run)):
            prev=SimpleNamespace(device=args.device,train_root=args.data_root/"tmp/analysis/c-local-update-normal-v1/train",
                                 hard_root=args.data_root/research_queue.HARD_ROOT.relative_to(ROOT),
                                 out=args.data_root/module.OUT.relative_to(ROOT))
            for stage in stages:
                for small in ([True] if args.smoke_only else [True,False]):
                    if module is reader_queue:
                        runner(stage,small,prev,receipt,plan)
                    else:
                        runner(prev,stage,small,receipt,plan)
    for stage in STAGES:
        for smoke in ([True] if args.smoke_only else [True,False]):
            folder=args.out/("smoke" if smoke else "formal")/stage
            records=[]
            for model,spec in specs.items():
                out=folder/model
                out.mkdir(parents=True,exist_ok=True)
                ident=dict(protocol="attribution-v1",checkpoint=spec["sha256"],model=model,stage=stage,
                           smoke=smoke,plan=P.sha(args.plan),config=spec["config"],
                           code={p.name:P.sha(p) for p in (Path(__file__),Path(__file__).with_name("attribution_cases.py"),
                               Path(__file__).with_name("research_probes.py"),Path(__file__).with_name("design.py"),
                               TOOLS/"c_update/runtime.py",TOOLS/"c_local_update/rebinding.py",
                               ROOT/"experiments/model.py",ROOT/"Simtransformer/simtransformer/model_base.py")})
                if (out/"identity.json").exists() and P.read(out/"identity.json")!=ident:
                    raise ValueError(f"identity mismatch at {out}")
                P.write(out/"identity.json",ident)
                rt=None
                cells=([(n,live,gap,spaced) for n,live,gap in itertools.product((1,2,4),(0,4,8),(0,1))
                        for spaced in ((1,) if n==1 else (0,1))] if stage=="workspace" else [(4,),(8,)])
                for cell in cells:
                    for gid in ([380] if smoke else range(352,368)):
                        key="-".join(map(str,(*cell,gid)))
                        dest=out/f"{key}.json.gz"
                        if dest.exists():
                            with gzip.open(dest,"rt",encoding="utf8") as f:
                                data=json.load(f)
                        else:
                            if rt is None:
                                rt=Runtime(spec,args.device)
                            with torch.inference_mode():
                                data=(workspace(rt,gid,cell) if stage=="workspace" else run_graph(rt,gid,cell[0],stage))
                            tmp=dest.with_suffix(".tmp")
                            with gzip.open(tmp,"wt",encoding="utf8") as f:
                                json.dump(data,f,separators=(",",":"))
                            tmp.replace(dest)
                        rows=data["rows"] if stage=="workspace" else counts(data)
                        records.extend(dict(model=model,gid=gid,cell="-".join(map(str,cell)),**r) for r in rows)
                del rt
                P.write(out/"complete.json",ident)
                print(stage,model,"complete",flush=True)
            csv_output(folder/"graph-counts.csv",records)
            if stage!="workspace":
                csv_output(folder/"paired.csv",paired(records))
                csv_output(folder/"four-way-common.csv",four_way(records))
            if stage=="pathway":
                csv_output(folder/"key-comparison.csv",pathway_contrasts(records))
            P.write(folder/"delivery.json",dict(stage=stage,models=list(specs),rows=len(records),
                    table_hashes={p.name:P.sha(p) for p in folder.glob("*.csv")}))
    export(args.out/"transfer-benchmarks")
    export(args.out/"transfer-benchmarks-forced",forced=True)


def paired(rows):
    groups=defaultdict(dict)
    for r in rows:
        if r["metric"]=="intervention":
            groups[r["gid"],r["cell"],r["timing"],r["condition"],r["block"]][r["model"]]=r
    pairs=[("A_s42","B_s42")]
    for seed in (42,1234):
        base=f"C_s{seed}"
        pairs.extend([(base,base+"-correction"),(base+"-correction",base+"-update")])
    totals=defaultdict(Counter)
    for (_,cell,timing,cond,block),models in groups.items():
        for a,b in pairs:
            x,y=models[a],models[b]
            c=totals[a,b,cell,timing,cond,block]
            common=x["eligible"] and y["eligible"]
            c.update(attempted=1,both_eligible=int(common),a_eligible=x["eligible"],b_eligible=y["eligible"])
            for suffix,r in (("a",x),("b",y)):
                for k in ("target_correct_next","direct_faults","direct_repaired_next","child_at_risk",
                          "child_ever_harmed","specific","damage_with_followup","stable_after_damage","child_damage_rounds"):
                    c[k+"_"+suffix]+=r[k] if common else 0
    return [dict(zip(("a","b","cell","timing","condition","block"),key),**c) for key,c in totals.items()]


if __name__=="__main__":
    main()
