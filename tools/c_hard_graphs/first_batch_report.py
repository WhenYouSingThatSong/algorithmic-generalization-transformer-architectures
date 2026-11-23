"""Build paired graph-level tables; denominators remain visible even with zero coverage."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import tarfile

from abc_scene_queue import C_RAW, DOC, DISCRETE, archive, saved
from abc_scene_report import standard_observations, grid_observations
from first_batch import ALL, ORIGINAL, POST, OUT, P, PROVENANCE, cells


def write(folder,name,rows):
    rows=list(rows)
    dest=folder/(name+".csv")
    tmp=dest.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        if rows:
            keys=dict.fromkeys(k for row in rows for k in row)
            w=csv.DictWriter(f,fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    tmp.replace(dest)


def graph_rows(root,stage,model,smoke):
    folder=root/stage/model
    idn=P.read(folder/"identity.json")
    original=model.split("-epoch")[0]
    if idn["model"]!=original or ("-epoch" not in model and idn["checkpoint"]!=P.read(PROVENANCE)["models"][original]):
        raise ValueError("Wrong identity at "+str(folder))
    if smoke and stage=="natural":
        for path in sorted(folder.glob("*.json.gz")):
            yield tuple(map(int,path.name.split(".")[0].split("-"))),saved(path)
        return
    for n,d,g in cells(stage,smoke):
        yield (n,d,g),saved(folder/f"{n}-{d}-{g}.json.gz")


def groups():
    return {"primary":("A_s42","B_s42","C_s42","C_s42-correction"),
            "four_with_C1234":("A_s42","B_s42","C_s1234","C_s1234-correction"),
            **{m:(m,m+"-correction",m+"-update") for m in ORIGINAL}}


def pair(models,key):
    for group,arms in groups().items():
        same=all(models[m]["eligible"] for m in arms)
        clock=same and len({models[m]["t"] for m in arms})==1
        fit=clock and all(models[m].get("fit_ok",False) for m in arms)
        for m in arms:
            z=models[m]
            for scope,ok in (("individual",z["eligible"]),("common",same),
                             ("common_clock",clock),("common_clock_fit",fit)):
                if scope=="common_clock_fit" and "fit_ok" not in z:
                    continue
                yield dict(**key,group=group,model=m,scope=scope,attempts=1,eligible=int(ok),
                           **{k:int(v) for k,v in z.items()
                              if ok and k not in ("eligible","t","fit_ok") and v is not None})


def summarize(out,name,records):
    counts=defaultdict(Counter)
    records=list(records)
    write(out,name+"-graphs",records)
    for row in records:
        key=tuple((k,v) for k,v in row.items() if isinstance(v,str) or k in ("fanout",))
        counts[key].update({k:v for k,v in row.items() if isinstance(v,int) and k not in
                            ("depth","gid","size","fanout")})
    write(out,name+"-counts",(dict(dict(k),**z) for k,z in counts.items()))
    paired=defaultdict(dict)
    for row in records:
        if row["scope"]!="common" or not row["eligible"]:
            continue
        key=tuple((k,row[k]) for k in ("suite","size","depth","gid","role","phase",
                                      "condition","fanout","group") if k in row)
        paired[key][row["model"]]=row
    effects=defaultdict(Counter)
    binary=("next_correct","target_next","specific","child_specific","child_damage","output_ever_wrong")
    for key,by_model in paired.items():
        candidates=[("A_s42","B_s42")]
        for model in ORIGINAL:
            candidates.extend((model,model+"-"+arm) for arm in ("correction","update"))
        for base,trained in candidates:
            if base not in by_model or trained not in by_model:
                continue
            a,b=by_model[base],by_model[trained]
            for metric in binary:
                if metric not in a or metric not in b:
                    continue
                cell=effects[tuple(kv for kv in key if kv[0]!="gid"),base,trained,metric]
                cell.update(common=1,base_only=int(bool(a[metric]) and not b[metric]),
                            trained_only=int(bool(b[metric]) and not a[metric]),
                            both=int(bool(a[metric]) and bool(b[metric])))
    write(out,name+"-paired",(dict(dict(k),base=base,trained=trained,metric=metric,**v)
          for (k,base,trained,metric),v in effects.items()))


def scenes(root,out,smoke):
    # Read the committed discrete archive without extracting paths to a shared worktree.
    folder=DOC/"abc-input-scenarios-results"
    packed=folder/"abc-ab-raw.tar.gz"
    if P.sha(packed)!=(folder/"abc-ab-raw-archive.sha256").read_text().split()[0]:
        raise ValueError("A/B archive checksum mismatch")
    blobs={}
    with tarfile.open(packed,"r|gz") as tar:
        for entry in tar:
            if entry.isfile() and entry.name.endswith(".json.gz"):
                blobs[entry.name]=tar.extractfile(entry).read()
    data={}
    fits=[]
    signatures={}
    for m in ALL:
        for stage in ("standard","expanded","layered"):
            old=m in ORIGINAL and stage=="layered"
            if old:
                history=archive(m)
                if history["checkpoint"]!=P.read(PROVENANCE)["models"][m]:
                    raise ValueError("layered checkpoint mismatch")
                bycase=defaultdict(list)
                for e in history["records"]:
                    bycase[e["case"]].append(e)
            if m in POST:
                items=graph_rows(root,stage,m,smoke)
            else:
                def existing():
                    for n,d,g in cells(stage,smoke):
                        name=f"{n}-{d}-{g}.json.gz"
                        if m in DISCRETE:
                            case=json.loads(gzip.decompress(blobs[f"{stage}/{m}/{name}"]))
                        elif old:
                            k=f"{g}-{d}"
                            case=dict(history["cases"][k],events=bycase[k])
                        elif stage=="expanded":
                            case=saved(DOC/"abc-input-scenarios/c-only-expanded/raw"/m/name)
                        else:
                            case=saved(C_RAW/"raw"/m/name)
                        yield (n,d,g),case
                items=existing()
            for (n,d,g),case in items:
                nodes=case["nodes"]
                sig=[(z["pos"],z["truth"],tuple(z["parents"])) for z in nodes]
                if signatures.setdefault((stage,n,d,g),sig)!=sig:
                    raise ValueError("unmatched graphs")
                fit=None
                if stage=="standard":
                    f=case["fit"][str(d+5)]
                    fit=f["value"]==len(nodes)
                    for t,metrics in case["fit"].items():
                        fits.append(dict(model=m,size=n,depth=d,gid=g,round=t,**metrics))
                    observations=standard_observations(case)
                else:
                    observations=grid_observations(case,old)
                for k,z in observations:
                    if fit is not None:
                        z["fit_ok"]=fit
                    models=data.setdefault((stage,len(nodes),d,g,*k),{})
                    if m in models:
                        raise ValueError("duplicate scene")
                    models[m]=z
    write(out,"same-graph-fit",fits)
    def records():
        for (stage,n,d,g,role,timing,condition),models in data.items():
            if set(models)!=set(ALL):
                raise ValueError("missing model in "+str((stage,g,condition)))
            yield from pair(models,dict(suite=stage,size=n,depth=d,gid=g,role=role,phase=timing,condition=condition))
    summarize(out,"scenes",records())


def propagation(z):
    u=z["u"]
    children=z["children"]
    gold=z["gold"]
    ref=z["clean"]
    path=z["changed"]
    def wrong(i,t):
        return ref[t][i]==gold[i] and path[t][i]!=gold[i]
    def numeric(i,t):
        return wrong(i,t) and 0<=path[t][i]<23
    rounds=range(len(path))
    first=next((t+1 for t in rounds if path[t][u]==gold[u]),None)
    first_child=next((t+1 for t in rounds if any(numeric(i,t) and path[t][i]==z["alt"][i]
                       for i in children)),None)
    return dict(eligible=z["eligible"],t=z["t"],target_next=path[0][u]==gold[u],
                target_recovered=first is not None,target_first_round_sum=first or 0,
                child_specific=first_child is not None,child_first_specific_round_sum=first_child or 0,
                child_at_risk=any(ref[t][i]==gold[i] for i in children for t in rounds),
                children_specific=sum(any(numeric(i,t) and path[t][i]==z["alt"][i] for t in rounds) for i in children),
                descendant_wrong_number_rounds=sum(numeric(i,t) for i in z["descendants"] for t in rounds),
                descendant_empty_rounds=sum(wrong(i,t) and path[t][i]==24 for i in z["descendants"] for t in rounds),
                output_at_risk=any(ref[t][z["output"]]==gold[z["output"]] for t in rounds),
                output_ever_wrong=any(wrong(z["output"],t) for t in rounds),
                output_wrong_end=wrong(z["output"],len(path)-1),
                followup=first_child is not None and first_child<=len(path)-2,
                stable_after_specific=first_child is not None and first_child<=len(path)-2 and all(
                    row[i]==gold[i] for row in path[-2:] for i in [u,*children]))


def fanout(root,out,smoke):
    bycase=defaultdict(dict)
    for m in ALL:
        for (_,d,g),case in graph_rows(root,"fanout",m,smoke):
            for z in case["records"]:
                bycase[d,g,z["fanout"],z["phase"]][m]=propagation(z)
    summarize(out,"fanout",(row for (d,g,f,phase),models in bycase.items() for row in
              pair(models,dict(depth=d,gid=g,fanout=f,phase=phase)) if row["scope"]!="common_clock_fit"))
    topology=[]
    for d,g,f,phase in bycase:
        if f!=1:
            continue
        for m in ALL:
            for multiplicity in (2,4):
                one=bycase[d,g,1,phase][m]
                many=bycase[d,g,multiplicity,phase][m]
                row=dict(model=m,depth=d,gid=g,phase=phase,fanout=multiplicity,
                         common=one["eligible"] and many["eligible"])
                if row["common"]:
                    for k in ("target_next","child_specific","output_ever_wrong",
                              "descendant_wrong_number_rounds","descendant_empty_rounds"):
                        row["one_"+k]=one[k]
                        row["many_"+k]=many[k]
                topology.append(row)
    write(out,"fanout-topology-pairs",topology)


def mechanisms(root,out,smoke):
    numerators=[]
    replay=[]
    history=[]
    for model in ORIGINAL:
        for (n,d,g),case in graph_rows(root,"natural",model,smoke):
            key=dict(model=model,size=n,depth=d,gid=g)
            numerators.append(dict(**key,opportunities=case["opportunities"],node_errors=case["node_events"],
                edge_records=case["events"],
                selected_edges=case["selected"],matched=sum(bool(z.get("paired_replay_agrees")) for z in case["records"])))
            for z in case["records"]:
                row=dict(**key,t=z["t"],u=z["u"],q=z["q"],wrong=z["wrong"],previously_confirmed=z["archived_specific"])
                if "skip" in z:
                    replay.append(dict(**row,skip=z["skip"]))
                    continue
                for bg,b in z["backgrounds"].items():
                    if "paths" not in b:
                        replay.append(dict(**row,background=bg,skip="no_historical_source"))
                        continue
                    for name,path in b["paths"].items():
                        replay.append(dict(**row,background=bg,arm=name,paired_clock_matches=z["paired_replay_agrees"],
                            changed_context_nodes=len(b["historical_rounds"]),target_next=path[0][0]==z["gold"][0],
                            child_specific_next=path[0][1]==z["expected_child"],
                            joint_last_two=path[-1]==path[-2]==z["gold"]))
        for (_,d,g),case in graph_rows(root,"history",model,smoke):
            arms=case["arms"]
            for arm in arms:
                first=arm["trace"][0][1]
                history.append(dict(model=model,depth=d,gid=g,t=case["t"],arm=arm["name"],
                    eligible=arm["eligible"],common=all(a["eligible"] for a in arms),
                    positive_control_used=arms[-1]["eligible"] and arms[-1]["trace"][0][1]==case["wrong_reader"],
                    clean_reader_ok=arms[0]["trace"][0][1]==case["reader_gold"],distance=arm["distance"],
                    reader_next=first,reader_next_correct=first==case["reader_gold"],
                    reader_end_correct=arm["trace"][-1][1]==case["reader_gold"]))
    write(out,"natural-coverage",numerators)
    write(out,"natural-pairs",replay)
    write(out,"same-value-new-reader",history)
    totals=defaultdict(Counter)
    for z in history:
        c=totals[z["model"],z["depth"],z["arm"]]
        c.update(attempts=1,own_eligible=int(z["eligible"]),common=int(z["common"]))
        if z["common"] and z["clean_reader_ok"] and z["positive_control_used"]:
            c.update(confirmed_read_opportunities=1,correct_next=int(z["reader_next_correct"]),
                     correct_end=int(z["reader_end_correct"]),effective=int(z["distance"]>0))
    write(out,"same-value-counts",(dict(model=k[0],depth=k[1],arm=k[2],**v) for k,v in totals.items()))
    totals=defaultdict(Counter)
    for z in replay:
        if "arm" not in z or not z["paired_clock_matches"]:
            continue
        c=totals[z["model"],z["background"],z["arm"],z["previously_confirmed"]]
        c.update(matched_edges=1,context_changed=int(z["changed_context_nodes"]>0),
                 target_next=int(z["target_next"]),child_specific_next=int(z["child_specific_next"]),
                 joint_last_two=int(z["joint_last_two"]))
    write(out,"natural-counts",(dict(model=k[0],background=k[1],arm=k[2],
          previously_confirmed=k[3],**v) for k,v in totals.items()))


def marks(root,out,smoke):
    stats=defaultdict(Counter)
    for model in sorted(p.name for p in (root/"mark").iterdir() if p.is_dir()):
        for (_,d,g),case in graph_rows(root,"mark",model,smoke):
            for row in case["rows"]:
                old=row["before"]
                for cut in (.1,.25,.5,.75,.9):
                    mark=row["score"]>cut
                    cell=stats[model,d,g,row["phase"],row["condition"],row["actual_mixed"],row["category"],row["target"],cut]
                    cell.update(attempts=1,eligible=int(row["eligible"]))
                    if not row["eligible"]:
                        continue
                    change=row["after"]!=old
                    correct=row["after"]==row["gold"]
                    cell.update(marked=int(mark),changed=int(change),original_correct=int(correct),
                        local_correct=int(row["after"]==row["expected"]),marked_changed=int(mark and change),
                        corrected_unmarked=int(not mark and old!=row["gold"] and correct),
                        healthy_false_positive=int(row["category"]=="healthy" and mark),
                        local_conflict_original_correct=int(row["category"]=="direct_conflict" and old==row["gold"]))
    seen={key[0] for key in stats}
    for m in (k+"-update" for k in ORIGINAL):
        for weight in (m,m+"-epoch10",m+"-epoch30"):
            if weight not in seen:
                raise ValueError("missing planned mark scores "+weight)
    keys=("model","depth","gid","phase","condition","actual_mixed","category","target","threshold")
    write(out,"mark-thresholds",(dict(zip(keys,k),**v) for k,v in stats.items()))
    totals=defaultdict(Counter)
    for (model,d,g,phase,cond,mixed,cat,target,cut),values in stats.items():
        totals[model,phase,cond,mixed,cat,target,cut].update(values)
    keys=("model","phase","condition","actual_mixed","category","target","threshold")
    write(out,"mark-counts",(dict(zip(keys,k),**v) for k,v in totals.items()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--tables",choices=("scenes","fanout","mechanisms","marks"),nargs="+",
                   default=["scenes","fanout","mechanisms","marks"])
    a=p.parse_args()
    root=a.out/("smoke" if a.smoke else "formal")
    dest=root/"tables"
    dest.mkdir(parents=True,exist_ok=True)
    for table in a.tables:
        {"scenes":scenes,"fanout":fanout,"mechanisms":mechanisms,"marks":marks}[table](root,dest,a.smoke)
