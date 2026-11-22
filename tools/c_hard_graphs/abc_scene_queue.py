"""Apply matched input scenarios to discrete A'/B and original continuous C."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import tarfile

import torch
from attribution_cases import exposure_case, patch_state
from attribution_queue import model_specs
from hard_eval import ROOT, PLAN, RECEIPT, P
from original_c_recovery import measure, MODELS, PROVENANCE
from research_probes import encode

DOC = ROOT/"docs/records/2026-11-22"
C_RAW = DOC/"original-c-recovery/c-only"
ARCHIVE = DOC/"c-original-layered-injection/raw-layered-16round.tar.gz"
OUT = ROOT/"tmp/analysis/abc-input-scenarios-v1"
DISCRETE = ("A_s42","B_s42")


def archive(model):
    with tarfile.open(ARCHIVE) as source:
        return json.load(source.extractfile(model+"-layered.json"))


def saved(path):
    with gzip.open(path,"rt",encoding="utf8") as f:
        return json.load(f)


def publish(path, data):
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp,"wt",encoding="utf8") as f:
        json.dump(data,f,separators=(",",":"))
    tmp.replace(path)


def recipes(case):
    """Construction uses topology, arithmetic and the intervention clock only."""
    nodes=case["base"]["nodes"]
    u,r,q,a=case["u"],case["r"],case["q"],case["ancestor"]
    v=case["unrelated"]
    p=next(i for i in nodes[u]["parents"] if i!=r)
    assert nodes[p]["depth"]==0
    due=nodes[u]["depth"]+1
    times={"early":due-2,"before":due-1,"after":due,"complete":case["D"]+4}
    result=[]
    for phase,t in times.items():
        def recipe(name, edits):
            result.append(dict(label=phase+"_"+name,start=t,phase=phase,edits=edits))
        def put(i, donor):
            return (i,max(t,nodes[i]["depth"]+1),donor)
        if phase=="early":
            recipe("clean",[])
            recipe("fill_correct",[put(u,0)])
            recipe("internal",[put(u,1)])
            recipe("direct_parent",[put(r,1)])
            recipe("older_ancestor",[put(a,1)])
            continue
        # Each donor row has its original position and spelling. For a pending slot,
        # obtain the number at its first scheduled computed round.
        recipe("clean",[])
        recipe("input_source",[put(0,1)])
        recipe("internal",[put(u,1)])
        recipe("direct_parent",[put(r,1)])
        recipe("older_ancestor",[put(a,1)])
        recipe("separate_branch",[put(v,2)])
        recipe("both_branches",[put(u,1),put(v,2)])
        recipe("both_parents_add",[put(r,1),put(p,3)])
        recipe("both_parents_cancel",[put(r,1),put(p,4)])
        recipe("parent_and_target_consistent",[put(r,1),put(u,1)])
        recipe("parent_and_target_conflict",[put(r,1),put(u,5)])
        if phase=="complete":
            recipe("sink",[put(len(nodes)-1,1)])
            recipe("restore_empty",[(u,due-1,0)])
        if phase=="before":
            recipe("fill_correct",[put(u,0)])
        if phase=="after":
            recipe("target_and_child",[put(u,1),put(q,1)])
    return result,p


def legacy_recipes(records,c):
    for rec in records:
        t=rec["start"]
        due=c["base"]["nodes"][c["u"]]["depth"]+1
        edits=[(p,src,0 if extra==["base"] else 1)
               for p,src,i,*extra in rec["patches"] if p==i]
        if len(edits)!=len(rec["patches"]):
            raise ValueError("Noncorresponding donor position")
        yield dict(label=rec["label"],start=t,edits=edits,
                   phase="before" if t<due else "after" if t==due else "complete")


def run_grid(rt,gid,depth,records=None):
    c=exposure_case(gid,depth)
    nodes=c["base"]["nodes"]
    pos=[n["pos"] for n in nodes]
    program,side=recipes(c)
    arms=[c["base"],c["donor"]]
    x=encode(rt,arms)
    gold=[c["gold"],c["alt"]]
    if records is None:
        # Edits to literal tokens build four extra donor problems, never presented
        # to the recipient. The original literal is two tokens before its definition.
        delta=c["delta"]
        for leaf,step in ((1,delta),(side,delta),(side,-delta),(0,2*delta)):
            xx=x[:1].clone()
            literal=pos[leaf]-2
            xx[0,literal,3]=(xx[0,literal,3]+step)%23
            x=torch.cat((x,xx),dim=0)
            vals=[]
            for i,n in enumerate(nodes):
                vals.append(sum(vals[p] for p in n["parents"])%23 if n["parents"]
                            else (n["truth"]+step*int(i==leaf))%23)
            gold.append(vals)
    else:
        program=list(legacy_recipes(records,c))
    rounds={s["start"] for s in program}
    rounds.update(t for s in program for _,t,_ in s["edits"])
    snapshots={}
    state=rt.init(x)
    for t in range(1,max(rounds)+1):
        state,_,_=rt.forward(state)
        if t in rounds:
            snapshots[t]=state.clone()
    labels=x[0,pos,:3].tolist()
    events=[]
    for scene in program:
        t=scene["start"]
        state=snapshots[t][:1].clone()
        before=rt.decode(state)[0,pos,:4].tolist()
        # Planned clock and actual start both remain visible, even on ineligible cases.
        old=[n["truth"] if n["depth"]<t else 24 for n in nodes]
        stage_ok=all(z[3]==y for z,y in zip(before,old))
        writes=[]
        for i,src,arm in scene["edits"]:
            donor=snapshots[src][arm:arm+1]
            z=rt.decode(donor)[0,pos[i],:4].tolist()
            want=gold[arm][i] if src>nodes[i]["depth"] else 24
            patch_state(rt,state,donor,[pos[i]])
            actual=rt.decode(state)[0,pos[i],:4].tolist()
            writes.append(dict(i=i,src=src,arm=arm,before=before[i],intended=want,actual=actual,
                               valid=z==labels[i]+[want] and actual==z and before[i]==labels[i]+[old[i]]))
        start_values=[z[3] for z in before]
        rows=[]
        for step in range(16):
            state,logits,_=rt.forward(state)
            rows.append(logits[3][0,pos].argmax(-1).tolist())
        events.append(dict(**scene,start_values=start_values,writes=writes,stage_ok=stage_ok,
                           eligible=stage_ok and all(w["valid"] for w in writes),
                           value_eligible=stage_ok and all(w["before"][3]==old[w["i"]] and
                               w["actual"][3]==w["intended"] for w in writes),rows=rows))
    return dict(gid=gid,depth=depth,nodes=nodes,u=c["u"],q=c["q"],events=events)


def fingerprint(spec,plan):
    files=[Path(__file__),Path(__file__).with_name("original_c_recovery.py"),
           Path(__file__).with_name("attribution_cases.py"),Path(__file__).with_name("research_probes.py"),
           ROOT/"tools/c_update/runtime.py",ROOT/"tools/c_local_update/rebinding.py",
           ROOT/"experiments/model.py",ROOT/"Simtransformer/simtransformer/model_base.py"]
    files.extend(sorted((ROOT/"experiments/configs"/spec["config"]).glob("*.yaml")))
    return dict(model=spec["id"],checkpoint=spec["sha256"],plan=P.sha(plan),config=spec["config"],
                code={f.name:P.sha(f) for f in files})


def run(args):
    plan=args.data_root/PLAN.relative_to(ROOT)
    if P.sha(plan)!=P.read(RECEIPT)["plan_sha256"]:
        raise ValueError("Plan mismatch")
    specs=model_specs(P.read(plan),args.data_root)
    receipt=P.read(PROVENANCE)["models"]
    old=archive("C_s42")
    grouped=defaultdict(list)
    for item in old["records"]:
        grouped[item["case"]].append(item)
    from runtime import Runtime
    root=args.out/("smoke" if args.smoke else "formal")
    for model in args.models:
        spec=specs[model]
        if spec["sha256"]!=receipt[model]:
            raise ValueError(model+" checkpoint mismatch")
        rt=Runtime(spec,args.device)
        for suite in ("standard","layered","expanded"):
            if model not in DISCRETE and suite!="expanded":
                continue  # Existing C large-grid and layered records have already shipped.
            folder=root/suite/model
            folder.mkdir(parents=True,exist_ok=True)
            identity=fingerprint(spec,plan)
            if (folder/"identity.json").exists() and P.read(folder/"identity.json")!=identity:
                raise ValueError(f"Incompatible saved run at {folder}")
            P.write(folder/"identity.json",identity)
            if suite=="standard":
                cells=[(g,n,d) for n in (32,64,128) for d in (4,10) for g in range(32)]
            elif suite=="layered":
                cells=[(g,0,d) for g in range(352,368) for d in (4,8)]
            else:
                cells=[(g,0,d) for g in range(400,432) for d in (4,8)]
            if args.smoke:
                cells=cells[:1]
            for gid,size,depth in cells:
                path=folder/f"{size}-{depth}-{gid}.json.gz"
                if path.exists():
                    continue
                with torch.inference_mode():
                    if suite=="standard":
                        result=measure(rt,gid,size,depth)
                    else:
                        template=grouped[f"{gid}-{depth}"] if suite=="layered" else None
                        result=run_grid(rt,gid,depth,template)
                publish(path,result)
            print(suite,model,len(cells),"done",flush=True)
        del rt
    return root


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--data-root",type=Path,default=ROOT)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--models",choices=MODELS,nargs="+",default=DISCRETE+MODELS[:2])
    run(parser.parse_args())
