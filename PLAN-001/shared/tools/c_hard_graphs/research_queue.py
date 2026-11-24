"""Run the existing stop pilot and four follow-ups in serial, one writer per output root."""
import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from design import COUNT, EXTENSION, P, PROFILES, SCALES, TOOLS, make_case
from hard_eval import MODELS, OUT as HARD_ROOT, PLAN, RECEIPT, ROOT, write_csv
from research_dynamics import signal_rows, transitions
from research_probes import fanout_probe, identity_probe
from research_tables import contrasts, matched_fanout
from runtime import Runtime
from stop_pilot import OUT as STOP_ROOT

HERE = Path(__file__).resolve().parent
OUT = ROOT / "tmp/analysis/c-research-sequence-v1"
PROVENANCE = ROOT / "docs/records/2026-11-21/hard-graphs-provenance.json"
STAGES = ("transitions","signals","identity","fanout")
# Remaining columns are additive counts. These keys define nonoverlapping cells.
GROUPS = {
    "transitions": ("arm","window","previous","following"),
    "signals": ("arm","window","margin_bin","repeat","update","previous","next_correct"),
    "identity": ("condition","extra","step","recipient_ok","source_ok","common_eligible",
                 "u_name_correct","u_value_correct","u_read_as_v"),
    "fanout": ("timing","role","coefficient","step","recipient_ok","source_ok"),
}


def specifications(stage, smoke):
    if stage == "transitions":
        cells = [(w,d,p) for w,d in SCALES for p in PROFILES]+[EXTENSION]
        ids = [0] if smoke else range(COUNT)
    elif stage == "signals":
        cells = [(w,d,p) for w,d in SCALES for p in ("uneven","balanced")]
        ids = [128] if smoke else range(80,88)
    elif stage == "identity":
        cells = [(2,d,p) for d in (4,8) for p in ("equal","different")]
        ids = [128] if smoke else range(80,96)
    else:
        cells = [(4,d,str(f)) for d in (4,8) for f in (1,2,4)]
        ids = [128] if smoke else range(80,96)
    return [(w,d,p,gid) for w,d,p in cells for gid in ids]


def compute(stage, spec, rt, hard_root, model, examples):
    w,d,p,gid = spec
    if stage == "transitions":
        src = hard_root/"formal"/model/f"w{w}-d{d}-{p}-{gid}.json.gz"
        with gzip.open(src,"rt",encoding="utf-8") as f:
            data=json.load(f)
        case=data["case"]
        if (case["width"],case["depth"],case["profile"],case["gid"]) != spec:
            raise ValueError(f"unexpected raw case in {src}")
        return transitions(case,data["result"]["traces"])
    if stage == "signals":
        return signal_rows(rt,make_case(w,d,p,gid))
    if stage == "identity":
        return identity_probe(rt,gid,d,p=="equal",examples)
    return fanout_probe(rt,gid,d,int(p),examples)


def grouped(rows, keys):
    bins=defaultdict(Counter)
    for r in rows:
        bins[tuple(r[k] for k in keys)].update({k:v for k,v in r.items() if k not in keys})
    return [dict(**dict(zip(keys,k)),**v) for k,v in sorted(bins.items())]


def checkpoint(args, run):
    if args.train_root:
        return args.train_root/run["model"]/"final.ckpt"
    return ROOT/run["final_ckpt"]["path"]


def source_hashes():
    sources = [Path(__file__),HERE/"research_dynamics.py",HERE/"research_probes.py",HERE/"research_tables.py",HERE/"design.py",
               HERE/"hard_eval.py",TOOLS/"c_update/runtime.py",TOOLS/"c_update/protocol.py",
               TOOLS/"c_local_update/rebinding.py"]
    return {p.name:P.sha(p) for p in sources}


def execute_stage(args, stage, smoke, receipt, plan):
    folder=args.out/("smoke" if smoke else "formal")/stage
    combined=[]
    specs=specifications(stage,smoke)
    sources=source_hashes()
    for model in MODELS:
        run=next(r for r in receipt["runs"] if r["model"]==model)
        ident=dict(stage=stage,model=model,checkpoint=run["final_ckpt"]["sha256"],
                   plan=receipt["plan_sha256"],sources=sources,smoke=smoke,cases=len(specs))
        if stage=="transitions":
            expected=P.read(PROVENANCE)["models"][model]["formal_complete_sha256"]
            if P.sha(args.hard_root/"formal"/model/"complete.json")!=expected:
                raise ValueError(f"wrong hard-graph raw directory for {model}")
            ident["hard_graph_receipt"]=expected
        out=folder/model
        out.mkdir(parents=True,exist_ok=True)
        identity=out/"identity.json"
        if identity.exists() and P.read(identity)!=ident:
            raise ValueError(f"incompatible run already at {out}")
        P.write(identity,ident)
        rt=None
        for spec in specs:
            w,d,profile,gid=spec
            file=out/f"{w}-{d}-{profile}-{gid}.json"
            if file.exists():
                rows=P.read(file)["counts"]
            else:
                if stage!="transitions" and rt is None:
                    base=next(m for m in plan["models"] if m["id"]==model)
                    rt=Runtime(dict(base,arm="update",path=str(checkpoint(args,run)),
                                    sha256=ident["checkpoint"]),args.device)
                with torch.inference_mode():
                    examples=[]
                    rows=grouped(compute(stage,spec,rt,args.hard_root,model,examples),GROUPS[stage])
                P.write(file,dict(counts=rows,trajectories=examples))
            for r in rows:
                combined.append(dict(model=model,width=w,depth=d,profile=profile,gid=gid,**r))
        del rt
        P.write(out/"complete.json",ident)
        print(stage,model,len(specs),"graphs complete",flush=True)
    # Keep per-graph cells in the shards. Summaries pool counts only within a
    # model and design cell. Contributing graphs are counted once in each row.
    keys=("model","width","depth","profile",*GROUPS[stage])
    def summary_input():
        for row in combined:
            yield {**{k:v for k,v in row.items() if k!="gid"},"contributing_graphs":1}
    table=folder/"summary.csv"
    temp=table.with_suffix(".tmp")
    summary=grouped(summary_input(),keys)
    write_csv(temp,summary)
    temp.replace(table)
    contrast_table=folder/"contrasts.csv"
    temp=contrast_table.with_suffix(".tmp")
    write_csv(temp,contrasts(stage,summary))
    temp.replace(contrast_table)
    delivery=dict(stage=stage,cases_per_model=len(specs),models=list(MODELS),
                  summary_sha256=P.sha(table),contrasts_sha256=P.sha(contrast_table),smoke=smoke)
    if stage=="fanout":
        paired=folder/"paired-fanout.csv"
        temp=paired.with_suffix(".tmp")
        write_csv(temp,matched_fanout(combined))
        temp.replace(paired)
        delivery["paired_fanout_sha256"]=P.sha(paired)
    P.write(folder/"delivery.json",delivery)


def finish_stop(args, receipt, smoke):
    folder=args.stop_root/("smoke" if smoke else "formal")
    option=["--smoke"] if smoke else []
    for model in MODELS:
        run=next(r for r in receipt["runs"] if r["model"]==model)
        completed=folder/model/"complete.json"
        if completed.exists():
            m=P.read(completed)
            ids=[64] if smoke else list(range(32,48))
            if (m["checkpoint"]!=run["final_ckpt"]["sha256"] or m["plan"]!=receipt["plan_sha256"]
                    or m["model"]!=model or m["ids"]!=ids or m["cases"]!=17*len(ids)):
                raise ValueError(f"unexpected stop-pilot receipt {completed}")
            continue
        subprocess.run([sys.executable,str(HERE/"stop_pilot.py"),"--model",model,"--device",args.device,
                        "--out",str(args.stop_root),"--plan",str(args.plan),"--checkpoint",
                        str(checkpoint(args,run)),*option],check=True)
    subprocess.run([sys.executable,str(HERE/"stop_report.py"),"--out",str(args.stop_root),*option],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in (("plan",PLAN),("hard-root",HARD_ROOT),("stop-root",STOP_ROOT),
                         ("train-root",None),("out",OUT)):
        p.add_argument("--"+name,type=Path,default=default)
    p.add_argument("--device",default="cuda")
    p.add_argument("--smoke-only",action="store_true")
    args=p.parse_args()
    if args.out.resolve() in (args.hard_root.resolve(),args.stop_root.resolve()):
        raise ValueError("new research results require their own output root")
    receipt=P.read(RECEIPT)
    if P.sha(args.plan)!=receipt["plan_sha256"]:
        raise ValueError("unexpected training plan")
    plan=P.read(args.plan)
    for stage in ("stop",*STAGES):
        for smoke in ([True] if args.smoke_only else [True,False]):
            if stage=="stop":
                finish_stop(args,receipt,smoke)
            else:
                execute_stage(args,stage,smoke,receipt,plan)


if __name__=="__main__":
    main()
