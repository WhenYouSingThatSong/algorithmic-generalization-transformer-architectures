"""Run the five fixed-weight studies approved on 2026-11-23."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from abc_scene_queue import (C_RAW, PLAN, P, PROVENANCE, RECEIPT, ROOT, archive, fingerprint,
                             measure, model_specs, publish, run_grid, saved)
from attribution_queue import fit_sources
from first_batch_probes import fanout_probe, history_probe, mark_probe, natural_probe
from runtime import Runtime

OUT=ROOT/"tmp/analysis/first-batch-20261123"
ORIGINAL=("C_s42","C_s1234")
POST=tuple(m+"-"+arm for m in ORIGINAL for arm in ("correction","update"))
ALL=("A_s42","B_s42")+ORIGINAL+POST
STAGES=("standard","layered","expanded","natural","history","mark","fanout")
STAGE_MODELS={"standard":POST,"layered":POST,"expanded":POST,"natural":ORIGINAL,
              "history":ORIGINAL,"mark":tuple(m+"-update" for m in ORIGINAL),"fanout":ALL}


def cells(stage,smoke):
    if stage in ("standard","natural"):
        rows=[(n,d,g) for n in (32,64,128) for d in (4,10) for g in range(32)]
        if smoke and stage=="natural":
            return rows  # scan small archived records until a real replay is found
    else:
        start,count=(352,16) if stage=="layered" else (400,32)
        rows=[(0,d,g) for d in (4,8) for g in range(start,start+count)]
    return rows[:1] if smoke else rows


def run(args):
    plan=args.data_root/PLAN.relative_to(ROOT)
    if P.sha(plan)!=P.read(RECEIPT)["plan_sha256"]:
        raise ValueError("plan mismatch")
    specs=model_specs(P.read(plan),args.data_root)
    ids=P.read(PROVENANCE)["models"]
    groups=defaultdict(list)
    if "layered" in args.stages:
        for row in archive("C_s42")["records"]:
            groups[row["case"]].append(row)
    for model in args.models:
        stages=[s for s in args.stages if model in STAGE_MODELS[s]]
        if not stages:
            raise ValueError("No applicable stage for "+model)
        spec=dict(specs[model],id=model)
        if spec["sha256"]!=ids[model]:
            raise ValueError("wrong checkpoint "+model)
        if args.mark_epoch is not None:
            if stages!=["mark"] or not 1<=args.mark_epoch<50:
                raise ValueError("--mark-epoch requires an update model and only stage mark, epoch 1..49")
            ckpt=Path(spec["path"]).with_name(f"epoch-{args.mark_epoch:02d}.ckpt")
            header=torch.load(ckpt,map_location="cpu",weights_only=False)
            source=model.split("-")[0]
            if (header["epoch"]!=args.mark_epoch or header["identity"]["model"]!=source or
                    header["identity"]["source"]!=ids[source] or
                    header["identity"]!=P.read(ckpt.parent/"meta.json")["identity"]):
                raise ValueError("Unexpected intermediate checkpoint identity")
            del header
            spec.update(path=str(ckpt),sha256=P.sha(ckpt))
        rt=Runtime(spec,args.device)
        for stage in stages:
            folder=args.out/("smoke" if args.smoke else "formal")/stage/model
            if args.mark_epoch is not None:
                folder=folder.parent/(model+f"-epoch{args.mark_epoch:02d}")
            folder.mkdir(parents=True,exist_ok=True)
            identity=fingerprint(spec,plan)
            for file in ("first_batch.py","first_batch_probes.py","abc_scene_queue.py","attribution_queue.py"):
                identity["code"][file]=P.sha(Path(__file__).with_name(file))
            path=folder/"identity.json"
            if path.exists() and P.read(path)!=identity:
                raise ValueError("Incompatible run "+str(folder))
            P.write(path,identity)
            fit=fit_sources({model:spec})[model]
            if args.mark_epoch is not None:
                fit["training_epoch"]=args.mark_epoch
                fit.pop("validation",None)  # the registry validation belongs to the final
            P.write(folder/"fit-source.json",fit)
            smoke_edges=0
            with torch.inference_mode():
                for n,d,g in cells(stage,args.smoke):
                    path=folder/f"{n}-{d}-{g}.json.gz"
                    if path.exists():
                        result=saved(path) if stage=="natural" and args.smoke else None
                    else:
                        if stage=="natural":
                            old=saved(C_RAW/"raw"/model/path.name)
                            if args.smoke and not any("replay_child" in e for e in old["natural"]):
                                continue
                            result=natural_probe(rt,old)
                        elif stage=="history":
                            result=history_probe(rt,g,d)
                        elif stage=="mark":
                            result=mark_probe(rt,g,d)
                        elif stage=="fanout":
                            result=fanout_probe(rt,g,d)
                        elif stage=="standard":
                            result=measure(rt,g,n,d)
                        else:
                            template=groups[f"{g}-{d}"] if stage=="layered" else None
                            result=run_grid(rt,g,d,template)
                        publish(path,result)
                    if stage=="natural" and args.smoke and any(
                            z.get("paired_replay_agrees") for z in result["records"]):
                        break
                    if stage=="natural" and args.smoke:
                        smoke_edges+=1
                        if smoke_edges>=4:
                            print("Natural smoke: no matched replay in first four replayable graphs",flush=True)
                            break
            print(stage,model,folder,flush=True)
        del rt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",choices=ALL,nargs="+",required=True)
    p.add_argument("--stages",choices=STAGES,nargs="+",required=True)
    p.add_argument("--data-root",type=Path,default=ROOT)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--device",default="cuda")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--mark-epoch",type=int)
    run(p.parse_args())
