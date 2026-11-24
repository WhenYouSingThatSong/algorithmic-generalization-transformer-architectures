"""Serial continuation after the first mechanism queue; two existing C weights."""
import argparse
from pathlib import Path

import torch

from hard_eval import MODELS, P, PLAN, RECEIPT, ROOT, write_csv
from research_queue import checkpoint, grouped, source_hashes
from runtime import Runtime
from version_readers import reader_probe, version_probe

OUT = ROOT/"tmp/analysis/c-reader-continuation-v1"
SETTINGS = {"versions": ("tight","delayed"), "readers": (4,8)}
GROUPS = {"versions": ("arm","condition","reader","extra"),
          "readers": ("timing","status","old_at_append","step")}


def run(stage, smoke, args, receipt, plan):
    ids = [256] if smoke else list(range(192,224))
    folder = args.out/("smoke" if smoke else "formal")/stage
    records = []
    sources = source_hashes()
    for path in (Path(__file__),Path(__file__).with_name("version_readers.py")):
        sources[path.name]=P.sha(path)
    for model in MODELS:
        out = folder/model
        out.mkdir(parents=True,exist_ok=True)
        weights = next(r for r in receipt["runs"] if r["model"]==model)
        identity = dict(stage=stage,model=model,ids=ids,settings=list(SETTINGS[stage]),sources=sources,
                        plan=receipt["plan_sha256"],checkpoint=weights["final_ckpt"]["sha256"])
        meta = out/"identity.json"
        if meta.exists() and P.read(meta)!=identity:
            raise ValueError(f"different study at {out}")
        P.write(meta,identity)
        rt=None
        for setting in SETTINGS[stage]:
            for gid in ids:
                dest=out/f"{setting}-{gid}.json"
                if dest.exists():
                    case=P.read(dest)
                    if (case["gid"],case["setting"])!=(gid,setting):
                        raise ValueError(f"case mismatch at {dest}")
                else:
                    if rt is None:
                        spec=next(m for m in plan["models"] if m["id"]==model)
                        rt=Runtime(dict(spec,arm="update",path=str(checkpoint(args,weights)),
                                        sha256=identity["checkpoint"]),args.device)
                    with torch.inference_mode():
                        rows,events=(version_probe(rt,gid,setting) if stage=="versions"
                                     else reader_probe(rt,gid,setting))
                    case=dict(setting=setting,gid=gid,rows=rows,events=events)
                    P.write(dest,case)
                records.extend(dict(model=model,setting=setting,**r) for r in case["rows"])
        del rt
        P.write(out/"complete.json",identity)
        print(stage,model,len(ids)*len(SETTINGS[stage]),"cases",flush=True)
    summary=grouped(records,("model","setting",*GROUPS[stage]))
    for r in summary:
        if stage=="versions":
            n=r["stale_testable_with_unique_correct"]
            r["tracked_old_rate_among_testable_unique_correct"]=r["tracks_older_with_unique_correct"]/n if n else None
        else:
            n=r["paired_clean"]
            if r["both_correct"]+r["old_only"]+r["new_only"]+r["both_wrong"]!=n:
                raise ValueError("inconsistent paired denominator")
            r["old_correct_rate"]=(r["old_only"]+r["both_correct"])/n if n else None
            r["new_correct_rate"]=(r["new_only"]+r["both_correct"])/n if n else None
    table=folder/"summary.csv"
    temp=table.with_suffix(".tmp")
    write_csv(temp,summary)
    temp.replace(table)
    P.write(folder/"delivery.json",dict(stage=stage,cases_per_model=len(ids)*len(SETTINGS[stage]),
            models=list(MODELS),smoke=smoke,summary_sha256=P.sha(table)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--train-root",type=Path)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--device",default="cuda")
    p.add_argument("--smoke-only",action="store_true")
    args=p.parse_args()
    receipt=P.read(RECEIPT)
    if P.sha(args.plan)!=receipt["plan_sha256"]:
        raise ValueError("training plan mismatch")
    plan=P.read(args.plan)
    for stage in SETTINGS:
        for smoke in ([True] if args.smoke_only else [True,False]):
            run(stage,smoke,args,receipt,plan)


if __name__=="__main__":
    main()
