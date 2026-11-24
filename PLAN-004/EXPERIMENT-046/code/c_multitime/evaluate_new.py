"""Same frozen evaluation as existing C models, with unambiguous new recipe identity."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import spec as S
P=S.P


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();plan=S.validate_plan(a.plan)
    from evaluate import load_split,run_graph
    from summarize import summarize_file
    from functional_probe import run_probe
    from runtime import Runtime
    import torch
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    identity=ck['identity']
    if identity['recipe_id']!=S.VERSION or identity['cohort']!='main' or identity['plan_sha256']!=P.sha(a.plan):
        raise ValueError('not a matching main multitime checkpoint')
    model=dict(identity['source'],path=str(a.checkpoint.resolve()),sha256=P.sha(a.checkpoint),arm='update')
    rt=Runtime(model,a.device)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',recipe_id=S.VERSION,display_name=S.DISPLAY,model=model,
              training_identity=identity,plan_sha256=P.sha(a.plan),code_hashes=S.hashes(),
              evaluation_core_hashes=P.source_hashes(),scope='test')
    P.write(a.out_dir/'meta.json',meta)
    with torch.no_grad(),(a.out_dir/'records.jsonl').open('w',encoding='utf-8') as f:
        def emit(row):f.write(json.dumps(row)+'\n');f.flush()
        for split in P.STRATA:
            ss=plan['splits'][split];x,y,gs=load_split(rt,ss)
            for gid,g in enumerate(gs):
                run_graph(rt,x[gid:gid+1].to(rt.device),y[gid].to(rt.device),g,split,gid,ss['targets'][gid],emit)
                print(f'{S.VERSION} {split} {gid+1}/{len(gs)}',flush=True)
    summarize_file(a.out_dir/'records.jsonl',a.out_dir/'summary.json')
    probe=[run_probe(rt,plan,split,limit=32) for split in ('N32_shallow','N32_deep','N128_shallow','N128_deep')]
    P.write(a.out_dir/'functional-probe.json',probe)
    meta.update(status='complete',records_sha256=P.sha(a.out_dir/'records.jsonl'),sample_forwards=rt.calls)
    P.write(a.out_dir/'meta.json',meta)


if __name__=='__main__':main()
