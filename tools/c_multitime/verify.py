"""Execution-agent checks: actual weights, GPU labels, BPTT, reload, six-step receipt."""
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
    p.add_argument('--smoke-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    plan=S.validate_plan(a.plan);meta=P.read(a.smoke_dir/'meta.json');ident=meta['identity']
    if meta['status']!='complete' or ident['cohort']!='smoke' or ident['recipe_id']!=S.VERSION:
        raise ValueError('wrong smoke run')
    if ident['code_hashes']!=S.hashes() or ident['plan_sha256']!=P.sha(a.plan):
        raise ValueError('smoke code/plan changed')
    rows=[json.loads(z) for z in (a.smoke_dir/'steps.jsonl').read_text().splitlines()]
    if len(rows)!=6 or {(r['mode'],r['enabled']) for r in rows}!={(m,b) for m in S.MODES for b in (False,True)}:
        raise ValueError('six maximum-depth clean/corrupt steps required')
    if any(r['graphs']!=32 for r in rows): raise ValueError('smoke must measure full effective batch32')
    # Preload old evaluator to avoid same-named modules in older tool directories.
    from evaluate import load_split,run_graph
    from summarize import summarize
    from functional_probe import run_probe
    from runtime import Runtime
    import torch
    from tensor_targets import Topology
    ckpath=a.smoke_dir/'smoke-final.ckpt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    if ck['identity']!=ident: raise ValueError('smoke checkpoint identity')
    model=dict(ident['source'],path=str(ckpath.resolve()),sha256=P.sha(ckpath),arm='update')
    rt=Runtime(model,a.device)
    x,y,gs=load_split(rt,plan['splits']['dev'])
    xx=x[:2].to(rt.device);yy=y[:2].to(rt.device)
    top=Topology(gs[:2]).batch([0,1],xx.shape[1],rt.device)
    with torch.no_grad():
        h=rt.init(xx)
        expected,_=rt.model(h,True);out,_,mid=rt.forward(h,middle=True)
        if not torch.equal(expected,out) or mid is None: raise ValueError('middle hook changed forward')
        current=yy[...,:4].clone()
        current[...,3]=torch.randint(0,25,current.shape[:2],device=rt.device)
        target,ref=top.local(current,yy)
        cc=current.cpu().tolist();original=yy.cpu().tolist()
        want=[P.local_step(g,c,o) for g,c,o in zip(gs[:2],cc,original)]
        if target.cpu().tolist()!=[v[0] for v in want] or ref.cpu().tolist()!=[v[1] for v in want]:
            raise ValueError('GPU synchronous labels differ')
    rt.model.train();rt.model.zero_grad(set_to_none=True)
    h1,_,_=rt.forward(rt.init(xx[:1]),middle=True);h1.retain_grad()
    _,logits,mid=rt.forward(h1,middle=True)
    (logits[3].square().mean()+mid.square().mean()).backward()
    if h1.grad is None or not torch.isfinite(h1.grad).all() or not h1.grad.abs().sum()>0:
        raise ValueError('two-loop gradient missing')
    if rt.model.output_layers[4].weight.grad is None: raise ValueError('UPDATE head gradient missing')
    rt.model.zero_grad(set_to_none=True);rt.model.eval()
    records=[]
    with torch.no_grad():
        for gid,g in enumerate(gs[:2]):
            run_graph(rt,x[gid:gid+1].to(rt.device),y[gid].to(rt.device),g,'dev',gid,
                      plan['splits']['dev']['targets'][gid],records.append,'dev')
    report=summarize(records)
    report['frozen_C_function']=run_probe(rt,plan,'dev',limit=2)
    P.write(a.smoke_dir/'reload-dev2.json',report)
    with (a.smoke_dir/'reload-dev2.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    P.write(a.output,dict(status='passed',plan_sha256=ident['plan_sha256'],code_hashes=ident['code_hashes'],
                         micro_batch=ident['micro_batch'],donor_batch=ident['donor_batch'],
                         smoke_checkpoint_sha256=P.sha(ckpath),
                         checks=['six_effective_batch32_steps','GPU_labels_equal','forward_unchanged',
                                 'two_loop_BPTT','middle_head_gradient','reload_dev2_and_frozen_C_probe'],
                         performance=[{k:r[k] for k in ('mode','enabled','train_step_seconds','peak_allocated_bytes',
                                                        'attempts','accepted','actual_value_changed')} for r in rows],
                         note='Technical checks only; source rejection and model errors are reported, not filtered away.'))


if __name__=='__main__':main()
