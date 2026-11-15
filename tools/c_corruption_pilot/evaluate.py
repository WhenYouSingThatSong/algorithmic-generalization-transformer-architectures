#!/usr/bin/env python3
"""Clean long rollouts plus paired same-value/semantic/noise interventions."""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pilot_core as C

def write_csv(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_models(study,plan):
    models=[];matches=[]
    for arm in C.ARMS:
        m=C.P.read_json(study/arm/'meta.json')
        if m['status']!='complete' or m['cohort']!='main' or m['arm']!=arm or m['plan']!=plan:
            raise ValueError('incomplete/unmatched main run')
        if m['source_sha256']!=C.P.SOURCE_SHA: raise ValueError('wrong initial weights')
        b={k:v for k,v in m['budget'].items() if k!='injected_events'}
        matches.append((m['optimizer'],m['batch_plan'],m['config_sha256'],m['seed'],m['epochs'],m['rate'],m['code_sha256'],b))
        for n,h in m['config_sha256'].items():
            if C.P.sha(C.P.ROOT/'experiments/configs'/C.P.CONFIG/n)!=h: raise ValueError('changed config')
        ck=m['checkpoint']
        if C.P.sha(ck['path'])!=ck['sha256']: raise ValueError('changed checkpoint')
        models.append(dict(arm=arm,**ck))
    if any(m!=matches[0] for m in matches): raise ValueError('unmatched recipes, schedules or budgets')
    return models

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=C.STUDY)
    p.add_argument('--data-dir',type=Path,default=C.P.STUDY/'data')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--source-ckpt',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--num_threads',type=int,default=4)
    p.add_argument('--limit-graphs',type=int,help='smoke only; not a main result')
    p.add_argument('--smoke-ckpt',type=Path,help='evaluate one smoke checkpoint; separate output only')
    a=p.parse_args()
    if (a.smoke_ckpt is not None)!=(a.limit_graphs is not None): p.error('smoke checkpoint and limit must be used together')
    if a.limit_graphs is not None and a.limit_graphs<1:p.error('positive limit')
    R=C.runtime()
    import torch,numpy as np
    torch.set_float32_matmul_precision('medium')
    plan=C.check_plan(a.study_dir/'plan',a.data_dir);cases=C.read_cases(a.study_dir/'plan/val.jsonl')
    source=a.source_ckpt or C.P.ROOT/C.P.source_spec()['weight_path']
    if C.P.sha(source)!=C.P.SOURCE_SHA:raise ValueError('wrong source')
    models=([dict(arm='smoke',path=str(a.smoke_ckpt),sha256=C.P.sha(a.smoke_ckpt))] if a.smoke_ckpt
            else [dict(arm='initial',path=str(source),sha256=C.P.SOURCE_SHA)]+load_models(a.study_dir,plan))
    a.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',protocol='C_corruption_eval_v1',models=models,plan=plan,
              cohort='smoke' if a.smoke_ckpt else 'main',horizon=C.HORIZON,
              code_sha256={n:C.P.sha(C.HERE/n) for n in ('evaluate.py','pilot_core.py')})
    C.P.write_json(a.out_dir/'meta.json',meta);clean_rows=[];coverage=[]
    with (a.out_dir/'records.jsonl').open('x',encoding='utf-8') as records,torch.no_grad():
        for spec in models:
            ctx=C.context(R,spec['path'],a.data_dir/'val.pt',a.device,1)
            lit=ctx['litmodel'];lit.eval();rawdir=a.out_dir/spec['arm'];rawdir.mkdir()
            count=0
            for gid,raw in enumerate(ctx['data_module'].val_dataloader()):
                if a.limit_graphs is not None and gid>=a.limit_graphs:break
                count+=1;c=cases[gid];d=c['D'];pos=c['positions'];gold=c['truth4']
                inputs,labels,info=R.split_batch(lit.reduce_batch(raw.to(ctx['device'])),ctx['model_config'])
                if labels[0,pos,:4].tolist()!=gold:raise ValueError('case truth mismatch')
                times={'hold':c['state_round'],'late':d+16}
                snapshots={};predictions=[];h=lit.model.token_embedder(inputs)
                for t in range(1,d+65):
                    h,logits=lit.model(h,True);pred=torch.stack([x.argmax(-1) for x in logits],-1)
                    predictions.append(pred[0,pos,:4].cpu().numpy())
                    for timing,tt in times.items():
                        if t==tt:snapshots[timing]=h.clone()
                array=np.stack(predictions);truth=np.array(gold);correct=array==truth[None,:,:]
                np.savez_compressed(rawdir/f'graph_{gid:04d}.npz',predictions=array,truth4=truth,positions=pos,D=d)
                for t,row in enumerate(correct,1):
                    clean_rows.append(dict(arm=spec['arm'],graph_id=gid,D=d,round=t,n=len(pos),
                        value_correct=int(row[:,3].sum()),first4_correct=int(row.all(-1).sum()),
                        syntax_correct=int(row[:,0].sum()),variable_correct=int(row[:,1].sum()),operation_correct=int(row[:,2].sum()),
                        value_all=bool(row[:,3].all()),first4_all=bool(row.all())))
                if c['status']!='selected':
                    coverage.append(dict(arm=spec['arm'],graph_id=gid,timing='none',selected=False,value_qualified=False,strict_qualified=False));continue
                u=c['target'];ui=pos.index(u)
                for timing,t in times.items():
                    h=snapshots[timing];donor,_=C.donors(lit,inputs,[(0,c,t)],include_same=True)
                    shift=donor[(0,f"shift_{c['delta']}")];same=donor[(0,'same')]
                    states=[];distances={}
                    for branch,mode,vector,scale in [('unchanged','hold',shift,1),('same','same',same,1),
                            ('semantic','semantic',shift,1),('noise','noise',shift,1),('noise3','noise',shift,3)]:
                        state,radius,norm=C.intervene(h,0,u,vector,mode,C.stable_seed(C.SEED,'evalnoise',gid,timing),scale)
                        states.append(state);distances[branch]=dict(reference_norm=radius,actual_norm=norm)
                    names=['unchanged','same','semantic','noise','noise3'];hs=torch.cat(states)
                    start=C.decode(lit,hs)[:,pos,:4].tolist()
                    branches={name:dict(initial=start[i],outputs=[]) for i,name in enumerate(names)}
                    for _ in range(C.HORIZON):
                        hs,ls=lit.model(hs,True);pr=torch.stack([x.argmax(-1) for x in ls],-1)
                        for i,name in enumerate(names):branches[name]['outputs'].append(pr[i,pos,:4].tolist())
                    orig=branches['unchanged']['initial'];dp=branches['semantic']['initial'][ui];sp=branches['same']['initial'][ui]
                    parent_positions={int(p) for q in c['children'] for p in c['parents'][str(q)]}
                    parent_ok=all(orig[pos.index(p)][3]==gold[pos.index(p)][3] for p in parent_positions)
                    q_ok=all(branches['unchanged']['outputs'][0][pos.index(q)][3]==gold[pos.index(q)][3] for q in c['children'])
                    value_ok=(orig[ui][3]==gold[ui][3] and dp[3]==c['shifted_value'] and sp[3]==gold[ui][3] and parent_ok and q_ok)
                    shifted=list(gold[ui]);shifted[3]=c['shifted_value']
                    strict=value_ok and orig[ui]==gold[ui] and dp==shifted and sp==gold[ui]
                    coverage.append(dict(arm=spec['arm'],graph_id=gid,timing=timing,selected=True,value_qualified=value_ok,strict_qualified=strict))
                    rec=dict(arm=spec['arm'],graph_id=gid,timing=timing,state_round=t,D=d,target=u,
                        positions=pos,truth4=gold,children=c['children'],descendants=c['descendants'],
                        delta=c['delta'],expected_q=c['expected_q'],value_qualified=value_ok,strict_qualified=strict,
                        distances=distances,branches=branches)
                    records.write(json.dumps(rec)+'\n');records.flush()
                if gid%16==0:print(json.dumps(dict(arm=spec['arm'],graph_id=gid)),flush=True)
            if count!=(a.limit_graphs or len(cases)):raise ValueError('missing validation graphs')
            del ctx,lit,h
    write_csv(a.out_dir/'clean_curves.csv',clean_rows);write_csv(a.out_dir/'coverage.csv',coverage)
    meta['status']='complete';C.P.write_json(a.out_dir/'meta.json',meta)
if __name__=='__main__':main()
