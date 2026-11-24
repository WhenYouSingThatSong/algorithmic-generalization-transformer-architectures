#!/usr/bin/env python3
"""Single and paired parent-child errors after solving; paired source controls."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import suite_core as U
C=U.C

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=C.STUDY)
    p.add_argument('--data-dir',type=Path,default=C.P.STUDY/'data')
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--source-ckpt',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--num_threads',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();R=C.runtime()
    import torch
    torch.set_float32_matmul_precision('medium')
    specs=U.models(a.study_dir,a.data_dir,a.source_ckpt);cases=C.read_cases(a.study_dir/'plan/val.jsonl')
    a.out_dir.mkdir(parents=True,exist_ok=False);limit=4 if a.smoke else len(cases)
    meta=dict(status='running',protocol='C_uq_pair_eval_v1',models=specs,cohort='smoke' if a.smoke else 'main',
        data_sha256=C.P.sha(a.data_dir/'val.pt'),plan_sha256=C.P.sha(a.study_dir/'plan/val.jsonl'),
        candidate_graphs=limit,horizon=16,branches=U.BRANCHES,
        code_hashes={n:C.P.sha(U.HERE/n) for n in ('run_pair.py','suite_core.py')})
    C.P.write_json(a.out_dir/'meta.json',meta)
    with (a.out_dir/'records.jsonl').open('x',encoding='utf-8') as output, (a.out_dir/'coverage.jsonl').open('x',encoding='utf-8') as coverage,torch.no_grad():
        for spec in specs:
            ctx=C.context(R,spec['path'],a.data_dir/'val.pt',a.device,1);lit=ctx['litmodel'];lit.eval();count=0
            for gid,raw in enumerate(ctx['data_module'].val_dataloader()):
                if gid>=limit:break
                count+=1
                x,y,info=R.split_batch(lit.reduce_batch(raw.to(ctx['device'])),ctx['model_config'])
                base=cases[gid];plan=U.pair_plan(base,y[0].tolist(),info[0].tolist())
                if plan['status']!='selected':
                    coverage.write(json.dumps(dict(arm=spec['arm'],graph_id=gid,**plan))+'\n');continue
                u,q=plan['u'],plan['q'];pos=plan['positions'];gold=y[0,pos,:4].tolist();ui=pos.index(u);qi=pos.index(q)
                # Donors computed in same source batch for every model and timing.
                inputs=[x];names=['original']
                for prefix,c in [('u',plan['u_case']),('q',plan['q_case'])]:
                    for kind,name in [('same','same'),('wrong',f"shift_{c['delta']}")]:
                        inputs.append(C.donor_input(x,c,name));names.append(prefix+'_'+kind)
                src=torch.cat(inputs);h=lit.model.token_embedder(src)
                times={'hold':base['state_round'],'late':base['D']+16};snapshots={}
                for t in range(1,max(times.values())+1):
                    h,_=lit.model(h,True)
                    for timing,tt in times.items():
                        if t==tt:snapshots[timing]=h.clone()
                for timing,t in times.items():
                    sources=snapshots[timing];source_decode=C.decode(lit,sources)[:,pos,:4].tolist()
                    states,norms=U.make_branches(sources,u,q,names)
                    start=C.decode(lit,states)[:,pos,:4].tolist()
                    branches={name:dict(initial=start[i],outputs=[]) for i,name in enumerate(U.BRANCHES)}
                    for _ in range(16):
                        states,ls=lit.model(states,True);pred=torch.stack([z.argmax(-1) for z in ls],-1)
                        for i,name in enumerate(U.BRANCHES):branches[name]['outputs'].append(pred[i,pos,:4].tolist())
                    if branches['sham']!=branches['unchanged']:raise ValueError('exact-writeback control diverged')
                    expected={'u_wrong':plan['expected_u'],'q_wrong':plan['expected_q'],'u_same':gold[ui][3],'q_same':gold[qi][3]}
                    for name in names[1:]:
                        ni=ui if name.startswith('u_') else qi
                        if branches[name]['initial'][ni]!=source_decode[names.index(name)][ni]:raise ValueError('row swap changed donor readout')
                    valid=all(source_decode[names.index(n)][ui if n.startswith('u_') else qi][3]==v for n,v in expected.items())
                    checkpos=set([u,q]+plan['parents'][u]+plan['parents'][q])
                    valid=valid and all(source_decode[0][pos.index(p0)][3]==gold[pos.index(p0)][3] for p0 in checkpos)
                    valid=valid and all(branches['unchanged']['outputs'][0][i][3]==gold[i][3] for i in (ui,qi))
                    strict=valid and all(source_decode[0][i]==gold[i] for i in (ui,qi))
                    for n,v in expected.items():
                        ni=ui if n.startswith('u_') else qi;target=list(gold[ni]);target[3]=v
                        strict=strict and source_decode[names.index(n)][ni]==target
                    coverage.write(json.dumps(dict(arm=spec['arm'],graph_id=gid,timing=timing,status='selected',u=u,q=q,
                        value_qualified=valid,strict_qualified=strict))+'\n')
                    rec=dict(arm=spec['arm'],graph_id=gid,timing=timing,state_round=t,D=base['D'],u=u,q=q,
                        positions=pos,truth4=gold,parents=plan['parents'],depths=plan['depths'],
                        children=base['children'],descendants=base['descendants'],delta_u=plan['delta_u'],delta_q=plan['delta_q'],
                        expected_u=plan['expected_u'],expected_q=plan['expected_q'],
                        value_qualified=valid,strict_qualified=strict,source_names=names,source_decode=source_decode,
                        source_edits={n:c['changes'] for n,c in [('u',plan['u_case']),('q',plan['q_case'])]},
                        start_graph_value_all=all(source_decode[0][i][3]==gold[i][3] for i in range(len(pos))),
                        start_graph_first4_all=source_decode[0]==gold,norms=norms,branches=branches)
                    output.write(json.dumps(rec)+'\n');output.flush();coverage.flush()
                if gid%16==0:print(json.dumps(dict(arm=spec['arm'],graph_id=gid)),flush=True)
            if count!=limit:raise ValueError('candidate graph count differs')
            del ctx,lit,h
    meta['status']='complete';C.P.write_json(a.out_dir/'meta.json',meta)
if __name__=='__main__':main()
