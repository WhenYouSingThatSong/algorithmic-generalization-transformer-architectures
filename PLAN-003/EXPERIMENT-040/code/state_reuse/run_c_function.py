#!/usr/bin/env python3
"""C: can a recovered parent support a historical unfinished child state?

Fixed graphs/times/lags; never select on the new intervention outcomes.
Whole vectors only. The historical child is a diagnostic reset, not an eraser
of all stored information. Clean and historical-solved controls are mandatory.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import core as P
sys.path.insert(0, str(HERE.parent/'priority_pilots'))
from io_utils import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--source-ckpt', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--candidate-count', type=int, default=32)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--arms', nargs='+', choices=tuple(P.C_NAMES), default=list(P.C_NAMES))
    p.add_argument('--snapshot-graphs', type=int, default=2, help='fixed prefix saved for server replay inspection')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if a.candidate_count < 1 or a.horizon < 2 or a.snapshot_graphs < 0 or len(set(a.arms)) != len(a.arms):
        p.error('invalid count/horizon/arms')
    sys.path.insert(0, str(HERE.parent/'c_evaluation_suite'))
    import suite_core as U
    C = U.C; R = C.runtime()
    import torch
    torch.set_num_threads(a.num_threads)
    torch.set_float32_matmul_precision('medium')
    study = a.study_dir or C.STUDY; data = a.data_dir or C.P.STUDY/'data'
    models = [s for s in U.models(study,data,a.source_ckpt) if s['arm'] in a.arms]
    if {s['arm'] for s in models} != set(a.arms): raise ValueError('missing requested model')
    cases = C.read_cases(study/'plan/val.jsonl')
    if len(cases) < a.candidate_count: raise ValueError('fixed candidates unavailable')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    (a.out_dir/'states').mkdir()
    deps = [Path(__file__),HERE/'core.py',HERE.parent/'c_evaluation_suite/suite_core.py',
            HERE.parent/'priority_pilots/io_utils.py',ROOT/'experiments/tokenizers.py']
    meta = dict(status='running',protocol='C_function_probe_v1',models=models,display_names=P.C_NAMES,
        candidate_ids=list(range(a.candidate_count)),horizon=a.horizon,grid=P.c_probe_grid(),branches=P.C_BRANCHES,
        data_sha256=sha(data/'val.pt'),plan_sha256=sha(study/'plan/val.jsonl'),
        source_data=str(data/'val.pt'),source_plan=str(study/'plan/val.jsonl'),
        code_hashes={str(f.relative_to(ROOT)):sha(f) for f in deps},dependencies=C.dependency_hashes(),
        num_threads=a.num_threads,matmul_precision='medium',snapshot_graphs=a.snapshot_graphs,
        note='Same seed2026 model variants; development data. Fixed lags, no successful-recovery selection; no decoded feedback.')
    write_json(a.out_dir/'meta.json',meta)
    written=0;sample_forwards=0;token_forwards=0
    with (a.out_dir/'records.jsonl').open('x',encoding='utf-8') as dst, (a.out_dir/'coverage.jsonl').open('x',encoding='utf-8') as cov, torch.no_grad():
        for spec in models:
            if sha(spec['path']) != spec['sha256']: raise ValueError('weight identity changed')
            ctx=C.context(R,spec['path'],data/'val.pt',a.device,1)
            lit=ctx['litmodel']; lit.eval(); device=ctx['device']; tok=ctx['tokenizer']
            if not lit.nointerm: raise ValueError('continuous feedback required')
            count=0
            for gid,raw in enumerate(ctx['data_module'].val_dataloader()):
                if gid>=a.candidate_count: break
                count+=1
                x,y,info=R.split_batch(lit.reduce_batch(raw.to(device)),ctx['model_config'])
                base=cases[gid]
                if base['graph_id']!=gid or y[0,base['positions'],:4].tolist()!=base['truth4']:
                    raise ValueError('plan/data mismatch')
                plan=U.pair_plan(base,y[0].tolist(),info[0].tolist())
                if plan['status']!='selected':
                    cov.write(json.dumps(dict(arm=spec['arm'],graph_id=gid,**plan))+'\n'); continue
                pos=plan['positions'];u,q=plan['u'],plan['q'];ui,qi=pos.index(u),pos.index(q)
                gold=y[0,pos,:4].tolist(); parents=plan['parents']
                # Early = just before q is theoretically first computable. This
                # keeps BOTH historical donors at or before every probe round.
                times=dict(early=plan['depths'][q],hold=base['state_round'],late=base['D']+16)
                pre_round=plan['depths'][q]; solved_round=pre_round+1
                if pre_round<1: raise ValueError('child must be internal')
                # Same graph, same weight and symbolic identities. Different source
                # problems only supply the u_same/u_wrong initial vectors.
                xs=torch.cat([x,C.donor_input(x,plan['u_case'],'same'),
                              C.donor_input(x,plan['u_case'],f"shift_{plan['u_case']['delta']}")])
                h=lit.model.token_embedder(xs)
                needed={pre_round,solved_round} | {t+j for t in times.values() for j in range(5)}
                sources={}; clean_readouts=[]
                for t in range(1,max(needed)+1):
                    h,_=lit.model(h,True)
                    sample_forwards+=3;token_forwards+=3*x.shape[1]
                    clean_readouts.append(C.decode(lit,h[0:1])[0,pos,:4].cpu().tolist())
                    if t in needed: sources[t]=h.clone()
                pre=sources[pre_round][0,q].clone(); solved=sources[solved_round][0,q].clone()
                pre_read=C.decode(lit,sources[pre_round][0:1])[0,q,:4].cpu().tolist()
                solved_read=C.decode(lit,sources[solved_round][0:1])[0,q,:4].cpu().tolist()
                cov.write(json.dumps(dict(arm=spec['arm'],graph_id=gid,status='selected',u=u,q=q,
                    times=times,donor_pre_round=pre_round,donor_solved_round=solved_round,
                    donor_pre=pre_read,donor_solved=solved_read))+'\n');cov.flush()
                for timing,t0 in times.items():
                    for mode in ('natural','same','semantic','noise2'):
                        start=sources[t0][0:1].clone();seed=C.stable_seed('functional-probe',gid,timing)
                        radius=actual=0.
                        if mode!='natural':
                            donor=sources[t0][1 if mode=='same' else 2,u]
                            start,radius,actual=C.intervene(start,0,u,donor,'noise' if mode=='noise2' else mode,seed,scale=2.)
                        initial=C.decode(lit,start)[0,pos,:4].cpu().tolist()
                        history=[initial]; observed={}; current=start
                        for lag in range(1,5):
                            current,_=lit.model(current,True)
                            sample_forwards+=1;token_forwards+=x.shape[1]
                            history.append(C.decode(lit,current)[0,pos,:4].cpu().tolist())
                            if lag in (1,2,4): observed[lag]=current.clone()
                        for lag in (1,2,4):
                            absolute=t0+lag; fault=observed[lag]; clean=sources[absolute][0:1]
                            if max(pre_round,solved_round)>absolute:
                                raise ValueError('future historical donor is not allowed')
                            fault_read=history[lag]; clean_read=C.decode(lit,clean)[0,pos,:4].cpu().tolist()
                            context=P.c_context(fault_read,clean_read,pre_read,solved_read,gold,pos,u,q,parents,tok.value_tok2idx['EMPTY'])
                            states=[];norms={}
                            for name in P.C_BRANCHES:
                                if name=='sham':
                                    s=fault.clone();s[:,q]=s[:,q].clone()
                                else:
                                    receiver,child=name.rsplit('_',1)
                                    s=(clean if receiver=='clean' else fault).clone()
                                    if receiver=='restore_u': s[:,u]=clean[:,u]
                                    if child=='pre': s[:,q]=pre
                                    if child=='solved': s[:,q]=solved
                                norms[name]=dict(u_l2=float((s[:,u]-fault[:,u]).norm()),q_l2=float((s[:,q]-fault[:,q]).norm()))
                                states.append(s)
                            batch=torch.cat(states)
                            branches={n:dict(inputs=[],outputs=[]) for n in P.C_BRANCHES}
                            for _ in range(a.horizon):
                                ip=C.decode(lit,batch)[:,pos,:4].cpu().tolist()
                                batch,ls=lit.model(batch,True)
                                sample_forwards+=len(P.C_BRANCHES);token_forwards+=len(P.C_BRANCHES)*x.shape[1]
                                op=torch.stack([z.argmax(-1) for z in ls],-1)[:,pos,:4].cpu().tolist()
                                for j,n in enumerate(P.C_BRANCHES):
                                    branches[n]['inputs'].append(ip[j]);branches[n]['outputs'].append(op[j])
                            if branches['sham']!=branches['observed_keep']: raise ValueError('exact-writeback differs')
                            key=f"{spec['arm']}:{gid}:{timing}:{mode}:{lag}"
                            snapshot=None
                            if gid<a.snapshot_graphs:
                                path=a.out_dir/'states'/f"{key.replace(':','_')}.pt"
                                torch.save(dict(observed=fault.cpu(),clean=clean.cpu(),pre=pre.cpu(),solved=solved.cpu(),
                                    u=u,q=q,positions=pos,case_id=key,absolute_round=absolute),path)
                                snapshot=dict(path=str(path.relative_to(a.out_dir)),sha256=sha(path))
                            observed_history=history[:lag+1]
                            clean_history=clean_readouts[:absolute]
                            rec=dict(case_id=key,arm=spec['arm'],display_name=P.C_NAMES[spec['arm']],graph_id=gid,
                                timing=timing,mode=mode,probe_lag=lag,state_round=t0,absolute_round=absolute,
                                horizon=a.horizon,D=base['D'],u=u,q=q,positions=pos,truth4=gold,parents=parents,
                                children=base['children'],descendants=base['descendants'],depths=plan['depths'],
                                context=context,donor_pre_round=pre_round,donor_solved_round=solved_round,
                                donor_pre_readout=pre_read,donor_solved_readout=solved_read,
                                perturbation=dict(seed=seed,semantic_radius=radius,actual_l2=actual,
                                    expected_u=plan['expected_u'],initial_u=initial[ui][3],
                                    changes=plan['u_case']['changes']),
                                u_had_wrong_readout=any(z[ui][3]!=gold[ui][3] for z in observed_history),
                                u_clean_history_had_wrong=any(z[ui][3]!=gold[ui][3] for z in clean_history),
                                q_had_wrong_readout=any(z[qi][3]!=gold[qi][3] for z in observed_history),
                                before_probe=observed_history,clean_before_probe=clean_history,
                                norms=norms,branches=branches,snapshot=snapshot)
                            dst.write(json.dumps(rec)+'\n');dst.flush();written+=1
                print(json.dumps(dict(arm=spec['arm'],graph_id=gid,written=written)),flush=True)
            if count!=a.candidate_count: raise ValueError('incomplete fixed candidate scan')
            del ctx,lit
    meta.update(status='complete',records=written,sample_forwards=sample_forwards,token_forwards=token_forwards,
        records_sha256=sha(a.out_dir/'records.jsonl'),
        coverage_sha256=sha(a.out_dir/'coverage.jsonl'))
    write_json(a.out_dir/'meta.json',meta)


if __name__=='__main__':main()
