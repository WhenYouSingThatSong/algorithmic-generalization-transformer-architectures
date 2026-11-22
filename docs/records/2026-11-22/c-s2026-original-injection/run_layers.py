"""Layered artificial-state injections on original C weights (supplementary pilot).

Grid: {target, parent, ancestor, all} x {pending, just-solved, later} plus
clean controls and a same-state self-patch control.  No source edits; every
patched row records source round, source node, donor value, and base value.

Records store ALL node values per branch step, keyed by node index, so the
summary script can separate target readout recovery, donor-specific propagation,
and downstream repair without rerunning the model.
"""
import argparse, json, sys
from pathlib import Path
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'tools/c_hard_graphs')); sys.path.insert(0,str(ROOT/'tools/c_update')); sys.path.insert(0,str(ROOT/'tools/c_local_update'))
from design import P
from attribution_cases import exposure_case,encode
from runtime import Runtime

ROUNDS=16

def rollout_snaps(rt,c):
    nodes=c['base']['nodes']; u=c['u']; due=nodes[u]['depth']+1; later=c['D']+4
    times=sorted(set((due-1,due,later)))
    h=rt.init(encode(rt,[c['base'],c['donor']]))
    snaps={}
    for t in range(1,max(times)+1):
        h,_,_=rt.forward(h)
        if t in times: snaps[t]=(h[:1].clone(),h[1:].clone())
    return snaps,due,later

def read(rt,h,nodes):
    dec=rt.decode(h)[0]
    return {i:int(dec[n['pos'],3]) for i,n in enumerate(nodes)}

def run_branch(rt,start,nodes,patches,donor_snaps):
    h=start.clone()
    actual={}
    for p,src_t,src_i,*rest in patches:
        batch=rest[0] if rest else 'donor'
        src=(donor_snaps[src_t][1] if batch=='donor' else donor_snaps[src_t][0])
        h[0,nodes[p]['pos']]=src[0,nodes[src_i]['pos']]
        actual[str(p)]=dict(source_round=src_t,source_node=src_i,source_batch=batch,
                           donor_value=int(rt.decode(src)[0,nodes[src_i]['pos'],3]),
                           base_value=int(rt.decode(start)[0,nodes[p]['pos'],3]))
    pre=read(rt,h,nodes)
    rows=[]
    for k in range(1,ROUNDS+1):
        h,_,_=rt.forward(h)
        rows.append(dict(step=k,values=read(rt,h,nodes)))
    return actual,pre,rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='C_s2026')
    ap.add_argument('--gids',default='352-367'); ap.add_argument('--depths',default='4,8')
    ap.add_argument('--out',type=Path,default=ROOT/'tmp/analysis/c-original-injection')
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args(); out=a.out; out.mkdir(parents=True,exist_ok=True)
    lo,hi=[int(z) for z in a.gids.split('-')]; depths=[int(z) for z in a.depths.split(',')]
    plan=P.read(ROOT/'tmp/analysis/c-update-20261119/data/plan.json')
    spec=next(m for m in plan['models'] if m['id']==a.model)
    rt=Runtime(spec,a.device); records=[]; cases={}
    with rt.torch.inference_mode():
        for gid in range(lo,hi+1):
            for depth in depths:
                c=exposure_case(gid,depth); nodes=c['base']['nodes']
                u,r,anc,q=c['u'],c['r'],c['ancestor'],c['q']; sink=len(nodes)-1
                due=nodes[u]['depth']+1; later=c['D']+4
                key=f'{gid}-{depth}'
                cases[key]=dict(gid=gid,depth=depth,D=c['D'],due=due,later=later,u=u,r=r,ancestor=anc,
                    q=q,sink=sink,closure=c['closure'],delta=c['delta'],
                    nodes=[dict(i=i,slot=n['slot'],depth=n['depth'],parents=n['parents'],pos=n['pos'],
                                truth=c['gold'][i],alt=c['alt'][i]) for i,n in enumerate(nodes)])
                snaps,_due,_later=rollout_snaps(rt,c)
                # Base state diagnostics at each start round.
                base_ok={}
                for t in (due-1,due,later):
                    vals=read(rt,snaps[t][0],nodes)
                    base_ok[str(t)]=dict(
                        computed_ok=all(vals[i]==c['gold'][i] for i,n in enumerate(nodes) if n['depth']<t),
                        computed_bad=[i for i,n in enumerate(nodes) if n['depth']<t and vals[i]!=c['gold'][i]],
                        pending_empty=all(vals[i]==24 for i,n in enumerate(nodes) if n['depth']>=t),
                        fully_correct=all(vals[i]==c['gold'][i] for i in range(len(nodes))))
                # label -> (start round, patches (node, source round, source node))
                conditions=[('clean_pending',due-1,[]),('clean_just',due,[]),('clean_later',later,[])]
                conditions += [
                    ('target_just_self',due,[(u,due,u,'base')]),
                    ('parent_just_self',due,[(r,due,r,'base')]),
                    ('ancestor_just_self',due,[(anc,due,anc,'base')]),
                    ('target_pending_correct',due-1,[(u,due,u,'base')]),
                    ('parent_pending_correct',due-1,[(r,due,r,'base')]),
                    ('ancestor_pending_correct',due-1,[(anc,due,anc,'base')]),
                    ('target_pending_empty',due-1,[(u,due-1,u)]),
                    ('target_pending_wrong',due-1,[(u,due,u)]),
                    ('target_just',due,[(u,due,u)]),
                    ('parent_pending',due-1,[(r,due-1,r)]),
                    ('ancestor_pending',due-1,[(anc,due-1,anc)]),
                    ('parent_just',due,[(r,due,r)]),
                    ('ancestor_just',due,[(anc,due,anc)]),
                    ('all_pending_wrong',due-1,[(u,due,u),(r,due-1,r),(anc,due-1,anc)]),
                    ('all_just',due,[(u,due,u),(r,due,r),(anc,due,anc)]),
                    ('later_target',later,[(u,later,u)]),
                    ('later_parent',later,[(r,later,r)]),
                    ('later_ancestor',later,[(anc,later,anc)]),
                    ('later_all',later,[(u,later,u),(r,later,r),(anc,later,anc)])]
                for label,t0,patches in conditions:
                    actual,pre,rows=run_branch(rt,snaps[t0][0],nodes,patches,snaps)
                    injected=[]
                    for p,src_t,src_i,*rest in patches:
                        v=actual[str(p)]
                        kind=('empty' if v['donor_value']>=24 else
                              'gold' if v['donor_value']==c['gold'][p] else 'wrong')
                        injected.append(dict(node=p,kind=kind,donor_matches_alt=v['donor_value']==c['alt'][p],
                                             donor_matches_gold=v['donor_value']==c['gold'][p]))
                    records.append(dict(model=a.model,gid=gid,depth=depth,case=key,label=label,start=t0,
                        patches=patches,actual=actual,injected=injected,base_ok=base_ok[str(t0)],
                        pre=pre,rows=rows))
                    print(a.model,gid,depth,label,'patched',[(z['node'],z['kind']) for z in injected],flush=True)
    P.write(out/f'{a.model}-layered.json',dict(model=a.model,checkpoint=spec['sha256'],rounds=ROUNDS,cases=cases,records=records))
    print('records',len(records),'cases',len(cases),'model',spec['sha256'])

if __name__=='__main__': main()
