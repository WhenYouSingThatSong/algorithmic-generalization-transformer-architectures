"""Fixed evaluation: keep failures, separate readout from downstream behavior."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P
from summarize import summarize_file

def observation(rt,h,g,y,raw=None):
    pred=(raw if raw is not None else rt.decode(h))[0].tolist();truth=y.tolist()
    return [[pred[p][3],pred[p][:4]==truth[p][:4],
             *[pred[p][f]==truth[p][f] for f in range(3)]] for p in g['nodes']]

def step(rt,h,g,y,raw=None):
    """Record actual inputs before forward; labels use only those inputs."""
    import torch
    decoded=raw if raw is not None else rt.decode(h)
    before=decoded[0,:,:4].tolist()
    out,logits,mid=rt.forward(h,middle=rt.update)
    after=torch.stack([z.argmax(-1) for z in logits],-1)
    row=dict(input=[before[p] for p in g['nodes']],
             output=observation(rt,out,g,y,after))
    if rt.update:
        target,ref=P.local_step(g,before,y.tolist());plan=mid.argmax(-1)[0].tolist()
        decoded_after=after[0].tolist()
        row['update']=[dict(reference=bool(ref[p]),plan=bool(plan[p]),
                           changed=decoded_after[p][3]!=before[p][3],
                           value_correct=decoded_after[p][3]==target[p][3],
                           cleared=decoded_after[p][4]==0) for p in g['nodes']]
    return out,row

def baseline(rt,x,y,g,rounds,keep):
    h=rt.init(x);snapshots={};rows=[]
    for t in range(rounds):
        if t in keep:snapshots[t]=h.detach().clone()
        h,row=step(rt,h,g,y,raw=x if t==0 else None);rows.append(row)
    return rows,snapshots

def branch(rt,start,g,y,positions,source,horizon=16,hold=1):
    h=start.clone();rows=[]
    for t in range(hold+horizon if hold>1 else horizon):
        if t<hold and source is not None:
            h=h.clone()
            for p in positions:h[0,p]=source[p]
        h,row=step(rt,h,g,y);rows.append(row)
    return rows

def header(split,gid,g,y):
    truth=y.tolist()
    return dict(split=split,gid=gid,nodes=g['nodes'],depths=[g['depths'][p] for p in g['nodes']],
                parents=[g['parents'][p] for p in g['nodes']],truth=[truth[p][:4] for p in g['nodes']],D=g['D'])

def source(rt,x,y,g,pos,edits,start):
    import torch
    if edits is None:return None,None,False
    xx=x.clone();yy=y.clone()
    for p,delta in edits.items():xx[0,p,3]=(xx[0,p,3]+delta)%23;yy[p,3]=xx[0,p,3]
    new_truth=P.truth(g,yy.tolist())
    for p,v in new_truth.items():yy[p,3]=v
    h=rt.init(xx)
    for _ in range(start):h,_,_=rt.forward(h)
    decoded=rt.decode(h)[0]
    good=all(bool((decoded[p,:4]==yy[p,:4]).all()) for p in pos)
    return {p:h[0,p].clone() for p in pos},{str(p):decoded[p,:4].tolist() for p in pos},good

def paired(rt,x,y,g,meta,target,start,base,clean,emit,conditions=None,hold=1,noise=None):
    """Same original graph and absolute source round. No replacement on failed source check."""
    u,q,delta=target['u'],target['q'],target['delta'];torch=rt.torch
    labels=y.tolist();conditions=conditions or ('u_wrong','q_wrong','both_wrong','same_value','exact_write')
    # Match same/different source edits when two nonzero ancestor coefficients exist.
    sys.path.insert(0,str(P.ROOT/'tools/counterfactual_control'))
    from design import matched_changes
    matched=matched_changes(g['coefs'][u],(delta,))
    if matched:
        edits={name:{g['constants'][p]:v for p,v in changes.items()} for name,changes in matched.items()}
        wrong_edits=edits[f'shift_{delta}'];same_edit=edits['same'];same_kind='different-source-matched'
    else:wrong_edits=P.donor_edits(g,u,delta);same_edit=P.same_edits(g,u);same_kind='original-source-fallback'
    input_read=rt.decode(base)[0]
    q_correct=bool((input_read[q,:4]==y[q,:4]).all())
    parents_correct=all(bool((input_read[p,:4]==y[p,:4]).all()) for p in {u,*g['parents'][q]})
    qualification=parents_correct and bool((input_read[q,:3]==y[q,:3]).all()) and (q_correct or int(input_read[q,3])==24)
    common=dict(**meta,start=start,u=u,q=q,delta=delta,input_qualified=qualification,
                q_input_correct=q_correct,parents_input_correct=parents_correct,
                graph_input_first4_correct=all(bool((input_read[p,:4]==y[p,:4]).all()) for p in g['nodes']),
                initial=observation(rt,base,g,y),hold=hold,clean=clean,downstream=sorted(P.descendants(g,u)))
    for name in conditions:
        label='same_value_original_source' if name=='same_value' and not matched else name
        ps=[u] if name in ('u_wrong','same_value','exact_write','noise') else [q] if name=='q_wrong' else [u,q]
        edit=P.donor_edits(g,q,delta) if name=='q_wrong' else same_edit if name=='same_value' else wrong_edits
        source_read=None;ok=True;src=None
        if name=='exact_write':src={p:base[0,p].clone() for p in ps}
        elif name=='noise':
            if not rt.continuous:
                emit(dict(**common,condition=label,status='not_applicable_discrete_noise',noise=noise));continue
            gen=torch.Generator(device='cpu');gen.manual_seed(P.seed('noise',meta['split'],meta['gid'],u,noise['direction']))
            z=torch.randn(base[0,u].shape,generator=gen).to(base.device)
            z=z/z.norm().clamp_min(1e-12)*base[0,u].norm()*noise['scale']
            src={u:base[0,u]+z}
        else:
            src,source_read,ok=source(rt,x,y,g,ps,edit,start)
        if ok and name=='both_wrong' and source_read[str(q)][3]==int(y[q,3]):
            emit(dict(**common,condition=label,status='counterfactual_q_unchanged',edits=edit,source_read=source_read));continue
        if not ok:
            emit(dict(**common,condition=label,status='source_rejected',edits=edit,source_read=source_read));continue
        rows=branch(rt,base,g,y,ps,src,16,hold)
        distance={str(p):float((src[p]-base[0,p]).float().norm()/base[0,p].float().norm().clamp_min(1e-12)) for p in ps} if rt.continuous else None
        emit(dict(**common,condition=label,status='ok',positions=ps,edits=edit if name not in ('noise','exact_write') else None,
                  source_read=source_read,same_kind=same_kind if name=='same_value' else None,
                  noise=noise,relative_vector_change=distance,rows=rows))

def append_compute(rt,x,y,g,h,u):
    """Append a NEW equation in unused variable/PAD space; keep all old hidden rows."""
    torch=rt.torch;tok=rt.tok;pad=tok.syntax_tok2idx[tok.PAD]
    used=set(x[0,:,1].tolist());free=[i for i in range(128) if i not in used]
    if not free:return None,'no_unused_variable'
    end=int((x[0,:,0]!=pad).sum());leaf=min(g['constants'])
    name=lambda p:tok.variable_idx2tok[int(y[p,1])]
    tokens=tok.encode_string(f'{tok.EQ_SEP} {name(u)} ADD {name(leaf)} {tok.EQUAL} x_{free[0]}')
    new=torch.tensor(tokens,device=x.device,dtype=x.dtype)
    # Official generator puts separators BETWEEN equations, not after the last.
    if end<1 or int(x[0,end-1,0])!=tok.syntax_tok2idx['VARIABLE']:raise ValueError('last result missing')
    length=max(x.shape[1],end+len(new))
    if length>1024:return None,'sequence_limit'
    padrow=torch.tensor(tok.encode_token(tok.PAD),device=x.device)
    xx=padrow.repeat(1,length,1);xx[:,:x.shape[1]]=x;xx[0,end:end+len(new)]=new
    hh=rt.init(xx);hh[:,:h.shape[1]]=h
    fresh=rt.init(xx[:,end:end+len(new)]);hh[:,end:end+len(new)]=fresh
    pos=end+5;yy=xx[0].clone()
    # y is the full padded label tensor; only copy the original prefix, otherwise
    # the appended equation rows would be overwritten by PAD labels.
    yy[:end]=y[:end]
    yy[pos,3]=(y[u,3]+y[leaf,3])%23
    gg=dict(g);gg['nodes']=g['nodes']+[pos];gg['parents']=dict(g['parents'])
    gg['parents'][pos]=[u,leaf];gg['depths']=dict(g['depths']);gg['depths'][pos]=g['depths'][u]+1
    gg['D']=max(g['D'],gg['depths'][pos]+1)
    return (hh,yy,gg,pos),None

def reuse(rt,x,y,g,meta,target,start,base,emit):
    """u recovered readout need not imply a new consumer can use it correctly."""
    u=target['u'];edits=P.donor_edits(g,u,target['delta'])
    src,read,ok=source(rt,x,y,g,[u],edits,start)
    if not ok:
        emit(dict(**meta,module='reuse',status='source_rejected',source_read=read));return
    for lag in (0,1,2,4):
        for name in ('clean','perturbed'):
            h=base.clone()
            if name=='perturbed':h[0,u]=src[u]
            for _ in range(lag):h,_,_=rt.forward(h)
            decoded=rt.decode(h);u_correct=bool((decoded[0,u,:4]==y[u,:4]).all())
            appended,reason=append_compute(rt,x,y,g,h,u)
            if reason:
                emit(dict(**meta,module='reuse',condition=name,lag=lag,status=reason));continue
            hh,yy,gg,p=appended;rows=branch(rt,hh,gg,yy,[],None)
            emit(dict(**header(meta['split'],meta['gid'],gg,yy),module='reuse',condition=name,
                      status='ok',lag=lag,u=u,new_node=p,start=start,u_first4_correct=u_correct,
                      u_readout=decoded[0,u,:4].tolist(),rows=rows))

def run_graph(rt,x,y,g,split,gid,target,emit,mode='full'):
    dev=mode=='dev';extra=16 if dev else 64
    early=g['depths'][target['u']]+1 if target else 1
    starts={'early':early,'complete_budget':g['D']+4,'late_budget':g['D']+16}
    # Budget labels do NOT assert the graph actually completed.
    count=g['D']+extra
    if dev and gid<64:count=max(count,starts['complete_budget']+16)
    rows,snaps=baseline(rt,x,y,g,count,set(starts.values()))
    meta=header(split,gid,g,y)
    emit(dict(**meta,module='clean',status='ok',rows=rows[:g['D']+extra]))
    if target is None:
        emit(dict(**meta,module='pair',status='no_structural_target'));return
    # 128 per N: first 64 of each depth band; stress 64 per N: first32 each band.
    if (dev and gid>=64) or (not dev and (split.startswith('N64') or gid>=64)):return
    for timing,t in starts.items():
        if dev and timing=='late_budget':continue
        m=dict(**meta,module='pair',timing=timing)
        paired(rt,x,y,g,m,target,t,snaps[t],rows[t:t+16],emit,
               conditions=('u_wrong','same_value','exact_write') if dev else None)
    if dev:return
    t=starts['complete_budget']
    if gid<32:
        for hold in (2,4):
            paired(rt,x,y,g,dict(**meta,module='sustained',timing='complete_budget'),target,t,snaps[t],
                   rows[t:t+hold+16],emit,conditions=('u_wrong',),hold=hold)
        for scale in (1,2,3,4,6):
            for direction in (0,1):
                paired(rt,x,y,g,dict(**meta,module='noise',timing='complete_budget'),target,t,snaps[t],
                       rows[t:t+16],emit,conditions=('noise',),noise=dict(scale=scale,direction=direction))
    if gid<8 and split.startswith('N32'):
        reuse(rt,x,y,g,meta,target,t,snaps[t],emit)

def load_split(rt,spec):
    from runtime import load_data,dataset_graphs
    if P.sha(spec['path'])!=spec['sha256']:raise ValueError('eval split hash')
    x,y,info=load_data(spec['path'],rt.tok)
    return x,y,dataset_graphs(y,info)

def small_check(rt,plan,epoch,path):
    from summarize import summarize
    training=rt.model.training;rt.model.eval();records=[]
    # Constructing frozen reference models must not change subsequent training dropout RNG.
    cpu_rng=rt.torch.get_rng_state()
    cuda_rng=rt.torch.cuda.get_rng_state_all() if rt.torch.cuda.is_available() else None
    try:
        with rt.torch.no_grad():
            spec=plan['splits']['dev'];x,y,gs=load_split(rt,spec)
            for gid,g in enumerate(gs):
                run_graph(rt,x[gid:gid+1].to(rt.device),y[gid].to(rt.device),g,'dev',gid,spec['targets'][gid],records.append,'dev')
        result=summarize(records);result.update(epoch=epoch,scope='dev_only')
        from functional_probe import run_probe
        result['frozen_C_function']=run_probe(rt,plan,'dev',limit=64)
        result['selection_score']=sum(z['mean_first4'] for z in result['clean'] if z['horizon']=='D')
        P.write(path,result)
        with path.with_suffix('.jsonl').open('w',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row)+'\n')
        return result
    finally:
        rt.model.train(training);rt.torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:rt.torch.cuda.set_rng_state_all(cuda_rng)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--arm',choices=('correction','update'))
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true');p.add_argument('--num_threads',type=int,default=4)
    a=p.parse_args();plan=P.read(a.plan)
    if plan['status']!='complete' or plan['protocol']!=P.VERSION:raise ValueError('plan')
    if bool(a.checkpoint)!=bool(a.arm):p.error('checkpoint and arm must be supplied together')
    spec=dict(next(z for z in plan['models'] if z['id']==a.model))
    if a.checkpoint:spec.update(path=str(a.checkpoint.resolve()),sha256=P.sha(a.checkpoint),arm=a.arm)
    from runtime import Runtime
    rt=Runtime(spec,a.device)
    if a.checkpoint:
        ck=rt.torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if ck['identity']['model']!=a.model or ck['identity']['arm']!=a.arm or ck['identity']['plan_sha256']!=P.sha(a.plan):raise ValueError('trained identity mismatch')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',protocol=P.VERSION,model=spec,plan_sha256=P.sha(a.plan),
              code_hashes=P.source_hashes(),scope='smoke' if a.smoke else 'test')
    P.write(a.out_dir/'meta.json',meta)
    with rt.torch.no_grad(),(a.out_dir/'records.jsonl').open('w',encoding='utf-8') as f:
        def emit(row):f.write(json.dumps(row)+'\n');f.flush()
        for split in (('N32_shallow',) if a.smoke else P.STRATA):
            ss=plan['splits'][split];x,y,gs=load_split(rt,ss)
            for gid,g in enumerate(gs[:2] if a.smoke else gs):
                run_graph(rt,x[gid:gid+1].to(rt.device),y[gid].to(rt.device),g,split,gid,ss['targets'][gid],emit)
                print(f'{a.model} {a.arm or "original"} {split} graph={gid+1}/{len(gs)}',flush=True)
    summarize_file(a.out_dir/'records.jsonl',a.out_dir/'summary.json')
    if rt.continuous:
        from functional_probe import run_probe
        probe=[]
        for split in (('N32_shallow',) if a.smoke else ('N32_shallow','N32_deep','N128_shallow','N128_deep')):
            probe.append(run_probe(rt,plan,split,limit=2 if a.smoke else 32))
        P.write(a.out_dir/'functional-probe.json',probe)
    meta.update(status='complete',sample_forwards=rt.calls,records_sha256=P.sha(a.out_dir/'records.jsonl'))
    P.write(a.out_dir/'meta.json',meta)

if __name__=='__main__':main()
