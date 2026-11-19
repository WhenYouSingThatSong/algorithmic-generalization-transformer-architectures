"""Approved 50-epoch training. Poor validation never stops training."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def recurrent_loss(rt,x,y,gs,events,micro_offset):
    import torch
    from losses import update_loss
    from runtime import node_masks
    d=gs[0]['D']
    if any(g['D']!=d for g in gs):raise ValueError('mixed D')
    local=[dict(e,bi=e['bi']-micro_offset) for e in events if micro_offset<=e['bi']<micro_offset+len(gs)]
    donors={};audit=[];donor_before=rt.calls
    generated=rt.donors(x,local)
    for e in local:
        bi,p=e['bi'],e['pos'];target=y[bi,p,:4].tolist();target[3]=(target[3]+e['delta'])%23
        vec,decoded=generated[bi,p]
        accepted=decoded==target
        audit.append(dict(**e,source_readout=decoded,intended=target,accepted=accepted))
        if accepted:donors[(bi,p)]=vec
    donor_cost=rt.calls-donor_before
    h=rt.init(x);previous=x[...,:4].detach();total=x.sum()*0.;metrics=[]
    nodes=node_masks(gs,x.shape[1],x.device);valid=y[...,0]!=rt.tok.syntax_tok2idx[rt.tok.PAD]
    orig=y.cpu().tolist()
    for t in range(d+1):
        fired=[]
        for e in local:
            key=e['bi'],e['pos']
            if e['round']==t and key in donors:fired.append(key)
        if fired:
            h=h.clone();previous=previous.clone()
            for bi,p in fired:
                before=previous[bi,p].tolist()
                ratio=float((donors[(bi,p)]-h[bi,p].detach()).norm()/h[bi,p].detach().norm().clamp_min(1e-12))
                h[bi,p]=donors[(bi,p)]
                after=rt.decode(h[bi:bi+1,p:p+1])[0,0,:4]
                previous[bi,p]=after
                next(a for a in audit if a['bi']==bi and a['pos']==p).update(before=before,after=after.tolist(),relative_vector_change=ratio)
        h,logits,mid=rt.forward(h,middle=rt.update)
        if rt.update:
            pairs=[P.local_step(g,c,o) for g,c,o in zip(gs,previous.cpu().tolist(),orig)]
            target=torch.tensor([z[0] for z in pairs],device=x.device)
            reference=torch.tensor([z[1] for z in pairs],device=x.device)
            loss,m=update_loss(logits,mid,previous,target,reference,nodes,valid)
        else:
            # Original teacher progression: all variable occurrences start EMPTY;
            # only definition positions at depth <= t acquire numeric labels.
            target=x.clone()
            for bi,g in enumerate(gs):
                for p in g['nodes']:
                    if g['depths'][p]<=t:target[bi,p]=y[bi,p]
            loss=sum(rt.lit.criterion(z.reshape(-1,z.size(-1)),target[...,f].reshape(-1)) for f,z in enumerate(logits[:4]))
            m=dict(state=float(loss.detach()))
        total=total+loss/(d+1);metrics.append(m)
        previous=torch.stack([z.argmax(-1) for z in logits[:4]],-1).detach()
    return total,metrics,audit,donor_cost

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--model',choices=('C_s42','C_s1234','C_s2026'),required=True)
    p.add_argument('--arm',choices=('correction','update'),required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--micro-batch',type=int,default=8)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke-batches',type=int)
    p.add_argument('--num_threads',type=int,default=4)
    a=p.parse_args()
    if a.arm=='update' and a.model!='C_s2026':p.error('only first UPDATE seed authorized')
    if not 1<=a.micro_batch<=32 or (a.smoke_batches is not None and a.smoke_batches<1):p.error('invalid batch size')
    from runtime import Runtime,load_data,dataset_graphs
    plan=P.read(a.plan)
    if plan['protocol']!=P.VERSION or plan['status']!='complete':raise ValueError('plan incomplete')
    if P.sha(plan['train']['path'])!=plan['train']['sha256'] or plan['train']['sha256']!=P.TRAIN_SHA:raise ValueError('training data')
    for name,h in plan['config_hashes'].items():
        if P.sha(P.ROOT/name)!=h:raise ValueError('changed config '+name)
    spec=dict(next(m for m in plan['models'] if m['id']==a.model),arm=a.arm)
    import runtime as R
    R.setup()
    import torch,numpy as np
    run_seed=P.seed('training',a.model)%(2**32)
    random.seed(run_seed);np.random.seed(run_seed);torch.manual_seed(run_seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(run_seed)
    rt=Runtime(spec,a.device,initialize_update=a.arm=='update')
    x,y,info=load_data(plan['train']['path'],rt.tok);gs=dataset_graphs(y,info);ds=[g['D'] for g in gs]
    if len(gs)!=50000 or any(len(g['nodes'])!=32 or g['D']>7 for g in gs):raise ValueError('wrong original training distribution')
    head=list(rt.model.output_layers[4].parameters()) if rt.update else []
    head_ids={id(z) for z in head}
    groups=[dict(params=[v for v in rt.model.parameters() if id(v) not in head_ids],lr=1e-4,initial_lr=1e-4)]
    if head:groups.append(dict(params=head,lr=5e-4,initial_lr=5e-4))
    opt=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01,eps=1e-8)
    per_epoch=len(P.batches(ds,1));total_steps=50*per_epoch
    fingerprint=dict(protocol=P.VERSION,arm=a.arm,model=a.model,source=spec,plan_sha256=P.sha(a.plan),
                     code_hashes=P.source_hashes(),micro_batch=a.micro_batch,cohort='smoke' if a.smoke_batches else 'main')
    step=0;start_epoch=1;start_batch=0;presentations=0;best=None
    if a.resume:
        old=P.read(a.out_dir/'meta.json')
        if old['identity']!=fingerprint or old['status']=='complete':raise ValueError('resume identity/status mismatch')
        ck=torch.load(a.out_dir/'last.ckpt',map_location='cpu',weights_only=False)
        rt.model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer_state'])
        step=ck['step'];presentations=ck['presentations'];start_epoch=ck['next_epoch'];start_batch=ck['next_batch'];best=ck['best']
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        # Logs beyond the last durable checkpoint belong to discarded updates.
        for name in ('steps.jsonl','injections.jsonl'):
            path=a.out_dir/name
            if path.exists():
                lines=[]
                for line in path.read_text(encoding='utf-8').splitlines():
                    try:row=json.loads(line)
                    except json.JSONDecodeError:break
                    if row['step']<=step:lines.append(line)
                path.write_text(''.join(line+'\n' for line in lines),encoding='utf-8')
    else:
        a.out_dir.mkdir(parents=True,exist_ok=False)
        P.write(a.out_dir/'meta.json',dict(status='running',identity=fingerprint,epochs=50,batches_per_epoch=per_epoch))
    def save(name,next_epoch,next_batch):
        dest=a.out_dir/(name+'.ckpt');tmp=dest.with_suffix('.tmp')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in rt.model.state_dict().items()},
                        optimizer_state=opt.state_dict(),step=step,presentations=presentations,
                        next_epoch=next_epoch,next_batch=next_batch,best=best,identity=fingerprint,
                        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])),tmp)
        tmp.replace(dest)
    epoch=start_epoch-1
    if a.resume and start_batch==0 and epoch in P.CHECKS:
        # An interruption during the epoch-end dev check must not silently skip it.
        devpath=a.out_dir/f'dev-{epoch:02d}.json'
        if devpath.exists():report=P.read(devpath)
        else:
            from evaluate import small_check
            report=small_check(rt,plan,epoch,devpath)
        score=report['selection_score']
        if best is None or score>best['score']:
            best=dict(epoch=epoch,score=score);save('best-dev',start_epoch,0)
        save('last',start_epoch,0)
    with (a.out_dir/'steps.jsonl').open('a',encoding='utf-8') as log, (a.out_dir/'injections.jsonl').open('a',encoding='utf-8') as inj:
        for epoch in range(start_epoch,51):
            bs=P.batches(ds,epoch);rt.model.train()
            for bi,ids in enumerate(bs):
                if epoch==start_epoch and bi<start_batch:continue
                if a.smoke_batches and bi>=a.smoke_batches:break
                enabled,events=P.events([gs[i] for i in ids],ids,epoch,bi,force=bool(a.smoke_batches))
                for g in opt.param_groups:g['lr']=g['initial_lr']*P.lr_factor(step,total_steps)
                opt.zero_grad(set_to_none=True);batch_loss=0.;audit_all=[];stats=[];d_cost=0;before_calls=rt.calls
                for off in range(0,len(ids),a.micro_batch):
                    ii=ids[off:off+a.micro_batch]
                    xx,yy=x[ii].to(rt.device),y[ii].to(rt.device)
                    loss,ms,au,dc=recurrent_loss(rt,xx,yy,[gs[i] for i in ii],events,off)
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    weight=len(ii)/len(ids);(loss*weight).backward();batch_loss+=float(loss.detach())*weight
                    stats.extend(ms);audit_all.extend(au);d_cost+=dc
                norm=torch.nn.utils.clip_grad_norm_(rt.model.parameters(),float('inf'),error_if_nonfinite=True)
                opt.step();step+=1;presentations+=len(ids)
                for row in audit_all:inj.write(json.dumps(dict(epoch=epoch,step=step,**row))+'\n')
                row=dict(epoch=epoch,batch=bi,step=step,graphs=len(ids),loss=batch_loss,grad_norm=float(norm),
                         candidates=sum(len(gs[i]['nodes']) for i in ids),
                         enabled=enabled,attempts=len(events),accepted=sum(z['accepted'] for z in audit_all),
                         donor_forwards=d_cost,recipient_forwards=rt.calls-before_calls-d_cost,metrics=stats)
                log.write(json.dumps(row)+'\n');log.flush();inj.flush()
                if step%100==0 or a.smoke_batches:
                    save('last',epoch,bi+1);print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
            if P.source_hashes()!=fingerprint['code_hashes']:raise ValueError('source changed during training')
            if a.smoke_batches:save('smoke',epoch+1,0);break
            save('last',epoch+1,0)
            if epoch in P.CHECKS:
                save(f'epoch-{epoch:02d}',epoch+1,0)
                from evaluate import small_check
                report=small_check(rt,plan,epoch,a.out_dir/f'dev-{epoch:02d}.json')
                score=report['selection_score']
                if best is None or score>best['score']:
                    best=dict(epoch=epoch,score=score);save('best-dev',epoch+1,0)
                save('last',epoch+1,0)
                # No early stopping or performance gate. All 50 epochs run.
        save('smoke-final' if a.smoke_batches else 'final',epoch+1,0)
    meta=P.read(a.out_dir/'meta.json');meta.update(status='complete',steps=step,presentations=presentations,best=best)
    P.write(a.out_dir/'meta.json',meta)
    from summarize_training import summarize_run
    summarize_run(a.out_dir)

if __name__=='__main__':main()
