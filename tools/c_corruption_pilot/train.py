#!/usr/bin/env python3
"""Three matched +D post-training arms, occasional single-node corruption."""
import argparse,json,random,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pilot_core as C

def recurrent_loss(lit,inputs,labels,info,events,donor_rows,arm,epoch):
    import torch
    ds=info[...,0].max(1).values+1
    if not torch.all(ds==ds[0]): raise ValueError('mixed D batch')
    d=int(ds[0]); teachers,_,_=lit.get_teacher_batch_with_depth(inputs,labels,info[...,0])
    hidden=lit.model.token_embedder(inputs); total=0; solve=hold=0.; audit=[]
    for t,w in enumerate(C.P.phase_weights(d,1),1):
        for bi,c,state_round in events:
            if t!=state_round+1: continue
            u=c['target']; donor=donor_rows[(bi,f"shift_{c['delta']}")]
            with torch.no_grad():
                before=C.decode(lit,hidden[bi:bi+1,u:u+1])[0,0,:4].tolist()
                dp=C.decode(lit,donor[None,None,:])[0,0,:4].tolist()
            hidden,radius,actual=C.intervene(hidden,bi,u,donor,arm,
                C.stable_seed(C.SEED,'noise',epoch,c['graph_id']))
            with torch.no_grad(): after=C.decode(lit,hidden[bi:bi+1,u:u+1])[0,0,:4].tolist()
            gold=labels[bi,u,:4].tolist(); shifted=list(gold);shifted[3]=c['shifted_value']
            audit.append(dict(graph_id=c['graph_id'],target=u,D=d,state_round=state_round,
                delta=c['delta'],before=before,donor=dp,after=after,truth=gold,
                before_value_correct=before[3]==gold[3],before_first4_correct=before==gold,
                donor_value_intended=dp[3]==shifted[3],donor_first4_intended=dp==shifted,
                after_value_wrong=after[3]!=gold[3],actual_readout_changed=after!=before,
                semantic_reference_norm=radius,actual_norm=actual))
        hidden,logits=lit.model(hidden,True)
        for row in audit:
            if row['state_round']+1==t:
                bi=next(bi for bi,c,_ in events if c['graph_id']==row['graph_id'])
                row['post1']=[int(x[bi,row['target']].argmax(-1)) for x in logits[:4]]
        target=teachers[min(t,d)]
        loss=sum(lit.factor_loss_scales[f]*lit.criterion(x.reshape(-1,x.size(-1)),target[...,f].reshape(-1))
                 for f,x in enumerate(logits) if lit.factor_loss_scales[f])
        total=total+w*loss
        if t<=d: solve+=float(loss.detach())/d
        else: hold+=float(loss.detach())/d
    return total,solve,hold,d,audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=C.ARMS,required=True)
    p.add_argument('--data-dir',type=Path,default=C.P.STUDY/'data')
    p.add_argument('--plan-dir',type=Path,default=C.STUDY/'plan')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--ckpt',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--num_threads',type=int,default=4)
    p.add_argument('--smoke-batches',type=int)
    a=p.parse_args()
    if a.smoke_batches is not None and a.smoke_batches<1: p.error('positive smoke count')
    R=C.runtime()
    import torch,numpy as np
    plan=C.check_plan(a.plan_dir,a.data_dir);cases=C.read_cases(a.plan_dir/'train.jsonl')
    ckpt=a.ckpt or C.P.ROOT/C.P.source_spec()['weight_path']
    if C.P.sha(ckpt)!=C.P.SOURCE_SHA: raise ValueError('wrong source')
    random.seed(C.SEED);np.random.seed(C.SEED);torch.manual_seed(C.SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(C.SEED)
    torch.set_float32_matmul_precision('medium')
    ctx=C.context(R,ckpt,a.data_dir/'train.pt',a.device)
    lit,dm,device=ctx['litmodel'],ctx['data_module'],ctx['device']
    if not lit.nointerm or list(lit.factor_loss_scales)!=[1,1,1,1,0,0]: raise ValueError('wrong model')
    opt_cfg=dict(lit.train_config.AdamW_optimizer_config);opt_cfg['lr']=1e-4
    optimizer=torch.optim.AdamW(lit.model.parameters(),**opt_cfg)
    batches=C.P.batches(dm.data_train_info[...,0].max(1).values.tolist(),8,C.SEED)
    if len(cases)!=len(dm.data_train): raise ValueError('case count')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',protocol='C_corruption_pilot_v1',arm=a.arm,source_sha256=C.P.SOURCE_SHA,
        source_checkpoint=str(ckpt),plan=plan,optimizer=opt_cfg,batch_plan=batches,
        seed=C.SEED,epochs=C.EPOCHS,rate=C.RATE,cohort='smoke' if a.smoke_batches else 'main',
        code_sha256={n:C.P.sha(C.HERE/n) for n in ('train.py','pilot_core.py','prepare.py')},
        config_sha256={n:C.P.sha(R.EXPERIMENTS/'configs'/C.P.CONFIG/n) for n in ('model_config.yaml','train_config.yaml','data_config.yaml')})
    budget=dict(steps=0,graph_presentations=0,recipient_sample_forwards=0,recipient_token_forwards=0,
                recipient_attention_units=0,donor_sample_forwards=0,donor_token_forwards=0,donor_attention_units=0,
                scheduled_events=0,injected_events=0)
    C.P.write_json(a.out_dir/'meta.json',meta);start=time.perf_counter();lit.train()
    with (a.out_dir/'steps.jsonl').open('x',encoding='utf-8') as log, (a.out_dir/'injections.jsonl').open('x',encoding='utf-8') as injections:
        for epoch in range(1,C.EPOCHS+1):
            order=C.P.epoch_order(len(batches),epoch,C.SEED)
            if a.smoke_batches: order=order[:a.smoke_batches]
            for batch_id in order:
                ids=batches[batch_id]['ids']
                raw=torch.cat([dm.data_train[ids],dm.data_train_label[ids],dm.data_train_info[ids]],-1)
                # Intentionally retain inherited PAD recipe; no model.py changes.
                raw=lit.reduce_batch(raw).to(device)
                inputs,labels,info=R.split_batch(raw,ctx['model_config'])
                events=[(bi,cases[gid],cases[gid]['state_round']) for bi,gid in enumerate(ids)
                        if cases[gid]['status']=='selected' and C.scheduled(gid,epoch)]
                donor_rows,db=C.donors(lit,inputs,events)
                optimizer.zero_grad(set_to_none=True)
                loss,solve,hold,d,audit=recurrent_loss(lit,inputs,labels,info,events,donor_rows,a.arm,epoch)
                if d!=batches[batch_id]['D'] or not torch.isfinite(loss): raise ValueError('D/nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(lit.model.parameters(),float('inf'),error_if_nonfinite=True)
                optimizer.step()
                b,s=inputs.shape[:2];budget['steps']+=1;budget['graph_presentations']+=b
                budget['recipient_sample_forwards']+=b*2*d;budget['recipient_token_forwards']+=b*s*2*d
                budget['recipient_attention_units']+=b*s*s*2*d
                for key,val in db.items(): budget['donor_'+key]+=val
                budget['scheduled_events']+=len(events);budget['injected_events']+=len(events)*(a.arm!='hold')
                for row in audit:
                    injections.write(json.dumps(dict(arm=a.arm,epoch=epoch,step=budget['steps'],**row))+'\n')
                log.write(json.dumps(dict(epoch=epoch,batch_id=batch_id,loss=float(loss.detach()),solve=solve,hold=hold,
                    grad_norm=float(norm),**budget))+'\n');log.flush();injections.flush()
                if budget['steps']%16==0 or a.smoke_batches: print(json.dumps(dict(arm=a.arm,epoch=epoch,**budget)),flush=True)
            if a.smoke_batches: break
    dest=a.out_dir/('smoke.ckpt' if a.smoke_batches else 'final.ckpt')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in lit.model.state_dict().items()},
        optimizer_state=optimizer.state_dict(),epoch=epoch,pilot_arm=a.arm,source_sha256=C.P.SOURCE_SHA),dest)
    meta.update(status='complete',budget=budget,wall_seconds=time.perf_counter()-start,
                checkpoint=dict(path=str(dest),sha256=C.P.sha(dest)))
    C.P.write_json(a.out_dir/'meta.json',meta)
if __name__=='__main__': main()
