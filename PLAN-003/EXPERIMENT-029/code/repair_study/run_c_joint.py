#!/usr/bin/env python3
"""Add joint target+parents repair to the SAME saved natural/healthy states.

No new event selection, donor search or 64-round replay. Reuses saved hidden
snapshots and original single-location branches; checks unchanged continuation
against the source before interpreting the added branch.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_dir',type=Path,required=True)
    p.add_argument('--out_dir',type=Path,required=True)
    p.add_argument('--limit_events',type=int,help='smoke: first N fixed event IDs including healthy controls')
    p.add_argument('--device',default='cuda')
    p.add_argument('--ckpt',type=Path)
    p.add_argument('--num_threads',type=int,default=4)
    a=p.parse_args()
    if a.limit_events is not None and a.limit_events<1:
        p.error('limit_events must be positive')
    source_meta=core.read_json(a.source_dir/'meta.json')
    if source_meta['status']!='complete' or source_meta['experiment']!='C_local_repair_v1':
        raise ValueError('complete original local-repair run required')
    cases_file=Path(source_meta['args']['cases'])
    if core.sha(cases_file)!=source_meta['cases_sha256']:
        raise ValueError('original case plan changed')
    plan=core.read_json(cases_file)
    rows=[json.loads(line) for line in (a.source_dir/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    events=sorted({r['paired_event_id'] for r in rows})
    if a.limit_events is not None:
        events=events[:a.limit_events]
    rows=[r for r in rows if r['paired_event_id'] in events]
    original=source_meta['source']['run_meta']
    ckpt=a.ckpt or Path(original['weight_path'])
    if core.sha(ckpt)!=source_meta['checkpoint_sha256']:
        raise ValueError('checkpoint differs from snapshot source')
    sys.path.insert(0,str(HERE.parent/'b_mechanism'))
    from trajectory_io import configure_runtime
    configure_runtime()
    sys.path.insert(0,str(HERE.parent/'c_capability'))
    import run_capability as R
    import torch
    a.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',experiment='C_joint_repair_v1',source_meta=source_meta,
              source_records_sha256=core.sha(a.source_dir/'records.jsonl'),
              script_sha256=core.sha(__file__),weight=source_meta['weight'],
              args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    core.write_json(a.out_dir/'meta.json',meta)
    written=0
    with (a.out_dir/'records.jsonl').open('x',encoding='utf-8') as out, torch.no_grad():
        for stratum in sorted({r['stratum'] for r in rows}):
            subset=[r for r in rows if r['stratum']==stratum]
            data_file=Path(plan['data_dir'])/f'{stratum}.pt'
            data_sha=core.sha(data_file)
            if any(r['data_sha256']!=data_sha for r in subset):
                raise ValueError('data identity differs from original experiment')
            cfg=dict(original['args']);cfg.update(ckpt=ckpt,weight_name=source_meta['weight'],device=a.device)
            ctx=R.build_model_and_data(SimpleNamespace(**cfg),data_file)
            lit,device=ctx['litmodel'],ctx['device']
            if not lit.nointerm:
                raise ValueError('continuous model required')
            def decode(h,positions):
                logits=[head(h) for head in lit.model.output_layers]
                return torch.stack(lit.generate_factor_decodes(logits,temperature=0.0),-1)[0,positions,:4].cpu().tolist()
            def rollout(h,positions,horizon):
                trace=dict(initial=decode(h,positions),outputs=[])
                for _ in range(horizon):
                    h,logits=lit.model(h,True)
                    pred=torch.stack(lit.generate_factor_decodes(logits,temperature=0.0),-1)
                    trace['outputs'].append(pred[0,positions,:4].cpu().tolist())
                return trace
            for r in subset:
                path=a.source_dir/r['hidden_snapshot']
                snapshot=torch.load(path,map_location='cpu',weights_only=True)
                selected=[r['target_pos']]+r['parent_positions']
                if snapshot['donor_positions']!=selected or snapshot['state_round']!=r['state_round'] or snapshot['donor_round']!=r['donor_round'] or snapshot['data_sha256']!=data_sha:
                    raise ValueError('snapshot identity mismatch')
                base=snapshot['base_hidden'].to(device)
                donor=snapshot['donor_vectors'].to(device)
                pos=r['node_positions'];horizon=len(r['branches']['unchanged']['outputs'])
                actual=rollout(base.clone(),pos,horizon)
                if actual!=r['branches']['unchanged']:
                    raise ValueError('saved base does not reproduce original unchanged trace')
                sham=base.clone();sham[:,selected,:]=base[:,selected,:]
                sham_trace=rollout(sham,pos,horizon)
                if sham_trace!=actual:
                    raise ValueError('joint sham differs from unchanged')
                joint=base.clone();joint[:,selected,:]=donor
                trace=rollout(joint,pos,horizon)
                lookup={p:i for i,p in enumerate(pos)}
                if any(trace['initial'][lookup[p]][3]!=r['truth4'][lookup[p]][3] for p in selected):
                    raise ValueError('saved donor does not read the original correct local values')
                rec=dict(r)
                rec['branches']=dict(r['branches'],restore_joint=trace,sham_joint=sham_trace)
                rec.update(joint_positions=selected,snapshot_sha256=core.sha(path),
                           hidden_snapshot=str(path.resolve()),
                           source_records_sha256=meta['source_records_sha256'],
                           reused_single_branches=True)
                if a.limit_events is not None:
                    rec['cohort']='smoke_only'
                out.write(json.dumps(rec)+'\n');out.flush();written+=1
                print(json.dumps(dict(case_id=r['case_id'],written=written)),flush=True)
            del lit,ctx
    meta.update(status='complete',written_records=written,events=len(events))
    core.write_json(a.out_dir/'meta.json',meta)


if __name__=='__main__':
    main()
