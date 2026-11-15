#!/usr/bin/env python3
"""Unified four-model six-stratum clean trajectories, node types and speed."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import suite_core as U
C=U.C

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=C.STUDY)
    p.add_argument('--pilot-data',type=Path,default=C.P.STUDY/'data')
    p.add_argument('--data-dir',type=Path,default=C.ROOT/'tmp/analysis/c0-capability-20261114/data')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--source-ckpt',type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--num_threads',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();R=C.runtime()
    import torch,numpy as np,json
    torch.set_float32_matmul_precision('medium')
    specs=U.models(a.study_dir,a.pilot_data,a.source_ckpt)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',protocol='C_basic_clean_v1',cohort='smoke' if a.smoke else 'main',models=specs,
        data_hashes={n:C.P.sha(a.data_dir/f'{n}.pt') for n in U.STRATA},batch_size=4,
        candidate_graphs_per_stratum=4 if a.smoke else 64,
        code_hashes={n:C.P.sha(U.HERE/n) for n in ('run_clean.py','suite_core.py')})
    C.P.write_json(a.out_dir/'meta.json',meta)
    with (a.out_dir/'index.jsonl').open('x',encoding='utf-8') as index,torch.no_grad():
        for spec in specs:
            for stratum in U.STRATA:
                ctx=C.context(R,spec['path'],a.data_dir/f'{stratum}.pt',a.device,4)
                lit=ctx['litmodel'];lit.eval();gid=0;limit=meta['candidate_graphs_per_stratum']
                dest=a.out_dir/spec['arm']/stratum;dest.mkdir(parents=True)
                for raw in ctx['data_module'].val_dataloader():
                    if gid>=limit:break
                    raw=raw[:min(len(raw),limit-gid)]
                    inputs,labels,info=R.split_batch(lit.reduce_batch(raw.to(ctx['device'])),ctx['model_config'])
                    ds=(info[...,0].max(1).values+1).tolist()
                    structures=[U.graph_structure(l.tolist(),i.tolist()) for l,i in zip(labels.cpu(),info.cpu())]
                    saved=[[] for _ in structures];h=lit.model.token_embedder(inputs)
                    for t in range(1,max(ds)+65):
                        h,ls=lit.model(h,True);pr=torch.stack([x.argmax(-1) for x in ls],-1)
                        for bi,(pos,_,_) in enumerate(structures):
                            if t<=ds[bi]+64:saved[bi].append(pr[bi,pos,:4].cpu().numpy())
                    for bi,(pos,depth,parents) in enumerate(structures):
                        path=dest/f'graph_{gid:04d}.npz';gold=labels[bi,pos,:4].cpu().numpy()
                        np.savez_compressed(path,predictions=np.stack(saved[bi]),truth4=gold,positions=pos,D=ds[bi])
                        index.write(json.dumps(dict(arm=spec['arm'],stratum=stratum,graph_id=gid,D=ds[bi],
                            path=str(path.relative_to(a.out_dir)),groups=U.node_groups(pos,depth,parents),
                            depths=depth,parents=parents))+'\n');gid+=1
                    index.flush()
                if gid!=limit:raise ValueError('fewer fixed candidate graphs than requested')
                print(json.dumps(dict(arm=spec['arm'],stratum=stratum,graphs=gid)),flush=True)
                del ctx,lit,h
    meta['status']='complete';C.P.write_json(a.out_dir/'meta.json',meta)
if __name__=='__main__':main()
