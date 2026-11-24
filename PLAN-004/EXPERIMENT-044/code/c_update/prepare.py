"""Pin original 50k training data/weights; generate disjoint dev and test graphs."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--models',type=Path,default=P.ROOT/'tmp/analysis/abc-workspace-20261119/data/models.json')
    p.add_argument('--train',type=Path,default=P.ROOT/'experiments/data/Tr32Test128-ADD/train_data.pt')
    p.add_argument('--extra-models',type=Path,help='optional JSON list of old C hold/noise/semantic specs')
    p.add_argument('--num_threads',type=int,default=4)
    a=p.parse_args()
    import runtime as R
    R.setup()
    import torch,random,numpy as np
    import b_mechanism_common as B
    sys.path.insert(0,str(P.ROOT/'tools/c_capability'))
    import c_common as CC
    from generate_data import generate_dataset
    _,_,dc=B.load_configs(P.CONFIG);tok=B.build_factored_tokenizer(dc)
    if P.sha(a.train)!=P.TRAIN_SHA:raise ValueError('original training file hash differs; do not silently replace it')
    models=P.read(a.models)['models']
    if a.extra_models:models+=P.read(a.extra_models)
    ids=[m['id'] for m in models]
    if len(ids)!=len(set(ids)) or not {'A_s42','B_s42','C_s42','C_s1234','C_s2026'}<=set(ids):raise ValueError('model registry')
    for m in models:
        if P.sha(m['path'])!=m['sha256']:raise ValueError('model hash: '+m['id'])
    _,btc,_=B.load_configs(next(m['config'] for m in models if m['id']=='B_s42'))
    bcor=dict(btc.recorrection_args)
    if not bcor['enable'] or bcor['corruption_strategy']!='last' or bcor['corruption_prob']!=.05 or bcor['tokenwise_corruption_prob']!=.5:
        raise ValueError('B reference corruption configuration differs')
    blob=torch.load(a.train,map_location='cpu',weights_only=False)
    y,info=blob['tokenized_dataset'],blob['dataset_info']
    if len(y)!=50000 or int(info[...,0].max())!=6:raise ValueError('training size/depth')
    pad=tok.syntax_tok2idx[tok.PAD]
    def fingerprints(ys):
        import hashlib
        return [hashlib.sha256(z[z[:,0]!=pad].to(torch.int64).contiguous().numpy().tobytes()).hexdigest() for z in ys]
    excluded=set(fingerprints(y));refs=[dict(path=str(a.train.resolve()),sha256=P.sha(a.train),count=len(y))]
    oldval=P.ROOT/'experiments'/dc.data_dir/dc.val_dag_config.data_file_name
    refpaths=[oldval]+list((P.ROOT/'tmp/analysis/c0-capability-20261114/data').glob('*.pt'))
    refpaths+=list((P.ROOT/'tmp/analysis/c-long-pilot-20261115/data').glob('*.pt'))
    for f in refpaths:
        if not f.exists():continue
        b=torch.load(f,map_location='cpu',weights_only=False)
        if 'tokenized_dataset' in b:
            excluded.update(fingerprints(b['tokenized_dataset']))
            refs.append(dict(path=str(f.resolve()),sha256=P.sha(f),count=len(b['tokenized_dataset'])))
    a.out_dir.mkdir(parents=True,exist_ok=False)
    plan=dict(protocol=P.VERSION,status='building',models=models,train=refs[0],
              epochs=50,effective_batch=32,micro_batch=8,train_rounds='D+1',
              batch_probability=.05,position_probability=.5,loss_weights=P.LOSS_WEIGHTS,b_reference_corruption=bcor,
              checkpoints=list(P.CHECKS),splits={},exclusions=refs,
              exclusion_note='exact final int64 tokens without PAD; not graph-isomorphism',
              config_hashes={str(f.relative_to(P.ROOT)):P.sha(f) for f in (P.ROOT/'experiments/configs'/P.CONFIG).glob('*.yaml')})
    for j,name in enumerate(('dev',*P.STRATA)):
        s=P.seed('data',name)%(2**32);random.seed(s);np.random.seed(s);torch.manual_seed(s)
        if name=='dev':cfg=B.EasyDict(dict(dc.dag_config));cfg.num_samples=128
        else:
            n=int(name.split('_')[0][1:]);band=name.split('_')[1]
            cfg=B.EasyDict(CC.generated_dag_config(n,band,1024))
        ys,ins=generate_dataset(cfg,tok)
        if name!='dev':
            inds,eligible=CC.select_indices(CC.max_depth_per_graph(ins),band,128,s)
            if len(inds)!=128:raise ValueError(f'{name}: fixed pool insufficient ({eligible}); report, no silent resampling')
            ys,ins=ys[inds],ins[inds]
        fs=fingerprints(ys)
        if len(set(fs))!=128 or excluded.intersection(fs):raise ValueError('overlap '+name)
        graphs=R.dataset_graphs(ys,ins)
        expected_n=32 if name=='dev' else n
        if any(len(g['nodes'])!=expected_n for g in graphs):raise ValueError('truncated/invalid generated graph '+name)
        excluded.update(fs)
        path=a.out_dir/(name+'.pt')
        torch.save(dict(tokenized_dataset=ys,dataset_info=ins),path)
        plan['splits'][name]=dict(path=str(path.resolve()),sha256=P.sha(path),count=128,seed=s,
                                 generator_config=dict(cfg),depths=ins[...,0].max(1).values.tolist())
        plan['splits'][name]['targets']=[P.select_target(g,name,i) for i,g in enumerate(graphs)]
        P.write(a.out_dir/'plan.json',plan)
    plan['status']='complete';P.write(a.out_dir/'plan.json',plan)
    print(a.out_dir/'plan.json')

if __name__=='__main__':main()
