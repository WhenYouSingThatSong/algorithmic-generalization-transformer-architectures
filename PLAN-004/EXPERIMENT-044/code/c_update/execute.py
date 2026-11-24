"""Serial execution; does not pull git, alter live files, or choose GPUs."""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','train','evaluate'),required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--num_threads',type=int,default=4)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).resolve().parent;plan=P.read(a.plan)
    def run(script,args):
        cmd=[sys.executable,str(code/script),*map(str,args)]
        print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=P.ROOT,check=True)
    common=['--plan',a.plan,'--device',a.device,'--num_threads',a.num_threads]
    jobs=[('C_s2026','correction'),('C_s2026','update'),('C_s42','correction'),('C_s1234','correction')]
    if a.phase=='smoke':
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(code/'tests'),'-v'],check=True,cwd=P.ROOT)
        run('verify_server.py',common+['--output',a.out_dir/'server-check.json'])
        for model,arm in jobs[:2]:
            dest=a.out_dir/'smoke'/f'{model}-{arm}'
            run('train_model.py',common+['--model',model,'--arm',arm,'--out-dir',dest,'--smoke-batches',1])
            run('evaluate.py',common+['--model',model,'--arm',arm,'--checkpoint',dest/'smoke-final.ckpt','--out-dir',a.out_dir/'smoke'/f'eval-{model}-{arm}','--smoke'])
        P.write(a.out_dir/'smoke-accepted.json',dict(status='passed',plan_sha256=P.sha(a.plan),code_hashes=P.source_hashes()))
    elif a.phase=='train':
        smoke=P.read(a.out_dir/'smoke-accepted.json')
        if smoke['plan_sha256']!=P.sha(a.plan) or smoke['code_hashes']!=P.source_hashes():raise ValueError('smoke belongs to different code/plan')
        for model,arm in jobs:
            probe=a.out_dir/'functional-baseline'/f'{model}.json'
            if not probe.exists():
                run('functional_probe.py',common+['--model',model,'--output',probe,'--limit',64])
            dest=a.out_dir/'train'/f'{model}-{arm}';args=common+['--model',model,'--arm',arm,'--out-dir',dest]
            if (dest/'meta.json').exists():
                meta=P.read(dest/'meta.json')
                if meta['status']=='complete':
                    if meta['identity']['cohort']!='main' or meta['identity']['plan_sha256']!=P.sha(a.plan) or meta['identity']['code_hashes']!=P.source_hashes() or not (dest/'final.ckpt').exists():raise ValueError('invalid completed run')
                    continue
                args+=['--resume']
            run('train_model.py',args)
    else:
        jobs_eval=[(z['id'],None,None) for z in plan['models']]
        jobs_eval += [(model,arm,a.out_dir/'train'/f'{model}-{arm}'/'final.ckpt') for model,arm in jobs]
        for model,arm,ck in jobs_eval:
            dest=a.out_dir/'eval'/f'{model}-{arm or "original"}'
            if (dest/'meta.json').exists() and P.read(dest/'meta.json')['status']=='complete':
                old=P.read(dest/'meta.json')
                if old['plan_sha256']!=P.sha(a.plan) or old['code_hashes']!=P.source_hashes():raise ValueError('previous eval uses other code/plan')
                if ck and old['model']['sha256']!=P.sha(ck):raise ValueError('previous eval uses other checkpoint')
                continue
            args=common+['--model',model,'--out-dir',dest]
            if arm:args+=['--arm',arm,'--checkpoint',ck]
            run('evaluate.py',args)
        run('compare.py',[a.out_dir/'eval',a.out_dir/'comparison.json'])

if __name__=='__main__':main()
