"""Serial handoff only: never pull, kill, select GPUs or migrate old training runs."""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import spec as S


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('smoke','train','evaluate'),required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--micro-batch',type=int,choices=(8,16,32),default=32)
    p.add_argument('--donor-batch',type=int,default=32)
    p.add_argument('--num-threads',type=int,default=4)
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    def run(script,args):
        command=[sys.executable,str(S.HERE/script),*map(str,args)]
        print('RUN',command,flush=True);subprocess.run(command,cwd=S.P.ROOT,check=True)
    common=['--plan',a.plan,'--device',a.device]
    training=common+['--micro-batch',a.micro_batch,'--donor-batch',a.donor_batch,'--num-threads',a.num_threads]
    if a.phase=='smoke':
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(S.HERE/'tests'),'-v'],cwd=S.P.ROOT,check=True)
        run('train.py',training+['--out-dir',a.out_dir/'smoke','--smoke'])
        run('verify.py',common+['--smoke-dir',a.out_dir/'smoke','--output',a.out_dir/'acceptance.json'])
    elif a.phase=='train':
        dest=a.out_dir/'train';meta=dest/'meta.json'
        extra=[]
        if meta.exists():
            data=S.P.read(meta)
            if data['status']=='complete':
                if data['identity']['code_hashes']!=S.hashes() or data['identity']['plan_sha256']!=S.P.sha(a.plan):
                    raise ValueError('completed run code/plan mismatch')
                if data['identity']['micro_batch']!=a.micro_batch or data['identity']['donor_batch']!=a.donor_batch:
                    raise ValueError('completed run batch configuration mismatch')
                if not (dest/'final.ckpt').exists():raise ValueError('final absent')
                print('Completed main run retained.');return
            extra=['--resume']
        run('train.py',training+['--out-dir',dest,'--acceptance',a.out_dir/'acceptance.json',*extra])
    else:
        run('evaluate_new.py',common+['--checkpoint',a.out_dir/'train'/'final.ckpt','--out-dir',a.out_dir/'eval'])


if __name__=='__main__':main()
