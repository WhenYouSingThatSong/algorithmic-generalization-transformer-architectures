"""Serial server handoff. Smoke acceptance is execution-agent evidence, not new user approval."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run import source_hashes
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'priority_pilots'))
from io_utils import read_json,write_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--lane',choices=('AB','C'),required=True)
    p.add_argument('--phase',choices=('smoke','main'),required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--smoke-dir',type=Path)
    p.add_argument('--checks',type=Path,help='execution agent smoke checks; required for main')
    a=p.parse_args();smoke=a.phase=='smoke'; hashes=source_hashes()
    tasks=a.data_dir.resolve()/'tasks.json';registry=a.data_dir.resolve()/'models.json'
    models=[m['id'] for m in read_json(registry)['models'] if (m['family']=='C')==(a.lane=='C')]
    if not models:raise ValueError('no models')
    if not smoke:
        if not a.smoke_dir or not a.checks or not a.checks.read_text(encoding='utf8').strip():
            raise ValueError('main requires execution-agent smoke checks')
        proof=read_json(a.smoke_dir/'complete.json')
        if (proof['code_hashes']!=hashes or proof['tasks_sha256']!=sha(tasks) or
            proof['models_sha256']!=sha(registry) or proof['lane']!=a.lane or proof['phase']!='smoke'):
            raise ValueError('smoke source/scope mismatch')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu;env['PYTHONUNBUFFERED']='1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='4'
    plan=dict(lane=a.lane,phase=a.phase,code_hashes=hashes,tasks_sha256=sha(tasks),models_sha256=sha(registry),
              models=models,checks_sha256=sha(a.checks) if a.checks else None,
              commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(a.out_dir/'plan.json',plan);dirs=[]
    def call(name,args):
        if hashes!=source_hashes():raise ValueError('source changed during queue')
        before=time.time()
        with (a.out_dir/(name+'.log')).open('x',encoding='utf8') as f:
            code=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        write_json(a.out_dir/(name+'.receipt.json'),dict(argv=args,exit_code=code,seconds=time.time()-before))
        if code:raise SystemExit('failed; preserve output, inspect log: '+name)
    for mode in ('assignment','program'):
        for model in models:
            folder=a.out_dir/(mode+'-'+model);dirs.append(folder)
            call(mode+'-'+model,['tools/abc_workspace/run.py','--tasks',str(tasks),'--models',str(registry),
                '--model',model,'--mode',mode,'--out-dir',str(folder),
                '--per-stratum',str(1 if smoke else (16 if mode=='assignment' else 8)),
                '--stages',str(2 if smoke else 20),'--probe-per-stratum',str(1 if smoke else 2),
                '--num_threads','4'])
    args=['tools/abc_workspace/summarize.py','--expect-models',*models,'--out-dir',str(a.out_dir/'summary')]
    for d in dirs:args+=['--run-dir',str(d)]
    call('summary',args)
    write_json(a.out_dir/'complete.json',dict(plan,outputs=[dict(path=str(d),meta_sha256=sha(d/'meta.json')) for d in dirs]))
    print('Completed '+a.lane+' '+a.phase+'. Commit small review tables, never weights/raw trajectories.')


if __name__=='__main__':main()
