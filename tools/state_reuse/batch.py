#!/usr/bin/env python3
"""Prepare and execute the approved pilots serially; no GPU work on prepare.

Plans use argument arrays, not shell strings. Resume skips only our own completed
receipts with matching output hashes. Incomplete outputs are preserved for review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'priority_pilots'))
from io_utils import read_json,write_json,sha


def code_identity():
    paths=subprocess.check_output(['git','ls-files','tools','experiments','Simtransformer'],cwd=ROOT,text=True).splitlines()
    return {p:sha(ROOT/p) for p in paths if Path(p).suffix in ('.py','.yaml','.yml') and (ROOT/p).is_file()}


def jobs_for(lane,phase,out,source_a,source_b):
    out=Path(out);jobs=[];smoke=phase=='smoke'
    def job(name,argv,dest,gpu):
        jobs.append(dict(name=name,argv=argv+['--out-dir',str(dest)],out=str(dest),gpu=gpu))
    if lane=='AB':
        for mode in ('repeat','reuse'):
            dirs=[]
            for model,source in [('A',source_a),('B',source_b)]:
                dest=out/f'{mode}-{model}';dirs.append(dest)
                argv=['tools/state_reuse/run_discrete.py','--source-dir',str(source),'--mode',mode]
                if smoke:argv+=['--limit-graphs','2','--horizons','4']
                job(f'{mode}-{model}',argv,dest,True)
            args=['tools/state_reuse/summarize.py']
            for d in dirs:args+=['--run-dir',str(d)]
            job(f'{mode}-summary',args,out/f'{mode}-summary',False)
            job(f'{mode}-delivery',['tools/state_reuse/deliver.py','--summary-dir',str(out/f'{mode}-summary')],out/f'{mode}-delivery',False)
    else:
        dirs=[]
        for arm in ('initial','hold','noise','semantic'):
            dest=out/f'C-{arm}';dirs.append(dest)
            args=['tools/state_reuse/run_c_function.py','--arms',arm]
            if smoke:args+=['--candidate-count','2','--horizon','4','--snapshot-graphs','1']
            job(f'C-{arm}',args,dest,True)
        args=['tools/state_reuse/summarize.py']
        for d in dirs:args+=['--run-dir',str(d)]
        job('C-summary',args,out/'C-summary',False)
        job('C-delivery',['tools/state_reuse/deliver.py','--summary-dir',str(out/'C-summary')],out/'C-delivery',False)
    return jobs


def completed(job):
    d=Path(job['out']);m=read_json(d/'meta.json')
    if m.get('status')!='complete':raise ValueError('incomplete output: '+str(d))
    for kind in ('records','coverage'):
        if kind+'_sha256' in m and sha(d/(kind+'.jsonl'))!=m[kind+'_sha256']:
            raise ValueError('output hash mismatch: '+str(d))
    for name,h in m.get('files',{}).items():
        if sha(d/name)!=h:raise ValueError('delivery file hash mismatch')
    return sha(d/'meta.json')


def prepare(a):
    out=a.out_dir.resolve()
    smoke=None
    if a.phase=='main':
        if a.smoke_bundle is None:raise ValueError('main requires the completed smoke bundle')
        smoke=a.smoke_bundle.resolve();proof=read_json(smoke/'accepted.json')
        plan=read_json(smoke/'plan.json')
        if proof['plan_sha256']!=sha(smoke/'plan.json') or plan['lane']!=a.lane or plan['phase']!='smoke':
            raise ValueError('wrong smoke acceptance')
        if plan['code_hashes']!=code_identity():raise ValueError('code changed since smoke; review and repeat affected smoke')
        for job in plan['jobs']:completed(job)
    out.mkdir(parents=True,exist_ok=False)
    sa=(ROOT/'tmp/analysis/priority-pilots-20261118/reassign-main-A') if a.source_a is None else a.source_a.resolve()
    sb=(ROOT/'tmp/analysis/priority-pilots-20261118/reassign-main-B') if a.source_b is None else a.source_b.resolve()
    if smoke and a.lane=='AB':
        previous=read_json(smoke/'plan.json')
        if [str(sa),str(sb)]!=previous['source_dirs']:raise ValueError('source changed since smoke')
    plan=dict(protocol='state_reuse_batch_v1',lane=a.lane,phase=a.phase,root=str(ROOT),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_hashes=code_identity(),source_dirs=[str(sa),str(sb)] if a.lane=='AB' else [],
        smoke_bundle=str(smoke) if smoke else None,jobs=jobs_for(a.lane,a.phase,out,sa,sb))
    write_json(out/'plan.json',plan)
    print(json.dumps(dict(prepared=str(out),jobs=len(plan['jobs']),gpu_started=False)))


def run(a):
    folder=a.bundle.resolve();plan=read_json(folder/'plan.json');ph=sha(folder/'plan.json')
    if plan['root']!=str(ROOT) or plan['code_hashes']!=code_identity():raise ValueError('workspace/code differs from frozen plan')
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['CUDA_VISIBLE_DEVICES']=a.gpu
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    for job in plan['jobs']:
        if plan['code_hashes']!=code_identity():raise ValueError('code changed while queue running')
        receipt=folder/(job['name']+'.receipt.json');dest=Path(job['out'])
        argv=[sys.executable,'-u',*job['argv']]
        if job['gpu']:argv+=['--num_threads','4']
        if receipt.exists():
            old=read_json(receipt)
            if old['plan_sha256']!=ph or old['argv']!=argv:raise ValueError('receipt identity mismatch')
            if old['status']=='complete':
                if completed(job)!=old['meta_sha256']:raise ValueError('completed output changed')
                print('skip complete: '+job['name'],flush=True);continue
            raise ValueError('previous incomplete/failed attempt retained: '+str(receipt)+'; inspect, create new bundle; do not overwrite')
        if dest.exists():raise ValueError('unowned output exists: '+str(dest))
        info=dict(status='running',plan_sha256=ph,argv=argv,gpu=a.gpu if job['gpu'] else None,start_time=time.time())
        write_json(receipt,info)
        log=folder/(job['name']+'.log')
        print('running: '+job['name'],flush=True)
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        info.update(exit_code=result.returncode,elapsed_seconds=time.time()-info['start_time'])
        if result.returncode:
            info['status']='failed';write_json(receipt,info);raise SystemExit('failed: '+str(log))
        info.update(status='complete',meta_sha256=completed(job));write_json(receipt,info)
    write_json(folder/'complete.json',dict(plan_sha256=ph,jobs=len(plan['jobs']),completed_at=time.time()))
    print('Queue complete. Smoke still requires the documented tensor/semantic checks.')


def accept(a):
    folder=a.bundle.resolve();plan=read_json(folder/'plan.json')
    if plan['phase']!='smoke':raise ValueError('only smoke is accepted here')
    done=read_json(folder/'complete.json')
    if done['plan_sha256']!=sha(folder/'plan.json'):raise ValueError('plan changed')
    if not a.notes.is_file() or not a.notes.read_text(encoding='utf-8').strip():raise ValueError('execution agent check notes required')
    for job in plan['jobs']:
        completed(job)
        if job['gpu'] and read_json(Path(job['out'])/'meta.json').get('records',0)==0:
            raise ValueError('zero-record smoke is coverage only; report it before preparing main')
    target=folder/'accepted.json'
    if target.exists():raise ValueError('acceptance already exists')
    write_json(target,dict(plan_sha256=sha(folder/'plan.json'),notes=str(a.notes.resolve()),notes_sha256=sha(a.notes),
        meaning='Execution agent attests the handoff checks, not model success or user approval.',accepted_at=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare');q.add_argument('--lane',choices=('AB','C'),required=True)
    q.add_argument('--phase',choices=('smoke','main'),required=True);q.add_argument('--out-dir',type=Path,required=True)
    q.add_argument('--source-a',type=Path);q.add_argument('--source-b',type=Path);q.add_argument('--smoke-bundle',type=Path)
    q=s.add_parser('run');q.add_argument('--bundle',type=Path,required=True);q.add_argument('--gpu',required=True)
    q=s.add_parser('accept-smoke');q.add_argument('--bundle',type=Path,required=True);q.add_argument('--notes',type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'run':run,'accept-smoke':accept}[a.command](a)


if __name__=='__main__':main()
