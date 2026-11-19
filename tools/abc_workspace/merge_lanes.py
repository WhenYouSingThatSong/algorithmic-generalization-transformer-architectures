"""Merge completed AB/C lanes with explicitly verified common task/budget identity."""
import argparse
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'priority_pilots'))
from io_utils import read_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ab-dir',type=Path,required=True)
    p.add_argument('--c-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();proofs=[read_json(d/'complete.json') for d in (a.ab_dir,a.c_dir)]
    for key in ('tasks_sha256','models_sha256','code_hashes','phase'):
        if proofs[0][key]!=proofs[1][key]:raise ValueError('different lane '+key)
    if [p['lane'] for p in proofs]!=['AB','C']:raise ValueError('wrong lanes')
    models=[m for p in proofs for m in p['models']]
    args=[sys.executable,str(HERE/'summarize.py'),'--expect-models',*models,'--out-dir',str(a.out_dir)]
    for proof in proofs:
        for item in proof['outputs']:
            folder=Path(item['path'])
            if sha(folder/'meta.json')!=item['meta_sha256']:raise ValueError('output metadata changed')
            args+=['--run-dir',str(folder)]
    subprocess.run(args,check=True)


if __name__=='__main__':main()
