"""Compare new final with existing same-seed C evaluations on the shared plan."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import spec as S
from compare import compact
P=S.P


def compare(left,right):
    lm=P.read(left/'meta.json');rm=P.read(right/'meta.json')
    if lm['status']!='complete' or rm['status']!='complete':raise ValueError('evaluation incomplete')
    if lm['scope']!='test' or rm['scope']!='test':raise ValueError('formal test only')
    if lm['plan_sha256']!=rm['plan_sha256']:raise ValueError('different evaluation data')
    if lm.get('evaluation_core_hashes',lm['code_hashes'])!=rm['evaluation_core_hashes']:
        raise ValueError('evaluation core differs; review before comparison')
    for path,meta in ((left,lm),(right,rm)):
        if P.sha(path/'records.jsonl')!=meta['records_sha256']:raise ValueError('records changed')
    aa=compact(left/'records.jsonl');bb=compact(right/'records.jsonl')
    coverage=defaultdict(lambda:defaultdict(int));values=defaultdict(list)
    for key in sorted(set(aa)|set(bb),key=str):
        group=(key[0],*key[2:]);c=coverage[group];c['fixed_cases']+=1
        a,b=aa.get(key),bb.get(key)
        if a is None or b is None:c['missing']+=1;continue
        if a['status']!='ok' or b['status']!='ok':c['not_both_source_accepted']+=1;continue
        c['both_source_accepted']+=1
        # Both all-accepted and common-qualified views; do not hide harder inputs.
        scopes=['both_source_accepted']
        if a['eligible'] and b['eligible']:
            c['common_input_qualified']+=1;scopes.append('common_input_qualified')
        for rk,(av,an) in a['roles'].items():
            bv,bn=b['roles'][rk]
            for scope in scopes:values[(*group,scope,*rk)].append((av,bv,bv-av,an,bn,bn-an))
    names=('left','right','difference','left_minus_clean','right_minus_clean','net_difference')
    endpoints=[dict(key=list(k),graphs=len(v),**{n:sum(z[i] for z in v)/len(v) for i,n in enumerate(names)})
               for k,v in values.items()]
    return dict(left=str(left),right=str(right),left_checkpoint=lm['model']['sha256'],
                right_checkpoint=rm['model']['sha256'],
                coverage=[dict(key=list(k),**v) for k,v in coverage.items()],endpoints=endpoints,
                clean_left=P.read(left/'summary.json')['clean'],clean_right=P.read(right/'summary.json')['clean'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-eval-dir',type=Path,required=True)
    p.add_argument('--new-eval-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    reports=[compare(a.old_eval_dir/f'C_s2026-{arm}',a.new_eval_dir) for arm in ('original','correction','update')]
    P.write(a.output,dict(comparisons=reports,note='Same seed, same plan. Separate source acceptance and receiver qualification. Last-two endpoint, not recovery conditional on unequal damaged sets; whole training recipes differ.'))
