"""Same-seed comparisons on fixed cases and shared qualification, not own damaged sets."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def compact(path):
    result={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['module'] not in ('pair','sustained','noise'):continue
            no=r.get('noise') or {}
            key=(r['split'],r['gid'],r['module'],r.get('timing'),r['condition'],r.get('hold',1),no.get('scale'),no.get('direction'))
            if key in result:raise ValueError('duplicate condition')
            row=dict(status=r['status'],eligible=r.get('input_qualified',False),roles={})
            if r['status']=='ok':
                ids={p:i for i,p in enumerate(r['nodes'])}
                for role,positions in dict(u=[r['u']],q=[r['q']],downstream=r['downstream'],all=r['nodes']).items():
                    if not positions:continue
                    for metric in ('value','first4'):
                        correct=lambda s,i: s['output'][i][0]==r['truth'][i][3] if metric=='value' else bool(s['output'][i][1])
                        vals=[all(correct(s,ids[p]) for s in r['rows'][-2:]) for p in positions]
                        clean=[all(correct(s,ids[p]) for s in r['clean'][-2:]) for p in positions]
                        row['roles'][role,metric]=(sum(vals)/len(vals),(sum(vals)-sum(clean))/len(vals))
            result[key]=row
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('eval_dir',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();pairs=[('A_s42-original','B_s42-original')]
    pairs += [(f'C_s{s}-original',f'C_s{s}-correction') for s in (2026,42,1234)]
    pairs += [('C_s2026-original','C_s2026-update'),('C_s2026-correction','C_s2026-update')]
    output=[]
    for left,right in pairs:
        lm=P.read(a.eval_dir/left/'meta.json');rm=P.read(a.eval_dir/right/'meta.json')
        if lm['plan_sha256']!=rm['plan_sha256'] or lm['code_hashes']!=rm['code_hashes']:raise ValueError('comparison protocol mismatch')
        aa=compact(a.eval_dir/left/'records.jsonl');bb=compact(a.eval_dir/right/'records.jsonl')
        counts=defaultdict(lambda:defaultdict(int));values=defaultdict(list)
        for key in sorted(set(aa)|set(bb),key=str):
            group=(key[0],*key[2:]);c=counts[group];c['fixed_cases']+=1
            arow,brow=aa.get(key),bb.get(key)
            if arow is None or brow is None:c['missing_record']+=1;continue
            if arow['status']!='ok' or brow['status']!='ok':c['not_both_source_accepted']+=1;continue
            c['both_source_accepted']+=1
            if not (arow['eligible'] and brow['eligible']):continue
            c['common_input_qualified']+=1
            for rk,(a_score,a_net) in arow['roles'].items():
                b_score,b_net=brow['roles'][rk]
                values[(*group,*rk)].append((a_score,b_score,b_score-a_score,a_net,b_net,b_net-a_net))
        output.append(dict(left=left,right=right,coverage=[dict(key=list(k),**v) for k,v in counts.items()],
                           endpoints=[dict(key=list(k),graphs=len(v),left=sum(x[0] for x in v)/len(v),right=sum(x[1] for x in v)/len(v),
                                           difference=sum(x[2] for x in v)/len(v),left_minus_clean=sum(x[3] for x in v)/len(v),
                                           right_minus_clean=sum(x[4] for x in v)/len(v),net_difference=sum(x[5] for x in v)/len(v)) for k,v in values.items()]))
    P.write(a.output,dict(comparisons=output,note='One fixed target per graph/condition; graph-equal endpoint last-two accuracy. Not conditional recovery on different damaged sets. Source acceptance and shared input qualification are explicitly restricted denominators.'))

if __name__=='__main__':main()
