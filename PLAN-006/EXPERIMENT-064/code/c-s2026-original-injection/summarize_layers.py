"""Aggregate layered-injection records without rerunning the model."""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'tools/c_hard_graphs'))
from design import P

ROLES=('u','r','anc','q','m1','sink')

def desc_sets(case):
    nodes=case['nodes']; parents={n['i']:n['parents'] for n in nodes}
    def desc(root):
        out=set();front={root}
        while front:
            nxt=set()
            for i in nodes:
                if any(p in front for p in i['parents']) and i['i'] not in out and i['i']!=root:
                    out.add(i['i']);nxt.add(i['i'])
            front=nxt
        return out
    u,r,a=case['u'],case['r'],case['ancestor']
    return dict(target=desc(u),parent=desc(r),ancestor=desc(a))

def affected(label,case,ds):
    u,r,a=case['u'],case['r'],case['ancestor']
    fam=('ancestor' if label.startswith(('ancestor','all')) else
         'parent' if label.startswith('parent') else 'target')
    if label.startswith(('clean','target_just_self')):
        fam='ancestor'  # widest clean control region
    return {u,r,a}|ds[fam]

def classify(v,g):
    if v>=24:return 'empty'
    if v==g:return 'correct'
    return 'wrong'

def metrics(vals,gold,alt,nodes):
    cor=spec=emp=wr=oth=0
    for d in nodes:
        v=vals[str(d)];g=gold[d];a=alt[d]
        if v>=24:emp+=1
        elif v==g:cor+=1
        elif v<23:
            wr+=1
            if a!=g and v==a:spec+=1
        else:oth+=1
    return dict(n=len(nodes),correct=cor,specific=spec,empty=emp,wrong=wr,other=oth)

def paired(vals,ref,gold,alt,nodes):
    """Separate injection damage from baseline drift and early computation."""
    out=dict(n=len(nodes),pair_wrong=0,pair_specific=0,premature_wrong=0,premature_correct=0,
             baseline_wrong=0,baseline_empty=0,correct=0)
    for d in nodes:
        v=vals[str(d)];b=ref[str(d)];g=gold[d];a=alt[d]
        if b>=24:
            out['baseline_empty']+=1
            if v<23:
                if v==g:out['premature_correct']+=1
                else:out['premature_wrong']+=1
        elif b==g:
            if v==g:out['correct']+=1
            elif v<23:
                out['pair_wrong']+=1
                if a!=g and v==a:out['pair_specific']+=1
        else:
            out['baseline_wrong']+=1
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='C_s2026')
    ap.add_argument('--src',type=Path,default=ROOT/'tmp/analysis/c-original-injection')
    ap.add_argument('--out',type=Path,default=ROOT/'docs/records/2026-11-22/c-s2026-original-injection')
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    data=P.read(a.src/f'{a.model}-layered.json'); cases=data['cases']; recs=data['records']; R=data.get('rounds',len(recs[0]['rows']))
    ds={k:desc_sets(c) for k,c in cases.items()}
    gold={k:{n['i']:n['truth'] for n in c['nodes']} for k,c in cases.items()}
    alt={k:{n['i']:n['alt'] for n in c['nodes']} for k,c in cases.items()}
    agg=defaultdict(lambda: defaultdict(int)); steps=defaultdict(lambda: defaultdict(int))
    psteps=defaultdict(lambda: defaultdict(int))
    ref_map={(r['case'],r['start']):r for r in recs if r['label'].startswith('clean_')}
    per=[]
    snaps=((0,'pre'),(1,'1'),(2,'2'),(8,'8'),(R,'last'))
    for r in recs:
        c=cases[r['case']];g=gold[r['case']];al=alt[r['case']]
        lab=r['label'];aff=sorted(affected(lab,c,ds[r['case']]))
        inj=[z['node'] for z in r['injected']]
        ref=ref_map[(r['case'],r['start'])]
        roles={name:c[key] for name,key in zip(('u','r','anc','q','sink'),('u','r','ancestor','q','sink'))}
        m1=sorted(set(affected('ancestor',c,ds[r['case']]))-{c['u'],c['r'],c['ancestor'],c['q'],c['sink']})
        roles['m1']=m1[0] if len(m1)==1 else -1
        A=agg[lab];A['graphs']+=1;A['base_ok']+=int(r['base_ok']['computed_ok'] and r['base_ok']['pending_empty'])
        A['patched_'+('_'.join(sorted({z['kind'] for z in r['injected']})) if r['injected'] else 'none')]+=1
        row=dict(model=a.model,gid=r['gid'],depth=r['depth'],label=lab,start=r['start'],
                 injected='|'.join(f'{z["node"]}:{z["kind"]}' for z in r['injected']),
                 base_ok=int(r['base_ok']['computed_ok'] and r['base_ok']['pending_empty']))
        for st,name in snaps:
            vals=r['pre'] if st==0 else r['rows'][st-1]['values']
            if inj:
                m=metrics(vals,g,al,inj)
                for k,v in m.items():A[f'inj_{k}_{name}']+=v
            for role,node in roles.items():
                if node<0:continue
                v=vals[str(node)]
                if name not in ('pre',):
                    A[f'{role}_correct_{name}']+=int(v==g[node])
                    A[f'{role}_donor_{name}']+=int(v<23 and al[node]!=g[node] and v==al[node])
                    A[f'{role}_empty_{name}']+=int(v>=24)
                if name in ('1','8','last'):
                    row[f'{role}_correct_{name}']=int(v==g[node])
            m=metrics(vals,g,al,aff)
            if name in ('1','8','last'):
                for k,v in m.items():A[f'aff_{k}_{name}']+=v
                A[f'graphs_any_wrong_{name}']+=int(m['wrong']>0)
                row[f'aff_wrong_{name}']=m['wrong']
        last=r['rows'][-1]['values'];last2=r['rows'][-2]['values']
        stable=all(classify(last[str(d)],g[d])=='correct' and classify(last2[str(d)],g[d])=='correct' for d in aff)
        uq=all(classify(last[str(d)],g[d])=='correct' and classify(last2[str(d)],g[d])=='correct' for d in (c['u'],c['q']))
        A['graphs_stable_last2']+=int(stable);A['uq_stable_last2']+=int(uq)
        first_all=None
        for st in range(1,R+1):
            vals=r['rows'][st-1]['values']
            if all(classify(vals[str(d)],g[d])=='correct' for d in aff):first_all=st;break
        for k,cut in (('all_correct_8',8),('first_all_le2',2),('first_all_le4',4),('first_all_le6',6),
                      ('first_all_le8',8),('first_all_le12',12),('all_correct_last',R)):
            A[k]+=int(first_all is not None and first_all<=cut)
        row['stable_last2']=int(stable);row['uq_stable_last2']=int(uq);row['first_all_correct']=first_all
        per.append(row)
        # time course
        for st in range(1,R+1):
            vals=r['rows'][st-1]['values']
            for scope,nodes in [('injected',inj),('affected',aff)]+[(z,[roles[z]]) for z in ROLES if roles[z]>=0]:
                m=metrics(vals,g,al,nodes);S=steps[(lab,str(st),scope)]
                for k,v in m.items():S[k]+=v
                S['graphs']+=1
            pm=paired(vals,ref['rows'][st-1]['values'],g,al,aff);S=psteps[(lab,str(st))]
            for k,v in pm.items():S[k]+=v
            S['graphs']+=1
            names=[]
            if st==1:names.append('1')
            if st==8:names.append('8')
            if st==R:names.append('last')
            for name in names:
                for k in ('n','correct','pair_wrong','pair_specific','premature_wrong','premature_correct',
                          'baseline_wrong','baseline_empty'):A[f'{k}_{name}']+=pm[k]
                A[f'graphs_pair_wrong_{name}']+=int(pm['pair_wrong']>0)
    order=[k for k in ('clean_pending','clean_just','clean_later','target_just_self','parent_just_self',
        'ancestor_just_self','target_pending_correct','parent_pending_correct','ancestor_pending_correct',
        'target_pending_empty','target_pending_wrong','target_just','parent_pending','ancestor_pending',
        'parent_just','ancestor_just','all_pending_wrong','all_just','later_target','later_parent',
        'later_ancestor','later_all')]
    keys=('graphs','base_ok','inj_n_pre','inj_correct_pre','inj_donor_pre','inj_n_1','inj_correct_1','inj_empty_1',
          'inj_donor_1','inj_n_2','inj_correct_2','inj_donor_2','inj_n_8','inj_correct_8','inj_donor_8')
    role_keys=[f'{r}_{k}_{s}' for r in ROLES for k in ('correct','donor','empty') for s in ('1','2','8','last')]
    aff_keys=[f'aff_{k}_{s}' for k in ('n','correct','specific','empty','wrong','other') for s in ('1','8','last')]
    tail=('graphs_any_wrong_1','graphs_any_wrong_8','graphs_any_wrong_last','graphs_stable_last2','uq_stable_last2',
          'all_correct_8','all_correct_last','first_all_le2','first_all_le4','first_all_le6',
          'first_all_le8','first_all_le12',
          'pair_wrong_1','pair_wrong_8','pair_wrong_last','pair_specific_1','pair_specific_8','pair_specific_last',
          'premature_wrong_1','premature_wrong_8','premature_wrong_last',
          'premature_correct_1','premature_correct_8','premature_correct_last',
          'baseline_wrong_1','baseline_wrong_8','baseline_wrong_last',
          'baseline_empty_1','baseline_empty_8','baseline_empty_last',
          'graphs_pair_wrong_1','graphs_pair_wrong_8','graphs_pair_wrong_last')
    for name in sorted({k for A in agg.values() for k in A if k.startswith('patched_')}):keys+= (name,)
    keys=tuple(keys)+tuple(role_keys)+tuple(aff_keys)+tail
    with (a.out/f'{a.model}-layered-summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=('model','label',*keys));w.writeheader()
        for lab in order:
            if lab in agg:w.writerow(dict(model=a.model,label=lab,**{k:agg[lab].get(k,0) for k in keys}))
    with (a.out/f'{a.model}-layered-per-graph.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    with (a.out/f'{a.model}-layered-steps.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=('model','label','step','scope','n','correct','specific','empty','wrong','other','graphs'))
        w.writeheader()
        for (lab,st,scope),S in sorted(steps.items(),key=lambda z:(order.index(z[0][0]) if z[0][0] in order else 99,int(z[0][1]),z[0][2])):
            w.writerow(dict(model=a.model,label=lab,step=st,scope=scope,**{k:S.get(k,0) for k in ('n','correct','specific','empty','wrong','other','graphs')}))
    with (a.out/f'{a.model}-layered-paired-steps.csv').open('w',newline='',encoding='utf-8') as f:
        flds=('model','label','step','n','correct','pair_wrong','pair_specific','premature_wrong',
              'premature_correct','baseline_wrong','baseline_empty','graphs')
        w=csv.DictWriter(f,fieldnames=flds);w.writeheader()
        for (lab,st),S in sorted(psteps.items(),key=lambda z:(order.index(z[0][0]) if z[0][0] in order else 99,int(z[0][1]))):
            w.writerow(dict(model=a.model,label=lab,step=st,**{k:S.get(k,0) for k in flds[3:]}))
    # convenience print
    for lab in order:
        A=agg.get(lab)
        if not A:continue
        print(f"{lab:22s} u1={A.get('u_correct_1',0):2d} u8={A.get('u_correct_8',0):2d} "
              f"r1={A.get('r_correct_1',0):2d} anc1={A.get('anc_correct_1',0):2d} "
              f"q1={A.get('q_correct_1',0):2d} q8={A.get('q_correct_8',0):2d} "
              f"affW1={A.get('aff_wrong_1',0):3d} affW8={A.get('aff_wrong_8',0):3d} "
              f"stable={A.get('graphs_stable_last2',0):2d} uq={A.get('uq_stable_last2',0):2d}")
    print('written',a.out)

if __name__=='__main__':main()
