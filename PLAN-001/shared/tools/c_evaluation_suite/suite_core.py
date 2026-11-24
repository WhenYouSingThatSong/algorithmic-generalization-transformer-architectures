"""Shared basic evaluation protocol; no GPU imports at module load."""
import csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'c_corruption_pilot'))
import pilot_core as C
E=C.load_module('corruption_eval_suite_adapter',C.HERE/'evaluate.py')
S=C.load_module('corruption_events_suite_adapter',C.HERE/'summarize.py')
STRATA=('N32_shallow','N32_deep','N64_shallow','N64_deep','N128_shallow','N128_deep')
OUT=C.ROOT/'tmp/analysis/c-evaluation-suite-20261115'
BRANCHES={'unchanged':(), 'sham':('u_original','q_original'),
          'u_same':('u_same',),'q_same':('q_same',),'both_same':('u_same','q_same'),
          'u_wrong':('u_wrong',),'q_wrong':('q_wrong',),'both_wrong':('u_wrong','q_wrong')}

def write_csv(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def models(study,data,source=None):
    plan=C.check_plan(study/'plan',data)
    ckpt=source or C.P.ROOT/C.P.source_spec()['weight_path']
    if C.P.sha(ckpt)!=C.P.SOURCE_SHA:raise ValueError('source checkpoint mismatch')
    return [dict(arm='initial',path=str(ckpt),sha256=C.P.SOURCE_SHA)]+E.load_models(study,plan)

def graph_structure(labels,info):
    pos=[p for p,x in enumerate(info) if x[0]>=0 and 0<=labels[p][3]<23]
    depth={p:int(info[p][0]) for p in pos}
    parents={p:[] if depth[p]==0 else [int(x) for x in info[p][1::2] if x!=-1] for p in pos}
    if any(q not in depth for ps in parents.values() for q in ps):raise ValueError('parent is not node')
    return pos,depth,parents

def node_groups(pos,depth,parents):
    children={p:[] for p in pos}
    for q,ps in parents.items():
        for p in set(ps):children[p].append(q)
    groups={'all':pos,'leaf':[p for p in pos if depth[p]==0],
            'internal':[p for p in pos if depth[p]>0 and children[p]],
            'sink':[p for p in pos if depth[p]>0 and not children[p]]}
    for d in sorted(set(depth.values())):groups[f'depth_{d}']=[p for p in pos if depth[p]==d]
    for n in sorted({len(parents[p]) for p in pos}):groups[f'fanin_{n}']=[p for p in pos if len(parents[p])==n]
    return groups

def pair_plan(case,labels,info):
    """Keep original structural u; select one q without model-dependent filtering."""
    if case['status']!='selected':return dict(status=case['status'])
    pos,depth,parents=graph_structure(labels,info)
    coefs=C.DESIGN.coefficients(parents,depth)
    qs=sorted(case['children'],key=lambda q:C.stable_seed('pair-q',case['graph_id'],q))
    for q in qs:
        # q wrong value matches substitution of wrong u, other parents correct.
        delta=(parents[q].count(case['target'])*case['delta'])%23
        if not delta:continue
        edits=C.DESIGN.matched_changes(coefs[q],(delta,))
        if edits is None:continue
        donor_case=dict(case,target=q,delta=delta,changes=edits)
        return dict(status='selected',u=case['target'],q=q,delta_u=case['delta'],delta_q=delta,
                    u_case=case,q_case=donor_case,positions=pos,depths=depth,parents=parents,
                    expected_u=(labels[case['target']][3]+case['delta'])%23,
                    expected_q=(labels[q][3]+delta)%23)
    return dict(status='no_q_counterfactual')

def input_parent_sum(row,positions,parents,node):
    ps=parents.get(str(node),parents.get(node,[]))
    if not ps:return None
    vals=[row[positions.index(p)][3] for p in ps]
    return sum(vals)%23 if all(0<=x<23 for x in vals) else None

def first_true(xs):return next((i+1 for i,x in enumerate(xs) if x),None)

def clean_events(flags,d):
    """All-graph completion and post-D events have separate denominators."""
    first=first_true(flags);bad=next((i for i in range(first,len(flags)) if not flags[i]),None) if first else None
    recovered=next((i for i in range(bad+1,len(flags)) if flags[i]),None) if bad is not None else None
    return dict(first_all=first,post_completion_observed=first is not None and first<len(flags),
        failure_after_completion=bad is not None,recovered_after_failure=recovered is not None,
        reerror=recovered is not None and not all(flags[recovered:]),
        all_at_D=bool(flags[d-1]),postD_failure=bool(not all(flags[d:])) if flags[d-1] else None,
        final_two_all=bool(all(flags[-2:])))


def make_branches(sources,u,q,names):
    import torch
    rows={'u_original':sources[0,u],'q_original':sources[0,q]}
    for name in names[1:]:rows[name]=sources[names.index(name),u if name.startswith('u_') else q]
    states=[];norms={}
    for branch,changes in BRANCHES.items():
        state=sources[0:1].clone();norm=0.
        for key in changes:
            pos=u if key.startswith('u_') else q
            norm+=float((rows[key]-state[0,pos]).square().sum());state[0,pos]=rows[key]
        states.append(state);norms[branch]=norm**0.5
    return torch.cat(states),norms
