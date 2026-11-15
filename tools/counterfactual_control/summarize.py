#!/usr/bin/env python3
"""Read completed matched-donor runs; preserve common q denominators and strata."""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import design
sys.path.insert(0,str(HERE.parent/'repair_study'))
import core


def save(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True,help='records.jsonl files')
    p.add_argument('--out_dir',type=Path,required=True)
    a=p.parse_args()
    coverage,curves,effects,refs,provenance=[],[],[],[],[]
    seen=set()
    for path in a.inputs:
        meta=core.read_json(path.with_name('meta.json'))
        if meta.get('status')!='complete' or meta.get('protocol')!='matched_counterfactual_v1':
            raise ValueError('completed matched-counterfactual run required')
        provenance.append(dict(path=str(path),sha256=core.sha(path),meta=meta))
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            key=(r['weight'],r['case_id'])
            if key in seen:
                raise ValueError('duplicate weight/case; do not mix smoke and main')
            seen.add(key)
            base={k:r[k] for k in ['weight','case_id','graph_id','stratum','fan_in','depth_band','depth','original_value','state_round']}
            base['same_vector_equal']=r['same_vector_equal']
            coverage.append(dict(**base,status=r['status'],eligible_children=len(r['eligible_children']),
                source_flags=json.dumps(r['source_qualification'],sort_keys=True),
                vector_distances=json.dumps(r['vector_distance'],sort_keys=True)))
            if r['status']!='eligible':
                continue
            for d in r['deltas']:
                bybranch={}
                for branch in ['unchanged','sham','same',f'shift_{d}']:
                    bybranch[branch]=design.branch_metrics(r,branch,d)
                    for row in bybranch[branch]:
                        curves.append(dict(**base,delta=d,branch=branch,**row))
                for same,different in zip(bybranch['same'],bybranch[f'shift_{d}']):
                    n=same['q_n']
                    effects.append(dict(**base,delta=d,round=same['round'],q_n=n,
                        same_expected_count=same['q_expected_shift'],
                        different_expected_count=different['q_expected_shift'],
                        expected_difference=different['q_expected_shift']-same['q_expected_shift'],
                        same_correct=same['q_correct'],different_correct=different['q_correct'],
                        same_first4=same['q_first4_correct'],different_first4=different['q_first4_correct'],
                        same_changed=same['q_changed_vs_unchanged'],different_changed=different['q_changed_vs_unchanged'],
                        same_harmed=same['q_harmed'],different_harmed=different['q_harmed']))
            # Full donor reference runs use full donor truths, never the local replacement target.
            for name,trace in r['references'].items():
                own_truth=r['donor_truth'][name.removeprefix('full_')]
                ids={q:r['node_positions'].index(q) for q in r['child_positions']}
                for t,row in enumerate(trace['outputs'],1):
                    refs.append(dict(**base,reference=name,round=t,
                        target_correct=int(row[r['node_positions'].index(r['target_pos'])][3]==own_truth[str(r['target_pos'])]),
                        q_n=len(ids),q_correct=sum(row[i][3]==own_truth[str(q)] for q,i in ids.items())))
    # Each case has >=1 qualified q. Average q within case, cases within graph,
    # then graphs equally. Keep zero-vector same donors in a separate cohort.
    grouped=defaultdict(list)
    for row in effects:
        for view in ['all','fan_in','depth_band','original_value']:
            level='all' if view=='all' else str(row[view])
            k=(row['weight'],row['stratum'],row['same_vector_equal'],view,level,row['delta'],row['round'])
            grouped[k].append(row)
    summary=[]
    for key,pool in sorted(grouped.items()):
        graphs=defaultdict(list)
        for r in pool:
            graphs[r['graph_id']].append(r)
        row=dict(zip(['weight','stratum','same_vector_equal','view','level','delta','round'],key))
        row.update(cases=len(pool),graphs=len(graphs),q_observations=sum(r['q_n'] for r in pool))
        for metric in ['same_expected_count','different_expected_count','expected_difference',
                       'same_correct','different_correct','same_first4','different_first4',
                       'same_changed','different_changed','same_harmed','different_harmed']:
            row[metric]=sum(r[metric] for r in pool)
            row[metric+'_graph_equal']=sum(sum(r[metric]/r['q_n'] for r in g)/len(g) for g in graphs.values())/len(graphs)
        summary.append(row)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in [('coverage',coverage),('case_curves',curves),('paired_effects',effects),('summary',summary),('full_reference_curves',refs)]:
        save(a.out_dir/(name+'.csv'),rows)
    core.write_json(a.out_dir/'meta.json',dict(inputs=provenance,cases=len(coverage),
        eligible=sum(r['status']=='eligible' for r in coverage),
        notes=['Conditional matched-source sample, not overall accuracy.',
               'Candidate selection and no-candidate graphs are in source selection.jsonl/meta.',
               'q_n is fixed by source and unchanged-branch checks, never treatment success.',
               'A child may later use changed other parents; local expected value is a diagnostic signature, not causal proof.',
               'Vector edit positions/count are matched; vector distances are measured, not matched.']))


if __name__=='__main__':
    main()
