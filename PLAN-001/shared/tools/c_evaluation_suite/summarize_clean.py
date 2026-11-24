#!/usr/bin/env python3
"""CPU clean capability summary; all graphs and post-completion cohorts separate."""
import argparse,sys,csv,json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import suite_core as U
C=U.C

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    meta=C.P.read_json(a.run_dir/'meta.json')
    if meta['status']!='complete':raise ValueError('incomplete clean run')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    buckets=defaultdict(list);curve_buckets=defaultdict(list);event_rows=[];seen=set()
    for row in C.read_cases(a.run_dir/'index.jsonl'):
        identity=(row['arm'],row['stratum'],row['graph_id'])
        if identity in seen:raise ValueError('duplicate graph')
        seen.add(identity)
        with np.load(a.run_dir/row['path']) as z:
            pred=z['predictions'];gold=z['truth4'];positions=z['positions'].tolist();d=int(z['D'])
        if len(pred)!=d+64:raise ValueError('wrong clean horizon')
        ok=pred==gold[None,:,:]
        matrices={'value':ok[:,:,3],'first4':ok.all(-1),'syntax':ok[:,:,0],'variable':ok[:,:,1],'operation':ok[:,:,2]}
        for group,ps in row['groups'].items():
            if not ps:continue
            indices=[positions.index(q) for q in ps]
            for metric in (matrices if group=='all' else ('value','first4')):
                matrix=matrices[metric][:,indices];flags=matrix.all(-1).tolist()
                ev=U.clean_events(flags,d)
                # Do not count accidental early matches before a node's supervised ready round.
                node_events=[U.clean_events(matrix[int(row['depths'][str(p0)]):,i].tolist(),
                    d-int(row['depths'][str(p0)])) for i,p0 in enumerate(ps)]
                event_rows.append(dict(arm=row['arm'],stratum=row['stratum'],graph_id=row['graph_id'],group=group,metric=metric,n=len(ps),
                    first_90=U.first_true((matrix.mean(-1)>=0.90).tolist()),first_99=U.first_true((matrix.mean(-1)>=0.99).tolist()),
                    **ev,nodes_ever_correct=sum(x['first_all'] is not None for x in node_events),
                    nodes_post_correct_observed=sum(x['post_completion_observed'] for x in node_events),
                    nodes_correct_then_wrong=sum(x['failure_after_completion'] for x in node_events),
                    nodes_recovered_after_failure=sum(x['recovered_after_failure'] for x in node_events)))
                for mark,t in C.P.evaluation_points(d).items():
                    count=int(matrix[t-1].sum())
                    buckets[(row['arm'],row['stratum'],group,metric,mark)].append((count,len(ps)))
                if group in ('all','leaf','internal','sink'):
                    times=[('absolute',t,t) for t in range(1,65)]+[('relative_D',r,d+r) for r in range(65)]
                    for kind,r,t in times:
                        curve_buckets[(row['arm'],row['stratum'],group,metric,kind,r)].append((int(matrix[t-1].sum()),len(ps)))
    expected=len(meta['models'])*len(U.STRATA)*meta['candidate_graphs_per_stratum']
    if len(seen)!=expected:raise ValueError('incomplete clean index')
    fields=['arm','stratum','group','metric','time'];summary=[]
    for key,rs in sorted(buckets.items()):
        n=sum(n for c,n in rs);correct=sum(c for c,n in rs)
        summary.append(dict(zip(fields,key),graphs=len(rs),correct=correct,n=n,node_acc=correct/n,
            graph_equal=sum(c/n for c,n in rs)/len(rs),graph_all=sum(c==n for c,n in rs)))
    U.write_csv(a.out_dir/'accuracy.csv',summary)
    curves=[]
    for key,rs in sorted(curve_buckets.items()):
        n=sum(n for c,n in rs);correct=sum(c for c,n in rs)
        curves.append(dict(zip(['arm','stratum','group','metric','time_basis','round'],key),graphs=len(rs),n=n,correct=correct,
            node_acc=correct/n,graph_equal=sum(c/n for c,n in rs)/len(rs),graph_all=sum(c==n for c,n in rs)))
    U.write_csv(a.out_dir/'curves.csv',curves)
    U.write_csv(a.out_dir/'graph_events.csv',event_rows)
    groups=defaultdict(list)
    for r in event_rows:groups[(r['arm'],r['stratum'],r['group'],r['metric'])].append(r)
    summary=[]
    for key,rs in sorted(groups.items()):
        observed=[r for r in rs if r['post_completion_observed']];atd=[r for r in rs if r['all_at_D']]
        completed=[r['first_all'] for r in rs if r['first_all'] is not None]
        summary.append(dict(zip(fields[:-1],key),graphs=len(rs),completed=len(completed),never_completed=len(rs)-len(completed),
            median_first_all_among_completed=float(np.median(completed)) if completed else None,
            post_completion_observed=len(observed),failure_after_completion=sum(r['failure_after_completion'] for r in observed),
            recovered_after_failure=sum(r['recovered_after_failure'] for r in observed),reerror=sum(r['reerror'] for r in observed),
            all_at_D=len(atd),postD_failure_among_all_at_D=sum(r['postD_failure'] for r in atd),
            final_two_all=sum(r['final_two_all'] for r in rs),
            nodes_ever_correct=sum(r['nodes_ever_correct'] for r in rs),
            nodes_post_correct_observed=sum(r['nodes_post_correct_observed'] for r in rs),
            nodes_correct_then_wrong=sum(r['nodes_correct_then_wrong'] for r in rs),
            nodes_recovered_after_failure=sum(r['nodes_recovered_after_failure'] for r in rs)))
    U.write_csv(a.out_dir/'stability.csv',summary)
    C.P.write_json(a.out_dir/'meta.json',dict(status='complete',source_meta=meta,code_sha256=C.P.sha(__file__),
        note='Completion medians conditional on completion; never_completed reported. Fixed-time metrics include every graph.'))
if __name__=='__main__':main()
