#!/usr/bin/env python3
"""CPU summaries: common fixed cohorts, dose audit, clean and damaged denominators."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pilot_core as C

def write_csv(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def trajectory_event(correct,reference):
    """Index0 is intervention input, index1 first forward; finite-window evidence."""
    if len(correct)!=len(reference) or len(correct)<3:raise ValueError('aligned input+outputs required')
    damage=[i for i,(x,b) in enumerate(zip(correct,reference)) if b and not x]
    first=damage[0] if damage else None
    recovery=next((i for i in range(first+1,len(correct)) if correct[i]),None) if first is not None else None
    stable=None
    if correct[-1]:
        stable=len(correct)-1
        while stable>0 and correct[stable-1]:stable-=1
    return dict(damaged=first is not None,first_damage=first,first_recovery=recovery,
                ever_recovered=recovery is not None,
                recovered_then_error=recovery is not None and not all(correct[recovery:]),
                last_stable_start=stable,final_two_correct=all(correct[-2:]),
                wrong_rounds=sum(not x for x in correct[1:]),
                excess_wrong_rounds=sum(b and not x for x,b in zip(correct[1:],reference[1:])))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--study-dir',type=Path,default=C.STUDY)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();meta=C.P.read_json(a.eval_dir/'meta.json')
    if meta['status']!='complete':raise ValueError('incomplete evaluation')
    arms={m['arm'] for m in meta['models']}
    records=C.read_cases(a.eval_dir/'records.jsonl');bycase=defaultdict(list)
    seen=set()
    for r in records:
        identity=(r['arm'],r['graph_id'],r['timing'])
        if identity in seen:raise ValueError('duplicate records')
        seen.add(identity);bycase[(r['graph_id'],r['timing'])].append(r)
    if any({r['arm'] for r in rs}!=arms for rs in bycase.values()):raise ValueError('missing paired model')
    common={key:all(r['value_qualified'] for r in rs) for key,rs in bycase.items()}
    strict={key:all(r['strict_qualified'] for r in rs) for key,rs in bycase.items()}
    a.out_dir.mkdir(parents=True,exist_ok=False)
    per_case=[];node_events=[];dose=[]
    for r in records:
        key=(r['graph_id'],r['timing']);pos=r['positions'];gold=r['truth4'];u=r['target']
        groups=dict(target=[u],children=r['children'],descendants=r['descendants'],
                    unrelated=sorted(set(pos)-{u}-set(r['descendants'])),all=pos)
        cohorts=['all_fixed']+(['common_value'] if common[key] else [])+(['common_strict'] if strict[key] else [])
        base=[r['branches']['unchanged']['initial']]+r['branches']['unchanged']['outputs']
        for branch,data in r['branches'].items():
            seq=[data['initial']]+data['outputs'];ui=pos.index(u)
            dose.append(dict(arm=r['arm'],graph_id=r['graph_id'],timing=r['timing'],branch=branch,
                before_correct=base[0][ui][3]==gold[ui][3],input_wrong=seq[0][ui][3]!=gold[ui][3],
                input_intended=seq[0][ui][3]==(gold[ui][3]+r['delta'])%23,
                input_nonnumeric=not 0<=seq[0][ui][3]<23,first_round_correct=seq[1][ui][3]==gold[ui][3],
                value_qualified=r['value_qualified'],common_value=common[key],**r['distances'][branch]))
            for metric in ('value','first4'):
                ok=lambda row,i:row[i][3]==gold[i][3] if metric=='value' else row[i]==gold[i]
                for cohort in cohorts:
                    for group,ps in groups.items():
                        for t in (1,2,4,8,16):
                            correct=sum(ok(seq[t],pos.index(q)) for q in ps)
                            per_case.append(dict(arm=r['arm'],graph_id=r['graph_id'],timing=r['timing'],branch=branch,
                                cohort=cohort,group=group,metric=metric,round=t,n=len(ps),correct=correct,
                                helped=sum(ok(seq[t],pos.index(q)) and not ok(base[t],pos.index(q)) for q in ps),
                                harmed=sum(not ok(seq[t],pos.index(q)) and ok(base[t],pos.index(q)) for q in ps),
                                nonnumeric=sum(not 0<=seq[t][pos.index(q)][3]<23 for q in ps),
                                expected_wrong=sum(seq[t][pos.index(q)][3]==r['expected_q'][str(q)] for q in ps) if group=='children' else 0))
                        for q in ps:
                            i=pos.index(q);ev=trajectory_event([ok(row,i) for row in seq],[ok(row,i) for row in base])
                            node_events.append(dict(arm=r['arm'],graph_id=r['graph_id'],timing=r['timing'],branch=branch,
                                cohort=cohort,group=group,metric=metric,pos=q,**ev))
    write_csv(a.out_dir/'case_rounds.csv',per_case);write_csv(a.out_dir/'node_events.csv',node_events)
    write_csv(a.out_dir/'evaluation_dose.csv',dose)
    summary=[];buckets=defaultdict(list)
    fields=['arm','timing','branch','cohort','group','metric','round']
    for r in per_case:buckets[tuple(r[k] for k in fields)].append(r)
    for key,rs in sorted(buckets.items()):
        n=sum(r['n'] for r in rs);valid=[r for r in rs if r['n']]
        summary.append(dict(zip(fields,key),graphs=len(rs),n=n,correct=sum(r['correct'] for r in rs),
            node_acc=sum(r['correct'] for r in rs)/n if n else None,
            graph_equal=sum(r['correct']/r['n'] for r in valid)/len(valid) if valid else None,
            helped=sum(r['helped'] for r in rs),harmed=sum(r['harmed'] for r in rs),
            nonnumeric=sum(r['nonnumeric'] for r in rs),expected_wrong=sum(r['expected_wrong'] for r in rs)))
    write_csv(a.out_dir/'intervention_summary.csv',summary)
    buckets=defaultdict(list);fields=fields[:-1];summary=[]
    for r in node_events:buckets[tuple(r[k] for k in fields)].append(r)
    for key,rs in sorted(buckets.items()):
        damaged=[r for r in rs if r['damaged']];graphs=defaultdict(list);allgraphs=defaultdict(list)
        for r in rs:allgraphs[r['graph_id']].append(r)
        for r in damaged:graphs[r['graph_id']].append(r)
        summary.append(dict(zip(fields,key),observed=len(rs),damaged=len(damaged),
            ever_recovered=sum(r['ever_recovered'] for r in damaged),
            recovered_then_error=sum(r['recovered_then_error'] for r in damaged),
            final_two_correct_among_damaged=sum(r['final_two_correct'] for r in damaged),
            graphs_with_damage=len(graphs),
            final_two_graph_equal=sum(sum(r['final_two_correct'] for r in xs)/len(xs) for xs in graphs.values())/len(graphs) if graphs else None,
            mean_excess_wrong_rounds=sum(r['excess_wrong_rounds'] for r in rs)/len(rs) if rs else None,
            graph_equal_excess_wrong_rounds=sum(sum(r['excess_wrong_rounds'] for r in xs)/len(xs) for xs in allgraphs.values())/len(allgraphs) if allgraphs else None))
    write_csv(a.out_dir/'event_summary.csv',summary)
    # Clean snapshots use all validation graphs, never intervention qualification.
    with (a.eval_dir/'clean_curves.csv').open(encoding='utf-8') as f:
        clean=list(csv.DictReader(f))
    buckets=defaultdict(list)
    for r in clean:
        d=int(r['D']);t=int(r['round'])
        for mark,tt in C.P.evaluation_points(d).items():
            if t==tt:buckets[(r['arm'],mark)].append(r)
    rows=[]
    for (arm,mark),rs in sorted(buckets.items()):
        n=sum(int(r['n']) for r in rs)
        row=dict(arm=arm,time=mark,graphs=len(rs),n=n)
        for field in ('value','first4','syntax','variable','operation'):
            cnt=sum(int(r[field+'_correct']) for r in rs);row[field+'_correct']=cnt;row[field+'_acc']=cnt/n
        rows.append(row)
    write_csv(a.out_dir/'clean_summary.csv',rows)
    training=[];budgets=[]
    if meta['cohort']=='main':
        for arm in C.ARMS:
            m=C.P.read_json(a.study_dir/arm/'meta.json');logs=C.read_cases(a.study_dir/arm/'injections.jsonl')
            if len(logs)!=m['budget']['scheduled_events']:raise ValueError('incomplete injection log')
            budgets.append(dict(arm=arm,**m['budget'],wall_seconds=m['wall_seconds'],checkpoint_sha256=m['checkpoint']['sha256']))
            for epoch in range(1,C.EPOCHS+1):
                rs=[r for r in logs if r['epoch']==epoch]
                training.append(dict(arm=arm,epoch=epoch,scheduled=len(rs),injected=len(rs)*(arm!='hold'),
                    before_value_correct=sum(r['before_value_correct'] for r in rs),
                    donor_value_intended=sum(r['donor_value_intended'] for r in rs),
                    donor_first4_intended=sum(r['donor_first4_intended'] for r in rs),
                    after_value_wrong=sum(r['after_value_wrong'] for r in rs),
                    changed_readout=sum(r['actual_readout_changed'] for r in rs),
                    avg_semantic_reference_norm=sum(r['semantic_reference_norm'] for r in rs)/len(rs) if rs else None))
        write_csv(a.out_dir/'training_dose.csv',training)
        write_csv(a.out_dir/'budget.csv',budgets)
    C.P.write_json(a.out_dir/'meta.json',dict(status='complete',cohort=meta['cohort'],eval_meta=meta,
        common_value_cases=sum(common.values()),common_strict_cases=sum(strict.values()),
        code_sha256=C.P.sha(__file__),note='Recovery conditional on paired damage; finite H16 only. Empty sets are NA, not zero success.'))
if __name__=='__main__':main()
