#!/usr/bin/env python3
"""Parent/child 2x2 metrics and decoded input/output evidence, finite H16."""
import argparse,sys,json,csv
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import suite_core as U
C=U.C

def pair_trajectory(r,branch):
    pos=r['positions'];gold=r['truth4'];ui=pos.index(r['u']);qi=pos.index(r['q'])
    b=r['branches'][branch];seq=[b['initial']]+b['outputs'];base=r['branches']['unchanged']['outputs'];rows=[]
    for t in range(1,len(seq)):
        ip,op=seq[t-1],seq[t]
        su=U.input_parent_sum(ip,pos,r['parents'],r['u']);sq=U.input_parent_sum(ip,pos,r['parents'],r['q'])
        rows.append(dict(arm=r['arm'],graph_id=r['graph_id'],timing=r['timing'],branch=branch,round=t,
            u=r['u'],q=r['q'],u_truth=gold[ui][3],q_truth=gold[qi][3],
            u_input=ip[ui][3],u_output=op[ui][3],q_input=ip[qi][3],q_output=op[qi][3],
            u_correct=op[ui][3]==gold[ui][3],q_correct=op[qi][3]==gold[qi][3],
            u_input_parent_sum=su,q_input_parent_sum=sq,
            u_matches_input_parent_readouts=op[ui][3]==su if su is not None else None,
            q_matches_input_parent_readouts=op[qi][3]==sq if sq is not None else None,
            clean_u=base[t-1][ui][3],clean_q=base[t-1][qi][3]))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();meta=C.P.read_json(a.run_dir/'meta.json')
    if meta['status']!='complete':raise ValueError('incomplete pair run')
    records=C.read_cases(a.run_dir/'records.jsonl');arms={s['arm'] for s in meta['models']};cases=defaultdict(dict)
    for r in records:
        key=(r['graph_id'],r['timing'])
        if r['arm'] in cases[key]:raise ValueError('duplicate case')
        cases[key][r['arm']]=r
    if any(set(x)!=arms for x in cases.values()):raise ValueError('unpaired case')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    curves=defaultdict(list);events=defaultdict(list);trajectory=[];coverage=[];dose=[]
    for key,rr in cases.items():
        common=all(r['value_qualified'] for r in rr.values());strict=all(r['strict_qualified'] for r in rr.values())
        for arm,r in rr.items():
            pos=r['positions'];gold=r['truth4'];u,q=r['u'],r['q'];ui=pos.index(u);qi=pos.index(q)
            coverage.append(dict(arm=arm,graph_id=key[0],timing=key[1],value_qualified=r['value_qualified'],
                strict_qualified=r['strict_qualified'],common_value=common,common_strict=strict,
                start_graph_value_all=r['start_graph_value_all'],start_graph_first4_all=r['start_graph_first4_all']))
            groups={'u':[u],'q':[q],'other_children':[p for p in r['children'] if p!=q],
                    'descendants':r['descendants'],'unrelated':sorted(set(pos)-{u}-set(r['descendants'])),'all':pos}
            cohorts=['all_fixed']+(['common_value'] if common else [])+(['common_strict'] if strict else [])
            clean=[r['branches']['unchanged']['initial']]+r['branches']['unchanged']['outputs']
            for branch,b in r['branches'].items():
                trajectory.extend(pair_trajectory(r,branch));seq=[b['initial']]+b['outputs']
                dose.append(dict(arm=arm,graph_id=key[0],timing=key[1],branch=branch,norm=r['norms'][branch],
                    u_changed=seq[0][ui][3]!=clean[0][ui][3],q_changed=seq[0][qi][3]!=clean[0][qi][3],
                    u_input_wrong=seq[0][ui][3]!=gold[ui][3],q_input_wrong=seq[0][qi][3]!=gold[qi][3]))
                for metric in ('value','first4'):
                    ok=lambda row,i:row[i][3]==gold[i][3] if metric=='value' else row[i]==gold[i]
                    for group,ps in groups.items():
                        if not ps:continue
                        local=[U.S.trajectory_event([ok(row,pos.index(p0)) for row in seq],[ok(row,pos.index(p0)) for row in clean]) for p0 in ps]
                        damaged=[x for x in local if x['damaged']]
                        event=dict(n=len(ps),damaged=len(damaged),ever_recovered=sum(x['ever_recovered'] for x in damaged),
                            reerror=sum(x['recovered_then_error'] for x in damaged),final_two=sum(x['final_two_correct'] for x in damaged),
                            excess_wrong=sum(x['excess_wrong_rounds'] for x in local))
                        for cohort in cohorts:
                            events[(arm,key[1],branch,cohort,group,metric)].append(event)
                            for t in (1,2,4,8,16):
                                curves[(arm,key[1],branch,cohort,group,metric,t)].append(dict(n=len(ps),
                                    correct=sum(ok(seq[t],pos.index(p0)) for p0 in ps),
                                    harmed=sum(not ok(seq[t],pos.index(p0)) and ok(clean[t],pos.index(p0)) for p0 in ps),
                                    helped=sum(ok(seq[t],pos.index(p0)) and not ok(clean[t],pos.index(p0)) for p0 in ps)))
    U.write_csv(a.out_dir/'joint_trajectories.csv',trajectory);U.write_csv(a.out_dir/'coverage.csv',coverage);U.write_csv(a.out_dir/'dose.csv',dose)
    fields=['arm','timing','branch','cohort','group','metric'];rows=[]
    for key,rs in sorted(curves.items()):
        n=sum(r['n'] for r in rs);correct=sum(r['correct'] for r in rs)
        rows.append(dict(zip(fields+['round'],key),graphs=len(rs),n=n,correct=correct,node_acc=correct/n,
            graph_equal=sum(r['correct']/r['n'] for r in rs)/len(rs),harmed=sum(r['harmed'] for r in rs),helped=sum(r['helped'] for r in rs)))
    U.write_csv(a.out_dir/'accuracy.csv',rows);rows=[]
    for key,rs in sorted(events.items()):
        ds=[r for r in rs if r['damaged']]
        rows.append(dict(zip(fields,key),graphs=len(rs),observed=sum(r['n'] for r in rs),damaged=sum(r['damaged'] for r in rs),
            ever_recovered=sum(r['ever_recovered'] for r in rs),reerror=sum(r['reerror'] for r in rs),
            final_two_among_damaged=sum(r['final_two'] for r in rs),graphs_with_damage=len(ds),
            final_two_graph_equal=sum(r['final_two']/r['damaged'] for r in ds)/len(ds) if ds else None,
            excess_wrong_graph_equal=sum(r['excess_wrong']/r['n'] for r in rs)/len(rs)))
    U.write_csv(a.out_dir/'events.csv',rows)
    C.P.write_json(a.out_dir/'meta.json',dict(status='complete',source_meta=meta,code_sha256=C.P.sha(__file__),
        records_sha256=C.P.sha(a.run_dir/'records.jsonl'),common_value_cases=sum(all(r['value_qualified'] for r in rr.values()) for rr in cases.values()),
        note='Decoded parent consistency is not proof that transformer uses decoded values; finite H16 only.'))
if __name__=='__main__':main()
