"""Streaming, per-graph equal-weight summaries; counts retain missing/failed cases."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def avg(xs):return sum(xs)/len(xs) if xs else None

def aggregate(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r.get(k) for k in keys)].append(r)
    out=[]
    for key,rr in sorted(groups.items(),key=lambda z:str(z[0])):
        names=set().union(*(r.keys() for r in rr))-set(keys)-{'gid'}
        record=dict(zip(keys,key));record['graphs']=len({r['gid'] for r in rr})
        for n in sorted(names):
            vals=[r[n] for r in rr if r.get(n) is not None]
            record[n]=avg(vals)
        out.append(record)
    return out

def summarize(records):
    clean=[];stratified=[];paired=[];coverage=defaultdict(int);counts=defaultdict(lambda:defaultdict(int));update=defaultdict(lambda:defaultdict(int));reuse=[]
    for r in records:
        module=r['module'];split=r['split'];condition=r.get('condition','none');status=r['status']
        coverage[(split,module,condition,status)]+=1
        if status!='ok':continue
        ns=r['nodes'];truth=r['truth'];rows=r['rows'];gid=r['gid'];D=r['D']
        if module=='clean':
            horizons={'D':D,'D+4':D+4,'D+16':D+16,'D+64':D+64}
            for h,t in horizons.items():
                if t>len(rows):continue
                output=rows[t-1]['output'];ok=[v[0]==y[3] for v,y in zip(output,truth)]
                clean.append(dict(split=split,gid=gid,horizon=h,mean_value=avg(ok),mean_first4=avg([v[1] for v in output]),
                                  mean_syntax=avg([v[2] for v in output]),mean_variable=avg([v[3] for v in output]),mean_operation=avg([v[4] for v in output]),
                                  graph_value_all=float(all(ok)),graph_first4_all=float(all(v[1] for v in output))))
                children={p for ps in r['parents'] for p in ps}
                bins=defaultdict(list)
                for i,p in enumerate(ns):
                    bins[('depth',r['depths'][i])].append(i);bins[('fanin',len(r['parents'][i]))].append(i)
                    bins[('role','leaf' if not r['parents'][i] else 'internal' if p in children else 'sink')].append(i)
                for (kind,value),indices in bins.items():
                    stratified.append(dict(split=split,gid=gid,horizon=h,kind=kind,bin=str(value),
                                           mean_value=avg([ok[i] for i in indices]),mean_first4=avg([output[i][1] for i in indices])))
            for a in rows:
                if 'update' in a:
                    for z in a['update']:
                        u=update[(split,'clean')];u['observed']+=1
                        for k in ('reference','plan','changed','value_correct','cleared'):u[k]+=int(z[k])
                        u['tp']+=int(z['reference'] and z['plan']);u['fp']+=int(not z['reference'] and z['plan']);u['fn']+=int(z['reference'] and not z['plan'])
                        u['plan_execution_agree']+=int(z['plan']==z['changed'])
            continue
        if module=='reuse':
            idx=ns.index(r['new_node']);correct=[s['output'][idx][0]==truth[idx][3] for s in rows]
            reuse.append(dict(split=split,gid=gid,condition=condition,lag=r['lag'],
                              input_u_first4=float(r['u_first4_correct']),new_node_first=float(correct[0]),
                              new_node_last_two=float(all(correct[-2:]))))
            continue
        timing=r.get('timing');hold=r.get('hold',1);noise=r.get('noise') or {}
        key=(split,module,timing,condition,hold,noise.get('scale'),noise.get('direction'))
        release=hold if hold>1 else 0
        cleanrows=r['clean']
        if len(cleanrows)!=len(rows):raise ValueError('clean/branch lengths differ')
        qidx=ns.index(r['q']);qparents=r['parents'][qidx];indices={p:i for i,p in enumerate(ns)}
        for t,(a,b) in enumerate(zip(rows,cleanrows)):
            ps=[a['input'][indices[p]][3] for p in qparents]
            wrong_parent=any(v!=truth[indices[p]][3] for v,p in zip(ps,qparents))
            if ps and all(0<=v<23 for v in ps) and wrong_parent:
                expected=sum(ps)%23
                counts[key]['q_wrong_parent_numeric_rounds']+=1
                counts[key]['q_matches_wrong_parent_sum']+=int(a['output'][qidx][0]==expected and expected!=truth[qidx][3])
            if 'update' in a:
                for z in a['update']:
                    u=update[key];u['observed']+=1
                    for k in ('reference','plan','changed','value_correct','cleared'):u[k]+=int(z[k])
                    u['tp']+=int(z['reference'] and z['plan']);u['fp']+=int(not z['reference'] and z['plan']);u['fn']+=int(z['reference'] and not z['plan'])
                    u['plan_execution_agree']+=int(z['plan']==z['changed'])
        roles={'all':ns,'direct':r['positions'],'u':[r['u']],'q':[r['q']],
               'downstream':r['downstream'],'unrelated':[p for p in ns if p not in {r['u'],*r['downstream']}]}
        for role,positions in roles.items():
            if not positions:continue
            for metric in ('value','first4'):
                observed=damaged=stable=first_round_repair=actual_injected=reerror=0
                first_restore=[];stable_start=[];correct_at=defaultdict(list)
                for p in positions:
                    i=indices[p];correct=lambda a:(a[0]==truth[i][3]) if metric=='value' else bool(a[1])
                    flags=[correct(z['output'][i]) for z in rows]
                    cleanflags=[correct(z['output'][i]) for z in cleanrows]
                    actual_bad=(rows[0]['input'][i][3]!=truth[i][3] if metric=='value' else rows[0]['input'][i]!=truth[i])
                    clean_input_good=(cleanrows[0]['input'][i][3]==truth[i][3] if metric=='value' else cleanrows[0]['input'][i]==truth[i])
                    direct_bad=p in r['positions'] and actual_bad and clean_input_good
                    observed+=1;actual_injected+=int(direct_bad);first_round_repair+=int(direct_bad and flags[0])
                    bad=[t for t,(a,b) in enumerate(zip(flags,cleanflags)) if not a and b]
                    was_damaged=bool(bad) or direct_bad;damaged+=int(was_damaged)
                    onset=-1 if direct_bad else min(bad) if bad else None
                    # Stable = last TWO RELEASED observations correct, after damage.
                    stable+=int(was_damaged and len(flags)-2>=release and onset<len(flags)-2 and all(flags[-2:]))
                    if onset is not None:
                        hf=P.horizon_flags(flags[onset+1:]);reerror+=int(hf['reerror'])
                        if hf['first'] is not None:first_restore.append(onset+hf['first']+1)
                        if hf['stable'] is not None:stable_start.append(onset+hf['stable']+1)
                    for hh in (1,2,4,8,16):
                        if release+hh<=len(flags):correct_at[hh].append(flags[release+hh-1])
                k=(*key,role,metric);c=counts[k]
                for n,v in dict(observed=observed,damaged=damaged,stable=stable,actual_injected=actual_injected,
                                first_round_repair=first_round_repair,reerror=reerror).items():c[n]+=v
                paired.append(dict(split=split,module=module,timing=timing,condition=condition,hold=hold,
                                   scale=noise.get('scale'),direction=noise.get('direction'),role=role,metric=metric,gid=gid,
                                   damage_rate=damaged/observed,stable_given_damaged=stable/damaged if damaged else None,
                                   first_repair_given_injected=first_round_repair/actual_injected if actual_injected else None,
                                   mean_first_restore_round=avg(first_restore),mean_stable_start_round=avg(stable_start),
                                   **{f'correct_r{h}':avg(v) for h,v in correct_at.items()},
                                   input_qualified=float(r['input_qualified'])))
    pairkeys=['split','module','timing','condition','hold','scale','direction','role','metric']
    def encoded(table):return [dict(key=list(k),counts=dict(v)) for k,v in sorted(table.items(),key=lambda z:str(z[0]))]
    return dict(clean=aggregate(clean,['split','horizon']),stratified=aggregate(stratified,['split','horizon','kind','bin']),paired=aggregate(paired,pairkeys),
                counts=encoded(counts),update=encoded(update),reuse=aggregate(reuse,['split','condition','lag']),
                coverage=[dict(split=k[0],module=k[1],condition=k[2],status=k[3],count=v) for k,v in sorted(coverage.items())],
                note='macro = equal graph weights within each fixed condition; raw counts separate. Stable means last two observed released rounds, not permanence. Missing conditional denominator => null.')

def summarize_file(src,dest):
    with Path(src).open(encoding='utf-8') as f:result=summarize(json.loads(line) for line in f if line.strip())
    P.write(dest,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('records',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();summarize_file(a.records,a.output)
