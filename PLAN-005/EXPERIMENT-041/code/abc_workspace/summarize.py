"""ABC per-task macro scores, common starts, program survival and paired probes."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import protocol as P
sys.path.insert(0,str(HERE.parent/'priority_pilots'))
from io_utils import read_json,write_json,jsonl,sha


def write_csv(path, rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,action='append',required=True)
    p.add_argument('--expect-models',nargs='+',required=True,help='Explicit comparison set; C seeds are not AB seeds')
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args(); sources=[]; identities=set(); start_good=defaultdict(dict)
    for folder in a.run_dir:
        m=read_json(folder/'meta.json');key=(m['mode'],m['model']['id'])
        if m['status']!='complete' or m['protocol']!=P.VERSION or key in identities:
            raise ValueError('incomplete/duplicate source')
        if sha(folder/'records.jsonl')!=m['records_sha256']:raise ValueError('records changed')
        identities.add(key);sources.append((folder,m))
        for r in jsonl(folder/'records.jsonl'):
            if m['mode']=='assignment' and r['branch']=='old_clean':
                start_good[r['model']][r['case']]=r['baseline']['nodes_first4_all']
    modes={m['mode'] for _,m in sources}
    for mode in modes:
        mm=[m for _,m in sources if m['mode']==mode]
        if {m['model']['id'] for m in mm}!=set(a.expect_models):raise ValueError('missing comparison model')
        signatures={(m['task_sha256'],m['per_stratum'],m['stages'],m['slack'],m['probe_per_stratum'],tuple(m['cases'])) for m in mm}
        if len(signatures)!=1:raise ValueError('incompatible candidates/budgets')
    common=set.intersection(*(set(k for k,v in d.items() if v) for d in start_good.values())) if start_good else set()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(lambda:defaultdict(float)); coverage=[]; events=[]; probes={}; examples={}
    programs=defaultdict(list); seen=set()
    for folder,m in sources:
        for r in jsonl(folder/'records.jsonl'):
            rid=(r['mode'],r['model'],r['case'],r['stage'],r['branch'])
            if rid in seen:raise ValueError('duplicate record')
            seen.add(rid)
            complete=r['status']=='complete'; ms=r.get('metrics',{})
            cm=r.get('conditional_metrics',{})
            cohorts=['all_candidates']
            if r['mode']=='assignment' and r['case'] in common:cohorts.append('common_initial_first4')
            if r['mode']=='program' and not r['diagnostic']:
                programs[(r['model'],r['case'],r['branch'],r['stratum'])].append(
                    {k:r[k] for k in ('stage','status','reference','metrics') if k in r})
            for cohort in cohorts:
                key=(r['mode'],r['model'],r['stratum'],r['stage'],r['branch'],cohort)
                g=groups[key];g['cases']+=1;g['observed']+=complete
                for reg in ('nodes','terminal'):
                    n=ms.get(reg+'_n',P.NODES if reg=='nodes' else P.STRATA[r['stratum']][0])
                    g[reg+'_n']+=n
                    for metric in ('value','first4'):
                        num=ms.get(reg+'_'+metric+'_n',0)
                        g[reg+'_'+metric+'_n']+=num
                        g[reg+'_'+metric+'_macro_sum']+=num/n
                        g[reg+'_'+metric+'_all']+=ms.get(reg+'_'+metric+'_all',False)
                        g[reg+'_'+metric+'_final_all']+=ms.get(reg+'_'+metric+'_final_all',False)
                        g[reg+'_'+metric+'_final_macro_sum']+=ms.get(reg+'_'+metric+'_final_n',0)/n
                if cm:
                    g['conditional_observed']+=1
                    g['conditional_nodes_value_macro_sum']+=cm['nodes_value_n']/cm['nodes_n']
                    g['conditional_terminal_all']+=cm['terminal_value_all']
                if complete:
                    g['task_observed']+=1
                    g['task_first4_macro_sum']+=ms.get('task_first4_n',0)/max(ms.get('task_n',1),1)
            ev=dict(mode=r['mode'],model=r['model'],case=r['case'],stratum=r['stratum'],
                    stage=r['stage'],branch=r['branch'],diagnostic=r['diagnostic'],status=r['status'],
                    node_value_first=ms.get('nodes_value_first'),node_value_stable=ms.get('nodes_value_stable'),
                    node_value_reerror=ms.get('nodes_value_reerror'),terminal_value_all=ms.get('terminal_value_all',False),
                    terminal_final_value_all=ms.get('terminal_value_final_all',False),
                    conditional_terminal_all=cm.get('terminal_value_all'),
                    cumulative_main_forwards=r.get('cumulative_main_forwards'),
                    cumulative_logical_nodes=r.get('reference',{}).get('cumulative_logical_nodes'))
            events.append(ev)
            if r['diagnostic']:
                correct={i for i in r['nodes'] if all(x[i][3]==r['truth4'][i][3] for x in r['trace']['outputs'][-2:])}
                probes[(r['mode'],r['model'],r['case'],r['stage'],r['branch'])]=(
                    dict(stratum=r['stratum'],nodes=r['nodes']),correct)
            if r['branch']=='carry' and complete:
                category='success' if ms['terminal_value_final_all'] else 'failure'
                k=(r['mode'],r['model'],category)
                if k not in examples:examples[k]=r
        coverage.append(dict(mode=m['mode'],model=m['model']['id'],candidates=len(m['cases']),
                             initial_first4_qualified=sum(start_good.get(m['model']['id'],{}).values()) if m['mode']=='assignment' else '',
                             common_initial_first4=len(common) if m['mode']=='assignment' else '',
                             compare_models='|'.join(a.expect_models)))
    table=[]
    for k,g in sorted(groups.items()):
        z=dict(zip(('mode','model','stratum','stage','branch','cohort'),k));z.update(g)
        for reg in ('nodes','terminal'):
            for metric in ('value','first4'):
                z[reg+'_'+metric+'_macro']=g[reg+'_'+metric+'_macro_sum']/g['cases']
                z[reg+'_'+metric+'_final_macro']=g[reg+'_'+metric+'_final_macro_sum']/g['cases']
        z['conditional_observed']=g.get('conditional_observed',0)
        z['conditional_nodes_value_macro']=g['conditional_nodes_value_macro_sum']/g['conditional_observed'] if g.get('conditional_observed') else None
        z['conditional_terminal_all']=g.get('conditional_terminal_all',0)
        z['task_observed']=g.get('task_observed',0)
        z['task_first4_macro']=g['task_first4_macro_sum']/g['task_observed'] if g.get('task_observed') else None
        z.pop('task_first4_macro_sum',None)
        z.pop('conditional_nodes_value_macro_sum',None)
        table.append(z)
    # Rectangular columns even when assignment has no conditional score.
    columns=sorted({k for r in table for k in r})
    table=[{k:r.get(k) for k in columns} for r in table]
    trajectories=[]
    for (model,cid,branch,stratum), rr in sorted(programs.items()):
        rr.sort(key=lambda r:r['stage'])
        n=next(m['stages'] for _,m in sources if m['mode']=='program' and m['model']['id']==model)
        if [r['stage'] for r in rr]!=list(range(1,n+1)):raise ValueError('missing program stages')
        first_bad=next((r['stage'] for r in rr if not r.get('metrics',{}).get('terminal_value_final_all',False)),None)
        last=rr[-1]
        row=dict(model=model,case=cid,stratum=stratum,branch=branch,stages=n,
                 first_terminal_failure=first_bad,all_stage_terminal_success=first_bad is None,
                 final_terminal_success=last.get('metrics',{}).get('terminal_value_final_all',False),
                 last_two_terminal_success=last.get('metrics',{}).get('terminal_value_all',False),
                 first_invalid_commit=next((r['stage'] for r in rr if r['status']=='invalid_commit'),None),
                 final_cumulative_logical_nodes=last['reference']['cumulative_logical_nodes'],
                 final_required_ancestor_operations=last['reference']['output_ancestor_operations'],
                 final_logical_depth=last['reference']['logical_depth'])
        for limit in (128,256,512):
            at=next((r for r in rr if r['reference']['cumulative_logical_nodes']>=limit),None)
            row[f'at_least_{limit}_actual_nodes']=at['reference']['cumulative_logical_nodes'] if at else None
            row[f'at_least_{limit}_terminal_success']=at.get('metrics',{}).get('terminal_value_final_all',False) if at else None
        trajectories.append(row)
    pairs=[]
    for k,(r,correct) in sorted(probes.items()):
        if k[-1]=='probe_continue':continue
        ref=probes[k[:-1]+('probe_continue',)]; other=ref[1]
        pairs.append(dict(mode=k[0],model=k[1],case=k[2],stage=k[3],branch=k[4],stratum=r['stratum'],
                          nodes=len(r['nodes']),reference_correct=len(other),branch_correct=len(correct),
                          repaired=len(correct-other),damaged=len(other-correct),
                          note='same starting state and extra rounds; subset diagnostics, not population recovery rate'))
    for name,rows in [('summary',table),('coverage',coverage),('events',events),('programs',trajectories),('probe_pairs',pairs)]:
        write_csv(a.out_dir/(name+'.csv'),rows)
    with (a.out_dir/'examples.jsonl').open('x',encoding='utf-8') as f:
        for k,r in sorted(examples.items()):f.write(json.dumps(dict(category=k,record=r))+'\n')
    (a.out_dir/'report.md').write_text('# ABC共同任务结果包\n\n'
        '这是开发任务结果，不自动给出机制结论。summary按任务等权，失败提交保留在分母；'
        'programs分别给最终成功、全阶段成功、首次失败与实际逻辑长度。'
        'conditional分数针对模型实际提交的输入，不能代替原程序正确率。'
        'probe_pairs在同一起点和额外预算下给修回/改坏，成功和失败起点均保留。'
        'C程序接口包含数值提交，不是始终连续的跨窗口工作空间。'
        'C的reset是整行隐藏重初始化，不等价于A/B的VALUE清空。\n\n'
        f'比较权重：{a.expect_models}。共同初始四属性合格任务：{len(common)}。\n',encoding='utf-8')
    files={p.name:sha(p) for p in a.out_dir.iterdir() if p.is_file()}
    write_json(a.out_dir/'meta.json',dict(protocol=P.VERSION,status='complete',files=files,
        sources=[dict(path=str(f),meta=m,meta_sha256=sha(f/'meta.json')) for f,m in sources],
        compare_models=a.expect_models,common_initial_cases=sorted(common)))


if __name__=='__main__':main()
