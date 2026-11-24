#!/usr/bin/env python3
"""Small tables with explicit cohorts, joint states, graph means and paired effects."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import core as P
sys.path.insert(0,str(HERE.parent/'priority_pilots'))
from io_utils import CSV,jsonl,read_json,sha,write_json


def identity(r,continuous):
    return ((r['graph_id'],r['timing'],r['mode'],r['probe_lag']) if continuous else
            (r['graph_id'],r['schedule'],r['horizon'],r['episode']))


def input_qualified(r):
    c=r['context']
    return (c['u_value_correct'] and c['other_parents_value_correct'] and
            c['clean_local_first4_correct'] and c['donor_pre_empty'] and c['donor_pre_identity_correct'])


def run(runs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sources=[];metas=[];seen=set();available=defaultdict(set);qualified=defaultdict(set);signatures={}
    continuous=None
    # First pass stores only keys, never all trajectories in RAM.
    for path in map(Path,runs):
        m=read_json(path/'meta.json')
        c=m['protocol']=='C_function_probe_v1'
        if m['status']!='complete' or m['protocol'] not in ('C_function_probe_v1','state_reuse_discrete_v1'):
            raise ValueError('unsupported/incomplete source')
        if continuous is not None and c!=continuous:raise ValueError('do not pool C and discrete experiments')
        continuous=c
        for f in ('records','coverage'):
            if sha(path/f'{f}.jsonl')!=m[f'{f}_sha256']:raise ValueError('source hash mismatch')
        metas.append(m);sources.append(dict(path=str(path),meta_sha256=sha(path/'meta.json'),meta=m))
        local=set()
        for r in jsonl(path/'records.jsonl'):
            arm=r['arm'] if c else m['model_key'];key=identity(r,c)
            if (arm,key) in seen:raise ValueError('duplicate arm/case')
            seen.add((arm,key));available[key].add(arm)
            local.add((arm,key))
            if c and input_qualified(r):qualified[key].add(arm)
            signature=(r['positions'],r['truth4'],r['u'],r['q']) if c else (r['positions'],r['old_truth4'],r['new_truth4'],r['plan'])
            serial=json.dumps(signature,sort_keys=True)
            if key in signatures and signatures[key]!=serial:raise ValueError('unmatched graph/plan/truth')
            signatures[key]=serial
        expected=set();scanned=set()
        for cv in jsonl(path/'coverage.jsonl'):
            arm=cv['arm'] if c else m['model_key'];gid=cv['graph_id']
            if (arm,gid) in scanned:raise ValueError('duplicate graph coverage')
            scanned.add((arm,gid))
            if cv['status']!='selected':continue
            if c:
                expected.update((arm,(gid,t,mode,lag)) for t,mode,lag in m['grid'])
            else:
                expected.update((arm,(gid,s,h,e)) for s,n in cv['episodes'].items()
                                for h in m['horizons'] for e in range(1,n+1))
        arms={s['arm'] for s in m['models']} if c else {m['model_key']}
        if scanned!={(arm,gid) for arm in arms for gid in m['candidate_ids']} or local!=expected:
            raise ValueError('missing candidate/episode/branch-grid records')
    if not continuous:
        required={'model_a','model_b'}
        if len(metas)!=2 or {m['model_key'] for m in metas}!=required:raise ValueError('one A and one B run required')
        a,b=metas
        for f in ('mode','candidate_ids','horizons','code_hashes'):
            if a[f]!=b[f]:raise ValueError('A/B protocol differs: '+f)
        for f in ('dataset_sha256','manifest_sha256'):
            if a['source_meta'][f]!=b['source_meta'][f]:raise ValueError('A/B source differs')
    else:
        required={s['arm'] for m in metas for s in m['models']}
        for m in metas[1:]:
            for f in ('data_sha256','plan_sha256','candidate_ids','horizon','grid','code_hashes','dependencies'):
                if m[f]!=metas[0][f]:raise ValueError('C protocol differs: '+f)
    common={key for key,arms in available.items() if arms==required}
    common_qualified={key for key,arms in qualified.items() if arms==required}
    streams={n:CSV(out/(n+'.csv')) for n in ('coverage','case_curves','summary','paired_effects','events','joint_states','update_retention','qualification_summary')}
    buckets=defaultdict(lambda:[0,0,0,0,0,0.,0.]);pairmodels=defaultdict(dict); written=0
    qualifications=defaultdict(lambda:defaultdict(int))
    def add(row,group):
        streams['case_curves'].add(row)
        # Each graph contributes once per condition (enforced by record identity).
        # Accumulate macro fractions, not millions of copies of trajectory rows.
        agg=buckets[group];agg[0]+=1
        if row['n']:
            agg[1]+=1;agg[2]+=row['n'];agg[3]+=row['value_n'];agg[4]+=row['first4_n']
            agg[5]+=row['value_n']/row['n'];agg[6]+=row['first4_n']/row['n']
    try:
        for src in sources:
            path=Path(src['path']);m=src['meta']
            for cov in jsonl(path/'coverage.jsonl'):
                # A single uniform schema, raw qualification details remain JSON.
                streams['coverage'].add(dict(source=str(path),model=cov.get('arm',m.get('model_key')),
                    graph_id=cov['graph_id'],status=cov['status'],details=json.dumps(cov,ensure_ascii=False)))
            for r in jsonl(path/'records.jsonl'):
                arm=r['arm'] if continuous else m['model_key'];key=identity(r,continuous)
                branch_names=set(P.C_BRANCHES if continuous else P.DISCRETE_BRANCHES)
                if set(r['branches'])!=branch_names:raise ValueError('missing branch')
                if any(len(b['outputs'])!=r['horizon'] or len(b['inputs'])!=r['horizon'] for b in r['branches'].values()):
                    raise ValueError('partial trajectory')
                pos=r['positions'];lookup={p:i for i,p in enumerate(pos)};h=r['horizon']
                if continuous:
                    gold=r['truth4'];ctx=r['context'];qi=lookup[r['q']];ui=lookup[r['u']]
                    protocol=(r['timing'],r['mode'],r['probe_lag'])
                    group_label=json.dumps(protocol)
                    regions=dict(all_nodes=pos,u=[r['u']],q=[r['q']],
                        other_children=[p for p in r['children'] if p!=r['q']],
                        descendants=r['descendants'],outside=[p for p in pos if p not in set(r['descendants']+[r['u']])])
                    eligible=input_qualified(r)
                    cohorts=['all_recorded']+(['common_recorded'] if key in common else [])
                    if eligible:
                        cohorts.extend(['input_qualified','input_qualified_q_correct' if ctx['q_value_correct'] else 'input_qualified_q_wrong'])
                    if key in common_qualified:cohorts.append('common_input_qualified')
                    clean_pre_ok=all(z[qi][3]==gold[qi][3] for z in r['branches']['clean_pre']['outputs'][-2:])
                    qual=qualifications[(arm,group_label)];qual['candidate_records']+=1
                    for field,ok in dict(ctx,input_qualified=eligible,common_input_qualified=key in common_qualified,
                                         clean_pre_last_two_correct=clean_pre_ok).items():
                        qual[field]+=int(ok)
                    # Joint counts per EXACT event, no division of separate marginal tables.
                    streams['joint_states'].add(dict(model=P.C_NAMES[arm],arm=arm,graph_id=r['graph_id'],
                        timing=r['timing'],mode=r['mode'],lag=r['probe_lag'],input_qualified=eligible,
                        common_recorded=key in common,common_input_qualified=key in common_qualified,
                        clean_pre_last_two_correct=clean_pre_ok,
                        u_had_wrong_readout=r['u_had_wrong_readout'],q_had_wrong_readout=r['q_had_wrong_readout'],**ctx))
                    contrasts=[('observed_pre','observed_keep'),('restore_u_pre','observed_pre'),
                        ('restore_u_keep','observed_keep'),('clean_pre','observed_pre'),
                        ('observed_solved','observed_pre'),('clean_pre','clean_keep')]
                else:
                    gold=r['new_truth4'];protocol=(r['schedule'],h,r['episode']);group_label=json.dumps(protocol)
                    regions=r['regions'];cohorts=['own_source_qualified']+(['common_source_qualified'] if key in common else [])
                    contrasts=[('warm','scratch'),('clear','warm'),('warm_pinned','warm')]
                for branch,b in r['branches'].items():
                    goal=(r['original_truth4'] if not continuous and branch in ('clean','sham') else gold)
                    for region,positions in regions.items():
                        indices=[lookup[p] for p in positions]
                        steps=[s for s in (1,2,4,8,16) if s<=h]+['last_two']
                        for step in steps:
                            counts=P.trace_counts(b['outputs'],goal,indices,step)
                            for cohort in cohorts:
                                row=dict(arm=arm,model=P.C_NAMES[arm] if continuous else m['display_name'],graph_id=r['graph_id'],
                                    condition=group_label,branch=branch,region=region,cohort=cohort,round_or_window=str(step),**counts)
                                add(row,(arm,group_label,branch,region,cohort,str(step)))
                        f=[all(z[i][3]==goal[i][3] for i in indices) for z in b['outputs']]
                        streams['events'].add(dict(arm=arm,graph_id=r['graph_id'],condition=group_label,branch=branch,region=region,
                            node_n=len(indices),first_all_correct=next((i+1 for i,v in enumerate(f) if v),None) if indices else None,
                            suffix_all_correct_from=P.stable_round(f) if indices else None,
                            correct_then_wrong=any(f[i] and not all(f[i+1:]) for i in range(len(f)-1)) if indices else None,
                            note='finite output window; >=2 samples for stable suffix; no convergence claim'))
                    if not continuous:
                        changed=[lookup[p] for p in r['plan']['changed']]
                        protected=[lookup[p] for p in r['nodes'] if r['original_truth4'][lookup[p]][3]==goal[lookup[p]][3]]
                        for step in [s for s in (1,2,4,8,16) if s<=h]+['last_two']:
                            inds=[h-2,h-1] if step=='last_two' else [step-1]
                            old=r['old_truth4'];clean=r['branches']['clean']['outputs']
                            streams['update_retention'].add(dict(arm=arm,graph_id=r['graph_id'],condition=group_label,
                                branch=branch,round_or_window=str(step),changed_n=len(changed),
                                retain_previous_n=sum(all(b['outputs'][s][i][3]==old[i][3] for s in inds) for i in changed),
                                accept_new_n=sum(all(b['outputs'][s][i][3]==gold[i][3] for s in inds) for i in changed),
                                original_truth_unchanged_n=len(protected),
                                clean_correct_n=sum(all(clean[s][i][3]==goal[i][3] for s in inds) for i in protected),
                                extra_damage_n=sum(all(clean[s][i][3]==goal[i][3] for s in inds) and
                                    any(b['outputs'][s][i][3]!=goal[i][3] for s in inds) for i in protected)))
                effectregion={'q':[r['q']]} if continuous else {k:regions[k] for k in ('target','changed','all_nodes')}
                for left,right in contrasts:
                    for region,positions in effectregion.items():
                        ix=[lookup[p] for p in positions]
                        for step in (1,'last_two'):
                            aa=P.trace_counts(r['branches'][left]['outputs'],gold,ix,step)
                            bb=P.trace_counts(r['branches'][right]['outputs'],gold,ix,step)
                            streams['paired_effects'].add(dict(arm=arm,graph_id=r['graph_id'],condition=group_label,
                                region=region,round_or_window=str(step),left=left,right=right,node_n=len(ix),
                                left_correct=aa['value_n'],right_correct=bb['value_n'],difference=aa['value_n']-bb['value_n'],
                                common_recorded=key in common,input_qualified=eligible if continuous else True))
                if not continuous:
                    for branch in P.DISCRETE_BRANCHES:
                        if branch in ('clean','sham'):continue
                        for region,positions in regions.items():
                            ix=[lookup[p] for p in positions]
                            pairmodels[(key,branch,region)][arm]=P.trace_counts(r['branches'][branch]['outputs'],gold,ix,'last_two')
                written+=1
        for group,agg in buckets.items():
            arm,condition,branch,region,cohort,step=group
            streams['summary'].add(dict(arm=arm,model=P.C_NAMES[arm] if continuous else ('A' if arm=='model_a' else 'B'),
                condition=condition,branch=branch,region=region,cohort=cohort,round_or_window=step,
                selected_graph_n=agg[0],nonempty_graph_n=agg[1],node_n=agg[2],
                value_n=agg[3],first4_n=agg[4],
                value_macro=agg[5]/agg[1] if agg[1] else None,
                first4_macro=agg[6]/agg[1] if agg[1] else None))
        for (arm,condition),qual in qualifications.items():
            streams['qualification_summary'].add(dict(arm=arm,model=P.C_NAMES[arm],condition=condition,**qual))
        if not continuous:
            f=CSV(out/'paired_models.csv')
            try:
                for (key,branch,region),ab in pairmodels.items():
                    if set(ab)!=required:continue
                    a,b=ab['model_a'],ab['model_b']
                    if a['n']!=b['n']:raise ValueError('paired denominator differs')
                    f.add(dict(graph_id=key[0],schedule=key[1],horizon=key[2],episode=key[3],branch=branch,region=region,
                        node_n=a['n'],A_value_n=a['value_n'],B_value_n=b['value_n'],B_minus_A=b['value_n']-a['value_n']))
            finally:f.close()
        write_json(out/'meta.json',dict(status='complete',protocol='state_reuse_summary_v1',sources=sources,
            records=written,common_cases=len(common),common_input_qualified=len(common_qualified),
            code_hashes={f.name:sha(f) for f in (Path(__file__),HERE/'core.py')},
            note='C common_recorded is shared structural cases; common_input_qualified intersects input checks across supplied arms. Report own and common coverage. No pooling across timing/dose/episode.'))
    finally:
        for stream in streams.values():stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,action='append',required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.out_dir)


if __name__=='__main__':main()
