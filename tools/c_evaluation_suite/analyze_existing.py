#!/usr/bin/env python3
"""CPU-only: joint readout/function observations from existing corruption records."""
import argparse,sys,json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import suite_core as U
C=U.C

def joint_rows(record):
    pos=record['positions'];ui=pos.index(record['target']);gold=record['truth4']
    base=record['branches']['unchanged'];qs=record['children'];rows=[]
    for branch,b in record['branches'].items():
        changed=b['initial'][ui][3]!=base['initial'][ui][3]
        for t,out in enumerate(b['outputs'],1):
            clean=base['outputs'][t-1]
            harmed=sum(clean[pos.index(q)][3]==gold[pos.index(q)][3] and out[pos.index(q)][3]!=gold[pos.index(q)][3] for q in qs)
            expected=sum(out[pos.index(q)][3]==record['expected_q'][str(q)] for q in qs)
            rows.append(dict(arm=record['arm'],graph_id=record['graph_id'],timing=record['timing'],branch=branch,round=t,
                input_readout_changed=changed,input_readout_wrong=b['initial'][ui][3]!=gold[ui][3],
                target_correct=out[ui][3]==gold[ui][3],child_n=len(qs),child_correct=sum(out[pos.index(q)][3]==gold[pos.index(q)][3] for q in qs),
                child_harmed=harmed,child_expected_wrong=expected))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',type=Path,default=C.STUDY/'evaluation')
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();meta=C.P.read_json(a.eval_dir/'meta.json')
    if meta['status']!='complete':raise ValueError('incomplete source')
    records=C.read_cases(a.eval_dir/'records.jsonl');models={r['arm'] for r in meta['models']}
    bycase=defaultdict(dict)
    for r in records:
        key=(r['graph_id'],r['timing'])
        if r['arm'] in bycase[key]:raise ValueError('duplicate case')
        bycase[key][r['arm']]=r
    if any(set(x)!=models for x in bycase.values()):raise ValueError('missing paired model')
    a.out_dir.mkdir(parents=True,exist_ok=False);rows=[];buckets=defaultdict(list);pairs=[]
    for key,arm_records in bycase.items():
        common=all(r['value_qualified'] for r in arm_records.values())
        perarm={arm:joint_rows(r) for arm,r in arm_records.items()}
        for arm,jrs in perarm.items():
            for r in jrs:
                if r['round'] not in (1,2,4,8,16):continue
                for cohort in ['all_fixed']+(['common_value'] if common else []):
                    rr=dict(r,cohort=cohort);rows.append(rr)
                    buckets[(arm,r['timing'],r['branch'],cohort,r['round'],r['input_readout_changed'],r['target_correct'],bool(r['child_harmed']))].append(rr)
        if 'hold' in perarm and 'semantic' in perarm:
            lookup={arm:{(r['branch'],r['round']):r for r in rr} for arm,rr in perarm.items()}
            for b,t in lookup['hold']:
                if t not in (1,2,4,8,16):continue
                h,s=lookup['hold'][(b,t)],lookup['semantic'][(b,t)]
                pairs.append(dict(graph_id=key[0],timing=key[1],branch=b,round=t,common_value=common,
                    hold_target_correct=h['target_correct'],semantic_target_correct=s['target_correct'],
                    hold_child_harmed=h['child_harmed'],semantic_child_harmed=s['child_harmed'],child_n=h['child_n'],
                    semantic_less_child_harm=s['child_harmed']<h['child_harmed'],
                    semantic_target_wrong_and_less_child_harm=not s['target_correct'] and s['child_harmed']<h['child_harmed']))
    U.write_csv(a.out_dir/'case_joint.csv',rows);U.write_csv(a.out_dir/'paired_models.csv',pairs)
    summary=[]
    for key,rs in sorted(buckets.items()):
        summary.append(dict(zip(['arm','timing','branch','cohort','round','input_readout_changed','target_correct','any_child_harmed'],key),
            cases=len(rs),child_n=sum(r['child_n'] for r in rs),child_harmed=sum(r['child_harmed'] for r in rs),
            child_expected_wrong=sum(r['child_expected_wrong'] for r in rs)))
    U.write_csv(a.out_dir/'joint_summary.csv',summary)
    C.P.write_json(a.out_dir/'meta.json',dict(status='complete',source_meta=meta,
        records_sha256=C.P.sha(a.eval_dir/'records.jsonl'),code_sha256=C.P.sha(__file__),
        note='Readout/function table is observational. No child effect does not prove no information use.'))
if __name__=='__main__':main()
