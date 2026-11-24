"""Actual dose/cost from durable training logs, including unavailable donors."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def summarize_run(path):
    counts=Counter();before=Counter();position=Counter();rounds=Counter()
    with (path/'steps.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);counts['batches']+=1;counts['enabled_batches']+=int(r['enabled'])
            for k in ('graphs','candidates','attempts','accepted','donor_forwards','recipient_forwards'):counts[k]+=r[k]
    with (path/'injections.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);position[str(r['pos'])]+=1;rounds[str(r['round'])]+=1
            if r['accepted']:
                value=r['before'][3];before['EMPTY' if value==24 else 'numeric' if 0<=value<23 else 'N/A']+=1
                counts['already_intended_wrong']+=int(r['before']==r['after'])
    div=lambda a,b:counts[a]/counts[b] if counts[b] else None
    result=dict(counts=dict(counts),enabled_batch_rate=div('enabled_batches','batches'),
                attempts_per_candidate=div('attempts','candidates'),accepted_per_candidate=div('accepted','candidates'),
                source_acceptance=div('accepted','attempts'),before_values=dict(before),positions=dict(position),rounds=dict(rounds))
    P.write(path/'training-audit.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();summarize_run(a.run)
