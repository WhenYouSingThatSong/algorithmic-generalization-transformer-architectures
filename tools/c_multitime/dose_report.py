"""Small per-mode training tables from logs; counts do not imply mechanisms."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec as S


def summarize_run(directory):
    groups=defaultdict(Counter)
    with (directory/'steps.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); c=groups[r['mode']]
            c['batches']+=1; c['enabled_batches']+=int(r['enabled'])
            for k in ('graphs','candidates','attempts','accepted','actual_value_changed',
                      'donor_forwards','recipient_forwards','train_step_seconds'):
                c[k]+=r[k]
            c['peak_allocated_bytes']=max(c['peak_allocated_bytes'], r['peak_allocated_bytes'] or 0)
            for m in r['metrics']:
                for k in ('nodes','reference_updates','tp','fp','fn','changed','disagreements',
                          'execution_correct','cleared'):
                    c[k]+=m[k]
    with (directory/'injections.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); c=groups[r['mode']]
            if not r['accepted']: continue
            v=r['before'][3]
            c['before_EMPTY' if v==24 else 'before_numeric' if 0<=v<23 else 'before_NA']+=1
            for k in ('before_first4_correct','before_value_correct','graph_first4_before','graph_value_before'):
                c[k]+=int(r[k])
            c['with_correct_direct_child']+=int(r['correct_children_at_input']>0)
            c['accepted_but_same_VALUE']+=int(not r['actual_value_changed'])
    rows=[]
    for mode, counts in groups.items():
        c=dict(counts)
        c['mode']=mode
        c['enabled_batch_rate']=c['enabled_batches']/c['batches']
        c['source_acceptance']=c['accepted']/c['attempts'] if c['attempts'] else None
        c['accepted_per_structural_candidate']=c['accepted']/c['candidates'] if c['candidates'] else None
        c['mean_train_step_seconds']=c['train_step_seconds']/c['batches']
        c['locate_precision']=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None
        c['locate_recall']=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None
        rows.append(c)
    S.P.write(directory/'training-audit.json', dict(recipe=S.VERSION, modes=rows,
              note='No recipient-success selection. Correct-child presence is not proof of causal use. Timings exclude checkpoint/dev; see timings.jsonl.'))
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('run', type=Path)
    summarize_run(p.parse_args().run)
