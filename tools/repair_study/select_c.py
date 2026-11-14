#!/usr/bin/env python3
"""Classify natural C errors and freeze a small, diverse local-repair sample.

Reads existing NPZ and original dataset_info (CPU, no model). All events remain
in candidates.jsonl, including events with no usable donor/control. The selected
diagnostic sample is conditional on an event, not a population accuracy estimate.
"""
import argparse
from collections import Counter
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--weights', nargs='+', default=['s42_e143', 's1234_e143'])
    p.add_argument('--per_group', type=int, default=1,
                   help='per weight/stratum/error-class/natural-outcome, distinct graphs')
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if a.per_group < 1:
        p.error('per_group must be positive')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'b_mechanism'))
    from trajectory_io import configure_runtime
    configure_runtime()
    import numpy as np
    import torch
    a.out_dir.mkdir(parents=True, exist_ok=False)
    all_cases, sources, cache = [], {}, {}
    totals = Counter()
    with (a.out_dir/'candidates.jsonl').open('x', encoding='utf-8') as dst:
        for weight in a.weights:
            run = a.run_dir/weight
            meta = core.read_json(run/'run_meta.json')
            if meta['status'] != 'complete':
                raise ValueError(f'incomplete source run: {run}')
            sources[weight] = dict(run_dir=str(run.resolve()), run_meta=meta,
                                   run_meta_sha256=core.sha(run/'run_meta.json'),
                                   index_sha256=core.sha(run/'index.jsonl'))
            for line in (run/'index.jsonl').read_text(encoding='utf-8').splitlines():
                index = json.loads(line)
                st = index['stratum']
                if st not in cache:
                    path = a.data_dir/f'{st}.pt'
                    cache[st] = (core.sha(path), torch.load(path, map_location='cpu', weights_only=True))
                ds_sha, blob = cache[st]
                if ds_sha != index['data_sha256']:
                    raise ValueError(f'dataset mismatch: {st}')
                # graph_id is stratum:local_index; source_index belongs to the original pool.
                gid = int(index['graph_id'].rsplit(':', 1)[1])
                info = blob['dataset_info'][gid].tolist()
                npz_path = run/index['path']
                with np.load(npz_path, allow_pickle=False) as z:
                    pred = np.stack([z['pred_'+f] for f in ('syntax', 'variable', 'operation', 'value')], -1).tolist()
                    truth = np.stack([z['true_'+f] for f in ('syntax', 'variable', 'operation', 'value')], -1).tolist()
                    positions = z['positions'].tolist()
                parents = {pos: [int(v) for v in info[pos][1::2] if v != -1] for pos in positions}
                if any(v != 0 for pos in positions for v in info[pos][2::2] if v != -1):
                    raise ValueError('this study is restricted to ADD operation ID 0')
                totals[(weight, st, 'graphs_scanned')] += 1
                for case in core.c_events(pred, truth, positions, parents):
                    category = case['category']
                    outcome = ('value_kept' if category == 'variable_only' else
                               'final_two_correct' if case['target_value_final_two_correct'] else 'not_final_two_correct')
                    case.update(weight=weight, stratum=st, graph_id=index['graph_id'],
                                graph_index=gid, source_index=index.get('source_index'),
                                case_id=f"C:{weight}:{st}:{gid}:{case['target_pos']}:{case['state_round']}:{category}",
                                selection_group=category+':'+outcome,
                                cohort='natural_event_sample', timing='natural_error_state',
                                npz_path=str(npz_path.resolve()), npz_sha256=core.sha(npz_path),
                                data_sha256=ds_sha, n_rounds=len(pred))
                    # Leave at least two existing future observations for replay checking.
                    case['selectable'] = (case['phase'] == 'post_complete' and
                                           case['donor_round'] is not None and case['state_round'] <= len(pred)-2)
                    totals[(weight, st, category)] += 1
                    dst.write(json.dumps(case)+'\n')
                    all_cases.append(case)
    selected = core.balanced_sample([c for c in all_cases if c['selectable']], a.per_group, a.seed)
    plan = dict(protocol='C_local_repair_v1', seed=a.seed, per_group=a.per_group,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                script_sha256=core.sha(__file__), core_sha256=core.sha(Path(__file__).with_name('core.py')),
                data_dir=str(a.data_dir.resolve()), sources=sources, cases=selected,
                counts=[dict(weight=k[0], stratum=k[1], kind=k[2], n=v) for k, v in sorted(totals.items())],
                all_events=len(all_cases), selectable_events=sum(c['selectable'] for c in all_cases),
                selected_cases=len(selected),
                missing_healthy_controls=sum(c['control_round'] is None for c in selected),
                notes=['Sampling stratifies natural outcomes for diagnosis, not prevalence.',
                       'Parent status uses decoded values BEFORE the failing forward.',
                       'No C+ training, no intervention-outcome-based selection.'])
    core.write_json(a.out_dir/'cases.json', plan)
    print(json.dumps(dict(selected=len(selected), events=len(all_cases),
                          missing_controls=plan['missing_healthy_controls'])))


if __name__ == '__main__':
    main()
