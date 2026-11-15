#!/usr/bin/env python3
"""Summarize paired branches without conflating repair, retention and harm.

Reports target trajectories/final-two accuracy, baseline-relative benefit/harm,
raw counts and graph-equal means. B primary results use the common eligible
case set across A'/B; diagnostic anchors and C healthy controls stay separate.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core


def save_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, nargs='+', required=True, help='records.jsonl files')
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--require_timings', nargs='+', help='B only: keep targets jointly eligible in both models at every listed timing')
    a = p.parse_args()
    records, identities, metadata = [], set(), []
    for path in a.inputs:
        meta = core.read_json(path.with_name('meta.json'))
        if meta['status'] != 'complete':
            raise ValueError(f'incomplete run: {path}')
        metadata.append(dict(path=str(path), sha256=core.sha(path), meta=meta))
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            key = (r['weight'], r['case_id'], r['data_sha256'])
            if key in identities:
                raise ValueError(f'duplicate case: {key}')
            identities.add(key)
            records.append(r)
    bsets = {w: {(r['case_id'], r['data_sha256']) for r in records if r['weight']==w
                 and r['cohort']=='fixed_candidates' and r['status']=='eligible'}
             for w in ('model_a', 'model_b')}
    if any(r['weight'] in bsets for r in records):
        if not all(any(r['weight']==w for r in records) for w in bsets):
            raise ValueError('B primary comparison requires both model_a and model_b files')
    common = bsets['model_a'] & bsets['model_b']
    if a.require_timings:
        availability = defaultdict(set)
        for r in records:
            if r['cohort']=='fixed_candidates' and (r['case_id'],r['data_sha256']) in common:
                availability[(r['graph_id'],r['target_pos'],r['data_sha256'])].add(r['timing'])
        allowed = {k for k,v in availability.items() if set(a.require_timings)<=v}
        common = {(r['case_id'],r['data_sha256']) for r in records
                  if r['cohort']=='fixed_candidates' and r['timing'] in a.require_timings
                  and (r['case_id'],r['data_sha256']) in common
                  and (r['graph_id'],r['target_pos'],r['data_sha256']) in allowed}
    metrics, target_rows, coverage, endpoints = [], [], [], []
    for r in records:
        included = r['status']=='eligible' and (r['cohort']!='fixed_candidates' or
                                                (r['case_id'], r['data_sha256']) in common)
        coverage.append(dict(weight=r['weight'], case_id=r['case_id'], cohort=r['cohort'],
                             status=r['status'], included=included,
                             reason='common_eligible' if included and r['cohort']=='fixed_candidates'
                             else 'eligible' if included else 'ineligible_or_not_common'))
        if not included:
            continue
        pos = r['node_positions']; i = pos.index(r['target_pos']); truth = r['truth4']
        focus = {r['target_pos']} | set(r['descendant_positions']) | set(r['parent_positions'])
        roles = dict(target=[r['target_pos']], parents=r['parent_positions'],
                     downstream=r['descendant_positions'], all_nodes=pos,
                     outside_local_region=[p for p in pos if p not in focus])
        base_fields = dict(weight=r['weight'], cohort=r['cohort'], timing=r['timing'],
                           category=r['category'], stratum=r.get('stratum','N32'),
                           selection_group=r.get('selection_group','fixed_candidates'),
                           graph_id=r['graph_id'], case_id=r['case_id'],
                           state_round=r['state_round'], donor_gap=r.get('donor_gap'))
        for name, branch in r['branches'].items():
            for role, positions in roles.items():
                for row in core.pair_metrics(r, branch, positions):
                    metrics.append(dict(**base_fields, branch=name, role=role, **row))
                if len(branch['outputs'])>=2:
                    ids=[pos.index(p) for p in positions if p in pos]
                    for metric in ('value','first4'):
                        endpoints.append(dict(**base_fields,branch=name,role=role,metric=metric,
                            **core.window_counts(truth,r['branches']['unchanged']['outputs'],branch['outputs'],ids,metric)))
            for metric in ('value', 'first4'):
                def ok(row):
                    return row[i][3]==truth[i][3] if metric=='value' else row[i]==truth[i]
                flags = [ok(row) for row in branch['outputs']]
                target_rows.append(dict(**base_fields, branch=name, metric=metric,
                    initial_correct=ok(branch['initial']), first_output_correct=flags[0],
                    first_correct_forward=next((t+1 for t,v in enumerate(flags) if v), None),
                    ever_output_wrong=not all(flags), final_two_correct=len(flags)>=2 and all(flags[-2:]),
                    initial_value=branch['initial'][i][3], final_value=branch['outputs'][-1][i][3],
                    true_value=truth[i][3]))
    group_names = ['weight','cohort','timing','category','selection_group','stratum','branch','role','round','metric']
    groups = defaultdict(list)
    for row in metrics:
        groups[tuple(row[k] for k in group_names)].append(row)
    aggregated = []
    for key, pool in sorted(groups.items()):
        gs = defaultdict(list)
        for row in pool:
            if row['n']:
                gs[row['graph_id']].append(row)
        result = dict(zip(group_names,key))
        result.update(cases=len(pool), graphs_with_nodes=len(gs),
                      graphs_without_nodes=len({r['graph_id'] for r in pool})-len(gs),
                      observed_nodes=sum(r['n'] for r in pool))
        for field in ('correct','baseline_correct','helped','harmed'):
            result[field+'_count'] = sum(r[field] for r in pool)
            means = [sum(r[field]/r['n'] for r in rows)/len(rows) for rows in gs.values()]
            result[field+'_graph_equal'] = sum(means)/len(means) if means else None
        aggregated.append(result)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    save_csv(a.out_dir/'coverage.csv', coverage)
    save_csv(a.out_dir/'target_cases.csv', target_rows)
    save_csv(a.out_dir/'node_curves.csv', aggregated)
    save_csv(a.out_dir/'endpoint_cases.csv', endpoints)
    save_csv(a.out_dir/'endpoint_summary.csv',core.summarize_endpoints(endpoints,
        ['weight','cohort','timing','category','selection_group','stratum','branch','role','metric']))
    tg = defaultdict(list)
    names = ['weight','cohort','timing','category','selection_group','stratum','branch','metric']
    for row in target_rows:
        tg[tuple(row[n] for n in names)].append(row)
    target_summary = []
    for key, pool in sorted(tg.items()):
        result = dict(zip(names, key))
        gs = defaultdict(list)
        for row in pool:
            gs[row['graph_id']].append(row)
        result.update(cases=len(pool), graphs=len(gs))
        for field in ('initial_correct','first_output_correct','ever_output_wrong','final_two_correct'):
            result[field+'_count'] = sum(r[field] for r in pool)
            result[field+'_graph_equal'] = sum(
                sum(r[field] for r in rows)/len(rows) for rows in gs.values())/len(gs)
        target_summary.append(result)
    save_csv(a.out_dir/'target_summary.csv', target_summary)
    core.write_json(a.out_dir/'meta.json', dict(inputs=metadata, common_b_cases=len(common),
        require_timings=a.require_timings,
        source_cases=len(records), included_cases=sum(r['included'] for r in coverage),
        notes=['No single pooled recovery rate across different roles/cohorts/windows.',
               'First correct forward can reflect initial replacement; inspect initial_correct.',
               'Final two correct is window-limited, not permanent stability.',
               'Graph-equal: average targets within graph, then average graphs.']))


if __name__ == '__main__':
    main()
