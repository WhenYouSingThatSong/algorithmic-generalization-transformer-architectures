#!/usr/bin/env python3
"""D1/D2/D3 solution accuracy against OLD and NEW truth, without clean pairing.

D2 starts from the new empty problem; its completion is not a recovery event.
Constants are a separate role, excluded from the core variable-node metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import b_mechanism_summary as S
import b_mechanism_horizon_summary as H

CONDITIONS = {'leaf_constant', 'leaf_constant_scratch', 'leaf_constant_pinned'}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def role_sets(rec):
    target = rec['target']
    const, leaf = int(target['const_pos']), int(target['var_pos'])
    nodes = {int(p) for p in rec['node_truth']}
    new = {int(p): v for p, v in rec['new_truth'].items()}
    if const not in new:
        raise ValueError('D1/D2/D3 new truth missing the changed input constant')
    changed = {p for p in nodes if new.get(p) is not None and
               str(new[p]) != str(rec['node_truth'][str(p)]['true'])}
    children = H.children_from_truth(rec['node_truth'])
    reachable = {leaf} | set(H.descendants(children, leaf))
    return dict(constant={const}, leaf={leaf}, reachable_variables=reachable & nodes,
                changed_value_variables=changed, unchanged_value_variables=nodes - changed,
                all_variables=nodes)


def correctness(rec, positions, truth_ref, metric):
    key = 'output_nodes_new' if truth_ref == 'new' else 'output_nodes'
    flag = 'value_correct' if metric == 'value' else 'first4_correct'
    series = []
    for row in rec['rounds']:
        if any(str(p) not in row.get(key, {}) for p in positions):
            raise ValueError(f'D1/D2/D3 missing {key} at tracked positions')
        series.append([bool(row[key][str(p)][flag]) for p in sorted(positions)])
    return series


def summarize_files(paths, out_dir, horizons=(8, 16, 64), stable_len=2):
    """One source/condition/truth/role/metric group; graph-equal plus node-weighted."""
    rounds = defaultdict(dict)
    windows = defaultdict(dict)
    events, coverage = [], []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        source, model = S.source_identity(path, data), H.model_label(data.get('args', {}))
        for rec in data.get('records', []):
            cond = rec.get('condition')
            if cond not in CONDITIONS:
                continue
            if not rec.get('rounds') or 'new_truth' not in rec:
                raise ValueError('D1/D2/D3 record missing rounds/new_truth')
            gid, tid = rec['graph_id'], rec['target_index']
            roles = role_sets(rec)
            coverage.append(dict(model=model, source=source, condition=cond,
                                 graph_id=gid, target_id=tid, rounds=len(rec['rounds']),
                                 has_clean_requirement=False))
            for ref in ('old', 'new'):
                for role, positions in roles.items():
                    for metric in ('value', 'first4'):
                        identity = (model, source, cond, ref, role, metric)
                        series = correctness(rec, positions, ref, metric)
                        n = len(positions)
                        for i, values in enumerate(series, 1):
                            H.add_graph(rounds[identity + (i,)], gid, [sum(values), n])
                        for horizon in horizons:
                            if horizon > len(series):
                                continue
                            tail = series[horizon - stable_len:horizon]
                            stable = (sum(all(row[j] for row in tail) for j in range(n))
                                      if horizon >= stable_len else 0)
                            H.add_graph(windows[identity + (horizon,)], gid,
                                        [sum(series[horizon - 1]), n, stable])
                        all_ok = [all(values) for values in series] if n else []
                        first = next((i + 1 for i, ok in enumerate(all_ok) if ok), None)
                        last_run = None
                        if all_ok and all_ok[-1]:
                            i = len(all_ok) - 1
                            while i > 0 and all_ok[i - 1]:
                                i -= 1
                            last_run = i + 1
                        events.append(dict(model=model, source=source, condition=cond,
                                           truth_ref=ref, role=role, metric=metric,
                                           graph_id=gid, target_id=tid, n_nodes=n,
                                           first_all_correct_round=first,
                                           final_all_correct=all_ok[-1] if all_ok else None,
                                           final_correct_run_start=last_run,
                                           full_window_rounds=len(series)))
    def identity_dict(key):
        return dict(zip(('model', 'source', 'condition', 'truth_ref', 'role', 'metric'), key[:6]))
    round_rows = [dict(**identity_dict(k), round=k[-1], **H.rate_view(v, 0, 1))
                  for k, v in sorted(rounds.items())]
    h_rows = []
    for key, value in sorted(windows.items()):
        for measure, idx in (('correct', 0), ('last_window_correct', 2)):
            h_rows.append(dict(**identity_dict(key), H=key[-1], measure=measure, stable_len=stable_len,
                               **H.rate_view(value, idx, 1)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (('d123_rounds.csv', round_rows), ('d123_by_H.csv', h_rows),
                           ('d123_events.csv', events), ('d123_coverage.csv', coverage)):
        write_csv(out_dir / filename, rows)
    return len(coverage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--horizons', default='8,16,64')
    args = parser.parse_args()
    summarize_files(args.inputs, args.out_dir, tuple(int(x) for x in args.horizons.split(',')))
