#!/usr/bin/env python3
"""Count VALUE/first4 and per-field errors from existing B trajectories.

The recorder stores pred4/true4 for each observed node.  This CPU-only pass
checks whether the aggregate identity between the VALUE and first4 summaries
comes from records that contain only VALUE errors, or from a recorder/metric
mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

FIELDS = ('SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE')


def model_from_path(path):
    name = Path(path).name
    if name.startswith('A_'):
        return 'A_prime'
    if name.startswith('B_'):
        return 'B_M6'
    raise SystemExit(f'cannot infer model from {path}')


def subset_from_path(path):
    parts = Path(path).parts
    for name in ('early', 'teacher', 'long32', 'late', 'controls'):
        if name in parts:
            return name
    return 'unknown'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--out_csv', required=True)
    a = p.parse_args()
    agg = defaultdict(lambda: dict(
        n_node_rounds=0, syntax_wrong=0, variable_wrong=0, operation_wrong=0,
        value_wrong=0, first4_wrong=0, value_ok_first4_wrong=0,
        first4_ok_value_wrong=0, stored_value_mismatch=0,
        stored_first4_mismatch=0))
    for path in a.inputs:
        subset = subset_from_path(path)
        model = model_from_path(path)
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                start = str(rec.get('start') or '')
                timing = str(rec.get('timing') or '')
                cond = str(rec.get('condition') or '')
                key = (subset, model, start, timing, cond)
                row = agg[key]
                for rr in rec.get('rounds', []):
                    for node in (rr.get('output_nodes') or {}).values():
                        pred, true = node.get('pred4'), node.get('true4')
                        if not pred or not true:
                            continue
                        wrong = {f: pred.get(f) != true.get(f) for f in FIELDS}
                        first4 = not any(wrong.values())
                        value = not wrong['VALUE']
                        row['n_node_rounds'] += 1
                        row['syntax_wrong'] += int(wrong['SYNTAX'])
                        row['variable_wrong'] += int(wrong['VARIABLE'])
                        row['operation_wrong'] += int(wrong['OPERATION'])
                        row['value_wrong'] += int(wrong['VALUE'])
                        row['first4_wrong'] += int(not first4)
                        row['value_ok_first4_wrong'] += int(value and not first4)
                        row['first4_ok_value_wrong'] += int(first4 and not value)
                        if 'value_correct' in node:
                            row['stored_value_mismatch'] += int(
                                bool(node['value_correct']) != value)
                        if 'first4_correct' in node:
                            row['stored_first4_mismatch'] += int(
                                bool(node['first4_correct']) != first4)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    rows = [dict(subset=k[0], model=k[1], start=k[2], timing=k[3],
                 condition=k[4], **v) for k, v in sorted(agg.items())]
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(rows=len(rows), out=a.out_csv,
                          n_node_rounds=sum(r['n_node_rounds'] for r in rows)),
                     ensure_ascii=True))


if __name__ == '__main__':
    main()
