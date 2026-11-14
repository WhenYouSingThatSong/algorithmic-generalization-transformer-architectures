#!/usr/bin/env python3
"""Export round-level raw rows for selected damaged nodes.

Reads a B runner JSONL sidecar plus its meta, pairs injected records to clean
records with the public target key, and writes one row per observed round for
nodes matching optional condition/graph/pos/role filters.  CPU-only; no model
execution and no interpretation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_main_summary as M  # noqa: E402


def meta_for(sidecar):
    base = str(sidecar)
    if not base.endswith('.jsonl'):
        raise SystemExit(f'expected JSONL sidecar: {sidecar}')
    return Path(base[:-6] + '.meta.json')


def node_fields(node):
    pred = (node or {}).get('pred4') or {}
    true = (node or {}).get('true4') or {}
    wrong = [f for f in ('SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE')
             if pred.get(f) != true.get(f)]
    return pred, true, ','.join(wrong)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--condition')
    p.add_argument('--graph', type=int)
    p.add_argument('--pos', type=int)
    p.add_argument('--role')
    a = p.parse_args()
    meta = json.loads(meta_for(a.input).read_text(encoding='utf-8'))
    args = dict(meta.get('args') or {})
    source = M.S.source_identity(a.input, {'args': args})
    clean = {}
    injected = []
    with open(a.input) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('condition') == 'clean':
                clean[M.clean_key(rec, source, args)] = rec
            else:
                injected.append(rec)
    rows = []
    for rec in injected:
        cond = str(rec.get('condition'))
        if a.condition and cond != a.condition:
            continue
        gid = int(rec.get('graph_id'))
        if a.graph is not None and gid != a.graph:
            continue
        c = clean.get(M.clean_key(rec, source, args))
        if c is None:
            continue
        roles = M.roles_for(rec)
        if not roles:
            continue
        target = rec.get('target') or {}
        injection = rec.get('injection') or {}
        chain = rec.get('chain') or {}
        event_id = rec.get('event_id') or target.get('event_id') or chain.get('event_id')
        node_truth = rec.get('node_truth', {})
        children_map = M.H.children_from_truth(node_truth)
        parents_map = {}
        for p, child_list in children_map.items():
            for child in child_list:
                parents_map.setdefault(child, []).append(p)
        n = min(len(c.get('rounds', [])), len(rec.get('rounds', [])))
        for role, positions in roles.items():
            if a.role and role != a.role:
                continue
            for pos in sorted(positions):
                if a.pos is not None and int(pos) != a.pos:
                    continue
                cb = [M.H.node_ok(c['rounds'][i], pos, 'value') for i in range(n)]
                jb = [M.H.node_ok(rec['rounds'][i], pos, 'value') for i in range(n)]
                damage_ts = [i for i in range(n) if cb[i] and not jb[i]]
                if not damage_ts:
                    continue
                first = damage_ts[0]
                first_rec = next((i for i in range(first + 1, n)
                                  if cb[i] and jb[i]), None)
                final_start = None
                if jb and jb[-1]:
                    s = n - 1
                    while s > 0 and jb[s - 1]:
                        s -= 1
                    final_start = s + 1
                for i in range(n):
                    cin = (c['rounds'][i].get('input_nodes') or {}).get(str(pos)) or {}
                    cout = (c['rounds'][i].get('output_nodes') or {}).get(str(pos)) or {}
                    jin = (rec['rounds'][i].get('input_nodes') or {}).get(str(pos)) or {}
                    jout = (rec['rounds'][i].get('output_nodes') or {}).get(str(pos)) or {}
                    _, true, wrong = node_fields(jout)
                    rows.append(dict(
                        source=source, graph_id=gid, condition=cond,
                        event_id='' if event_id is None else event_id,
                        target_index='' if rec.get('target_index') is None
                        else rec.get('target_index'),
                        target_pos='' if target.get('pos') is None else target.get('pos'),
                        injection_pos='' if injection.get('pos') is None
                        else injection.get('pos'),
                        role=role, node_pos=pos, round=i + 1,
                        clean_input=cin.get('pred4', {}).get('VALUE'),
                        clean_output=cout.get('pred4', {}).get('VALUE'),
                        injected_input=jin.get('pred4', {}).get('VALUE'),
                        injected_output=jout.get('pred4', {}).get('VALUE'),
                        true_value=true.get('VALUE'),
                        value_correct=jout.get('value_correct'),
                        first4_correct=jout.get('first4_correct'),
                        wrong_fields=wrong,
                        is_damage_round=bool(cb[i] and not jb[i]),
                        first_damage=first + 1,
                        first_recovery='' if first_rec is None else first_rec + 1,
                        final_run_start='' if final_start is None else final_start,
                        children=json.dumps(children_map.get(int(pos), [])),
                        parents=json.dumps(sorted(parents_map.get(int(pos), []))),
                    ))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    if not rows:
        print(json.dumps(dict(rows=0, out=a.out_csv), ensure_ascii=True))
        return
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(rows=len(rows), out=a.out_csv), ensure_ascii=True))


if __name__ == '__main__':
    main()