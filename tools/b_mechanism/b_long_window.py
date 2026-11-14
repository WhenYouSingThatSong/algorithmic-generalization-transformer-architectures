#!/usr/bin/env python3
"""Long-window B stability analysis from existing JSONL trajectories.

For each node in all_affected (and optionally other roles), using VALUE output
correctness against the matched clean trajectory, reports:

- nodes correct at round 16 and whether they error again in 17..H;
- nodes that were damaged, recovered, and then errored again;
- whether the final contiguous correct run starts after round 16;
- the usual last-two-round stable flag.

CPU-only; reads the runner JSONL sidecars, not the large final JSON files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_main_summary as M  # noqa: E402


def rec_event_id(rec):
    eid = rec.get('event_id')
    if eid is None:
        eid = (rec.get('chain') or rec.get('target') or {}).get('event_id')
    return eid


def event(rec):
    for key in ('event_id',):
        if rec.get(key) is not None:
            return str(rec[key])
    chain = rec.get('target') or rec.get('chain') or {}
    return chain.get('event_id')


def clean_key(rec):
    return (int(rec['graph_id']), event(rec), str(rec.get('timing')))


def model_from_path(path):
    name = Path(path).name
    if name.startswith('A_'):
        return 'A_prime'
    if name.startswith('B_'):
        return 'B_M6'
    raise SystemExit(f'cannot infer model from {path}')


def node_series(rounds, pos, metric='value'):
    return [M.H.node_ok(r, pos, metric) for r in rounds]


def analyze_pair(clean, rec, roles, n_rounds):
    out = {}
    n = n_rounds
    for role, positions in roles.items():
        for pos in sorted(positions):
            cb = node_series(clean.get('rounds', [])[:n], pos)
            jb = node_series(rec.get('rounds', [])[:n], pos)
            if len(cb) < n or len(jb) < n:
                continue
            ok = [bool(cb[i] and jb[i]) for i in range(n)]
            err = [bool(cb[i] and not jb[i]) for i in range(n)]
            if not cb[15]:
                continue
            row = out.setdefault((role, pos), dict(
                n16=0, post16=0, damaged=0, recovered=0, rec_err=0, stable=0,
                final_le16=0, final_17_32=0, final_33_48=0, final_49_64=0,
                final_none=0, final_starts=[]))
            if jb[15]:
                row['n16'] += 1
                row['post16'] += int(any(err[i] for i in range(16, n)))
            damage_ts = [i for i in range(n) if err[i]]
            if damage_ts:
                row['damaged'] += 1
                row['stable'] += int(all(ok[-2:]))
                first = damage_ts[0]
                first_rec = next((i for i in range(first + 1, n) if ok[i]), None)
                if first_rec is not None:
                    row['recovered'] += 1
                    row['rec_err'] += int(any(err[i]
                                              for i in range(first_rec + 1, n)))
                if ok[-1]:
                    start = n - 1
                    while start > 0 and ok[start - 1]:
                        start -= 1
                    start1 = start + 1
                    row['final_starts'].append(start1)
                    if start1 <= 16:
                        row['final_le16'] += 1
                    elif start1 <= 32:
                        row['final_17_32'] += 1
                    elif start1 <= 48:
                        row['final_33_48'] += 1
                    else:
                        row['final_49_64'] += 1
                else:
                    row['final_none'] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True,
                   help='JSONL sidecars from one long-window run set')
    p.add_argument('--out_csv', required=True)
    p.add_argument('--out_nodes_csv', default=None,
                   help='optional per-node event rows')
    p.add_argument('--rounds', type=int, default=64)
    p.add_argument('--checkpoint_round', type=int, default=16)
    a = p.parse_args()

    # group[(model, timing, condition, role, graph_id)] = counts dict
    groups = defaultdict(lambda: dict(
        n16=0, post16=0, damaged=0, recovered=0, rec_err=0, stable=0,
        final_le16=0, final_17_32=0, final_33_48=0, final_49_64=0,
        final_none=0, final_starts=[]))
    node_rows = []
    for path in a.inputs:
        model = model_from_path(path)
        cleans = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('condition') == 'clean':
                    cleans[clean_key(rec)] = rec
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                cond = rec.get('condition')
                if cond == 'clean':
                    continue
                clean = cleans.get(clean_key(rec))
                if clean is None:
                    continue
                roles = M.roles_for(rec)
                if not roles:
                    continue
                timing = str(rec.get('timing'))
                n_rounds = min(a.rounds, len(clean.get('rounds', [])),
                               len(rec.get('rounds', [])))
                if n_rounds <= a.checkpoint_round:
                    continue
                per_node = analyze_pair(clean, rec, roles, n_rounds)
                for (role, pos), v in per_node.items():
                    gid = int(rec['graph_id'])
                    g = groups[(model, timing, cond, role, gid)]
                    for key in ('n16', 'post16', 'damaged', 'recovered',
                                'rec_err', 'stable', 'final_le16',
                                'final_17_32', 'final_33_48', 'final_49_64',
                                'final_none'):
                        g[key] += v[key]
                    g['final_starts'].extend(v['final_starts'])
                    if a.out_nodes_csv:
                        node_rows.append(dict(model=model, timing=timing,
                                              condition=cond, role=role,
                                              graph_id=gid, pos=pos,
                                              n16=v['n16'], post16=v['post16'],
                                              damaged=v['damaged'],
                                              recovered=v['recovered'],
                                              rec_err=v['rec_err'],
                                              stable=v['stable'],
                                              final_run_start=(v['final_starts'][0]
                                                               if v['final_starts'] else '')))

    rows = []
    for (model, timing, cond, role, graph_id), g in sorted(groups.items()):
        # raw counts and node-weighted rates; graph-equal is computed from the
        # per-graph table below.
        rows.append(dict(model=model, timing=timing, condition=cond, role=role,
                         graph_id=graph_id, **{k: g[k] for k in
                         ('n16', 'post16', 'damaged', 'recovered', 'rec_err',
                          'stable', 'final_le16', 'final_17_32', 'final_33_48',
                          'final_49_64', 'final_none')},
                         final_starts=g['final_starts']))

    # Aggregate by group key, keeping per-graph ratios for graph-equal.
    agg = {}
    for r in rows:
        k = (r['model'], r['timing'], r['condition'], r['role'])
        arow = agg.setdefault(k, dict(n16=0, post16=0, damaged=0, recovered=0,
                                      rec_err=0, stable=0, final_le16=0,
                                      final_17_32=0, final_33_48=0,
                                      final_49_64=0, final_none=0,
                                      ratios=defaultdict(list),
                                      final_starts=[]))
        for key in ('n16', 'post16', 'damaged', 'recovered', 'rec_err',
                    'stable', 'final_le16', 'final_17_32', 'final_33_48',
                    'final_49_64', 'final_none'):
            arow[key] += r[key]
        arow['final_starts'].extend(r['final_starts'])
        if r['n16']:
            arow['ratios']['post16'].append(r['post16'] / r['n16'])
        if r['recovered']:
            arow['ratios']['rec_err'].append(r['rec_err'] / r['recovered'])
        if r['damaged']:
            arow['ratios']['stable'].append(r['stable'] / r['damaged'])

    def rate(num, den):
        return None if not den else num / den

    def ge(vals):
        return None if not vals else sum(vals) / len(vals)

    out_rows = []
    for k, v in sorted(agg.items()):
        starts = sorted(v['final_starts'])
        median_start = starts[len(starts) // 2] if starts else None
        out_rows.append(dict(
            model=k[0], timing=k[1], condition=k[2], role=k[3],
            n_nodes_correct_round16=v['n16'],
            post16_error_num=v['post16'], post16_error_den=v['n16'],
            post16_error_node_weighted=rate(v['post16'], v['n16']),
            post16_error_graph_equal=ge(v['ratios']['post16']),
            damaged=v['damaged'], recovered=v['recovered'],
            recovered_then_error_num=v['rec_err'],
            recovered_then_error_den=v['recovered'],
            recovered_then_error_node_weighted=rate(v['rec_err'], v['recovered']),
            recovered_then_error_graph_equal=ge(v['ratios']['rec_err']),
            stable_num=v['stable'], stable_den=v['damaged'],
            stable_node_weighted=rate(v['stable'], v['damaged']),
            stable_graph_equal=ge(v['ratios']['stable']),
            final_run_start_den=v['damaged'],
            final_run_start_le16=v['final_le16'],
            final_run_start_17_32=v['final_17_32'],
            final_run_start_33_48=v['final_33_48'],
            final_run_start_49_64=v['final_49_64'],
            final_run_start_none=v['final_none'],
            final_run_start_median=median_start,
        ))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps(dict(rows=len(out_rows), out=a.out_csv), ensure_ascii=True))
    if a.out_nodes_csv:
        with open(a.out_nodes_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()))
            w.writeheader()
            w.writerows(node_rows)
        print(json.dumps(dict(node_rows=len(node_rows), out=a.out_nodes_csv),
                         ensure_ascii=True))


if __name__ == '__main__':
    main()
