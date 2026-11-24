#!/usr/bin/env python3
"""Summarize B mechanism trajectories.

Computes the provisional N0 -> N1 -> N2 denominators from a clean continuation
and an injected continuation, separately for VALUE and first-four fields.

N0 = nodes correct in the clean continuation at post_round 1
N1 = N0 nodes that are wrong under the injection at post_round 1
N2 = N1 nodes that become correct in any later injection round

Scope warning: N1 is post_round_1 only, N2 is any-later only, and
max_propagation_edges is measured from the first-round queue.  These are not
full-window propagation or sustained-recovery statistics.

D1/D2/D3 leaf-constant conditions are skipped here because their truth changes
or their start is the new problem; their raw records carry ``new_truth`` and
``output_nodes_new`` for a later, separately-defined summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict, deque

LEAF_DYNAMIC = {'leaf_constant', 'leaf_constant_pinned', 'leaf_constant_scratch'}


def source_identity(path, data):
    """Stable per-file run identity, used so A' and B never share a key."""
    args = data.get('args', {}) or {}
    keys = ('config', 'group', 'run', 'ckpt', 'ds_file',
            'value_mode', 'r_delta', 'u_delta', 'delta', 'out')
    vals = [args.get(k) for k in keys]
    if any(v is not None for v in vals):
        return '|'.join('' if v is None else str(v) for v in vals)
    return os.path.abspath(path)


def record_kind(rec):
    if rec.get('mode'):
        return 'control'
    if 'chain_index' in rec:
        return 'early'
    return 'late'


def record_start_timing(rec, args):
    kind = record_kind(rec)
    if kind == 'control':
        return 'control', str(rec.get('mode') or args.get('mode') or 'control')
    if kind == 'early':
        return (str(rec.get('start') or args.get('start') or 'own'),
                str(rec.get('timing') or args.get('timing') or 'unknown'))
    return 'late', 'complete'


def target_key(rec, source, args):
    """Pair clean and injected records; condition is intentionally excluded."""
    kind = record_kind(rec)
    start, timing = record_start_timing(rec, args)
    gid = rec.get('graph_id')
    if kind == 'early':
        eid = rec.get('event_id') or (rec.get('chain') or {}).get('event_id')
        return (source, kind, gid, rec.get('chain_index'), eid, start, timing)
    target_json = json.dumps(rec.get('target'), sort_keys=True)
    return (source, kind, gid, rec.get('target_index'), target_json, start, timing)


def child_map(node_truth):
    out = {}
    for k, v in node_truth.items():
        out[int(k)] = [int(x) for x in v.get('children', [])]
    return out


def distances_from(start, children):
    dist = {}
    q = deque([(start, 0)])
    seen = {start}
    while q:
        x, d = q.popleft()
        for y in children.get(x, []):
            if y in seen:
                continue
            seen.add(y)
            dist[y] = d + 1
            q.append((y, d + 1))
    return dist


def first_nodes(rounds):
    return rounds[0].get('output_nodes', {}) if rounds else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--out_json', required=True)
    a = p.parse_args()

    records = []
    for path in a.inputs:
        with open(path) as f:
            data = json.load(f)
        args = data.get('args', {}) or {}
        source = source_identity(path, data)
        for rec in data.get('records', []):
            records.append((source, args, rec))

    cleans = {}
    duplicate_cleans = 0
    for source, args, rec in records:
        if rec.get('condition') != 'clean' or not rec.get('rounds'):
            continue
        key = target_key(rec, source, args)
        if key in cleans:
            old = cleans[key]
            same = (json.dumps(old.get('rounds'), sort_keys=True) ==
                    json.dumps(rec.get('rounds'), sort_keys=True))
            if not same:
                raise SystemExit(
                    f'conflicting duplicate clean records for target {key}; '
                    'check inputs or regenerate without duplicate cleans')
            duplicate_cleans += 1
            continue
        cleans[key] = rec

    rows = defaultdict(list)
    skipped = 0
    for source, args, rec in records:
        cond = rec.get('condition')
        if cond in (None, 'clean'):
            continue
        if cond in LEAF_DYNAMIC:
            skipped += 1
            continue
        clean = cleans.get(target_key(rec, source, args))
        if clean is None:
            skipped += 1
            continue
        cr = clean.get('rounds', [])
        ir = rec.get('rounds', [])
        if not cr or not ir:
            skipped += 1
            continue
        c0, i0 = first_nodes(cr), first_nodes(ir)
        nodes = sorted(set(c0) & set(i0), key=int)
        if not nodes:
            skipped += 1
            continue
        later = {}
        for rr in ir[1:]:
            for k, v in rr.get('output_nodes', {}).items():
                later.setdefault(k, []).append(bool(v.get('first4_correct')))
        later_val = {}
        for rr in ir[1:]:
            for k, v in rr.get('output_nodes', {}).items():
                later_val.setdefault(k, []).append(bool(v.get('value_correct')))
        n0 = n1 = n2 = 0
        n0v = n1v = n2v = 0
        empty = 0
        corrupted_nodes = []
        for k in nodes:
            clean_ok = bool(c0[k].get('first4_correct'))
            clean_okv = bool(c0[k].get('value_correct'))
            if clean_ok:
                n0 += 1
                if not bool(i0[k].get('first4_correct')):
                    n1 += 1
                    corrupted_nodes.append(int(k))
                    if any(later.get(k, [])):
                        n2 += 1
            if clean_okv:
                n0v += 1
                if not bool(i0[k].get('value_correct')):
                    n1v += 1
                    if any(later_val.get(k, [])):
                        n2v += 1
            pred = i0[k].get('pred4', {}).get('VALUE')
            if pred in ('EMPTY', 'N/A'):
                empty += 1
        inj = rec.get('injection', {}) or {}
        start = inj.get('pos')
        if start is None:
            ch = rec.get('chain') or {}
            if rec.get('mode') or cond in ('r', 'ur'):
                start = ch.get('r_pos')
            else:
                start = ch.get('pos')
        max_edge = None
        if start is not None:
            dist = distances_from(int(start), child_map(rec.get('node_truth', {})))
            ds = [dist[x] for x in corrupted_nodes if x in dist]
            if ds:
                max_edge = max(ds)
        start_label, timing = record_start_timing(rec, args)
        key = (source, record_kind(rec), start_label, timing, cond)
        rows[key].append(dict(
            graph_id=rec.get('graph_id'), target_index=rec.get('target_index'),
            n0=n0, n1=n1, n2=n2, n0v=n0v, n1v=n1v, n2v=n2v,
            empty=empty, max_edge=max_edge,
        ))

    out = []
    for (source, kind, start, timing, cond), rr in sorted(rows.items(), key=lambda kv: str(kv[0])):
        n0 = sum(x['n0'] for x in rr)
        n1 = sum(x['n1'] for x in rr)
        n2 = sum(x['n2'] for x in rr)
        n0v = sum(x['n0v'] for x in rr)
        n1v = sum(x['n1v'] for x in rr)
        n2v = sum(x['n2v'] for x in rr)
        edges = [x['max_edge'] for x in rr if x['max_edge'] is not None]
        out.append(dict(
            source=source, kind=kind, start=start, timing=timing, condition=cond,
            n_graphs=len({x['graph_id'] for x in rr}),
            n_targets=len(rr), n_nodes=n0,
            first4_n0=n0, first4_n1=n1, first4_n2=n2,
            first4_corrupt_rate=(n1 / n0 if n0 else None),
            first4_recovery_rate=(n2 / n1 if n1 else None),
            first4_n1_scope='post_round_1',
            first4_n2_scope='any_later_round',
            value_n0=n0v, value_n1=n1v, value_n2=n2v,
            value_corrupt_rate=(n1v / n0v if n0v else None),
            value_recovery_rate=(n2v / n1v if n1v else None),
            value_n1_scope='post_round_1',
            value_n2_scope='any_later_round',
            empty_at_first_round=sum(x['empty'] for x in rr),
            max_propagation_edges=(max(edges) if edges else None),
            max_propagation_edges_scope='post_round_1',
            skipped_total=skipped,
            duplicate_cleans=duplicate_cleans,
        ))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else [])
        if out:
            w.writeheader()
            w.writerows(out)
    with open(a.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print('WROTE', a.out_csv, a.out_json)


if __name__ == '__main__':
    main()
