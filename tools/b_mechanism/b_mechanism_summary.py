#!/usr/bin/env python3
"""Summarize B mechanism trajectories.

Computes the fixed N0 -> N1 -> N2 denominators from a clean continuation and an
injected continuation, separately for VALUE and first-four fields.

N0 = nodes correct in the clean continuation at post_round 1
N1 = N0 nodes that are wrong under the injection at post_round 1
N2 = N1 nodes that become correct in any later injection round

Also reports EMPTY/N/A at the first round and the maximum descendant edge
distance of N1 nodes from the injected position.

D1/D2/D3 leaf-constant conditions are skipped here because their truth changes
or their start is the new problem; analyze them separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict, deque


def target_key(rec):
    if 'chain_index' in rec:
        return (rec['graph_id'], 'early', rec.get('chain_index'))
    return (rec['graph_id'], 'late', rec.get('target_index'), json.dumps(rec.get('target'), sort_keys=True))


def clean_key(rec):
    return target_key(rec)


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
            d = json.load(f)
        for r in d.get('records', []):
            records.append(r)

    cleans = {}
    for r in records:
        if r.get('condition') == 'clean' and r.get('rounds'):
            cleans[clean_key(r)] = r

    rows = defaultdict(list)
    skipped = 0
    for r in records:
        cond = r.get('condition')
        if cond in (None, 'clean'):
            continue
        if cond in ('leaf_constant', 'leaf_constant_pinned', 'leaf_constant_scratch'):
            # truth changes / start changes; analyzed separately
            skipped += 1
            continue
        c = cleans.get(clean_key(r))
        if c is None:
            skipped += 1
            continue
        cr = c.get('rounds', [])
        ir = r.get('rounds', [])
        if not cr or not ir:
            skipped += 1
            continue
        c0, i0 = first_nodes(cr), first_nodes(ir)
        nodes = sorted(set(c0) & set(i0), key=int)
        if not nodes:
            skipped += 1
            continue
        # later injection correctness by node
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
        # propagation edge distance from injected position, if known
        inj = r.get('injection', {}) or {}
        start = inj.get('pos')
        if start is None:
            ch = r.get('chain') or {}
            start = ch.get('r_pos') if cond in ('r', 'ur') else ch.get('pos')
        max_edge = None
        if start is not None:
            dist = distances_from(int(start), child_map(r.get('node_truth', {})))
            ds = [dist[x] for x in corrupted_nodes if x in dist]
            if ds:
                max_edge = max(ds)
        key = (r.get('start', ''), r.get('timing', ''), cond)
        rows[key].append(dict(
            graph_id=r.get('graph_id'), target_index=r.get('target_index'),
            n0=n0, n1=n1, n2=n2, n0v=n0v, n1v=n1v, n2v=n2v,
            empty=empty, max_edge=max_edge,
        ))

    out = []
    for (start, timing, cond), rr in sorted(rows.items(), key=lambda kv: str(kv[0])):
        n0 = sum(x['n0'] for x in rr)
        n1 = sum(x['n1'] for x in rr)
        n2 = sum(x['n2'] for x in rr)
        n0v = sum(x['n0v'] for x in rr)
        n1v = sum(x['n1v'] for x in rr)
        n2v = sum(x['n2v'] for x in rr)
        edges = [x['max_edge'] for x in rr if x['max_edge'] is not None]
        out.append(dict(
            start=start, timing=timing, condition=cond,
            n_graphs=len({x['graph_id'] for x in rr}),
            n_targets=len(rr), n_nodes=n0,
            first4_n0=n0, first4_n1=n1, first4_n2=n2,
            first4_corrupt_rate=(n1 / n0 if n0 else None),
            first4_recovery_rate=(n2 / n1 if n1 else None),
            value_n0=n0v, value_n1=n1v, value_n2=n2v,
            value_corrupt_rate=(n1v / n0v if n0v else None),
            value_recovery_rate=(n2v / n1v if n1v else None),
            empty_at_first_round=sum(x['empty'] for x in rr),
            max_propagation_edges=(max(edges) if edges else None),
            skipped=skipped,
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