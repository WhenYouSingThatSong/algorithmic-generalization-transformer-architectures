#!/usr/bin/env python3
"""Long-window B stability analysis from existing JSONL trajectories.

For each node in all_affected (and optionally other roles), using VALUE output
correctness against the matched clean trajectory, reports:

- nodes correct at the checkpoint round and whether they error after it;
- nodes that were damaged, recovered, and then errored again;
- whether the final contiguous correct run starts after the checkpoint;
- the usual last-two-round stable flag.

CPU-only; reads the runner JSONL sidecars and their meta files.  Pairing uses
the public b_main_summary target key, which distinguishes early event ids from
late target identity and guards duplicate clean records.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_main_summary as M  # noqa: E402


def meta_for(sidecar):
    base = str(sidecar)
    if not base.endswith('.jsonl'):
        raise SystemExit(f'expected JSONL sidecar: {sidecar}')
    return Path(base[:-6] + '.meta.json')


def load_source(path):
    meta_path = meta_for(path)
    if not meta_path.exists():
        raise SystemExit(f'missing meta for {path}: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    args = dict(meta.get('args') or {})
    source = M.S.source_identity(str(path), {'args': args})
    model = M.H.model_label(args)
    return source, model, args


def target_identity(rec):
    chain = rec.get('chain') or {}
    target = rec.get('target') or {}
    event_id = rec.get('event_id') or chain.get('event_id') or target.get('event_id')
    target_index = rec.get('target_index')
    if target_index is None:
        target_index = rec.get('chain_index')
    target_pos = target.get('pos')
    if target_pos is None:
        target_pos = chain.get('pos')
    return event_id, target_index, target_pos


def node_series(rounds, pos, metric='value'):
    return [M.H.node_ok(r, pos, metric) for r in rounds]


def median_final_start(starts):
    return statistics.median(starts) if starts else None


def analyze_pair(clean, rec, roles, n_rounds, checkpoint_round):
    out = {}
    n = n_rounds
    cp = int(checkpoint_round)
    cp_idx = cp - 1
    for role, positions in roles.items():
        for pos in sorted(positions):
            cb = node_series(clean.get('rounds', [])[:n], pos)
            jb = node_series(rec.get('rounds', [])[:n], pos)
            if len(cb) < n or len(jb) < n:
                continue
            ok = [bool(cb[i] and jb[i]) for i in range(n)]
            err = [bool(cb[i] and not jb[i]) for i in range(n)]
            if not cb[cp_idx]:
                continue
            row = out.setdefault((role, pos), dict(
                n_cp=0, post=0, damaged=0, recovered=0, rec_err=0,
                rec_err_after_cp=0, rec_err_before_cp=0, stable=0,
                le_cp=0, cp1_2cp=0, cp2_3cp=0, gt_3cp=0, none=0,
                final_starts=[], first_damage=None, first_recovery=None))
            if jb[cp_idx]:
                row['n_cp'] += 1
                row['post'] += int(any(err[i] for i in range(cp, n)))
            damage_ts = [i for i in range(n) if err[i]]
            if damage_ts:
                row['damaged'] += 1
                row['stable'] += int(all(ok[-2:]))
                first = damage_ts[0]
                row['first_damage'] = first + 1
                first_rec = next((i for i in range(first + 1, n) if ok[i]), None)
                if first_rec is not None:
                    row['recovered'] += 1
                    row['first_recovery'] = first_rec + 1
                    later_err = [i for i in range(first_rec + 1, n) if err[i]]
                    row['rec_err'] += int(bool(later_err))
                    row['rec_err_after_cp'] += int(
                        any(i >= cp for i in later_err))
                    row['rec_err_before_cp'] += int(
                        any(i < cp for i in later_err))
                if ok[-1]:
                    start = n - 1
                    while start > 0 and ok[start - 1]:
                        start -= 1
                    start1 = start + 1
                    row['final_starts'].append(start1)
                    if start1 <= cp:
                        row['le_cp'] += 1
                    elif start1 <= 2 * cp:
                        row['cp1_2cp'] += 1
                    elif start1 <= 3 * cp:
                        row['cp2_3cp'] += 1
                    else:
                        row['gt_3cp'] += 1
                else:
                    row['none'] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True,
                   help='JSONL sidecars from a single or multiple run sources')
    p.add_argument('--out_csv', required=True)
    p.add_argument('--out_nodes_csv', default=None,
                   help='optional per-node event rows')
    p.add_argument('--rounds', type=int, default=64)
    p.add_argument('--checkpoint_round', type=int, default=16)
    a = p.parse_args()
    if a.checkpoint_round < 1 or a.rounds <= a.checkpoint_round:
        p.error('need 1 <= checkpoint_round < rounds')

    # group[(source, model, start, timing, condition, role, graph_id)]
    groups = defaultdict(lambda: dict(
        n_cp=0, post=0, damaged=0, recovered=0, rec_err=0,
        rec_err_after_cp=0, rec_err_before_cp=0, stable=0,
        le_cp=0, cp1_2cp=0, cp2_3cp=0, gt_3cp=0, none=0,
        final_starts=[]))
    node_rows = []
    duplicates = 0
    missing_clean = 0
    for path in a.inputs:
        source, model, args = load_source(path)
        cleans = {}
        injected = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                key = M.clean_key(rec, source, args)
                if rec.get('condition') == 'clean':
                    if key in cleans:
                        duplicates += 1
                        if json.dumps(cleans[key].get('rounds'), sort_keys=True) != \
                                json.dumps(rec.get('rounds'), sort_keys=True):
                            raise SystemExit(
                                f'conflicting clean records in {path}: {key}')
                    else:
                        cleans[key] = rec
                else:
                    injected.append(rec)
        for rec in injected:
            key = M.clean_key(rec, source, args)
            clean = cleans.get(key)
            if clean is None:
                missing_clean += 1
                continue
            roles = M.roles_for(rec)
            if not roles:
                continue
            start, timing = M.S.record_start_timing(rec, args)
            cond = str(rec.get('condition'))
            event_id, target_index, target_pos = target_identity(rec)
            n_rounds = min(a.rounds, len(clean.get('rounds', [])),
                           len(rec.get('rounds', [])))
            if n_rounds <= a.checkpoint_round:
                continue
            per_node = analyze_pair(clean, rec, roles, n_rounds,
                                    a.checkpoint_round)
            gid = int(rec['graph_id'])
            for (role, pos), v in per_node.items():
                g = groups[(source, model, start, timing, cond, role, gid)]
                for key in ('n_cp', 'post', 'damaged', 'recovered',
                            'rec_err', 'rec_err_after_cp', 'rec_err_before_cp',
                            'stable', 'le_cp', 'cp1_2cp',
                            'cp2_3cp', 'gt_3cp', 'none'):
                    g[key] += v[key]
                g['final_starts'].extend(v['final_starts'])
                if a.out_nodes_csv:
                    node_rows.append(dict(
                        source=source, model=model, start=start, timing=timing,
                        condition=cond, role=role, graph_id=gid,
                        event_id='' if event_id is None else event_id,
                        target_index='' if target_index is None else target_index,
                        target_pos='' if target_pos is None else target_pos,
                        pos=pos, checkpoint_round=a.checkpoint_round,
                        n_correct_checkpoint=v['n_cp'],
                        post_checkpoint_error=v['post'],
                        damaged=v['damaged'], recovered=v['recovered'],
                        recovered_then_error=v['rec_err'],
                        recovered_then_error_after_checkpoint=v['rec_err_after_cp'],
                        recovered_then_error_before_checkpoint=v['rec_err_before_cp'],
                        first_damage='' if v['first_damage'] is None else v['first_damage'],
                        first_recovery='' if v['first_recovery'] is None else v['first_recovery'],
                        stable=v['stable'],
                        final_run_start=(v['final_starts'][0]
                                         if v['final_starts'] else '')))
    if missing_clean:
        raise SystemExit(f'{missing_clean} injected records missing clean pair; '
                         'refusing to silently omit them')

    rows = []
    for (source, model, start, timing, cond, role, graph_id), g in sorted(groups.items()):
        rows.append(dict(source=source, model=model, start=start, timing=timing,
                         condition=cond, role=role, graph_id=graph_id,
                         **{k: g[k] for k in
                            ('n_cp', 'post', 'damaged', 'recovered',
                             'rec_err', 'rec_err_after_cp',
                             'rec_err_before_cp', 'stable', 'le_cp',
                             'cp1_2cp', 'cp2_3cp', 'gt_3cp', 'none')},
                         final_starts=g['final_starts']))

    agg = {}
    for r in rows:
        k = (r['source'], r['model'], r['start'], r['timing'], r['condition'],
             r['role'])
        arow = agg.setdefault(k, dict(
            n_cp=0, post=0, damaged=0, recovered=0, rec_err=0,
            rec_err_after_cp=0, rec_err_before_cp=0, stable=0,
            le_cp=0, cp1_2cp=0, cp2_3cp=0, gt_3cp=0, none=0,
            ratios=defaultdict(list), final_starts=[]))
        for key in ('n_cp', 'post', 'damaged', 'recovered', 'rec_err',
                    'rec_err_after_cp', 'rec_err_before_cp', 'stable',
                    'le_cp', 'cp1_2cp', 'cp2_3cp', 'gt_3cp', 'none'):
            arow[key] += r[key]
        arow['final_starts'].extend(r['final_starts'])
        if r['n_cp']:
            arow['ratios']['post'].append(r['post'] / r['n_cp'])
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
        median_start = median_final_start(v['final_starts'])
        out_rows.append(dict(
            source=k[0], model=k[1], start=k[2], timing=k[3],
            condition=k[4], role=k[5], checkpoint_round=a.checkpoint_round,
            n_nodes_correct_checkpoint=v['n_cp'],
            post_checkpoint_error_num=v['post'],
            post_checkpoint_error_den=v['n_cp'],
            post_checkpoint_error_node_weighted=rate(v['post'], v['n_cp']),
            post_checkpoint_error_graph_equal=ge(v['ratios']['post']),
            damaged=v['damaged'], recovered=v['recovered'],
            recovered_then_error_num=v['rec_err'],
            recovered_then_error_den=v['recovered'],
            recovered_then_error_node_weighted=rate(v['rec_err'], v['recovered']),
            recovered_then_error_graph_equal=ge(v['ratios']['rec_err']),
            recovered_then_error_after_checkpoint_num=v['rec_err_after_cp'],
            recovered_then_error_before_checkpoint_num=v['rec_err_before_cp'],
            stable_num=v['stable'], stable_den=v['damaged'],
            stable_node_weighted=rate(v['stable'], v['damaged']),
            stable_graph_equal=ge(v['ratios']['stable']),
            final_run_start_den=v['damaged'],
            final_run_start_1_cp=v['le_cp'],
            final_run_start_cp1_2cp=v['cp1_2cp'],
            final_run_start_2cp1_3cp=v['cp2_3cp'],
            final_run_start_gt_3cp=v['gt_3cp'],
            final_run_start_none=v['none'],
            final_run_start_median=median_start,
        ))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps(dict(rows=len(out_rows), duplicates=duplicates,
                          out=a.out_csv), ensure_ascii=True))
    if a.out_nodes_csv:
        with open(a.out_nodes_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()))
            w.writeheader()
            w.writerows(node_rows)
        print(json.dumps(dict(node_rows=len(node_rows), out=a.out_nodes_csv),
                         ensure_ascii=True))


if __name__ == '__main__':
    main()