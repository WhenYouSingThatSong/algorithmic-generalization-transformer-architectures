#!/usr/bin/env python3
"""Main B mechanism summary for early, complete and sustained trajectories.

This is a CPU-only summary.  It covers the condition matrix planned for B2
(before_u/after_u x clean/U/R/U+R, complete internal/sink/leaf/unrelated and
sustained wrong-parent controls) while preserving the four evaluation choices:
full-window round-by-round, stable = last two rounds, graph-equal primary plus
node-weighted, VALUE and first-four in parallel.

It does not replace the pilot-only ``b_mechanism_horizon_summary.py``; that
file remains the reference for the six pilot runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_mechanism_horizon_summary as H  # noqa: E402
import b_mechanism_summary as S  # noqa: E402

HORIZONS = (8, 16, 64)
METRICS = ('value', 'first4')
ROLES = ('direct', 'target_u', 'downstream_d1', 'downstream_d2plus', 'all_affected')


def condition_label(rec):
    cond = rec.get('condition')
    timing = rec.get('timing')
    if timing in ('before_u', 'after_u') and cond in ('u', 'r', 'ur'):
        suffix = {'u': 'U', 'r': 'R', 'ur': 'UR'}[cond]
        return f'{timing}_{suffix}'
    if cond == 'late_u':
        return 'complete_U'
    if cond == 'late_r':
        return 'complete_R'
    if cond == 'sink':
        return 'complete_sink'
    if cond == 'leaf_value':
        return 'complete_leaf_value'
    if cond == 'unrelated':
        return 'complete_unrelated'
    if rec.get('mode') == 'sustained_r' or cond == 'sustained_r':
        return 'sustained_R'
    return None


def run_label(rec):
    cond = condition_label(rec) or ''
    if cond.startswith('before_u') or cond.startswith('after_u'):
        return str(rec.get('start') or 'own'), str(rec.get('timing'))
    if cond.startswith('sustained'):
        return 'control', 'sustained_r'
    return 'late', 'complete'


def clean_key(rec, source, args):
    return S.target_key(rec, source, args)


def pair_file(path, data):
    args = data.get('args', {}) or {}
    source = S.source_identity(path, data)
    model = H.model_label(args)
    cleans = {}
    injected = []
    for rec in data.get('records', []):
        cond = rec.get('condition')
        if cond == 'clean' and rec.get('rounds'):
            key = clean_key(rec, source, args)
            if key in cleans:
                old = cleans[key]
                if json.dumps(old.get('rounds'), sort_keys=True) != \
                        json.dumps(rec.get('rounds'), sort_keys=True):
                    raise SystemExit(f'conflicting clean records for {key}')
                continue
            cleans[key] = rec
        elif condition_label(rec) is not None:
            injected.append(rec)
    for rec in injected:
        cond = condition_label(rec)
        clean = cleans.get(clean_key(rec, source, args))
        yield model, source, args, rec, clean, cond


def center_for(rec):
    chain = rec.get('chain') or {}
    target = rec.get('target') or {}
    if 'pos' in chain:
        return int(chain['pos'])
    if 'pos' in target:
        return int(target['pos'])
    if 'var_pos' in target:
        return int(target['var_pos'])
    inj = rec.get('injection') or {}
    if 'pos' in inj:
        return int(inj['pos'])
    return None


def roles_for(rec):
    children = H.children_from_truth(rec.get('node_truth', {}))
    direct = H.injected_positions(rec)
    center = center_for(rec)
    if center is None:
        return None
    dist = H.descendants(children, center)
    roles = {
        'direct': set(direct),
        'target_u': {center} - set(direct),
        'downstream_d1': {p for p, d in dist.items() if d == 1} - set(direct),
        'downstream_d2plus': {p for p, d in dist.items() if d >= 2} - set(direct),
    }
    roles['all_affected'] = set().union(*roles.values())
    return roles


def build_cache(rec, clean, roles, max_rounds):
    offset = int(rec.get('hold_rounds', 0) or 0)
    cr, ir = clean.get('rounds', []), rec.get('rounds', [])
    n = min(max_rounds, max(0, len(cr) - offset), max(0, len(ir) - offset))
    if n < 1:
        return None
    cache = {}
    for role in ROLES:
        for metric in METRICS:
            entries = []
            for pos in sorted(roles[role]):
                cb = [H.node_ok(cr[offset + i], pos, metric) for i in range(n)]
                jb = [H.node_ok(ir[offset + i], pos, metric) for i in range(n)]
                entries.append((int(pos), cb, jb))
            cache[(role, metric)] = entries
    all_nodes = sorted(int(k) for k in (rec.get('node_truth') or {}))
    all_cache = {}
    for metric in METRICS:
        entries = []
        for pos in all_nodes:
            cb = [H.node_ok(cr[offset + i], pos, metric) for i in range(n)]
            jb = [H.node_ok(ir[offset + i], pos, metric) for i in range(n)]
            entries.append((int(pos), cb, jb))
        all_cache[metric] = entries
    return dict(roles=roles, offset=offset, n=n, cache=cache,
                all_cache=all_cache, all_nodes=all_nodes)


def input_value(round_rec, pos):
    return (round_rec.get('input') or {}).get(str(pos))


def direct_first_round(rec, clean, roles):
    if not roles['direct']:
        return None
    phase = 'hold' if int(rec.get('hold_rounds', 0) or 0) else 'post_injection'
    actual = empty_to_value = empty_correct = empty_wrong = 0
    corrected = wrong = 0
    for pos in sorted(roles['direct']):
        cin = input_value(clean['rounds'][0], pos)
        jin = input_value(rec['rounds'][0], pos)
        out_ok = H.node_ok(rec['rounds'][0], pos, 'first4')
        if cin in ('EMPTY', 'N/A') and jin not in (None, 'EMPTY', 'N/A'):
            empty_to_value += 1
            if out_ok:
                empty_correct += 1
            else:
                empty_wrong += 1
        if cin not in (None, 'EMPTY', 'N/A') and jin not in (None, 'EMPTY', 'N/A') \
                and cin != jin:
            actual += 1
            if out_ok:
                corrected += 1
            else:
                wrong += 1
    return dict(phase=phase, n_direct=len(roles['direct']), actual=actual,
                empty_to_value=empty_to_value,
                empty_first_round_correct=empty_correct,
                empty_first_round_wrong=empty_wrong,
                first_round_corrected=corrected, first_round_wrong=wrong)


def clean_drift_rows(cache, model, source, start, timing, cond):
    out = []
    for metric, entries in cache['all_cache'].items():
        for Hn in HORIZONS:
            if Hn > cache['n']:
                continue
            clean_ok = sum(1 for _p, cb, _j in entries if cb[Hn - 1])
            total = len(entries)
            out.append(dict(model=model, source=source, start=start, timing=timing,
                            condition=cond, metric=metric, H=Hn,
                            clean_correct=clean_ok, clean_total=total,
                            clean_node_acc=(clean_ok / total if total else None)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--stable_len', type=int, default=2)
    p.add_argument('--max_rounds', type=int, default=64)
    p.add_argument('--horizons', default=','.join(str(x) for x in HORIZONS))
    a = p.parse_args()
    horizons = [int(x) for x in a.horizons.split(',') if x.strip()]
    os.makedirs(a.out_dir, exist_ok=True)

    h_stats = defaultdict(dict)
    h_targets = defaultdict(set)
    round_stats = defaultdict(dict)
    acc_stats = defaultdict(dict)
    hold_stats = defaultdict(dict)
    coverage = defaultdict(lambda: dict(n_targets=0, n_graphs=set(),
                                        n_skipped_no_clean=0, n_skipped_short=0))
    direct_rows = []
    drift_rows = []
    pair_keys = defaultdict(lambda: dict(early=set(), complete=set()))

    for path in a.inputs:
        with open(path) as f:
            data = json.load(f)
        for model, source, args, rec, clean, cond in pair_file(path, data):
            start, timing = run_label(rec)
            key = (model, source, start, timing, cond)
            if clean is None:
                coverage[key]['n_skipped_no_clean'] += 1
                continue
            roles = roles_for(rec)
            if roles is None:
                coverage[key]['n_skipped_short'] += 1
                continue
            cache = build_cache(rec, clean, roles, a.max_rounds)
            if cache is None:
                coverage[key]['n_skipped_short'] += 1
                continue
            gid = int(rec.get('graph_id'))
            tid = rec.get('target_index')
            if tid is None:
                tid = rec.get('chain_index')
            tid = int(tid) if tid is not None else -1
            coverage[key]['n_targets'] += 1
            coverage[key]['n_graphs'].add(gid)

            d = direct_first_round(rec, clean, roles)
            if d:
                direct_rows.append(dict(
                    model=model, source=source, start=start, timing=timing,
                    condition=cond, graph_id=gid, target_id=tid, **d))

            # Clean drift once per record.
            drift_rows.extend(clean_drift_rows(
                cache, model, source, start, timing, cond))

            if cache['offset'] > 0:
                for (role, metric), entries in H.hold_entries(
                        rec, clean, roles, cache['offset']).items():
                    for i in range(cache['offset']):
                        obs = dam = forced = obs_in = 0
                        for _pos, cb, jb, ib, cin in entries:
                            if cin[i]:
                                obs_in += 1
                                if not ib[i]:
                                    forced += 1
                            if cb[i]:
                                obs += 1
                                if not jb[i]:
                                    dam += 1
                        if obs or obs_in:
                            hkey = (model, source, start, timing, cond, role, metric, i + 1)
                            H.add_graph(hold_stats[hkey], gid,
                                        [obs, dam, forced, obs_in])

            for Hn in horizons:
                if Hn > cache['n']:
                    continue
                ev = H.summarize_events(cache, Hn, stable_len=a.stable_len)
                for (role, metric), v in ev.items():
                    if not v['obs']:
                        continue
                    hkey = (model, source, start, timing, cond, role, metric, Hn)
                    h_targets[hkey].add((gid, tid))
                    H.add_graph(h_stats[hkey], gid, [
                        v['obs'], v['damaged'], v['stable'], v['ever'],
                        v['only_last'], v['stable_clean']])

            for (role, metric), entries in cache['cache'].items():
                for i in range(cache['n']):
                    obs = dam = 0
                    for _pos, cb, jb in entries:
                        if cb[i]:
                            obs += 1
                            if not jb[i]:
                                dam += 1
                    if obs:
                        rkey = (model, source, start, timing, cond, role, metric, i + 1)
                        H.add_graph(round_stats[rkey], gid, [obs, dam])

            for metric, entries in cache['all_cache'].items():
                for i in range(cache['n']):
                    clean_ok = sum(1 for _p, cb, _j in entries if cb[i])
                    inj_ok = sum(1 for _p, _c, jb in entries if jb[i])
                    total = len(entries)
                    if total:
                        akey = (model, source, start, timing, cond, metric, i + 1)
                        H.add_graph(acc_stats[akey], gid, [clean_ok, inj_ok, total])

            eid = (rec.get('chain') or rec.get('target') or {}).get('event_id')
            if eid and start == 'own' and cond.startswith(('before_u', 'after_u')):
                pair_keys[model]['early'].add((gid, eid))
            if eid and cond in ('complete_U', 'complete_R'):
                pair_keys[model]['complete'].add((gid, eid))

    def write_csv(name, rows):
        if not rows:
            return
        with open(os.path.join(a.out_dir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    main_rows = []
    for (model, source, start, timing, cond, role, metric, Hn), gstats in sorted(h_stats.items()):
        n_targets = len(h_targets[(model, source, start, timing, cond, role, metric, Hn)])
        for measure, num_i, den_i in (
                ('damage', 1, 0), ('stable', 2, 1), ('ever', 3, 1),
                ('only_last', 4, 1), ('stable_clean', 5, 1)):
            r = H.rate_view(gstats, num_i, den_i)
            main_rows.append(dict(
                model=model, source=source, start=start, timing=timing,
                condition=cond, role=role, metric=metric, H=Hn,
                measure=measure, n_targets=n_targets, n_graphs=len(gstats), **r))
    write_csv('main_table.csv', main_rows)

    r_rows = []
    for (model, source, start, timing, cond, role, metric, rnd), gstats in sorted(round_stats.items()):
        r = H.rate_view(gstats, 1, 0)
        r_rows.append(dict(model=model, source=source, start=start, timing=timing,
                           condition=cond, role=role, metric=metric, round=rnd, **r))
    write_csv('per_round_curves.csv', r_rows)

    acc_rows = []
    for (model, source, start, timing, cond, metric, rnd), gstats in sorted(acc_stats.items()):
        def acc(idx):
            num = sum(v[idx] for v in gstats.values())
            den = sum(v[2] for v in gstats.values())
            per = [v[idx] / v[2] for v in gstats.values() if v[2]]
            return (num / den if den else None,
                    (sum(per) / len(per)) if per else None)
        c_node, c_graph = acc(0)
        j_node, j_graph = acc(1)
        acc_rows.append(dict(
            model=model, source=source, start=start, timing=timing,
            condition=cond, metric=metric, round=rnd,
            clean_acc_node_weighted=c_node, clean_acc_graph_equal=c_graph,
            inj_acc_node_weighted=j_node, inj_acc_graph_equal=j_graph,
            n_nodes=sum(v[2] for v in gstats.values())))
    write_csv('per_round_accuracy.csv', acc_rows)

    hold_rows = []
    for (model, source, start, timing, cond, role, metric, rnd), gstats in sorted(hold_stats.items()):
        hold_rows.append(dict(
            model=model, source=source, start=start, timing=timing,
            condition=cond, role=role, metric=metric, round=rnd,
            damage=H.rate_view(gstats, 1, 0),
            forced=H.rate_view(gstats, 2, 3)))
    # Flatten nested rate views for CSV.
    flat_hold = []
    for row in hold_rows:
        base = {k: v for k, v in row.items() if k not in ('damage', 'forced')}
        for prefix in ('damage', 'forced'):
            for k, v in row[prefix].items():
                base[prefix + '_' + k] = v
        flat_hold.append(base)
    write_csv('sustained_hold.csv', flat_hold)
    write_csv('direct_first_round.csv', direct_rows)
    write_csv('clean_drift.csv', drift_rows)

    cov_rows = []
    for (model, source, start, timing, cond), v in sorted(coverage.items()):
        cov_rows.append(dict(model=model, source=source, start=start, timing=timing,
                             condition=cond, n_targets=v['n_targets'],
                             n_graphs=len(v['n_graphs']),
                             n_skipped_no_clean=v['n_skipped_no_clean'],
                             n_skipped_short=v['n_skipped_short']))
    write_csv('coverage.csv', cov_rows)

    pair_rows = []
    for model in sorted(pair_keys):
        e = pair_keys[model]['early']
        c = pair_keys[model]['complete']
        pair_rows.append(dict(model=model, n_early=len(e), n_complete=len(c),
                              n_intersection=len(e & c)))
    write_csv('paired_intersection.csv', pair_rows)

    print(json.dumps(dict(main_rows=len(main_rows), direct=len(direct_rows),
                          coverage=len(cov_rows), pairs=pair_rows), indent=2))
    print('WROTE', a.out_dir)


if __name__ == '__main__':
    main()
