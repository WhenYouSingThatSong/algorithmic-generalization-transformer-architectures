#!/usr/bin/env python3
"""Long-horizon B mechanism summary.

Consumes the long-horizon early/late/controls JSON files and implements the
evaluation choices recorded in ``ai-collab/consensus.md``:

- full-window round-by-round curves, with first-round and ever-damaged summaries;
- stable recovery = correct in the last ``L`` rounds of the window (L=2 by
  default); ever-recovered and only-last-round are reported separately;
- graph-equal weighting is primary, node-weighted is secondary;
- VALUE and first-four node correctness are reported separately;
- a node is called damaged only when the same-round clean branch is correct and
  the injected branch is wrong.  Early EMPTY/not-yet nodes are therefore not
  silently counted as damage;
- for sustained-parent controls, H is measured after release.

This is a provisional analysis helper, not a model runner.  It reads existing
trajectory JSONs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_mechanism_summary as S  # noqa: E402

HORIZONS = (2, 4, 8, 16, 32, 64)
METRICS = ('value', 'first4')
ROLES = ('direct', 'target_u', 'downstream_d1', 'downstream_d2plus', 'all_affected')
PILOT_CONDS = ('after_u_U', 'after_u_R', 'complete_U', 'complete_R', 'sustained_R')


def model_label(args):
    cfg = str(args.get('config', ''))
    if 'A-L4D384fast' in cfg:
        return 'A_prime'
    if 'B-M6' in cfg:
        return 'B_M6'
    return cfg or 'unknown'


def condition_label(rec):
    timing = rec.get('timing')
    cond = rec.get('condition')
    if timing == 'after_u' and cond in ('u', 'r'):
        return f'after_u_{cond.upper()}'
    if cond == 'late_u':
        return 'complete_U'
    if cond == 'late_r':
        return 'complete_R'
    if rec.get('mode') == 'sustained_r' or cond == 'sustained_r':
        return 'sustained_R'
    return None


def node_ok(round_rec, pos, metric, state='output'):
    node = (round_rec.get(f'{state}_nodes') or {}).get(str(pos))
    if node is None:
        return False
    key = 'value_correct' if metric == 'value' else 'first4_correct'
    return bool(node.get(key))


def children_from_truth(node_truth):
    out = {}
    for k, v in (node_truth or {}).items():
        out[int(k)] = [int(x) for x in v.get('children', [])]
    return out


def descendants(children, start):
    out = {}
    q = deque([(int(start), 0)])
    seen = {int(start)}
    while q:
        x, d = q.popleft()
        for y in children.get(x, []):
            if y in seen:
                continue
            seen.add(y)
            out[y] = d + 1
            q.append((y, d + 1))
    return out


def injected_positions(rec):
    """Positions actually overwritten before the first rollout forward.

    Early records predate the ``injection.pos`` field, so their injected set
    must come from the 2x2 condition; late/control records carry it directly.
    """
    chain = rec.get('chain') or rec.get('target') or {}
    cond = rec.get('condition')
    if rec.get('timing') in ('before_u', 'after_u'):
        u, r = chain.get('pos'), chain.get('r_pos')
        if cond == 'u':
            return {int(u)}
        if cond == 'r':
            return {int(r)}
        if cond == 'ur':
            return {int(u), int(r)}
        return set()
    if rec.get('mode') in ('sustained_r', 'switch_held_r'):
        r = chain.get('r_pos')
        return set() if r is None else {int(r)}
    pos = (rec.get('injection') or {}).get('pos')
    return set() if pos is None else {int(pos)}


def role_sets(rec):
    chain = rec.get('chain') or rec.get('target') or {}
    u = chain.get('pos')
    if u is None:
        return None
    u = int(u)
    direct = injected_positions(rec)
    dist = descendants(children_from_truth(rec.get('node_truth', {})), u)
    roles = {
        'direct': direct,
        'target_u': {u} - direct,
        'downstream_d1': {p for p, d in dist.items() if d == 1} - direct,
        'downstream_d2plus': {p for p, d in dist.items() if d >= 2} - direct,
    }
    roles['all_affected'] = set().union(*roles.values())
    return roles


def pair_file(path, data):
    """Yield (model, source, args, injected, clean, skipped_reason)."""
    args = data.get('args', {}) or {}
    source = S.source_identity(path, data)
    model = model_label(args)
    cleans = {}
    injected = []
    for rec in data.get('records', []):
        cond = rec.get('condition')
        if cond == 'clean' and rec.get('rounds'):
            key = S.target_key(rec, source, args)
            if key in cleans:
                old = cleans[key]
                if json.dumps(old.get('rounds'), sort_keys=True) != \
                        json.dumps(rec.get('rounds'), sort_keys=True):
                    raise SystemExit(f'conflicting clean records for {key}')
                continue
            cleans[key] = rec
        elif cond in ('u', 'r', 'late_u', 'late_r', 'sustained_r') or \
                rec.get('mode') == 'sustained_r':
            injected.append(rec)
    for rec in injected:
        cond_label = condition_label(rec)
        if cond_label is None:
            yield model, source, args, rec, None, 'unknown_condition'
            continue
        clean = cleans.get(S.target_key(rec, source, args))
        if clean is None:
            yield model, source, args, rec, None, 'no_clean'
            continue
        yield model, source, args, rec, clean, None


def build_cache(rec, clean, max_rounds=64):
    roles = role_sets(rec)
    if roles is None:
        return None
    offset = int(rec.get('hold_rounds', 0) or 0)
    cr = clean.get('rounds', [])
    ir = rec.get('rounds', [])
    n = min(max_rounds, max(0, len(cr) - offset), max(0, len(ir) - offset))
    if n < 1:
        return None
    cache = {}
    for role in ROLES:
        for metric in METRICS:
            entries = []
            for pos in sorted(roles[role]):
                cb = [node_ok(cr[offset + i], pos, metric) for i in range(n)]
                jb = [node_ok(ir[offset + i], pos, metric) for i in range(n)]
                entries.append((int(pos), cb, jb))
            cache[(role, metric)] = entries
    all_nodes = sorted(int(k) for k in (rec.get('node_truth') or {}))
    all_cache = {}
    for metric in METRICS:
        entries = []
        for pos in all_nodes:
            cb = [node_ok(cr[offset + i], pos, metric) for i in range(n)]
            jb = [node_ok(ir[offset + i], pos, metric) for i in range(n)]
            entries.append((int(pos), cb, jb))
        all_cache[metric] = entries
    return dict(roles=roles, offset=offset, n=n, cache=cache,
                all_cache=all_cache, all_nodes=all_nodes)


def hold_entries(rec, clean, roles, n_hold):
    """Output/input correctness during the forced-hold window itself.

    These rounds are reported separately from post-release recovery; the H
    table never uses them.
    """
    out = {}
    for role in ROLES:
        for metric in METRICS:
            entries = []
            for pos in sorted(roles[role]):
                cb = [node_ok(clean['rounds'][i], pos, metric) for i in range(n_hold)]
                jb = [node_ok(rec['rounds'][i], pos, metric) for i in range(n_hold)]
                ib = [node_ok(rec['rounds'][i], pos, metric, 'input')
                      for i in range(n_hold)]
                cin = [node_ok(clean['rounds'][i], pos, metric, 'input')
                       for i in range(n_hold)]
                entries.append((int(pos), cb, jb, ib, cin))
            out[(role, metric)] = entries
    return out


def summarize_events(cache, H, stable_len=2):
    out = {}
    for (role, metric), entries in cache['cache'].items():
        obs = damaged = stable = ever = only_last = stable_clean = 0
        events = []
        for pos, cb, jb in entries:
            if H > len(cb):
                continue
            c = cb[:H]
            j = jb[:H]
            if any(c):
                obs += 1
            damage_ts = [i for i in range(H) if c[i] and not j[i]]
            if not damage_ts:
                continue
            damaged += 1
            first = damage_ts[0]
            first_recovery = next((i for i in range(first + 1, H) if j[i]), None)
            ev = first_recovery is not None
            st = H >= stable_len and all(j[H - stable_len:])
            ol = H >= 2 and bool(j[-1]) and not bool(j[-2])
            sc = st and all(c[H - stable_len:])
            last_run_start = None
            if j and j[-1]:
                last_run_start = H - 1
                while last_run_start > 0 and j[last_run_start - 1]:
                    last_run_start -= 1
            ever += int(ev)
            stable += int(st)
            only_last += int(ol)
            stable_clean += int(sc)
            events.append(dict(
                pos=pos, first_damage=first + 1, first_recovery=None if first_recovery is None else first_recovery + 1,
                ever_recovered=ev, stable_recovered=st, only_last=ol,
                stable_clean=sc,
                last_run_start=None if last_run_start is None else last_run_start + 1,
            ))
        out[(role, metric)] = dict(obs=obs, damaged=damaged, stable=stable,
                                   ever=ever, only_last=only_last,
                                   stable_clean=stable_clean, events=events)
    return out


def add_graph(stats, gid, values):
    cur = stats.setdefault(int(gid), [0] * len(values))
    for i, v in enumerate(values):
        cur[i] += int(v)


def rate_view(graph_stats, num_idx, den_idx):
    n_num = sum(v[num_idx] for v in graph_stats.values())
    n_den = sum(v[den_idx] for v in graph_stats.values())
    per_graph = []
    zero = 0
    for v in graph_stats.values():
        if v[den_idx]:
            per_graph.append(v[num_idx] / v[den_idx])
        elif v[num_idx] == 0:
            zero += 1
    graph_equal = (sum(per_graph) / len(per_graph)) if per_graph else None
    node_weighted = (n_num / n_den) if n_den else None
    return dict(numerator=n_num, denominator=n_den,
                node_weighted=node_weighted, graph_equal=graph_equal,
                n_graphs_with_denominator=len(per_graph),
                n_zero_denominator_graphs=zero)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--manifest', default=None,
                   help='optional selection manifest for coverage summary')
    p.add_argument('--stable_len', type=int, default=2)
    p.add_argument('--horizons', default=','.join(str(x) for x in HORIZONS))
    p.add_argument('--max_rounds', type=int, default=64)
    a = p.parse_args()
    horizons = [int(x) for x in a.horizons.split(',') if x.strip()]
    max_h = max(horizons) if horizons else 0
    os.makedirs(a.out_dir, exist_ok=True)

    h_stats = defaultdict(dict)
    h_targets = defaultdict(set)
    round_stats = defaultdict(dict)
    all_round_stats = defaultdict(dict)
    hold_stats = defaultdict(dict)
    event_rows = []
    target_rows = []
    cases = []
    coverage = defaultdict(lambda: dict(n_targets=0, n_graphs=set(),
                                        n_skipped_no_clean=0,
                                        n_skipped_short=0))
    case_seen = set()

    for path in a.inputs:
        with open(path) as f:
            data = json.load(f)
        for model, source, args, rec, clean, reason in pair_file(path, data):
            cond = condition_label(rec) if rec is not None else None
            if reason is not None or cond not in PILOT_CONDS:
                if cond is not None:
                    key = (model, cond)
                    coverage[key]['n_skipped_' + ('no_clean' if reason == 'no_clean' else 'short')] += 1
                continue
            cache = build_cache(rec, clean, max_rounds=a.max_rounds)
            if cache is None:
                coverage[(model, cond)]['n_skipped_short'] += 1
                continue
            gid = int(rec.get('graph_id'))
            target_id = rec.get('target_index')
            if target_id is None:
                target_id = rec.get('chain_index')
            target_id = int(target_id) if target_id is not None else -1
            key = (model, cond)
            coverage[key]['n_targets'] += 1
            coverage[key]['n_graphs'].add(gid)

            # Forced-hold window is reported separately from post-release H.
            if cache['offset'] > 0:
                for (role, metric), entries in hold_entries(
                        rec, clean, cache['roles'], cache['offset']).items():
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
                            hkey = (model, cond, role, metric, i + 1)
                            add_graph(hold_stats[hkey], gid,
                                      [obs, dam, forced, obs_in])

            # H-table events.
            for H in horizons:
                if H > cache['n']:
                    continue
                ev = summarize_events(cache, H, stable_len=a.stable_len)
                for (role, metric), v in ev.items():
                    if not v['obs']:
                        continue
                    hkey = (model, cond, role, metric, H)
                    h_targets[hkey].add((gid, target_id))
                    add_graph(h_stats[hkey], gid, [
                        v['obs'], v['damaged'], v['stable'], v['ever'],
                        v['only_last'], v['stable_clean']])
                    if H == max_h:
                        for e in v['events']:
                            event_rows.append(dict(
                                model=model, condition=cond, role=role,
                                metric=metric, graph_id=gid,
                                target_id=target_id, H=H, pos=e['pos'],
                                first_damage=e['first_damage'],
                                first_recovery=e['first_recovery'],
                                ever_recovered=e['ever_recovered'],
                                stable_recovered=e['stable_recovered'],
                                only_last=e['only_last'],
                                stable_clean=e['stable_clean'],
                                last_run_start=e['last_run_start']))

            # Per-round curves.
            for (role, metric), entries in cache['cache'].items():
                for i in range(cache['n']):
                    obs = dam = 0
                    for _pos, cb, jb in entries:
                        c, j = cb[i], jb[i]
                        if c:
                            obs += 1
                            if not j:
                                dam += 1
                    if obs:
                        rkey = (model, cond, role, metric, i + 1)
                        add_graph(round_stats[rkey], gid, [obs, dam])

            # All-numeric accuracy per round.
            for metric, entries in cache['all_cache'].items():
                for i in range(cache['n']):
                    clean_ok = sum(1 for _p, cb, _j in entries if cb[i])
                    inj_ok = sum(1 for _p, _c, jb in entries if jb[i])
                    total = len(entries)
                    if total:
                        rkey = (model, cond, metric, i + 1)
                        add_graph(all_round_stats[rkey], gid,
                                  [clean_ok, inj_ok, total])

            # Target-level outcome at H=64 (first-four, all affected nodes).
            if cache['n'] >= 64:
                ev = summarize_events(cache, 64, stable_len=a.stable_len)
                v = ev[('all_affected', 'first4')]
                if v['damaged']:
                    if v['stable']:
                        cat = 'stable_fast' if any(
                            e['first_recovery'] is not None and e['first_recovery'] <= 2
                            for e in v['events']) else 'stable_delayed'
                    elif v['only_last']:
                        cat = 'only_last'
                    elif v['ever']:
                        cat = 'recover_then_error'
                    else:
                        cat = 'unrecovered'
                else:
                    cat = 'no_damage'
                firsts = [e['first_recovery'] for e in v['events'] if e['first_recovery'] is not None]
                target_rows.append(dict(
                    model=model, condition=cond, graph_id=gid,
                    target_id=target_id, category=cat,
                    n_observed=v['obs'], n_damaged=v['damaged'],
                    n_stable=v['stable'], n_ever=v['ever'],
                    n_only_last=v['only_last'], n_stable_clean=v['stable_clean'],
                    first_recovery_min=(min(firsts) if firsts else None),
                ))
                ckey = (model, cond, cat)
                if cat in ('stable_fast', 'stable_delayed', 'only_last',
                           'recover_then_error', 'unrecovered') and ckey not in case_seen:
                    case_seen.add(ckey)
                    roles = cache['roles']
                    rounds = []
                    for i in range(cache['n']):
                        item = dict(round=i + 1,
                                    phase='post_release' if cond == 'sustained_R'
                                          else 'post_injection')
                        for role in ('direct', 'target_u', 'downstream_d1',
                                     'downstream_d2plus', 'all_affected'):
                            entries = cache['cache'][(role, 'first4')]
                            item[role] = dict(
                                n_correct=sum(1 for _p, cb, jb in entries if jb[i]),
                                n_total=len(entries))
                        rounds.append(item)
                    cases.append(dict(model=model, condition=cond, category=cat,
                                      graph_id=gid, target_id=target_id,
                                      roles={k: sorted(v) for k, v in roles.items()},
                                      rounds=rounds))

    # H table.
    h_rows = []
    for (model, cond, role, metric, H), gstats in sorted(h_stats.items()):
        n_targets = len(h_targets[(model, cond, role, metric, H)])
        for measure, num_i, den_i in (
                ('damage', 1, 0), ('stable', 2, 1), ('ever', 3, 1),
                ('only_last', 4, 1), ('stable_clean', 5, 1)):
            r = rate_view(gstats, num_i, den_i)
            h_rows.append(dict(
                model=model, condition=cond, role=role, metric=metric, H=H,
                measure=measure, n_targets=n_targets,
                n_graphs=len(gstats), **r))
    with open(os.path.join(a.out_dir, 'horizon_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(h_rows[0].keys()) if h_rows else [])
        if h_rows:
            w.writeheader()
            w.writerows(h_rows)
    with open(os.path.join(a.out_dir, 'horizon_table.json'), 'w') as f:
        json.dump(h_rows, f, indent=2)

    # Per-round damage curves.
    r_rows = []
    for (model, cond, role, metric, rnd), gstats in sorted(round_stats.items()):
        r = rate_view(gstats, 1, 0)
        r_rows.append(dict(model=model, condition=cond, role=role,
                           metric=metric, round=rnd,
                           phase='post_release' if cond == 'sustained_R'
                                 else 'post_injection',
                           **r))
    with open(os.path.join(a.out_dir, 'per_round_curves.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(r_rows[0].keys()) if r_rows else [])
        if r_rows:
            w.writeheader()
            w.writerows(r_rows)

    # Per-round all-numeric accuracy.
    acc_rows = []
    for (model, cond, metric, rnd), gstats in sorted(all_round_stats.items()):
        def acc(idx):
            num = sum(v[idx] for v in gstats.values())
            den = sum(v[2] for v in gstats.values())
            per = [v[idx] / v[2] for v in gstats.values() if v[2]]
            return (num / den if den else None,
                    (sum(per) / len(per)) if per else None)
        c_node, c_graph = acc(0)
        j_node, j_graph = acc(1)
        acc_rows.append(dict(
            model=model, condition=cond, metric=metric, round=rnd,
            phase='post_release' if cond == 'sustained_R' else 'post_injection',
            clean_acc_node_weighted=c_node, clean_acc_graph_equal=c_graph,
            inj_acc_node_weighted=j_node, inj_acc_graph_equal=j_graph,
            n_nodes=sum(v[2] for v in gstats.values())))
    with open(os.path.join(a.out_dir, 'per_round_accuracy.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(acc_rows[0].keys()) if acc_rows else [])
        if acc_rows:
            w.writeheader()
            w.writerows(acc_rows)

    # Forced-hold window (sustained controls only), kept out of the H table.
    hold_rows = []
    for (model, cond, role, metric, rnd), gstats in sorted(hold_stats.items()):
        dam = rate_view(gstats, 1, 0)
        forced = rate_view(gstats, 2, 3)
        hold_rows.append(dict(
            model=model, condition=cond, role=role, metric=metric, round=rnd,
            n_graphs=len(gstats),
            n_observed=sum(v[0] for v in gstats.values()),
            n_damaged=sum(v[1] for v in gstats.values()),
            n_forced=sum(v[2] for v in gstats.values()),
            damage_graph_equal=dam['graph_equal'],
            damage_node_weighted=dam['node_weighted'],
            forced_graph_equal=forced['graph_equal']))
    with open(os.path.join(a.out_dir, 'sustained_hold.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hold_rows[0].keys()) if hold_rows else [])
        if hold_rows:
            w.writeheader()
            w.writerows(hold_rows)

    # Target outcomes and cases.
    with open(os.path.join(a.out_dir, 'target_outcomes.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(target_rows[0].keys()) if target_rows else [])
        if target_rows:
            w.writeheader()
            w.writerows(target_rows)
    with open(os.path.join(a.out_dir, 'events.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()) if event_rows else [])
        if event_rows:
            w.writeheader()
            w.writerows(event_rows)
    with open(os.path.join(a.out_dir, 'cases.json'), 'w') as f:
        json.dump(cases, f, indent=2)

    cov = []
    for (model, cond), v in sorted(coverage.items()):
        cov.append(dict(model=model, condition=cond,
                        n_targets=v['n_targets'], n_graphs=len(v['n_graphs']),
                        n_skipped_no_clean=v['n_skipped_no_clean'],
                        n_skipped_short=v['n_skipped_short']))
    if a.manifest:
        with open(a.manifest) as f:
            m = json.load(f)
        cov.append(dict(model='selection_manifest', condition='all',
                        n_targets=m.get('counts', {}).get('chains', 0),
                        n_graphs=len(m.get('graphs', [])),
                        n_skipped_no_clean=0, n_skipped_short=0,
                        selection_counts=m.get('counts'),
                        ds_sha256=m.get('ds_sha256')))
    with open(os.path.join(a.out_dir, 'coverage.json'), 'w') as f:
        json.dump(cov, f, indent=2)
    print(json.dumps(cov, indent=2))
    print('WROTE', a.out_dir)


if __name__ == '__main__':
    main()
