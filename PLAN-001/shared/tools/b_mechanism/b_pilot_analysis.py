#!/usr/bin/env python3
"""CPU-only supplements for the B long-horizon pilot.

This script does NOT change the pilot result.  It adds the denominators,
direct-injection bookkeeping, full-trajectory recurrence counts and a few
representative r/u/q1/q2 sequences requested in B1.  Inputs are the six v2
trajectory JSONs produced by the optimized runners.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_mechanism_horizon_summary as H  # noqa: E402
import b_mechanism_summary as S  # noqa: E402

CONDS = ('after_u_U', 'after_u_R', 'complete_U', 'complete_R', 'sustained_R')
METRICS = ('value', 'first4')
ROLES = ('direct', 'target_u', 'downstream_d1', 'downstream_d2plus', 'all_affected')


def load(path):
    with open(path) as f:
        return json.load(f)


def role_nodes(role, metric, cache):
    return cache['cache'][(role, metric)]


def node_flags(cb, jb):
    """Per round: correct/wrong/unknown against the clean branch."""
    wrong = [bool(c) and not bool(j) for c, j in zip(cb, jb)]
    correct = [bool(c) and bool(j) for c, j in zip(cb, jb)]
    return wrong, correct


def recurrence(cb, jb, stable_len=2):
    """Count wrong episodes and recovery->re-error in one node trajectory."""
    Hn = len(cb)
    wrong, correct = node_flags(cb, jb)
    episodes = 0
    prev_wrong = False
    for w in wrong:
        if w and not prev_wrong:
            episodes += 1
        prev_wrong = w
    first_damage = next((i for i, w in enumerate(wrong) if w), None)
    first_recovery = None
    if first_damage is not None:
        first_recovery = next((i for i in range(first_damage + 1, Hn) if jb[i]), None)
    reerror = False
    if first_recovery is not None:
        reerror = any(wrong[i] for i in range(first_recovery + 1, Hn))
    last_run_start = None
    if jb and jb[-1]:
        last_run_start = Hn - 1
        while last_run_start > 0 and jb[last_run_start - 1]:
            last_run_start -= 1
    stable = Hn >= stable_len and all(jb[Hn - stable_len:])
    stable_clean = stable and all(correct[Hn - stable_len:])
    return dict(
        first_damage=None if first_damage is None else first_damage + 1,
        first_recovery=None if first_recovery is None else first_recovery + 1,
        last_run_start=None if last_run_start is None else last_run_start + 1,
        damage_episodes=episodes,
        ever_recovered=(first_recovery is not None),
        recovery_then_error=reerror,
        stable=stable,
        stable_clean=stable_clean,
    )


def direct_first_round(rec, clean, roles, offset):
    """Actual injected direct positions and output at their first observed round."""
    d = roles['direct']
    if not d:
        return None
    if offset:
        idx = 0  # first forced-hold round
        phase = 'hold'
    else:
        idx = 0
        phase = 'post_injection'
    actual = corrected = wrong = 0
    details = []
    for p in sorted(d):
        cin = H.node_ok(clean['rounds'][idx], p, 'first4', 'input')
        jin = H.node_ok(rec['rounds'][idx], p, 'first4', 'input')
        out = H.node_ok(rec['rounds'][idx], p, 'first4')
        if cin and not jin:
            actual += 1
            if out:
                corrected += 1
            else:
                wrong += 1
        details.append(dict(pos=p, clean_input=cin, injected_input=jin,
                            output=out))
    return dict(phase=phase, round=idx + 1, actual=actual,
                first_round_corrected=corrected, first_round_wrong=wrong,
                positions=details)


def case_record(rec, roles):
    chain = rec.get('chain') or rec.get('target') or {}
    keys = {}
    for name in ('pos', 'r_pos', 'q1', 'q2'):
        p = chain.get(name)
        if p is not None:
            keys[name] = str(p)
    rounds = []
    for i, row in enumerate(rec['rounds']):
        item = dict(round=i + 1)
        for name, p in keys.items():
            item[name] = dict(
                input=(row.get('input') or {}).get(p),
                output=(row.get('output') or {}).get(p),
                output_ok=(row.get('output_nodes') or {}).get(p, {}).get('value_correct'),
            )
        rounds.append(item)
    return dict(graph_id=rec.get('graph_id'), target_index=rec.get('target_index', rec.get('chain_index')),
                condition=rec.get('condition'), keys=keys, rounds=rounds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--stable_len', type=int, default=2)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    denom = []
    direct = []
    rec_stats = []
    exceptions = []
    cases = {}
    seen = set()

    for path in a.inputs:
        data = load(path)
        for model, source, args, rec, clean, reason in H.pair_file(path, data):
            cond = H.condition_label(rec) if rec is not None else None
            if reason is not None or cond not in H.PILOT_CONDS:
                continue
            cache = H.build_cache(rec, clean, max_rounds=64)
            if cache is None:
                continue
            gid = int(rec.get('graph_id'))
            tid = rec.get('target_index', rec.get('chain_index'))
            tid = -1 if tid is None else int(tid)
            roles = cache['roles']
            # Denominator bookkeeping at every H.
            for Hn in H.HORIZONS:
                if Hn > cache['n']:
                    continue
                ev = H.summarize_events(cache, Hn, stable_len=a.stable_len)
                for (role, metric), v in ev.items():
                    if not v['obs']:
                        continue
                    denom.append(dict(
                        model=model, condition=cond, role=role, metric=metric,
                        H=Hn, graph_id=gid, target_id=tid,
                        observed=v['obs'], damaged=v['damaged'],
                        stable=v['stable'], ever=v['ever'],
                        only_last=v['only_last'], stable_clean=v['stable_clean']))
            # Direct injection at its first observed round.
            d = direct_first_round(rec, clean, roles, cache['offset'])
            if d:
                direct.append(dict(model=model, condition=cond, graph_id=gid,
                                   target_id=tid, **d))
            # Full-trajectory recurrence at H=64.
            if cache['n'] >= 64:
                for role in ROLES:
                    for metric in METRICS:
                        for pos, cb, jb in role_nodes(role, metric, cache):
                            r = recurrence(cb, jb, stable_len=a.stable_len)
                            r.update(model=model, condition=cond, role=role,
                                     metric=metric, graph_id=gid, target_id=tid,
                                     pos=pos)
                            rec_stats.append(r)
                # A-prime R-only direct exceptions: r output correct at round 1.
                if model == 'A_prime' and cond == 'after_u_R':
                    for pos in sorted(roles['direct']):
                        if H.node_ok(rec['rounds'][0], pos, 'first4'):
                            nt = (rec.get('node_truth') or {}).get(str(pos), {})
                            chain = rec.get('chain') or {}
                            exceptions.append(dict(
                                graph_id=gid, target_id=tid, pos=pos,
                                chain_d=chain.get('d'), node_depth=nt.get('depth'),
                                n_children=nt.get('n_children'),
                                parents=nt.get('parents'),
                                r_true=chain.get('r_true'),
                                r_wrong=(rec.get('injection') or {}).get('r_wrong'),
                            ))
            # Representative sequences, one per (model, condition).
            if (model, cond) not in seen and cache['n'] >= 8:
                seen.add((model, cond))
                cases[f'{model}:{cond}'] = case_record(rec, roles)

    def write_csv(name, rows):
        if not rows:
            return
        with open(os.path.join(a.out_dir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv('b1_denominators.csv', denom)
    write_csv('b1_direct_first_round.csv', [
        {k: v for k, v in r.items() if k != 'positions'} for r in direct])
    with open(os.path.join(a.out_dir, 'b1_direct_positions.json'), 'w') as f:
        json.dump(direct, f, indent=2)
    write_csv('b1_recurrence.csv', rec_stats)
    write_csv('b1_exceptions.csv', exceptions)
    with open(os.path.join(a.out_dir, 'b1_sequences.json'), 'w') as f:
        json.dump(cases, f, indent=2)
    print('WROTE', a.out_dir, 'denom', len(denom), 'direct', len(direct),
          'recurrence', len(rec_stats), 'exceptions', len(exceptions),
          'cases', len(cases))


if __name__ == '__main__':
    main()
