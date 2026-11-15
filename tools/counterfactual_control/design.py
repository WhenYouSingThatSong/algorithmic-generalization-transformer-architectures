"""Torch-free construction of matched same/different-value ADD donors."""
from itertools import combinations
from collections import defaultdict
import hashlib


def evaluate(parents, depths, leaf_values, modulus=23):
    values = dict(leaf_values)
    for pos in sorted(parents, key=lambda p: (depths[p], p)):
        if depths[pos] == 0:
            continue
        if not parents[pos] or any(p not in values for p in parents[pos]):
            raise ValueError('not a depth-ordered numeric ADD DAG')
        values[pos] = sum(values[p] for p in parents[pos]) % modulus
    return values


def coefficients(parents, depths, modulus=23):
    """Count leaf-to-node paths, including repeated parents, modulo modulus."""
    result = {}
    for pos in sorted(parents, key=lambda p: (depths[p], p)):
        if depths[pos] == 0:
            result[pos] = {pos: 1}
        else:
            row = {}
            for parent in parents[pos]:
                for leaf, coef in result[parent].items():
                    row[leaf] = (row.get(leaf, 0) + coef) % modulus
            result[pos] = row
    return result


def matched_changes(coefs, deltas=(1, 3), modulus=23):
    """Two nonzero leaf edits in BOTH donors, at exactly the same positions.

    Same donor: a*x+b*y=0; different donors: a*x+b*z=delta. x is shared.
    Choose deterministically, minimizing the larger modular edit L1 magnitude.
    This matches edit positions/count, NOT hidden-vector distance.
    """
    if not deltas or any(d % modulus == 0 for d in deltas) or len(set(d % modulus for d in deltas)) != len(deltas):
        raise ValueError('distinct nonzero deltas required')
    candidates = []
    for left, right in combinations(sorted(p for p,c in coefs.items() if c % modulus), 2):
        a, b = coefs[left], coefs[right]
        inv = pow(b, -1, modulus)
        for x in range(1, modulus):
            ys = tuple(((d-a*x)*inv) % modulus for d in (0, *deltas))
            if any(y == 0 for y in ys):
                continue
            mag = lambda v: min(v, modulus-v)
            score = max(mag(x)+mag(y) for y in ys)
            candidates.append((score, left, right, x, ys))
    if not candidates:
        return None
    _, left, right, x, ys = min(candidates)
    return {name: {left: x, right: y} for name,y in
            zip(['same']+[f'shift_{d}' for d in deltas], ys)}


def choose_candidates(parents, depths, leaf_values, graph_key, limit=2,
                      deltas=(1, 3), modulus=23, seed=20261115):
    """Value-agnostic, graph-local stratified sample; no model observations.

    Rotate through (fan-in, lower/upper depth) groups in fixed hash order.
    Return every structural rejection/count as well as the selected candidates.
    """
    if limit < 1:
        raise ValueError('positive limit required')
    truth = evaluate(parents, depths, leaf_values, modulus)
    coef = coefficients(parents, depths, modulus)
    pool, coverage = [], []
    for u in sorted(parents):
        if depths[u] < 1:
            continue
        qs = sorted(q for q, ps in parents.items() if u in ps)
        ready = [q for q in qs if depths[q] == depths[u]+1]
        row = dict(target_pos=u, depth=depths[u], fan_in=len(parents[u]),
                   value=truth[u], selected=False)
        if not ready:
            coverage.append(dict(row, reason='no_next_round_child'))
            continue
        changes = matched_changes(coef[u], deltas, modulus)
        if changes is None:
            coverage.append(dict(row, reason='no_matched_two_leaf_edit'))
            continue
        coverage.append(dict(row, reason='structurally_eligible'))
        donor_truth = {}
        for name, edits in changes.items():
            leaves = dict(leaf_values)
            for p, delta in edits.items():
                leaves[p] = (leaves[p] + delta) % modulus
            donor_truth[name] = evaluate(parents, depths, leaves, modulus)
        expected_values = dict(same=truth[u], **{f'shift_{d}': (truth[u]+d)%modulus for d in deltas})
        if any(donor_truth[n][u] != value for n,value in expected_values.items()):
            raise ValueError('counterfactual arithmetic check failed')
        # Only u changes in recipient hidden. Do NOT use the full donor q label!
        expected = {str(d): {str(q): sum((truth[u]+d)%modulus if p == u else truth[p]
                           for p in parents[q]) % modulus for q in qs} for d in deltas}
        pool.append(dict(target_pos=u, depth=depths[u], fan_in=len(parents[u]),
                    state_round=depths[u]+1, original_value=truth[u],
                    child_positions=qs, ready_children=ready, changes=changes,
                    target_coefficients=coef[u], truth=truth, donor_truth=donor_truth,
                    donor_values=expected_values, local_expected_q=expected))
    buckets = defaultdict(list)
    max_depth = max(depths.values())
    rank = lambda x: hashlib.sha256(f'{seed}|{graph_key}|{x}'.encode()).hexdigest()
    for c in pool:
        group = (c['fan_in'], 'lower' if 2*c['depth'] <= max_depth else 'upper')
        c['depth_band'] = group[1]
        buckets[group].append(c)
    keys = sorted(buckets, key=rank)
    for group in keys:
        buckets[group].sort(key=lambda c: rank(c['target_pos']))
    selected = []
    while len(selected) < min(limit, len(pool)):
        for group in keys:
            if buckets[group] and len(selected) < limit:
                selected.append(buckets[group].pop(0))
    chosen = {c['target_pos'] for c in selected}
    for row in coverage:
        row['selected'] = row['target_pos'] in chosen
    return selected, coverage


def qualify(truth4, positions, parents, candidate, source_initial, baseline_first, value_only=False):
    """Common source qualification; q eligibility uses unchanged branch only."""
    lookup = {p: i for i,p in enumerate(positions)}
    u = candidate['target_pos']; ui = lookup[u]
    flags = {}
    for name in ('original', *candidate['changes']):
        expected = list(truth4[ui])
        if name != 'original':
            expected[3] = candidate['donor_truth'][name][u]
        flags[name+'_target_first4'] = source_initial[name][ui] == expected
        flags[name+'_target_value'] = source_initial[name][ui][3] == expected[3]
        flags[name+'_nonvalue_matches_original'] = source_initial[name][ui][:3] == source_initial['original'][ui][:3]
    eligible_q = []
    source_ok = all(v for k,v in flags.items() if k.endswith('_target_value')) if value_only else all(flags.values())
    if source_ok:
        for q in candidate['ready_children']:
            qi = lookup[q]
            parents_ok = all(source_initial['original'][lookup[p]][3] == truth4[lookup[p]][3]
                             for p in parents[q])
            if parents_ok and baseline_first[qi][3] == truth4[qi][3]:
                eligible_q.append(q)
    return flags, eligible_q


def branch_metrics(record, branch, delta):
    """Per-round direct-child observations on the same qualified q set."""
    lookup = {p: i for i,p in enumerate(record['node_positions'])}
    truth = record['truth4']; qs = record['eligible_children']
    ui = lookup[record['target_pos']]
    base = record['branches']['unchanged']['outputs']
    out = record['branches'][branch]['outputs']
    result = []
    for t, (row, baseline) in enumerate(zip(out, base), 1):
        wrong = lambda p, r: r[lookup[p]][3] != truth[lookup[p]][3]
        result.append(dict(round=t, q_n=len(qs),
            q_correct=sum(not wrong(q,row) for q in qs),
            q_first4_correct=sum(row[lookup[q]] == truth[lookup[q]] for q in qs),
            q_expected_shift=sum(row[lookup[q]][3] == record['local_expected_q'][str(delta)][str(q)] for q in qs),
            q_nonnumeric=sum(not 0 <= row[lookup[q]][3] < record['modulus'] for q in qs),
            q_changed_vs_unchanged=sum(row[lookup[q]][3] != baseline[lookup[q]][3] for q in qs),
            q_harmed=sum(not wrong(q,baseline) and wrong(q,row) for q in qs),
            q_helped=sum(wrong(q,baseline) and not wrong(q,row) for q in qs),
            u_original=int(row[ui][3] == truth[ui][3]),
            u_first4_correct=int(row[ui] == truth[ui]),
            u_shifted=int(row[ui][3] == (truth[ui][3]+delta)%record['modulus'])))
    return result
