"""Pure-Python definitions for B value updates and C local state repair.

All rounds are 1-based completed forwards. An intervention at state_round=t
changes the OUTPUT state of forward t; outputs[0] is forward t+1.
Recorded four-field IDs are observations, never continuous feedback.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def descendants(parents, pos):
    children = defaultdict(list)
    for child, ps in parents.items():
        for p in ps:
            children[p].append(child)
    seen, queue = {pos}, [pos]
    for p in queue:
        for c in children[p]:
            if c not in seen:
                seen.add(c)
                queue.append(c)
    return sorted(seen - {pos})


def b_variants(value, modulus, empty_id, deltas=(1, 3)):
    values = [('unchanged', value), ('correct_write', value), ('empty', empty_id)]
    used = {value}
    for delta in deltas:
        wrong = (value + delta) % modulus
        if wrong in used:
            raise ValueError('wrong-value variants must be distinct and not equal truth')
        used.add(wrong)
        values.append((f'wrong_plus_{delta}', wrong))
    return values


def first_transition(values):
    return next((t for t in range(1, len(values)) if values[t-1] and not values[t]), None)


def select_donor_and_control(pred, truth, target_i, parent_i, recipient):
    """Choose last earlier jointly VALUE-correct donor; same-lag healthy control.

    recipient is a 1-based state round. Healthy recipient additionally requires
    all graph VALUEs correct. No donor/control comes from future rounds.
    """
    selected = [target_i] + list(parent_i)
    def jointly_correct(t):
        return all(pred[t-1][i][3] == truth[i][3] for i in selected)
    donor = next((t for t in range(recipient-1, 0, -1) if jointly_correct(t)), None)
    if donor is None:
        return None, None
    gap = recipient - donor
    control = next((t for t in range(recipient-1, gap, -1)
                    if jointly_correct(t) and jointly_correct(t-gap)
                    and all(row[3] == truth[i][3] for i, row in enumerate(pred[t-1]))), None)
    return donor, control


def c_events(pred, truth, positions, parents, modulus=23):
    """First post-completion VALUE correct->wrong transition per non-leaf node.

    Parent status is measured at t-1, before the failing forward t. Classification
    is decoded behavior, not a claim that discrete values are the causal input.
    VARIABLE-only controls use future observation explicitly, for diagnostics.
    Graphs that never complete are described from their entire trajectory but
    are not included in the primary post-completion intervention sample.
    """
    lookup = {p: i for i, p in enumerate(positions)}
    graph_ok = [all(row[i][3] == truth[i][3] for i in range(len(truth))) for row in pred]
    first_complete = next((i+1 for i, x in enumerate(graph_ok) if x), None)
    start = first_complete-1 if first_complete is not None else 0
    events = []
    for i, pos in enumerate(positions):
        ps = parents.get(pos, [])
        if not ps or any(p not in lookup for p in ps):
            continue  # leaves reference constants, outside this local-parent experiment
        pi = [lookup[p] for p in ps]
        value_ok = [row[i][3] == truth[i][3] for row in pred]
        variable_ok = [row[i][1] == truth[i][1] for row in pred]
        relative = first_transition(value_ok[start:])
        t = None if relative is None else relative+start
        kind = None
        if t is not None:
            parent_values = [pred[t-1][j][3] for j in pi]
            if any(v < 0 or v >= modulus for v in parent_values):
                kind = 'value_parent_nonnumeric'
            elif all(v == truth[j][3] for v, j in zip(parent_values, pi)):
                kind = 'value_parent_correct'
            else:
                kind = 'value_parent_wrong'
        else:
            relative = first_transition(variable_ok[start:])
            t = None if relative is None else relative+start
            if t is not None and all(value_ok[t-1:]):
                kind = 'variable_only'
        if kind is None:
            continue
        donor, control = select_donor_and_control(pred, truth, i, pi, t+1)
        pvals = [pred[t-1][j][3] for j in pi]
        numeric = all(0 <= v < modulus for v in pvals)
        events.append(dict(target_pos=pos, parent_positions=ps, category=kind,
                           state_round=t+1, donor_round=donor, control_round=control,
                           donor_gap=None if donor is None else t+1-donor,
                           donor_target_first4_correct=None if donor is None else pred[donor-1][i] == truth[i],
                           donor_parents_first4_correct=None if donor is None else all(pred[donor-1][j] == truth[j] for j in pi),
                           graph_first_complete=first_complete,
                           phase='post_complete' if first_complete is not None and first_complete < t+1
                                 else 'not_yet_complete',
                           parent_values_before_error=pvals,
                           expected_from_parent_readouts=sum(pvals) % modulus if numeric else None,
                           output_at_event=pred[t][i][3],
                           target_value_final_correct=value_ok[-1],
                           target_value_final_two_correct=all(value_ok[-2:]),
                           target_first_recovery=next((j+1 for j in range(t+1, len(pred)) if value_ok[j]), None),
                           descendant_positions=descendants(parents, pos)))
    return events


def balanced_sample(cases, per_group, seed):
    """Per weight/stratum/category quota, max one event per graph per category.

    Rank by fixed hash, not by repair success. Includes both successful and failed
    natural outcomes; export all candidates separately for denominators.
    """
    buckets = defaultdict(list)
    for c in cases:
        buckets[(c['weight'], c['stratum'], c.get('selection_group', c['category']))].append(c)
    selected = []
    for key, pool in sorted(buckets.items()):
        ordered = sorted(pool, key=lambda c: hashlib.sha256(
            f"{seed}|{c['case_id']}".encode()).hexdigest())
        seen = set()
        for c in ordered:
            if c['graph_id'] in seen:
                continue
            selected.append(c)
            seen.add(c['graph_id'])
            if len(seen) == per_group:
                break
    return selected


def pair_metrics(rec, branch, role):
    """Same initial cases; baseline-relative benefit/harm; zero denominators None."""
    positions, truth = rec['node_positions'], rec['truth4']
    lookup = {p: i for i, p in enumerate(positions)}
    ids = [lookup[p] for p in role if p in lookup]
    base = rec['branches']['unchanged']['outputs']
    out = branch['outputs']
    if len(base) != len(out) or not base:
        raise ValueError('unpaired or empty observation windows')
    result = []
    for t, (br, rr) in enumerate(zip(base, out), 1):
        for metric in ('value', 'first4'):
            def ok(row, i):
                return row[i][3] == truth[i][3] if metric == 'value' else row[i] == truth[i]
            b = [ok(br, i) for i in ids]
            v = [ok(rr, i) for i in ids]
            result.append(dict(round=t, metric=metric, n=len(ids),
                               correct=sum(v), baseline_correct=sum(b),
                               helped=sum(not x and y for x, y in zip(b, v)),
                               harmed=sum(x and not y for x, y in zip(b, v))))
    return result
