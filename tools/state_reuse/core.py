"""Pure protocols for repeated updates, dead-slot reuse and C functional probes."""
import copy
import hashlib

SEED = 20261118
DISCRETE_BRANCHES = ('warm', 'clear', 'scratch', 'warm_pinned', 'clean', 'sham')
C_BRANCHES = tuple(f'{receiver}_{child}' for receiver in ('observed', 'restore_u', 'clean')
                   for child in ('keep', 'pre', 'solved')) + ('sham',)
C_NAMES = dict(initial='C（原始）', hold='C（保持后训练）',
               noise='C（噪声后训练）', semantic='C（语义后训练）')


def order(items, *key):
    return sorted(items, key=lambda x: hashlib.sha256(repr((SEED, key, x)).encode()).hexdigest())


def evaluate(nodes, parents, constants, modulus=23):
    """Independent recursive evaluator, including cycle/missing-definition checks."""
    values = dict(constants); visiting = set(); depths = {}
    def visit(p):
        if p in values:
            return values[p]
        if p in visiting or p not in parents or not parents[p]:
            raise ValueError('cyclic or undefined assignment')
        visiting.add(p)
        values[p] = sum(visit(q) for q in parents[p]) % modulus
        depths[p] = 0 if all(q in constants for q in parents[p]) else 1 + max(depths.get(q, -1) for q in parents[p])
        visiting.remove(p)
        return values[p]
    for p in nodes:
        visit(p)
    return values, depths


def closure(nodes, parents, starts):
    seen = set(starts)
    while True:
        new = seen | {p for p in nodes if any(q in seen for q in parents[p])}
        if new == seen:
            return sorted(seen & set(nodes))
        seen = new


def schedules(nodes, parents, constants, gid):
    leaves = order([p for p in nodes if len(parents[p]) == 1 and parents[p][0] in constants
                    and len(closure(nodes, parents, [p])) > 1], gid, 'leaves')
    if not leaves:
        return {}
    a = leaves[0]
    result = {'same_input': [(a, d) for d in (1, 3, 7, -11)]}
    if len(leaves) >= 2:
        b = leaves[1]
        result['alternating_inputs'] = [(a, 1), (b, 3), (a, 7), (b, -3), (a, -8)]
    return result


def assignment_plans(nodes, parents, constants, gid, schedule):
    cs = dict(constants); prev, _ = evaluate(nodes, parents, cs)
    for step, (leaf, delta) in enumerate(schedule, 1):
        const = parents[leaf][0]
        before = cs[const]; cs[const] = (before + delta) % 23
        new, depths = evaluate(nodes, parents, cs)
        affected = closure(nodes, parents, [leaf])
        yield dict(episode=step, target=leaf, constant=const, delta=delta,
                   old_constant=before, new_constant=cs[const], constants=dict(cs),
                   old_truth=prev, new_truth=new, depths=depths, parents=copy.deepcopy(parents),
                   affected=affected, changed=[p for p in nodes if prev[p] != new[p]],
                   unchanged=[p for p in nodes if prev[p] == new[p]])
        prev = new


def reuse_plans(nodes, parents, constants, gid):
    """Two dead sinks become new logical versions; second consumes first.

    Operands precede definitions, retaining the generator's first-occurrence
    naming convention. No old references to reclaimed sinks can remain.
    Selection is structural and hash-based, never based on new answers.
    """
    used = {q for ps in parents.values() for q in ps}
    sinks = [p for p in nodes if p not in used and all(q in nodes for q in parents[p])]
    pairs = [(a, b) for a in sinks for b in sinks if a < b and
             len([p for p in nodes if p < a and p not in (a, b)]) >= len(parents[a]) and
             len([p for p in nodes if p < b and p not in (a, b)]) >= len(parents[b]) - 1]
    if not pairs:
        return []
    a, b = order(pairs, gid, 'dead-sinks')[0]
    ps = copy.deepcopy(parents); prev, _ = evaluate(nodes, ps, constants); result = []
    for episode, target in enumerate((a, b), 1):
        choices = order([p for p in nodes if p < target and p not in (a, b)], gid, target, 'operands')
        arity = len(ps[target])
        newps = choices[:arity] if target == a else [a] + choices[:arity-1]
        # Force a symbolic change when a different structural choice exists.
        if target == a and newps == ps[target] and len(choices) > arity:
            newps = choices[1:arity+1]
        if newps == ps[target]:
            return []
        oldps = list(ps[target]); ps[target] = newps
        new, depth = evaluate(nodes, ps, constants)
        result.append(dict(episode=episode, target=target, reclaimed=[a, b],
            old_parents=oldps, new_parents=newps, parents=copy.deepcopy(ps), constants=dict(constants),
            old_truth=prev, new_truth=new, depths=depth, affected=[target],
            changed=[p for p in nodes if prev[p] != new[p]],
            unchanged=[p for p in nodes if prev[p] == new[p]],
            logical_version=f'{target}:v1', old_version=f'{target}:v0',
            consumes_recycled_version=(target == b), cumulative_logical_nodes=len(nodes)+episode))
        prev = new
    return result


def expression_positions(raw, target, parents, syntax, var_index=1):
    """Actual format: x ADD y = z; target is the z token, not x."""
    n = len(parents); refs = list(range(target - 2*n, target - 1, 2))
    if len(refs) != n or min(refs) < 0 or raw[target-1][0] != syntax['=']:
        raise ValueError('unexpected equation extent')
    if refs[0] and raw[refs[0]-1][0] != syntax['<EQ_SEP>']:
        raise ValueError('equation boundary mismatch')
    for ref, p in zip(refs, parents):
        if raw[ref][0] != syntax['VARIABLE'] or raw[ref][var_index] != raw[p][var_index]:
            raise ValueError('operand identity mismatch')
    return refs


def flags(trace, truth, indices, factor=None):
    return [[row[i] == truth[i] if factor is None else row[i][factor] == truth[i][factor]
             for i in indices] for row in trace]


def stable_round(xs):
    """First round starting an all-correct suffix of >=2 observed outputs."""
    return next((i+1 for i in range(len(xs)-1) if all(xs[i:])), None)


def trace_counts(outputs, gold, indices, step):
    rows = outputs[-2:] if step == 'last_two' else [outputs[int(step)-1]]
    return dict(n=len(indices), value_n=sum(all(r[i][3] == gold[i][3] for r in rows) for i in indices),
                first4_n=sum(all(r[i] == gold[i] for r in rows) for i in indices))


def c_probe_grid():
    return [(timing, mode, lag) for timing in ('early', 'hold', 'late')
            for mode in ('natural', 'same', 'semantic', 'noise2') for lag in (1, 2, 4)]


def c_context(row, clean, donor_pre, donor_solved, gold, pos, u, q, parents, empty):
    ix = {p: i for i, p in enumerate(pos)}; ui, qi = ix[u], ix[q]
    other = [p for p in parents[q] if p != u]
    return dict(u_value_correct=row[ui][3] == gold[ui][3],
        u_first4_correct=row[ui] == gold[ui], q_value_correct=row[qi][3] == gold[qi][3],
        q_first4_correct=row[qi] == gold[qi],
        other_parents_value_correct=all(row[ix[p]][3] == gold[ix[p]][3] for p in other),
        other_parents_first4_correct=all(row[ix[p]] == gold[ix[p]] for p in other),
        clean_local_first4_correct=all(clean[ix[p]] == gold[ix[p]] for p in set([u,q]+parents[q])),
        donor_pre_empty=donor_pre[3] == empty,
        donor_pre_identity_correct=donor_pre[:3] == gold[qi][:3],
        donor_solved_first4_correct=donor_solved == gold[qi])
