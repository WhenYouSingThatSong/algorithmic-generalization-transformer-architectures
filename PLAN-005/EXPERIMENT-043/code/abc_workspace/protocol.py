"""Fixed-space ADD programs and legitimate updates. No torch and no model answers."""
import copy
import hashlib
import json
import random

VERSION = 'abc_workspace_v1'
SEED = 2026111903
MOD = 23
NODES = 32
LEAVES = 6
# Width is the layer/frontier width, not a claim about minimum live memory.
STRATA = {'w2_f2': (2, 2), 'w4_f2': (4, 2), 'w6_f2': (6, 2),
          'w4_f1': (4, 1), 'w4_f3': (4, 3)}
BRANCHES = ('carry', 'reset_results', 'scratch')
PROBES = ('continue', 'refresh_task', 'reset_results', 'refresh_and_reset', 'scratch_same_input')


def seed(*parts):
    return int(hashlib.sha256(repr((SEED, parts)).encode()).hexdigest()[:15], 16)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frame(case, stage):
    """Same physical layout each stage; operands only refer to earlier definitions."""
    width, fanin = STRATA[case['stratum']]
    rng = random.Random(seed(case['id'], 'structure', stage))
    parents = [[] for _ in range(LEAVES)]
    previous = list(range(LEAVES))
    for start in range(LEAVES, NODES, width):
        layer = list(range(start, min(start + width, NODES)))
        for j, node in enumerate(layer):
            # All active boundary leaves are reached by first-layer outputs.
            anchor = previous[j % len(previous)]
            others = [p for p in previous if p != anchor]
            # Last partial layer can have fewer parents than fan-in.
            pool = others + [p for p in range(start) if p != anchor and p not in others]
            parents.append([anchor] + rng.sample(pool, fanin - 1))
        previous = layer
    # Last width nodes are distinct, valid boundary values, including partial layer.
    outputs = list(range(NODES - width, NODES))
    constants = [rng.randrange(MOD) for _ in range(LEAVES)]
    return dict(parents=parents, constants=constants, outputs=outputs,
                width=width, fanin=fanin, stage=stage)


def evaluate(parents, inputs):
    if len(inputs) != LEAVES or any(not isinstance(v, int) or not 0 <= v < MOD for v in inputs):
        raise ValueError('six valid numeric boundary inputs required; never replace invalid commits')
    values, depths = list(inputs), [0] * LEAVES
    for i in range(LEAVES, len(parents)):
        ps = parents[i]
        if not ps or any(p < 0 or p >= i for p in ps):
            raise ValueError('cyclic, missing, or forward reference')
        values.append(sum(values[p] for p in ps) % MOD)
        depths.append(1 + max(depths[p] for p in ps))
    return values, depths


def reference_program(case, stages):
    """Oracle isolated for scoring. Runner receives only exogenous inputs/layout."""
    result, previous, previous_support, previous_depth = [], None, None, None
    for s in range(stages):
        f = frame(case, s)
        inputs = list(f['constants'])
        w = f['width']
        if previous is not None:
            inputs[:w] = previous
        values, depths = evaluate(f['parents'], inputs)
        support = [{f'input:{s}:{i}'} for i in range(LEAVES)]
        logical_depth = [0] * LEAVES
        if previous is not None:
            support[:w] = [set(x) for x in previous_support]
            logical_depth[:w] = previous_depth
        for i in range(LEAVES, NODES):
            support.append(set.union(*(support[p] for p in f['parents'][i])) | {f'op:{s}:{i}'})
            logical_depth.append(1 + max(logical_depth[p] for p in f['parents'][i]))
        previous = [values[i] for i in f['outputs']]
        previous_support = [support[i] for i in f['outputs']]
        previous_depth = [logical_depth[i] for i in f['outputs']]
        result.append(dict(inputs=inputs, truth=values, local_depth=max(depths),
                           output=previous, logical_depth=max(previous_depth),
                           output_ancestor_operations=sum(x.startswith('op:') for x in set.union(*previous_support)),
                           executed_operations=(s + 1) * (NODES - LEAVES),
                           cumulative_logical_nodes=NODES + s * (NODES - LEAVES)))
    return result


def emitted_inputs(f, previous_commit):
    """Runtime boundary: only actual model output, no access to reference_program."""
    inputs = list(f['constants'])
    if previous_commit is not None:
        if len(previous_commit) != f['width'] or any(v is None or not 0 <= v < MOD for v in previous_commit):
            return None
        inputs[:f['width']] = previous_commit
    return inputs


def render(f, names, inputs):
    tokens, node_pos, constant_pos, task_rows = [], [], [], []
    for i, ps in enumerate(f['parents']):
        if i:
            task_rows.append(len(tokens)); tokens.append('<EQ_SEP>')
        if i < LEAVES:
            constant_pos.append(len(tokens)); expr = [str(inputs[i])]
        else:
            expr = []
            for j, p in enumerate(ps):
                if j: expr.append('ADD')
                expr.append(names[p])
        start = len(tokens)
        tokens.extend(expr + ['=', names[i]])
        node_pos.append(len(tokens) - 1)
        task_rows.extend(range(start, len(tokens) - 1))
    return dict(tokens=tokens, nodes=node_pos, constants=constant_pos, task_rows=task_rows)


def update_inputs(original):
    x = list(original); result = []
    for edits in ({0: 1}, {1: 3}, {0: 7, 1: 11}, None):
        before = list(x)
        if edits is None: x = list(original)
        else:
            for i, d in edits.items(): x[i] = (x[i] + d) % MOD
        result.append(dict(inputs=list(x), changed_leaves=[i for i in range(LEAVES) if x[i] != before[i]]))
    return result


def affected_nodes(parents, starts):
    seen = set(starts)
    for i, ps in enumerate(parents):
        if any(p in seen for p in ps): seen.add(i)
    return sorted(seen)


def make_cases(count=16):
    cases = []
    for stratum in STRATA:
        for i in range(count):
            cid = f'{stratum}:{i:03d}'
            names = random.Random(seed(cid, 'names')).sample(range(128), NODES)
            cases.append(dict(id=cid, stratum=stratum, ordinal=i, names=[f'x_{n}' for n in names]))
    return cases


def measures(outputs, gold4, node_indices, terminal_indices):
    def correct(row, i, fields):
        return row[i][3] == gold4[i][3] if fields == 1 else row[i] == gold4[i]
    out = {}
    for label, inds in [('nodes', node_indices), ('terminal', terminal_indices)]:
        for fields, name in [(1, 'value'), (4, 'first4')]:
            flags = [all(correct(row, i, fields) for i in inds) for row in outputs]
            out[f'{label}_{name}_n'] = sum(all(correct(row, i, fields) for row in outputs[-2:]) for i in inds)
            out[f'{label}_{name}_all'] = all(flags[-2:]) and len(flags) >= 2
            out[f'{label}_{name}_final_n'] = sum(correct(outputs[-1], i, fields) for i in inds)
            out[f'{label}_{name}_final_all'] = flags[-1]
            out[f'{label}_{name}_first'] = next((i+1 for i, v in enumerate(flags) if v), None)
            out[f'{label}_{name}_stable'] = next((i+1 for i in range(len(flags)-1) if all(flags[i:])), None)
            out[f'{label}_{name}_reerror'] = any(flags[i] and not all(flags[i+1:]) for i in range(len(flags)-1))
        out[label + '_n'] = len(inds)
    return out
