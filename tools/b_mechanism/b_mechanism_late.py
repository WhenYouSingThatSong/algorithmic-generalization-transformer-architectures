#!/usr/bin/env python3
"""Late-timing B mechanism runner (integrated first-batch conditions).

Conditions (choose with --conditions):
  late_r               : r wrong, u stays correct, all downstream correct
  late_u               : u wrong (4->5), r correct, downstream correct
  sink                 : numeric sink VALUE wrong
  leaf_value           : leaf variable VALUE wrong, constant unchanged (old truth)
  leaf_constant        : constant VALUE wrong, old leaf/downstream kept (new truth)
  leaf_constant_scratch: new problem solved from empty node values
  leaf_constant_pinned : new constant forced every round
  unrelated            : matched unrelated node VALUE wrong

This runner uses the manifest from b_mechanism_select.py.  It is one model at a
time; the manifest was built from A' and B clean-complete states.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import b_mechanism_common as C


def delta_value(tok, value, delta):
    return tok.value_tok2idx[str((int(value) + int(delta)) % tok.mod_val)]


def set_value(state, pos, token_idx):
    state[0, pos, C.V] = token_idx


def token_values(state, tok, positions):
    return {str(p): C.tok_str(tok, int(state[0, p, C.V])) for p in positions}


def recompute_truth(meta, label_b, tok, teacher, constant_override=None):
    """Topological replay of node values; constant_override maps c_pos -> value."""
    constant_override = constant_override or {}
    vals = {}
    nodes = sorted(meta['nodes'], key=lambda p: (int(meta['depth'][p]), p))
    for p in nodes:
        d = int(meta['depth'][p])
        if d == 0:
            cpos = None
            for pp in meta['parents'].get(p, []):
                if pp in constant_override:
                    cpos = pp
                    break
            vals[p] = constant_override[cpos] if cpos is not None else int(label_b[p, C.V])
            continue
        parents = [pp for pp in meta['parents'].get(p, []) if pp in meta['nodes']]
        if parents and all(pp in vals for pp in parents):
            vals[p] = C.apply_expr(meta['info_b'], p, vals, tok, teacher)
        else:
            vals[p] = None
    return vals


def tracked_positions(meta, manifest_graph, target, kind):
    pos = set(meta['nodes'])
    if kind.startswith('late'):
        pos.add(target['r_pos'])
        pos.add(target['pos'])
        if target.get('q1') is not None:
            pos.add(target['q1'])
        if target.get('q2') is not None:
            pos.add(target['q2'])
    if kind.startswith('leaf'):
        pos.add(target['var_pos'])
        pos.add(target['const_pos'])
    if kind == 'unrelated':
        pos.add(target['pos'])
    return sorted(pos)


def make_injected(base, manifest_graph, meta, tok, label_b, kind, target, delta):
    """Return (start_state, old_truth_new_truth, metadata)."""
    start = base.clone()
    info = dict(kind=kind)
    if kind == 'late_r':
        r_pos = target['r_pos']
        wrong = (int(target['r_true']) + delta) % tok.mod_val
        set_value(start, r_pos, delta_value(tok, target['r_true'], delta))
        info.update(pos=r_pos, target_pos=target['pos'], r_pos=r_pos,
                    from_value=target['r_true'], to_value=wrong, truth_unchanged=True)
    elif kind == 'late_u':
        u_pos = target['pos']
        wrong = (int(target['a']) + delta) % tok.mod_val
        set_value(start, u_pos, delta_value(tok, target['a'], delta))
        info.update(pos=u_pos, r_pos=target['r_pos'], from_value=target['a'],
                    to_value=wrong, truth_unchanged=True)
    elif kind == 'sink':
        set_value(start, target['pos'], delta_value(tok, target['true'], delta))
        info.update(pos=target['pos'], from_value=target['true'],
                    to_value=(int(target['true']) + delta) % tok.mod_val,
                    truth_unchanged=True)
    elif kind == 'leaf_value':
        set_value(start, target['var_pos'], delta_value(tok, target['true'], delta))
        info.update(pos=target['var_pos'], const_pos=target['const_pos'],
                    from_value=target['true'],
                    to_value=(int(target['true']) + delta) % tok.mod_val,
                    truth_unchanged=True)
    elif kind in ('leaf_constant', 'leaf_constant_pinned'):
        new_const = (int(target['true']) + delta) % tok.mod_val
        set_value(start, target['const_pos'], tok.value_tok2idx[str(new_const)])
        info.update(pos=target['const_pos'], var_pos=target['var_pos'],
                    from_value=target['true'], to_value=new_const,
                    truth_unchanged=False)
    elif kind == 'unrelated':
        set_value(start, target['pos'], delta_value(tok, target['true'], delta))
        info.update(pos=target['pos'], from_value=target['true'],
                    to_value=(int(target['true']) + delta) % tok.mod_val,
                    truth_unchanged=True)
    else:
        raise KeyError(kind)
    return start, info


def rollout_records(lit, teacher, tok, start, meta, manifest_graph, target, kind,
                    horizon, label_b, pre_forward_round, input_hook=None):
    positions = tracked_positions(meta, manifest_graph, target, kind)
    inputs, outputs = C.rollout(lit, teacher, start, horizon, input_hook=input_hook)
    rows = []
    for t in range(horizon):
        rows.append(dict(
            post_round=t + 1,
            pre_forward_round=pre_forward_round,
            input=token_values(inputs[t], tok, positions),
            output=token_values(outputs[t], tok, positions),
            input_nodes=C.node_state(inputs[t], label_b, tok, positions),
            output_nodes=C.node_state(outputs[t], label_b, tok, positions),
        ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--ds_file', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--group', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--ckpt', default='last.ckpt')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--conditions', default='late_r,late_u,sink,leaf_value,leaf_constant,leaf_constant_scratch,unrelated')
    p.add_argument('--delta', type=int, default=3, help='r/leaf/sink delta')
    p.add_argument('--u_delta', type=int, default=1, help='u injection delta when value_mode=delta')
    p.add_argument('--value_mode', choices=['a_plus_1', 'four_to_five', 'delta'],
                    default='a_plus_1')
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--n_graphs', type=int, default=0,
                    help='0 = all manifest graphs')
    p.add_argument('--out', required=True)
    a = p.parse_args()

    with open(a.manifest) as f:
        manifest = json.load(f)
    C.verify_manifest_identity(manifest, a.ds_file, C.checkpoint_path(a.base, a.group, a.run, a.ckpt))
    graphs = {g['graph_id']: g for g in manifest['graphs']}
    ctx = C.load_eval_context(a.config, a.group, a.run, a.base, a.ckpt,
                              a.ds_file, batch_size=a.batch_size, num_workers=0)
    lit, tok, teacher, mc = ctx['lit'], ctx['tok'], ctx['teacher'], ctx['mc']
    selected = set(graphs)
    done = 0
    records = []
    clean_done = set()
    counts = dict(processed=0, emitted=0, skipped_not_complete=0)

    with torch.no_grad():
        for batch_idx, raw in enumerate(ctx['dm'].val_dataloader()):
            batch = lit.reduce_batch(raw).to(ctx['device'])
            data, label, info = C.split_batch(batch, mc)
            B = data.size(0)
            for bi in range(B):
                gid = batch_idx * a.batch_size + bi
                if gid not in selected:
                    continue
                mg = graphs[gid]
                meta = C.build_meta(data[bi], label[bi], info[bi], tok, teacher)
                meta['info_b'] = info[bi].cpu()
                # Recompute clean rollout and require the preselected T to be complete.
                n_loops = mg['n_loops']
                _, outs = C.rollout(lit, teacher, data[bi:bi + 1], n_loops)
                T = mg['T']
                if T >= len(outs) or not C.all_numeric_correct(outs[T][0], label[bi], meta['numeric']):
                    counts['skipped_not_complete'] += 1
                    continue
                base = outs[T].clone()
                counts['processed'] += 1
                for cond in a.conditions.split(','):
                    cond = cond.strip()
                    if not cond:
                        continue
                    if cond == 'late_r':
                        targets = mg['chains']
                    elif cond == 'late_u':
                        targets = mg['chains']
                    elif cond == 'sink':
                        targets = mg['late']['sinks']
                    elif cond.startswith('leaf_'):
                        targets = mg['late']['leaves']
                    elif cond == 'unrelated':
                        targets = mg['unrelated']
                    else:
                        raise SystemExit(f'unknown condition {cond}')
                    for ti, target in enumerate(targets):
                        if cond == 'late_u':
                            delta = a.u_delta if a.value_mode == 'delta' else 1
                            if a.value_mode == 'four_to_five' and int(target['a']) != 4:
                                continue
                        else:
                            delta = a.delta
                        # Same-start, no-injection continuation; one per target.
                        clean_key = (gid, cond, ti)
                        if cond not in ('leaf_constant_scratch',) and clean_key not in clean_done:
                            clean_done.add(clean_key)
                            clean_rows = rollout_records(lit, teacher, tok, base[0:1], meta, mg,
                                                         target, 'clean', a.horizon, label[bi],
                                                         pre_forward_round=T + 1)
                            records.append(dict(graph_id=gid, condition='clean',
                                                target_index=ti, target=target,
                                                injection=dict(kind='clean'), T=T,
                                                pre_forward_round=T + 1,
                                                rounds=clean_rows, node_truth=mg['nodes']))
                        if cond == 'leaf_constant_scratch':
                            # D2: solve the new problem from an empty node state.
                            start = data[bi:bi + 1].clone()
                            for p in meta['nodes']:
                                start[0, p, C.V] = tok.value_tok2idx['EMPTY']
                            new_const = (int(target['true']) + delta) % tok.mod_val
                            start[0, target['const_pos'], C.V] = tok.value_tok2idx[str(new_const)]
                            inj = dict(kind=cond, pos=target['var_pos'],
                                       const_pos=target['const_pos'],
                                       from_value=target['true'], to_value=new_const,
                                       truth_unchanged=False)
                            scratch_horizon = max(a.horizon, mg['n_loops'])
                            rows = rollout_records(lit, teacher, tok, start, meta, mg,
                                                   target, cond, scratch_horizon, label[bi], 0)
                            records.append(dict(graph_id=gid, condition=cond,
                                                target_index=ti, target=target,
                                                injection=inj, T=None, pre_forward_round=0,
                                                rounds=rows, new_truth=recompute_truth(
                                                    meta, label[bi], tok, teacher,
                                                    {target['const_pos']: new_const}),
                                                node_truth=mg['nodes']))
                            counts['emitted'] += 1
                            continue
                        if cond == 'leaf_constant_pinned':
                            # D3: keep the new constant forced every round.
                            new_const = (int(target['true']) + delta) % tok.mod_val
                            start = base[0:1].clone()
                            start[0, target['const_pos'], C.V] = tok.value_tok2idx[str(new_const)]
                            def pin_hook(t, cur, cpos=target['const_pos'], v=new_const):
                                cur[0, cpos, C.V] = tok.value_tok2idx[str(v)]
                                return cur
                            rows = rollout_records(lit, teacher, tok, start, meta, mg,
                                                   target, cond, a.horizon, label[bi],
                                                   T + 1, input_hook=pin_hook)
                            inj = dict(kind=cond, pos=target['var_pos'],
                                       const_pos=target['const_pos'],
                                       from_value=target['true'], to_value=new_const,
                                       truth_unchanged=False)
                            records.append(dict(graph_id=gid, condition=cond,
                                                target_index=ti, target=target,
                                                injection=inj, T=T, pre_forward_round=T + 1,
                                                rounds=rows, new_truth=recompute_truth(
                                                    meta, label[bi], tok, teacher,
                                                    {target['const_pos']: new_const}),
                                                node_truth=mg['nodes']))
                            counts['emitted'] += 1
                            continue
                        start, inj = make_injected(base[0:1], mg, meta, tok, label[bi],
                                                   cond, target, delta)
                        rows = rollout_records(lit, teacher, tok, start, meta, mg,
                                               target, cond, a.horizon, label[bi], T + 1)
                        extra = {}
                        if cond == 'leaf_constant':
                            new_const = int(inj['to_value'])
                            extra['new_truth'] = recompute_truth(
                                meta, label[bi], tok, teacher,
                                {target['const_pos']: new_const})
                        records.append(dict(graph_id=gid, condition=cond, target_index=ti,
                                            target=target, injection=inj,
                                            T=T, pre_forward_round=T + 1, rounds=rows,
                                            node_truth=mg['nodes'], **extra))
                        counts['emitted'] += 1
                done += 1
                if a.n_graphs and done >= a.n_graphs:
                    break
            if a.n_graphs and done >= a.n_graphs:
                break

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(dict(args=vars(a), counts=counts, records=records), f, indent=2)
    print(json.dumps(counts, indent=2))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()