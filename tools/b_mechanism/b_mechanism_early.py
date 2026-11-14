#!/usr/bin/env python3
"""Early-timing 2x2 trajectories with full per-round records.

This is the early counterpart of b_mechanism_late.py.  It uses the frozen
manifest chains and can start from teacher S[d]/S[d+1] or model own rollout.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import b_mechanism_common as C


def set_value(state, pos, token_idx):
    state[0, pos, C.V] = token_idx
    return state


def values_at(state, tok, positions):
    return {str(p): C.tok_str(tok, int(state[0, p, C.V])) for p in positions}


def make_state(base, tok, chain, cond, a_plus_one=1, r_delta=3, value_mode='a_plus_1'):
    st = base.clone()
    a = int(chain['a'])
    r_true = int(chain['r_true'])
    if value_mode == 'four_to_five' and a != 4:
        return None, None
    a_inj = 5 if value_mode == 'four_to_five' else (a + a_plus_one) % tok.mod_val
    r_wrong = (r_true + r_delta) % tok.mod_val
    info = dict(a=a, a_inj=a_inj, r_true=r_true, r_wrong=r_wrong)
    if cond in ('u', 'ur'):
        set_value(st, chain['pos'], tok.value_tok2idx[str(a_inj)])
    if cond in ('r', 'ur'):
        set_value(st, chain['r_pos'], tok.value_tok2idx[str(r_wrong)])
    return st, info


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
    p.add_argument('--start', choices=['teacher', 'own'], default='own')
    p.add_argument('--timing', choices=['before_u', 'after_u'], required=True)
    p.add_argument('--value_mode', choices=['a_plus_1', 'four_to_five'], default='a_plus_1')
    p.add_argument('--r_delta', type=int, default=3)
    p.add_argument('--conditions', default='clean,u,r,ur',
                   help='comma-separated subset of clean,u,r,ur')
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    conditions = [c.strip() for c in a.conditions.split(',') if c.strip()]
    bad = [c for c in conditions if c not in ('clean', 'u', 'r', 'ur')]
    if bad:
        raise SystemExit(f'unknown early conditions: {bad}')
    if not conditions:
        raise SystemExit('no early conditions selected')

    with open(a.manifest) as f:
        manifest = json.load(f)
    C.verify_manifest_identity(manifest, a.ds_file, C.checkpoint_path(a.base, a.group, a.run, a.ckpt))
    C.validate_manifest_graphs(manifest)
    graphs = {g['graph_id']: g for g in manifest['graphs']}
    ctx = C.load_eval_context(a.config, a.group, a.run, a.base, a.ckpt,
                              a.ds_file, batch_size=a.batch_size, num_workers=0)
    lit, tok, teacher, mc = ctx['lit'], ctx['tok'], ctx['teacher'], ctx['mc']
    records = []
    counts = dict(processed=0, emitted=0, skipped=0)

    with torch.no_grad():
        for batch_idx, raw in enumerate(ctx['dm'].val_dataloader()):
            batch = lit.reduce_batch(raw).to(ctx['device'])
            data, label, info = C.split_batch(batch, mc)
            B = data.size(0)
            for bi in range(B):
                gid = batch_idx * a.batch_size + bi
                if gid not in graphs:
                    continue
                mg = graphs[gid]
                meta = C.build_meta(data[bi], label[bi], info[bi], tok, teacher)
                meta['info_b'] = info[bi].cpu()
                _, outs = C.rollout(lit, teacher, data[bi:bi + 1], mg['n_loops'])
                counts['processed'] += 1
                for ci, ch in enumerate(mg['chains']):
                    if not C.early_common_flag(ch, a.start, a.timing):
                        counts['skipped'] += 1
                        continue
                    d = int(ch['d'])
                    if a.start == 'own':
                        if a.timing == 'before_u':
                            if d == 0:
                                continue
                            base = outs[d - 1].clone()
                        else:
                            base = outs[d].clone()
                    else:
                        S, _, _ = lit.get_teacher_batch_with_depth(data, label, info[..., 0].long())
                        # S[0] is the original batch; S[1:] are depth-wise states.
                        idx = d if a.timing == 'before_u' else d + 1
                        if idx >= S.size(0):
                            continue
                        base = S[idx][bi:bi + 1].clone()
                    # Eligibility: r correct before injection; u EMPTY before or
                    # exactly correct after.
                    if int(base[0, ch['r_pos'], C.V]) != int(ch['r_true']):
                        counts['skipped'] += 1
                        continue
                    if a.timing == 'before_u':
                        if C.tok_str(tok, int(base[0, ch['pos'], C.V])) not in ('EMPTY', 'N/A'):
                            counts['skipped'] += 1
                            continue
                    else:
                        if int(base[0, ch['pos'], C.V]) != int(ch['a']):
                            counts['skipped'] += 1
                            continue
                    positions = sorted(set(meta['nodes']) | {ch['r_pos'], ch['pos'],
                                        ch.get('q1'), ch.get('q2')} - {None})
                    for cond in conditions:
                        start, inj = make_state(base, tok, ch, cond,
                                                r_delta=a.r_delta,
                                                value_mode=a.value_mode)
                        if start is None:
                            continue
                        inputs, outputs = C.rollout(lit, teacher, start, a.horizon)
                        rounds = []
                        for t in range(a.horizon):
                            rounds.append(dict(
                                post_round=t + 1,
                                pre_forward_round=(d if a.timing == 'before_u' else d + 1),
                                input=values_at(inputs[t], tok, positions),
                                output=values_at(outputs[t], tok, positions),
                                input_nodes=C.node_state(inputs[t], label[bi], tok, positions),
                                output_nodes=C.node_state(outputs[t], label[bi], tok, positions),
                            ))
                        records.append(dict(
                            graph_id=gid, chain_index=ci, event_id=ch.get('event_id'),
                            chain=ch, timing=a.timing,
                            start=a.start, condition=cond, injection=inj,
                            base_round=(d if a.timing == 'before_u' else d + 1),
                            rounds=rounds, node_truth=mg['nodes']))
                        counts['emitted'] += 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(dict(args=vars(a), counts=counts, records=records), f, indent=2)
    print(json.dumps(counts, indent=2))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()