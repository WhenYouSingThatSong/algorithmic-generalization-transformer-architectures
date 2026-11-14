#!/usr/bin/env python3
"""Second-batch B controls: sustained wrong parent and switch from held state.

Local self-consistent-error is deliberately deferred (see
tmp/records/2026-11-14/b-local-self-consistent-deferred.md) and is NOT
implemented here.

Modes:
  sustained_r   : force r wrong for hold_rounds, then release.
  switch_held_r : force r wrong for hold_rounds, then at switch_at set r to a
                  second wrong value once and release.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_mechanism_common as C  # noqa: E402


def set_r_value(cur, tok, r_pos, value):
    cur[0, r_pos, C.V] = tok.value_tok2idx[str(int(value) % tok.mod_val)]
    return cur


def values_at(state, tok, positions):
    return {str(p): C.tok_str(tok, int(state[0, p, C.V])) for p in positions}


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
    p.add_argument('--mode', choices=['sustained_r', 'switch_held_r'], required=True)
    p.add_argument('--hold_rounds', type=int, default=4,
                   help='rounds with r held at the first wrong value')
    p.add_argument('--switch_at', type=int, default=0,
                   help='1-based round at which r switches to the second wrong value; 0 disables')
    p.add_argument('--delta', type=int, default=3)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    with open(a.manifest) as f:
        manifest = json.load(f)
    C.verify_manifest_identity(manifest, a.ds_file, C.checkpoint_path(a.base, a.group, a.run, a.ckpt))
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
                T = mg['T']
                if T >= len(outs) or not C.all_numeric_correct(outs[T][0], label[bi], meta['numeric']):
                    counts['skipped'] += 1
                    continue
                base = outs[T].clone()
                counts['processed'] += 1
                for ci, ch in enumerate(mg['chains']):
                    # Only early-eligible chains are meaningful for this control.
                    flag = ch.get('early_own_after')
                    if flag is False:
                        counts['skipped'] += 1
                        continue
                    r_pos = ch['r_pos']
                    r_true = int(ch['r_true'])
                    r_wrong = (r_true + a.delta) % tok.mod_val
                    r_other = (r_wrong + a.delta) % tok.mod_val
                    positions = sorted((set(meta['nodes']) |
                                        {r_pos, ch['pos'], ch.get('q1'), ch.get('q2')}) - {None})

                    def hook(t, cur, r_pos=r_pos, r_wrong=r_wrong, r_other=r_other):
                        # Hold the first wrong value for hold_rounds.
                        if t < a.hold_rounds:
                            return set_r_value(cur, tok, r_pos, r_wrong)
                        # Switch once, at the requested round, then release.
                        if a.mode == 'switch_held_r' and a.switch_at and (t + 1) == a.switch_at:
                            return set_r_value(cur, tok, r_pos, r_other)
                        return cur

                    inputs, outputs = C.rollout(lit, teacher, base[0:1], a.horizon,
                                                input_hook=hook)
                    rounds = []
                    for t in range(a.horizon):
                        rounds.append(dict(
                            post_round=t + 1,
                            pre_forward_round=T + 1,
                            input=values_at(inputs[t], tok, positions),
                            output=values_at(outputs[t], tok, positions),
                            input_nodes=C.node_state(inputs[t], label[bi], tok, positions),
                            output_nodes=C.node_state(outputs[t], label[bi], tok, positions),
                        ))
                    records.append(dict(
                        graph_id=gid, mode=a.mode, target_index=ci, chain=ch,
                        r_true=r_true, r_wrong=r_wrong, r_other=r_other,
                        hold_rounds=a.hold_rounds, switch_at=a.switch_at,
                        pre_forward_round=T + 1, rounds=rounds,
                        node_truth=mg['nodes']))
                    counts['emitted'] += 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(dict(args=vars(a), counts=counts, records=records), f, indent=2)
    print(json.dumps(counts, indent=2))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()