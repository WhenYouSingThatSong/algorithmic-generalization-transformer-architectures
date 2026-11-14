#!/usr/bin/env python3
"""Audit B's `last` corruption mask against the actual forward input.

This does not train and does not modify weights.  It exports every candidate
position under the `last` strategy and reads the *pre-corruption input VALUE*,
then separately samples the normal stochastic corruption.

Key point: forcing `corruption_prob=1` only enters the branch; it does not
enumerate all candidates because `tokenwise_corruption_prob` still drops some.
The exhaustive table here comes directly from `candidates`.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import b_mechanism_common as C


def cat(tok, x):
    s = C.tok_str(tok, x)
    if s == 'EMPTY':
        return 'EMPTY'
    if s == 'N/A':
        return 'NA'
    if s.isdigit():
        return 'numeric'
    return 'other'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ds_file', required=True, help='training data file')
    p.add_argument('--config', required=True)
    p.add_argument('--group', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--ckpt', default='last.ckpt')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--n_batches', type=int, default=4)
    p.add_argument('--trials', type=int, default=100,
                   help='normal-probability sampling trials per batch')
    p.add_argument('--seed', type=int, default=20261114)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    ctx = C.load_eval_context(a.config, a.group, a.run, a.base, a.ckpt,
                              a.ds_file, batch_size=a.batch_size, num_workers=0)
    lit, tok, teacher, mc = ctx['lit'], ctx['tok'], ctx['teacher'], ctx['mc']
    val = tok.value_tok2idx
    value_vocab = tok.vocab_sizes[C.V]
    all_candidates = []
    actual = []
    batch_count = 0

    with torch.no_grad():
        for raw in ctx['dm'].val_dataloader():
            if batch_count >= a.n_batches:
                break
            batch = lit.reduce_batch(raw).to(ctx['device'])
            data, label, info = C.split_batch(batch, mc)
            depths = info[..., 0].long()
            # The trainer's local `batch_ext` is the returned concatenation;
            # `batch_ext[:-1]` is therefore S[:-1], not S[1:-1].
            S, value_avail_pos, _ = lit.get_teacher_batch_with_depth(data, label, depths)
            inputs = S[:-1]
            if value_avail_pos.size(0) != inputs.size(0):
                raise RuntimeError('depth/input shape mismatch')
            cand = torch.cat([
                value_avail_pos[0].unsqueeze(0),
                torch.logical_and(value_avail_pos[:-1].logical_not(), value_avail_pos[1:])
            ], dim=0)
            # Exhaustive candidate table: pre-corruption input value.
            for d in range(cand.size(0)):
                for b in range(cand.size(1)):
                    for t in range(cand.size(2)):
                        if not bool(cand[d, b, t]):
                            continue
                        before = int(inputs[d, b, t, C.V])
                        true = int(label[b, t, C.V])
                        all_candidates.append(dict(
                            batch=batch_count, b=b, row=d, pos=t,
                            node_depth=int(depths[b, t]),
                            before=cat(tok, before), before_token=int(before),
                            true=cat(tok, true), true_token=true,
                        ))
            # Actual stochastic sampling, normal tokenwise=0.5.
            for trial_id in range(a.trials):
                # B's outer coinflip is 0.05; keep it explicit.
                if np.random.random() >= float(lit.train_config.recorrection_args.corruption_prob):
                    continue
                corrupted, mask = lit.randomly_corrupt_values(inputs, value_avail_pos)
                for d in range(mask.size(0)):
                    for b in range(mask.size(1)):
                        for t in range(mask.size(2)):
                            if not bool(mask[d, b, t]):
                                continue
                            before = int(inputs[d, b, t, C.V])
                            after = int(corrupted[d, b, t, C.V])
                            true = int(label[b, t, C.V])
                            actual.append(dict(
                                trial_id=trial_id, batch=batch_count, b=b, row=d, pos=t,
                                node_depth=int(depths[b, t]),
                                before=cat(tok, before), before_token=before,
                                after=cat(tok, after), after_token=after,
                                true=cat(tok, true), true_token=true,
                                after_is_true=(after == true),
                            ))
            batch_count += 1

    def summarize(rows, with_after=False):
        out = dict(n=len(rows))
        out['before'] = dict(Counter(r['before'] for r in rows))
        out['node_depth'] = dict(Counter(str(r['node_depth']) for r in rows))
        if with_after:
            out['after'] = dict(Counter(r['after'] for r in rows))
            out['after_is_true'] = int(sum(1 for r in rows if r['after_is_true']))
        return out

    result = dict(
        args=vars(a),
        run_identity=dict(
            ds_sha256=C.sha256_file(a.ds_file),
            ckpt_sha256=C.sha256_file(C.checkpoint_path(a.base, a.group, a.run, a.ckpt)),
            script_sha256=C.sha256_file(os.path.abspath(__file__)),
            seed=a.seed,
        ),
        candidates=summarize(all_candidates),
        actual=summarize(actual, with_after=True),
        # Keep a bounded sample for manual inspection; full rows are useful but
        # large.  The summary above is exact over the processed batches.
        candidate_sample=all_candidates[:2000],
        actual_sample=actual[:2000],
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(candidates=result['candidates'], actual=result['actual']), indent=2))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()