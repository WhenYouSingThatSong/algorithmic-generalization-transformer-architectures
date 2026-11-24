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
import csv
import json
import os
from collections import Counter, defaultdict

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


CATS = ('EMPTY', 'NA', 'numeric', 'other')


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def group_graph_round(rows, with_after=False, batch_size=8):
    """Aggregate full candidate/actual rows by (batch, graph-in-batch, row)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(int(r['batch']), int(r['b']), int(r['row']))].append(r)
    out = []
    for (batch, b, row), rs in sorted(groups.items()):
        item = dict(batch=batch, b=b, row=row, graph_index=batch * batch_size + b,
                    n_records=len(rs),
                    min_depth=min(int(x['node_depth']) for x in rs),
                    max_depth=max(int(x['node_depth']) for x in rs))
        for c in CATS:
            item['before_' + c] = sum(1 for x in rs if x['before'] == c)
        if with_after:
            for c in CATS:
                item['after_' + c] = sum(1 for x in rs if x['after'] == c)
            item['after_is_true'] = sum(1 for x in rs if x['after_is_true'])
            item['before_numeric_after_numeric'] = sum(
                1 for x in rs if x['before'] == 'numeric' and x['after'] == 'numeric')
            item['before_4_after_5'] = sum(
                1 for x in rs if int(x['before_token']) == 4 and
                int(x['after_token']) == 5)
        out.append(item)
    return out


def transition_rows(rows):
    counts = Counter((r['before'], r['after']) for r in rows)
    out = []
    for (before, after), n in sorted(counts.items()):
        out.append(dict(before=before, after=after, n=n))
    return out


def group_position(rows, with_after=False, batch_size=8):
    """Aggregate candidate/actual rows by graph, round and position."""
    groups = defaultdict(list)
    for r in rows:
        groups[(int(r['batch']), int(r['b']), int(r['row']), int(r['pos']))].append(r)
    out = []
    for (batch, b, row, pos), rs in sorted(groups.items()):
        item = dict(batch=batch, b=b, row=row, pos=pos,
                    graph_index=batch * batch_size + b,
                    n_records=len(rs),
                    node_depth=int(rs[0]['node_depth']))
        for c in CATS:
            item['before_' + c] = sum(1 for x in rs if x['before'] == c)
        if with_after:
            for c in CATS:
                item['after_' + c] = sum(1 for x in rs if x['after'] == c)
            item['after_is_true'] = sum(1 for x in rs if x['after_is_true'])
            item['before_numeric_after_numeric'] = sum(
                1 for x in rs if x['before'] == 'numeric' and x['after'] == 'numeric')
            item['before_4_after_5'] = sum(
                1 for x in rs if int(x['before_token']) == 4 and
                int(x['after_token']) == 5)
        out.append(item)
    return out


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
        out['before_numeric'] = int(sum(1 for r in rows if r['before'] == 'numeric'))
        if with_after:
            out['after'] = dict(Counter(r['after'] for r in rows))
            out['after_is_true'] = int(sum(1 for r in rows if r['after_is_true']))
            out['after_numeric'] = int(sum(1 for r in rows if r['after'] == 'numeric'))
            out['before_numeric_after_numeric'] = int(sum(
                1 for r in rows if r['before'] == 'numeric' and r['after'] == 'numeric'))
            out['before_4_after_5'] = int(sum(
                1 for r in rows if int(r['before_token']) == 4 and
                int(r['after_token']) == 5))
        return out

    base = os.path.splitext(os.path.abspath(a.out))[0]
    candidate_by_graph_round = group_graph_round(
        all_candidates, batch_size=a.batch_size)
    actual_by_graph_round = group_graph_round(
        actual, with_after=True, batch_size=a.batch_size)
    candidate_by_position = group_position(
        all_candidates, batch_size=a.batch_size)
    actual_by_position = group_position(
        actual, with_after=True, batch_size=a.batch_size)
    transitions = transition_rows(actual)
    write_csv(base + '_candidate_by_graph_round.csv', candidate_by_graph_round)
    write_csv(base + '_actual_by_graph_round.csv', actual_by_graph_round)
    write_csv(base + '_candidate_by_position.csv', candidate_by_position)
    write_csv(base + '_actual_by_position.csv', actual_by_position)
    write_csv(base + '_actual_transitions.csv', transitions)

    result = dict(
        args=vars(a),
        run_identity=dict(
            ds_sha256=C.sha256_file(a.ds_file),
            ckpt_sha256=C.sha256_file(C.checkpoint_path(a.base, a.group, a.run, a.ckpt)),
            script_sha256=C.sha256_file(os.path.abspath(__file__)),
            seed=a.seed,
        ),
        coverage=dict(
            n_batches=batch_count,
            batch_size=a.batch_size,
            graphs_seen=batch_count * a.batch_size,
            candidate_rows=len(all_candidates),
            actual_rows=len(actual),
        ),
        candidates=summarize(all_candidates),
        actual=summarize(actual, with_after=True),
        # Keep a bounded sample for manual inspection; the aggregate CSV files
        # and the counters above are computed over all processed rows.
        candidate_sample=all_candidates[:2000],
        actual_sample=actual[:2000],
        aggregate_files=dict(
            candidate_by_graph_round=base + '_candidate_by_graph_round.csv',
            actual_by_graph_round=base + '_actual_by_graph_round.csv',
            candidate_by_position=base + '_candidate_by_position.csv',
            actual_by_position=base + '_actual_by_position.csv',
            actual_transitions=base + '_actual_transitions.csv',
        ),
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(candidates=result['candidates'], actual=result['actual']), indent=2))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()