#!/usr/bin/env python3
"""Select fixed candidates and freeze independently eligible mechanism targets.

Runs A' and B clean free rollouts on the same data. Fixed mode retains early
eligibility independently of the first common complete round T; late/control
runners only use graphs with T. Legacy topup remains complete-only.  It also freezes:

- primary chains: single-parent u, q1 not directly reading r, q2 present and not
  directly reading r;
- late sinks, leaf var/constant pairs, and unrelated nodes that cannot reach the
  observed u/q1/q2 path.

The manifest is used by early/late/control runners.  No injection occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_io import configure_runtime, add_runtime_arguments  # noqa: E402
if __name__ == '__main__':
    configure_runtime()

import torch  # noqa: E402

import b_mechanism_common as C  # noqa: E402


def model_args(prefix, p):
    return dict(
        config_dir_name=getattr(p, f'{prefix}_config'),
        group_name=getattr(p, f'{prefix}_group'),
        run_name=getattr(p, f'{prefix}_run'),
        ckpt_base_dir=getattr(p, f'{prefix}_base'),
        ckpt_name=getattr(p, f'{prefix}_ckpt'),
    )


def ckpt_path(args):
    return os.path.join(args['ckpt_base_dir'],
                        f"{args['group_name']}-{args['run_name']}",
                        args['ckpt_name'])


def find_common_T(outputs_a, outputs_b, label, numeric, max_rounds=None):
    """First common complete round within the per-graph budget."""
    n = min(len(outputs_a), len(outputs_b))
    if max_rounds is not None:
        n = min(n, int(max_rounds))
    for t in range(n):
        if C.all_numeric_correct(outputs_a[t], label, numeric) and \
           C.all_numeric_correct(outputs_b[t], label, numeric):
            return t
    return None


def check_early_state(base, ch, timing, tok):
    """Eligibility at an early start state for one model."""
    r_ok = int(base[0, ch['r_pos'], C.V]) == int(ch['r_true'])
    u_val = C.tok_str(tok, int(base[0, ch['pos'], C.V]))
    if timing == 'before_u':
        u_ok = u_val in ('EMPTY', 'N/A')
    else:
        u_ok = int(base[0, ch['pos'], C.V]) == int(ch['a'])
    return bool(r_ok and u_ok)


def build_late_targets(meta, label_b, tok, k):
    sinks = [p for p in meta['sinks']
             if int(meta['depth'][p]) >= 1 and meta['parents'].get(p)]
    sinks = sorted(sinks, key=lambda p: (-int(meta['depth'][p]), p))[:k]
    leaves = sorted(meta['leaves'], key=lambda z: (-int(meta['depth'][z[0]]), z[0]))[:k]
    return dict(
        sinks=[dict(pos=p, d=int(meta['depth'][p]),
                    true=C.tok_str(tok, int(label_b[p, C.V])))
               for p in sinks],
        leaves=[dict(var_pos=v, const_pos=c, d=int(meta['depth'][v]),
                     true=C.tok_str(tok, int(label_b[v, C.V])))
                for v, c in leaves],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ds_file', required=True)
    p.add_argument('--n_graphs', type=int, default=1024,
                   help='maximum selected common-complete graphs')
    p.add_argument('--selected_graphs', type=int, default=0,
                   help='alias for n_graphs; 0 uses --n_graphs')
    p.add_argument('--scan_graphs', type=int, default=0,
                   help='maximum graphs scanned; 0 = no separate scan cap')
    p.add_argument('--candidate_start', type=int, default=0,
                   help='first graph_id in fixed-candidate mode (default 0)')
    p.add_argument('--candidate_count', type=int, default=0,
                   help='number of graph_ids in fixed-candidate mode; 0 = disabled')
    p.add_argument('--selection_mode', choices=['topup', 'fixed'], default='topup',
                   help='topup = old select-until-full behavior; fixed = scan a '
                        'predeclared graph_id range and never top up')
    p.add_argument('--no_topup', action='store_true',
                   help='alias for --selection_mode fixed')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--k', type=int, default=4, help='per-graph targets per type')
    p.add_argument('--extra_rounds', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--a_config', required=True)
    p.add_argument('--a_group', required=True)
    p.add_argument('--a_run', required=True)
    p.add_argument('--a_base', required=True)
    p.add_argument('--a_ckpt', default='last.ckpt')
    p.add_argument('--b_config', required=True)
    p.add_argument('--b_group', required=True)
    p.add_argument('--b_run', required=True)
    p.add_argument('--b_base', required=True)
    p.add_argument('--b_ckpt', default='last.ckpt')
    add_runtime_arguments(p)
    a = p.parse_args()
    if a.selected_graphs:
        a.n_graphs = a.selected_graphs
    if a.no_topup:
        a.selection_mode = 'fixed'
    if a.selection_mode == 'fixed':
        if a.candidate_count <= 0:
            p.error('--candidate_count must be positive in fixed mode')
        if a.candidate_start < 0:
            p.error('--candidate_start must be non-negative')
        if a.scan_graphs:
            p.error('--scan_graphs cannot be combined with fixed mode; '
                    'use --candidate_count')
        candidate_end = a.candidate_start + a.candidate_count
    else:
        if a.candidate_start or a.candidate_count:
            p.error('--candidate_start/--candidate_count require fixed mode')
        candidate_end = None

    ma, mb = model_args('a', a), model_args('b', a)
    ctx_a = C.load_eval_context(**ma, ds_file=a.ds_file,
                                batch_size=a.batch_size, num_workers=0)
    ctx_b = C.load_eval_context(**mb, ds_file=a.ds_file,
                                batch_size=a.batch_size, num_workers=0)
    lit_a, lit_b = ctx_a['lit'], ctx_b['lit']
    tok, teacher, mc = ctx_a['tok'], ctx_a['teacher'], ctx_a['mc']
    if tuple(ctx_a['mc'].vocab_sizes) != tuple(ctx_b['mc'].vocab_sizes):
        raise SystemExit(f'vocab mismatch: {ctx_a["mc"].vocab_sizes} vs {ctx_b["mc"].vocab_sizes}')

    graphs = []
    candidates = []
    counts = dict(scanned=0, candidates_total=0, not_common_complete=0,
                  candidates_with_chains=0, common_complete=0,
                  chains=0, sinks=0, leaves=0,
                  same_all=0, same_numeric=0, skipped_no_chain=0,
                  early_own_before=0, early_own_after=0,
                  early_teacher_before=0, early_teacher_after=0,
                  early_own_before_a=0, early_own_before_b=0,
                  early_own_after_a=0, early_own_after_b=0,
                  early_teacher_before_a=0, early_teacher_before_b=0,
                  early_teacher_after_a=0, early_teacher_after_b=0,
                  early_own_before_disagree=0, early_own_after_disagree=0,
                  early_teacher_before_disagree=0, early_teacher_after_disagree=0)
    selected = 0
    scanned = 0
    with torch.no_grad():
        for batch_idx, raw in enumerate(ctx_a['dm'].val_dataloader()):
            batch_start = batch_idx * a.batch_size
            if a.selection_mode == 'fixed':
                if batch_start >= candidate_end:
                    break
            else:
                if a.scan_graphs and scanned >= a.scan_graphs:
                    break
                if selected >= a.n_graphs:
                    break
            batch = lit_a.reduce_batch(raw).to(ctx_a['device'])
            data, label, info = C.split_batch(batch, mc)
            depths = info[..., 0].long()
            B = data.size(0)
            batch_loops = int(depths.max().item()) + 1 + a.extra_rounds
            _, out_a = C.rollout(lit_a, teacher, data, batch_loops)
            _, out_b = C.rollout(lit_b, teacher, data, batch_loops)
            S_a, _, _ = lit_a.get_teacher_batch_with_depth(data, label, depths)
            S_b, _, _ = lit_b.get_teacher_batch_with_depth(data, label, depths)
            for bi in range(B):
                gid = batch_start + bi
                if a.selection_mode == 'fixed':
                    if gid < a.candidate_start:
                        continue
                    if gid >= candidate_end:
                        continue
                else:
                    if a.scan_graphs and scanned >= a.scan_graphs:
                        break
                scanned += 1
                counts['scanned'] += 1
                graph_loops = int(depths[bi].max().item()) + 1 + a.extra_rounds
                meta = C.build_meta(data[bi], label[bi], info[bi], tok, teacher)
                T = find_common_T([o[bi] for o in out_a], [o[bi] for o in out_b],
                                  label[bi], meta['numeric'], max_rounds=graph_loops)
                common_complete = T is not None
                # Fixed candidate mode retains early-eligible graphs even when
                # no common complete state exists. Legacy topup still selects
                # complete graphs until its requested count is reached.
                if T is None and a.selection_mode == 'topup':
                    candidates.append(dict(graph_id=gid, common_complete=False,
                                           status='not_common_complete', T=None,
                                           n_loops=graph_loops, n_chains=0))
                    continue
                if common_complete:
                    if not (0 <= T < graph_loops):
                        raise AssertionError(f'graph {gid}: T={T} outside [0,{graph_loops})')
                    counts['common_complete'] += 1
                chains = C.pick_chain(meta, data[bi], label[bi], tok, k=a.k)
                # Early eligibility is recorded per model, then stored as a
                # common flag.  Both runners must use the common flag.
                for ci, ch in enumerate(chains):
                    ch['event_id'] = f'g{gid}:c{ci}'
                    d = int(ch['d'])
                    a_before = b_before = False
                    if d > 0:
                        a_before = check_early_state(out_a[d - 1][bi:bi + 1], ch, 'before_u', tok)
                        b_before = check_early_state(out_b[d - 1][bi:bi + 1], ch, 'before_u', tok)
                    a_after = check_early_state(out_a[d][bi:bi + 1], ch, 'after_u', tok)
                    b_after = check_early_state(out_b[d][bi:bi + 1], ch, 'after_u', tok)
                    a_teacher_before = b_teacher_before = False
                    a_teacher_after = b_teacher_after = False
                    if (d < S_a.size(0) and d + 1 < S_a.size(0) and
                            d < S_b.size(0) and d + 1 < S_b.size(0)):
                        a_teacher_before = check_early_state(
                            S_a[d][bi:bi + 1], ch, 'before_u', tok)
                        b_teacher_before = check_early_state(
                            S_b[d][bi:bi + 1], ch, 'before_u', tok)
                        a_teacher_after = check_early_state(
                            S_a[d + 1][bi:bi + 1], ch, 'after_u', tok)
                        b_teacher_after = check_early_state(
                            S_b[d + 1][bi:bi + 1], ch, 'after_u', tok)
                    C.set_early_flag(ch, 'own', 'before_u', a_before, b_before)
                    C.set_early_flag(ch, 'own', 'after_u', a_after, b_after)
                    C.set_early_flag(ch, 'teacher', 'before_u',
                                     a_teacher_before, b_teacher_before)
                    C.set_early_flag(ch, 'teacher', 'after_u',
                                     a_teacher_after, b_teacher_after)
                    counts['early_own_before'] += int(ch['early_own_before'])
                    counts['early_own_after'] += int(ch['early_own_after'])
                    counts['early_teacher_before'] += int(ch['early_teacher_before'])
                    counts['early_teacher_after'] += int(ch['early_teacher_after'])
                    counts['early_own_before_a'] += int(a_before)
                    counts['early_own_before_b'] += int(b_before)
                    counts['early_own_after_a'] += int(a_after)
                    counts['early_own_after_b'] += int(b_after)
                    counts['early_teacher_before_a'] += int(a_teacher_before)
                    counts['early_teacher_before_b'] += int(b_teacher_before)
                    counts['early_teacher_after_a'] += int(a_teacher_after)
                    counts['early_teacher_after_b'] += int(b_teacher_after)
                    counts['early_own_before_disagree'] += int(a_before != b_before)
                    counts['early_own_after_disagree'] += int(a_after != b_after)
                    counts['early_teacher_before_disagree'] += int(
                        a_teacher_before != b_teacher_before)
                    counts['early_teacher_after_disagree'] += int(
                        a_teacher_after != b_teacher_after)
                late = (build_late_targets(meta, label[bi], tok, k=a.k)
                        if common_complete else dict(sinks=[], leaves=[]))
                unrelated = []
                for ch in (chains if common_complete else []):
                    forbidden = {ch['pos'], ch['r_pos']}
                    for node in (ch['pos'], ch.get('q1'), ch.get('q2')):
                        if node is None:
                            continue
                        forbidden |= set(C.ancestors(meta['parents'], node))
                        forbidden |= set(C.descendants(meta['children'], node))
                        forbidden.add(node)
                    cand = [x for x in meta['nodes'] if x not in forbidden]
                    if not cand:
                        continue
                    cand.sort(key=lambda x: (abs(int(meta['depth'][x]) - ch['d']), x))
                    unrelated.append(dict(pos=cand[0], d=int(meta['depth'][cand[0]]),
                                          true=C.tok_str(tok, int(label[bi, cand[0], C.V])),
                                          for_u=ch['pos'], for_chain_event_id=ch['event_id'],
                                          for_chain=dict(ch)))
                same_all = common_complete and bool((out_a[T][bi] == out_b[T][bi]).all().item())
                same_numeric = common_complete and bool((out_a[T][bi][..., C.V][meta['numeric']] ==
                                     out_b[T][bi][..., C.V][meta['numeric']]).all().item())
                if same_all:
                    counts['same_all'] += 1
                if same_numeric:
                    counts['same_numeric'] += 1
                candidates.append(dict(
                    graph_id=gid, common_complete=common_complete,
                    status='selected' if common_complete else 'early_only',
                    T=T, n_loops=graph_loops, n_chains=len(chains),
                    n_sinks=len(late['sinks']), n_leaves=len(late['leaves']),
                    n_unrelated=len(unrelated),
                    early_own_before=sum(int(ch['early_own_before']) for ch in chains),
                    early_own_after=sum(int(ch['early_own_after']) for ch in chains),
                    early_teacher_before=sum(int(ch['early_teacher_before']) for ch in chains),
                    early_teacher_after=sum(int(ch['early_teacher_after']) for ch in chains),
                    early_own_before_a=sum(int(ch.get('early_own_before_a', False)) for ch in chains),
                    early_own_before_b=sum(int(ch.get('early_own_before_b', False)) for ch in chains),
                    early_own_after_a=sum(int(ch.get('early_own_after_a', False)) for ch in chains),
                    early_own_after_b=sum(int(ch.get('early_own_after_b', False)) for ch in chains),
                    early_teacher_before_a=sum(int(ch.get('early_teacher_before_a', False)) for ch in chains),
                    early_teacher_before_b=sum(int(ch.get('early_teacher_before_b', False)) for ch in chains),
                    early_teacher_after_a=sum(int(ch.get('early_teacher_after_a', False)) for ch in chains),
                    early_teacher_after_b=sum(int(ch.get('early_teacher_after_b', False)) for ch in chains),
                    same_all=same_all, same_numeric=same_numeric))
                graphs.append(dict(
                    graph_id=gid, T=T, common_complete=common_complete,
                    T_kind='first_common_complete' if common_complete else 'no_common_complete',
                    n_loops=graph_loops,
                    n_numeric=int(meta['numeric'].sum().item()),
                    chains=chains, late=late, unrelated=unrelated,
                    same_all=same_all, same_numeric=same_numeric,
                    nodes=C.graph_records(meta, label[bi], tok),
                ))
                counts['chains'] += len(chains)
                counts['sinks'] += len(late['sinks'])
                counts['leaves'] += len(late['leaves'])
                if not chains:
                    counts['skipped_no_chain'] += 1
                selected += 1
                if a.selection_mode == 'topup' and selected >= a.n_graphs:
                    break

    counts['candidates_total'] = len(candidates)
    counts['not_common_complete'] = sum(
        1 for c in candidates if not c['common_complete'])
    counts['candidates_with_chains'] = sum(
        1 for c in candidates if c['n_chains'] > 0)
    for key in ('early_own_before', 'early_own_after', 'early_teacher_before', 'early_teacher_after'):
        counts[key + '_graphs'] = sum(int(c.get(key, 0) > 0) for c in candidates)
    if a.selection_mode == 'fixed':
        if len(candidates) != a.candidate_count:
            raise SystemExit(
                f'fixed candidate range incomplete: expected '
                f'{a.candidate_count}, got {len(candidates)}')
        if counts['scanned'] != a.candidate_count:
            raise SystemExit(
                f'fixed candidate scan mismatch: expected '
                f'{a.candidate_count}, scanned {counts["scanned"]}')
        n_graphs_out = len(graphs)
        candidate_end_out = candidate_end
    else:
        n_graphs_out = a.n_graphs
        candidate_end_out = None

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    manifest = dict(
        ds_file=a.ds_file, ds_sha256=C.sha256_file(a.ds_file),
        n_graphs=n_graphs_out, requested_n_graphs=a.n_graphs,
        scan_graphs=a.scan_graphs, batch_size=a.batch_size, k=a.k,
        extra_rounds=a.extra_rounds,
        selection_mode=a.selection_mode,
        eligibility_protocol="independent_early_complete_v2",
        candidate_start=a.candidate_start,
        candidate_count=a.candidate_count,
        candidate_end=candidate_end_out,
        model_a={**ma, 'ckpt_sha256': C.sha256_file(ckpt_path(ma))},
        model_b={**mb, 'ckpt_sha256': C.sha256_file(ckpt_path(mb))},
        code_sha256={
            'common': C.sha256_file(os.path.join(os.path.dirname(__file__), 'b_mechanism_common.py')),
            'select': C.sha256_file(os.path.abspath(__file__)),
        },
        counts=counts, candidates=candidates, graphs=graphs,
    )
    with open(a.out, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(counts, indent=2))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()