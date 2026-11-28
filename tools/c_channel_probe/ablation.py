#!/usr/bin/env python3
"""Freeze ladder + four-vector-difference measures.

Design: docs/records/2026-11-28/vector-difference-and-freeze-ablation-design.md

Conditions (see the design doc for why each exists):
  A   no freeze
  B   pre_grace=0, start=0    the old freeze: pins positions at h_0
  C   pre_grace=0, start=1    lets the round-0 transform happen, then pins
  D2  pre_grace=0, start=2
  D3  pre_grace=0, start=3
  E   post_grace=0            no update at all once a position is computed
  F1  post_grace=1            one round of update after computing, then pinned
  F2  post_grace=2
  G   pre_grace=0, start=1 with extra rounds (default +8)

Measures per (round, position), for the state that produced the round's readout:
  one_minus_cos  direction change, scale free
  norm_ratio     ||h_{t+1}|| / ||h_t||          (1 = length unchanged)
  rel_l2         ||delta|| / ||h_t||            direction and length together
  radial         (delta . h_hat) / ||h_t||      signed component along h_t
  tangential     sqrt(||delta||^2 - (delta.h_hat)^2) / ||h_t||
with the identity rel_l2^2 = radial^2 + tangential^2 checked on every record.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe as P

CONDS = {
    'A':   dict(pre=None, post=None, start=0, extra=0),
    'B':   dict(pre=0,    post=None, start=0, extra=0),
    'C':   dict(pre=0,    post=None, start=1, extra=0),
    'D2':  dict(pre=0,    post=None, start=2, extra=0),
    'D3':  dict(pre=0,    post=None, start=3, extra=0),
    'E':   dict(pre=None, post=0,    start=0, extra=0),
    'F1':  dict(pre=None, post=1,    start=0, extra=0),
    'F2':  dict(pre=None, post=2,    start=0, extra=0),
    'G':   dict(pre=0,    post=None, start=1, extra=8),
    'A8':  dict(pre=None, post=None, start=0, extra=8),
}


def run_cond(rt, batch, depths, res, cond):
    bi, pos = torch.where(res)
    dv = depths[bi, pos]
    h = rt.token_embedder(batch)
    L = min(int(depths.max().item()) + 1, 36) + cond['extra']
    acc = {k: [] for k in ('t', 'lead', 'one_minus_cos', 'norm_ratio', 'rel_l2',
                           'radial', 'tangential')}
    worst_identity = 0.0
    logits = None
    with torch.no_grad():
        for t in range(L):
            h_prev = h
            h, preds = rt(h, True)
            logits = preds
            a_, b_ = h_prev[bi, pos], h[bi, pos]
            d = b_ - a_
            an = a_.norm(dim=1)
            bn = b_.norm(dim=1)
            dn = d.norm(dim=1)
            par = (d * a_).sum(1) / an.clamp_min(1e-12)
            perp = (dn ** 2 - par ** 2).clamp_min(0).sqrt()
            cos = F.cosine_similarity(a_, b_, dim=1)
            one_minus_cos = 1 - cos
            norm_ratio = bn / an.clamp_min(1e-12)
            rel_l2 = dn / an.clamp_min(1e-12)
            radial = par / an.clamp_min(1e-12)
            tangential = perp / an.clamp_min(1e-12)
            worst_identity = max(worst_identity,
                                 float(((rel_l2 ** 2 - radial ** 2 - tangential ** 2).abs() /
                                        (rel_l2 ** 2).clamp_min(1e-9)).max()))
            acc['t'].append(np.full(len(cos), t))
            acc['lead'].append((dv - t).cpu().numpy())
            for k, v in (('one_minus_cos', one_minus_cos), ('norm_ratio', norm_ratio),
                         ('rel_l2', rel_l2), ('radial', radial), ('tangential', tangential)):
                acc[k].append(v.cpu().numpy())
            if t >= cond['start']:
                fr = torch.zeros_like(res)
                if cond['pre'] is not None:
                    fr |= res & (depths > (t + cond['pre']))
                if cond['post'] is not None:
                    fr |= res & (depths < (t - cond['post']))
                if fr.any():
                    h = h.clone()
                    h[fr] = h_prev[fr]
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out['logits'] = logits
    out['identity_err'] = worst_identity
    return out


def summarise(rec):
    t_all, lead_all = rec['t'], rec['lead']
    def mean(k, s):
        return float(rec[k][s].mean())
    by_round, by_lead = {}, {}
    for t in sorted(set(t_all.tolist())):
        s = t_all == t
        by_round[int(t)] = dict(n=int(s.sum()), one_minus_cos=mean('one_minus_cos', s),
                                norm_ratio=mean('norm_ratio', s), rel_l2=mean('rel_l2', s),
                                radial=mean('radial', s), tangential=mean('tangential', s))
    for Ld in sorted(set(lead_all.tolist())):
        s = lead_all == Ld
        if s.sum() < 50:
            continue
        by_lead[int(Ld)] = dict(n=int(s.sum()), one_minus_cos=mean('one_minus_cos', s),
                                norm_ratio=mean('norm_ratio', s), rel_l2=mean('rel_l2', s),
                                radial=mean('radial', s), tangential=mean('tangential', s))
    return dict(by_round=by_round, by_lead=by_lead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--n-graphs', type=int, default=8)
    ap.add_argument('--conds', default=','.join(CONDS))
    ap.add_argument('--out', default='tmp/analysis/channel-probe/ablation.json')
    a = ap.parse_args()

    data = torch.load(os.path.join(P.REPO, P.DATA), map_location='cpu', weights_only=True)
    tok_t, info = data['tokenized_dataset'][:a.n_graphs], data['dataset_info'][:a.n_graphs]
    label = tok_t.clone()
    batch = tok_t.clone()
    batch[..., P.F_VALUE] = torch.where(batch[..., P.F_SYNTAX] == P.SYN_VAR,
                                        torch.full_like(batch[..., P.F_VALUE], P.EMPTY),
                                        batch[..., P.F_VALUE])
    depths = info[..., 0].clone()
    res = (depths >= 0) & (batch[..., P.F_SYNTAX] != P.SYN_PAD)

    out = {}
    for seed in [int(s) for s in a.seeds.split(',')]:
        d, ck = P.WEIGHTS[seed]
        lit, tk, dev = P.load_model(os.path.join(P.REPO, d, ck))
        rt = lit.model
        b, dp, lb, rs = (x.to(dev) for x in (batch, depths, label, res))
        out[str(seed)] = {}
        for name in a.conds.split(','):
            rec = run_cond(rt, b, dp, rs, CONDS[name])
            m = P.metrics_from_logits(rec['logits'], lb, (res.to(dev)))
            s = summarise(rec)
            s['metrics'] = dict(node_acc=m['value_node_acc'], graph_all=m['value_graph_all'])
            s['identity_err'] = rec['identity_err']
            out[str(seed)][name] = s
            print(f"seed {seed} {name:3} node_acc={m['value_node_acc']:.6f} "
                  f"graph_all={m['value_graph_all']:.6f}  identity_err={rec['identity_err']:.2e}",
                  flush=True)
        del lit
        torch.cuda.empty_cache()

    p = os.path.join(P.REPO, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('已写', os.path.relpath(p, P.REPO))


if __name__ == '__main__':
    main()
