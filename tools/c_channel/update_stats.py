#!/usr/bin/env python3
"""How much does each position's hidden state move between consecutive rounds?

Measured under several conditions on the SAME graphs:

  none        plain loop, no intervention (the whole graph is computed)
  freeze0     freeze every result position with depth > t      (plain freeze)
  freeze1     ... depth > t + 1                                (one round of slack)
  freeze2     ... depth > t + 2

Two numbers per (round, position):

  one_minus_cos = 1 - cos(h_t[u], h_{t+1}[u])      direction change
  rel_norm      = ||h_{t+1}[u] - h_t[u]|| / ||h_t[u]||   also sees rescaling,
                                                          which cosine is blind to

Both say only that the state MOVED, not that it computed.

`lead = depth(u) - t`.  The marginal table by lead is NOT interpretable on its own:
a large lead can only be observed in early rounds, so lead and round are confounded.
The (round x lead) cross table is reported for that reason.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as P


def run_with_stats(rt, batch, depths, res, variant='none', grace=0):
    """Run the loop, optionally freezing, and return per-round movement records."""
    bi, pos = torch.where(res)
    dv = depths[bi, pos]
    h = rt.token_embedder(batch)
    L = min(int(depths.max().item()) + 1, 36)
    acc = dict(t=[], lead=[], one_minus_cos=[], rel_norm=[], pred=None)
    logits = None
    with torch.no_grad():
        for t in range(L):
            h_prev = h
            h, preds = rt(h, True)
            logits = preds
            a_, b_ = h_prev[bi, pos], h[bi, pos]
            cos = F.cosine_similarity(a_, b_, dim=1)
            rel = (b_ - a_).norm(dim=1) / a_.norm(dim=1).clamp_min(1e-12)
            acc['t'].append(np.full(len(cos), t))
            acc['lead'].append((dv - t).cpu().numpy())
            acc['one_minus_cos'].append((1 - cos).cpu().numpy())
            acc['rel_norm'].append(rel.cpu().numpy())
            if variant == 'freeze':
                h = h.clone()
                fr = depths > (t + grace)
                h[fr] = h_prev[fr]
    out = {k: np.concatenate(v) for k, v in acc.items() if k != 'pred'}
    out['logits'] = logits
    return out


def summarise(rec):
    t_all, lead_all = rec['t'], rec['lead']
    o, r = rec['one_minus_cos'], rec['rel_norm']
    by_round, by_lead, by_round_lead = {}, {}, {}
    for t in sorted(set(t_all.tolist())):
        s = t_all == t
        by_round[int(t)] = dict(n=int(s.sum()), mean_one_minus_cos=float(o[s].mean()),
                                mean_rel_norm=float(r[s].mean()))
    for Ld in sorted(set(lead_all.tolist())):
        s = lead_all == Ld
        if s.sum() < 50:
            continue
        by_lead[int(Ld)] = dict(n=int(s.sum()), mean_one_minus_cos=float(o[s].mean()),
                                mean_rel_norm=float(r[s].mean()))
    for t in sorted(set(t_all.tolist())):
        cells = {}
        for Ld in (0, 1, 2, 3, 4):
            s = (t_all == t) & ((lead_all == Ld) if Ld < 4 else (lead_all >= 4))
            if s.sum() < 30:
                continue
            cells[str(Ld) if Ld < 4 else '4+'] = dict(n=int(s.sum()),
                                                     mean_one_minus_cos=float(o[s].mean()),
                                                     mean_rel_norm=float(r[s].mean()))
        if cells:
            by_round_lead[int(t)] = cells
    return dict(by_round=by_round, by_lead=by_lead, by_round_lead=by_round_lead,
                overall=dict(n=int(len(o)), mean_one_minus_cos=float(o.mean()),
                             mean_rel_norm=float(r.mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1234,2026')
    ap.add_argument('--conditions', default='none,freeze0,freeze1,freeze2')
    ap.add_argument('--n-graphs', type=int, default=8)
    ap.add_argument('--out', default='tmp/analysis/channel/update_stats.json')
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
    valid = res

    results = {}
    for seed in [int(s) for s in a.seeds.split(',')]:
        d, ck = P.WEIGHTS[seed]
        lit, tk, dev = P.load_model(os.path.join(P.REPO, d, ck))
        rt = lit.model
        b, dp, lb, vd = (x.to(dev) for x in (batch, depths, label, valid))
        results[str(seed)] = {}
        for cond in a.conditions.split(','):
            if cond == 'none':
                rec = run_with_stats(rt, b, dp, res.to(dev))
            else:
                g = int(cond.replace('freeze', ''))
                rec = run_with_stats(rt, b, dp, res.to(dev), variant='freeze', grace=g)
            m = P.metrics_from_logits(rec['logits'], lb, vd)
            summ = summarise(rec)
            summ['metrics'] = dict(node_acc=m['value_node_acc'], graph_all=m['value_graph_all'])
            results[str(seed)][cond] = summ
            print(f"\nseed {seed}  condition {cond}: node_acc={m['value_node_acc']:.6f} "
                  f"graph_all={m['value_graph_all']:.6f}  总体 1-cos={summ['overall']['mean_one_minus_cos']:.5f} "
                  f"rel_norm={summ['overall']['mean_rel_norm']:.5f}", flush=True)
            print('  交叉表 (1-cos / rel_norm)：行=轮次  列=lead')
            print('    t ' + ''.join(f'{c:>19}' for c in ('lead0', 'lead1', 'lead2', 'lead3', 'lead4+')))
            for t, cells in summ['by_round_lead'].items():
                line = f'  {t:>3} '
                for c in ('0', '1', '2', '3', '4+'):
                    line += (f"{cells[c]['mean_one_minus_cos']:.4f}/{cells[c]['mean_rel_norm']:.4f}".rjust(19)
                             if c in cells else '-'.rjust(19))
                print(line)
        del lit
        torch.cuda.empty_cache()

    p = os.path.join(P.REPO, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(results, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n已写', os.path.relpath(p, P.REPO))


if __name__ == '__main__':
    main()
