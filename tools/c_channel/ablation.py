#!/usr/bin/env python3
"""冻结消融阶梯 + 信息传递指标（只推理，不重训）。

条件名刻意不复用 A/B/C 这套字母，避免与模型名（A′ / B / C / D）冲突。

每个条件都在同一批题、同一套权重量上跑；按 `--chunk` 分批前向以免显存不足，
记录按位置-轮次累积后统一汇总。对每个 (轮次 t, 位置 u) 记五个量：

  one_minus_cos  1 - cos(h_t, h_{t+1})          方向变化，尺度无关
  norm_ratio     ||h_{t+1}|| / ||h_t||          模长比，1 表示长度不变
  rel_l2         ||Δ|| / ||h_t||                方向与模长合起来的相对变化
  radial         (Δ·ĥ) / ||h_t||                沿原方向的分量（有符号）
  tangential     sqrt(||Δ||² - (Δ·ĥ)²) / ||h_t||  垂直分量

恒等式 rel_l2² = radial² + tangential² 逐条自检。
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as P

# 条件名 -> (pre, post, start, extra)
#   pre=g   每轮把 depth > t+g 的结果位置退回上一轮（未到轮次的）
#   post=p  每轮把 depth < t-p 的结果位置退回上一轮（已经算出的）
#   start   从第几轮起才施加冻结
#   extra   在 max_depth+1 之外多跑几轮
CONDS = {
    'baseline':                    dict(pre=None, post=None, start=0, extra=0),
    'hold_unreached_from0':        dict(pre=0,    post=None, start=0, extra=0),
    'hold_unreached_from1':        dict(pre=0,    post=None, start=1, extra=0),
    'hold_unreached_from2':        dict(pre=0,    post=None, start=2, extra=0),
    'hold_unreached_from3':        dict(pre=0,    post=None, start=3, extra=0),
    'hold_computed_immediately':   dict(pre=None, post=0,    start=0, extra=0),
    'hold_computed_after1':        dict(pre=None, post=1,    start=0, extra=0),
    'hold_computed_after2':        dict(pre=None, post=2,    start=0, extra=0),
    'hold_unreached_from1_extra8': dict(pre=0,    post=None, start=1, extra=8),
    'baseline_extra8':             dict(pre=None, post=None, start=0, extra=8),
}
KEYS = ('one_minus_cos', 'norm_ratio', 'rel_l2', 'radial', 'tangential')


def run_chunk(rt, batch, depths, res, cond):
    bi, pos = torch.where(res)
    dv = depths[bi, pos]
    h = rt.token_embedder(batch)
    L = min(int(depths.max().item()) + 1, 36) + cond['extra']
    acc = {k: [] for k in ('t', 'lead') + KEYS}
    worst_id = 0.0
    logits = None
    with torch.no_grad():
        for t in range(L):
            h_prev = h
            h, preds = rt(h, True)
            logits = preds
            a_, b_ = h_prev[bi, pos], h[bi, pos]
            d = b_ - a_
            an, bn, dn = a_.norm(dim=1), b_.norm(dim=1), d.norm(dim=1)
            anc = an.clamp_min(1e-12)
            par = (d * a_).sum(1) / anc
            perp = (dn ** 2 - par ** 2).clamp_min(0).sqrt()
            vals = {
                'one_minus_cos': 1 - F.cosine_similarity(a_, b_, dim=1),
                'norm_ratio': bn / anc,
                'rel_l2': dn / anc,
                'radial': par / anc,
                'tangential': perp / anc,
            }
            rel2 = vals['rel_l2'] ** 2
            worst_id = max(worst_id, float(((rel2 - vals['radial'] ** 2 - vals['tangential'] ** 2).abs()
                                            / rel2.clamp_min(1e-9)).max()))
            acc['t'].append(np.full(len(vals['rel_l2']), t))
            acc['lead'].append((dv - t).cpu().numpy())
            for k in KEYS:
                acc[k].append(vals[k].cpu().numpy())
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
    out['identity_err'] = worst_id
    return out


def accumulate(bank, rec, logits, label, res, chunk_graphs):
    bank.setdefault('t', []).append(rec['t'])
    bank.setdefault('lead', []).append(rec['lead'])
    for k in KEYS:
        bank.setdefault(k, []).append(rec[k])
    vp = logits[P.F_VALUE].argmax(-1)
    tgt = label[..., P.F_VALUE]
    ok = (vp == tgt) & res
    bank.setdefault('node_ok', []).append(int(ok.sum()))
    bank.setdefault('node_num', []).append(int(res.sum()))
    per_graph = ok.sum(1).float() / res.sum(1).clamp_min(1).float()
    bank.setdefault('graph_ok', []).append(int((per_graph == 1.0).sum()))
    bank.setdefault('graph_num', []).append(chunk_graphs)
    bank.setdefault('identity_err', []).append(rec['identity_err'])


def summarise(bank):
    t_all = np.concatenate(bank['t']); lead_all = np.concatenate(bank['lead'])
    arr = {k: np.concatenate(bank[k]) for k in KEYS}
    def cell(s):
        return dict(n=int(s.sum()), **{k: float(arr[k][s].mean()) for k in KEYS})
    by_round = {int(t): cell(t_all == t) for t in sorted(set(t_all.tolist()))}
    by_lead = {int(L): cell(lead_all == L) for L in sorted(set(lead_all.tolist()))
               if (lead_all == L).sum() >= 50}
    by_round_lead = {}
    for t in sorted(set(t_all.tolist())):
        cells = {}
        for L in (0, 1, 2, 3, 4):
            s = (t_all == t) & ((lead_all == L) if L < 4 else (lead_all >= 4))
            if s.sum() >= 30:
                cells[str(L) if L < 4 else '4+'] = cell(s)
        if cells:
            by_round_lead[int(t)] = cells
    nn = sum(bank['node_num']); gnum = sum(bank['graph_num'])
    return dict(by_round=by_round, by_lead=by_lead, by_round_lead=by_round_lead,
                metrics=dict(node_acc=sum(bank['node_ok']) / max(nn, 1),
                             graph_all=sum(bank['graph_ok']) / max(gnum, 1),
                             node_num=nn, graph_num=gnum),
                identity_err=max(bank['identity_err']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1234,2026')
    ap.add_argument('--n-graphs', type=int, default=32)
    ap.add_argument('--chunk', type=int, default=8, help='每次前向的题数，控制显存')
    ap.add_argument('--conds', default=','.join(CONDS))
    ap.add_argument('--out', default='tmp/analysis/channel/ablation.json')
    a = ap.parse_args()

    data = torch.load(os.path.join(P.REPO, P.DATA), map_location='cpu', weights_only=True)
    n = min(a.n_graphs, data['tokenized_dataset'].size(0))
    tok_t, info = data['tokenized_dataset'][:n], data['dataset_info'][:n]
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
        out[str(seed)] = dict(n_graphs=n, chunk=a.chunk, conditions={})
        for name in a.conds.split(','):
            cond = CONDS[name]
            bank = {}
            for lo in range(0, n, a.chunk):
                hi = min(lo + a.chunk, n)
                b = batch[lo:hi].to(dev); dp = depths[lo:hi].to(dev)
                rs = res[lo:hi].to(dev); lb = label[lo:hi].to(dev)
                rec = run_chunk(rt, b, dp, rs, cond)
                accumulate(bank, rec, rec['logits'], lb, rs, hi - lo)
            s = summarise(bank)
            s['params'] = cond
            out[str(seed)]['conditions'][name] = s
            print(f"seed {seed:>4} {name:28} node_acc={s['metrics']['node_acc']:.6f} "
                  f"graph_all={s['metrics']['graph_all']:.6f} "
                  f"(n={s['metrics']['node_num']}, id_err={s['identity_err']:.1e})", flush=True)
        del lit
        torch.cuda.empty_cache()

    p = os.path.join(P.REPO, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('已写', os.path.relpath(p, P.REPO))


if __name__ == '__main__':
    main()
