#!/usr/bin/env python3
"""在一张小图上完整记录每一步的向量更新（只推理）。

默认生成一张约 16 个节点、深度 3–4 的图（与训练分布的节点数不同，仅用于过程观察），
然后逐轮、逐位置记录：读出值、五栏向量差异、以及“父节点是否已完备”。

输出 markdown 与 CSV；屏幕上给一份紧凑摘要。
"""
import argparse, csv, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments'))
import common as P

from generate_data import generate_dataset
from Simtransformer.simtransformer.utils import EasyDict


def make_small_graph(seed, num_nodes, max_depth, max_length):
    import yaml, random
    with open(os.path.join(P.REPO, P.CFG, 'data_config.yaml')) as f:
        dc = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    if hasattr(P, 'build_factored_tokenizer'):
        tok = P.build_factored_tokenizer(dc)
    else:
        from tokenizers import build_factored_tokenizer
        tok = build_factored_tokenizer(dc)
    cfg = dict(dc.val_dag_config)
    cfg.update(dict(num_samples=1, num_nodes=num_nodes, max_depth=max_depth,
                    max_comp_depth=max_depth, max_length=max_length))
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tok_ds, info = generate_dataset(EasyDict(cfg), tok)
    return tok_ds, info, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-c', type=int, default=42, help='用哪个 C 权重')
    ap.add_argument('--num-nodes', type=int, default=16)
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--gen-seed', type=int, default=1)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--out', default='tmp/analysis/channel/trace')
    a = ap.parse_args()

    tok_ds, info, tk = make_small_graph(a.gen_seed, a.num_nodes, a.max_depth, a.max_length)
    label = tok_ds[0].clone()
    batch = label.clone()
    batch[..., P.F_VALUE] = torch.where(batch[..., P.F_SYNTAX] == P.SYN_VAR,
                                        torch.full_like(batch[..., P.F_VALUE], P.EMPTY),
                                        batch[..., P.F_VALUE])
    depths = info[0, :, 0].clone()

    g = P.parse_graph(label.numpy(), label.numpy())
    dv = {v: int(depths[g['res_pos'][v]]) for v in g['parents']}
    dv.update({v: -1 for v in g['leaf_val']})
    val = P.evaluate(g['leaf_val'], g['parents'], dv)
    bad = sum(1 for v, p in g['res_pos'].items() if int(label[p, P.F_VALUE]) != val.get(v, -1))

    lit, tk2, dev = P.load_model(os.path.join(P.REPO, *P.WEIGHTS[a.seed_c]))
    rt = lit.model
    b, dp = batch.unsqueeze(0).to(dev), depths.unsqueeze(0).to(dev)
    res = (dp >= 0)

    pos_of = {v: g['res_pos'][v] for v in g['res_pos']}
    order = sorted(pos_of.values(), key=lambda p: (int(dp[0, p]), p))
    rows = []
    h = rt.token_embedder(b)
    L = min(int(dp.max().item()) + 1, 36)
    with torch.no_grad():
        for t in range(L):
            h_prev = h
            h, preds = rt(h, True)
            vp = preds[P.F_VALUE].argmax(-1)[0]
            for p in order:
                v = g['res_var_of'][p]
                d = int(dp[0, p])
                a_, b_ = h_prev[0, p], h[0, p]
                dd = b_ - a_
                an = float(a_.norm()); bn = float(b_.norm()); dn = float(dd.norm())
                par = float((dd * a_).sum() / max(an, 1e-12))
                perp = max(dn ** 2 - par ** 2, 0.0) ** 0.5
                ps = g['parents'].get(v, [])
                avail = sum(1 for q in ps if dv[q] == -1 or dv[q] <= t)
                rows.append(dict(t=t, var=v, pos=p, depth=d, reached=int(d <= t),
                                 parents_total=len(ps), parents_available=avail,
                                 read=int(vp[p]), label=int(label[p, P.F_VALUE]),
                                 one_minus_cos=1 - float(F.cosine_similarity(a_, b_, dim=0)),
                                 norm_ratio=bn / max(an, 1e-12), rel_l2=dn / max(an, 1e-12),
                                 radial=par / max(an, 1e-12), tangential=perp / max(an, 1e-12)))
    os.makedirs(os.path.join(P.REPO, a.out), exist_ok=True)
    csv_p = os.path.join(P.REPO, a.out, f'trace_{a.num_nodes}nodes_seed{a.seed_c}.csv')
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'图: {len(g["res_pos"])} 个结果位置，最大 depth {int(dp.max())}，标签复现失败 {bad}')
    print(f'轮数 {L}，记录 {len(rows)} 行 -> {os.path.relpath(csv_p, P.REPO)}')
    # 摘要：按是否已到轮次分组
    import statistics as st
    for label_, sel in (('未到轮次 (depth>t)', [r for r in rows if not r['reached']]),
                        ('本轮该算 (depth==t)', [r for r in rows if r['depth'] == r['t']])):
        if not sel: continue
        print(f"  {label_}: n={len(sel)}  1-cos={st.mean(r['one_minus_cos'] for r in sel):.4f}  "
              f"norm_ratio={st.mean(r['norm_ratio'] for r in sel):.4f}  rel_l2={st.mean(r['rel_l2'] for r in sel):.4f}")
    post = [r for r in rows if r['depth'] < r['t']]
    if post:
        print(f"  已算完 (depth<t): n={len(post)}  1-cos={st.mean(r['one_minus_cos'] for r in post):.4f}  "
              f"norm_ratio={st.mean(r['norm_ratio'] for r in post):.4f}  rel_l2={st.mean(r['rel_l2'] for r in post):.4f}")


if __name__ == '__main__':
    main()
