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



def apply_prefill(batch, label, g, dv, val, mode, ratio, rng):
    """把结果格的 VALUE 预填成指定数值（实验 073 / 074 的输入构造）。

    mode: none | correct | random | all_wrong
      correct    填真值
      random     均匀随机（可能碰巧正确）——073
      all_wrong  既不等于真值、也不等于父值当前推出的数——074
    按拓扑顺序处理，父值使用已经确定下来的输入值。
    """
    if mode == 'none':
        return {}, batch
    import random as _r
    order = sorted(g['res_pos'], key=lambda v: dv[v])
    avail = dict(g['leaf_val'])
    chosen = {}
    for v in order:
        p = g['res_pos'][v]
        if mode == 'correct':
            x = val[v]
        elif mode == 'random':
            x = _r.randrange(0, 23)
        else:  # all_wrong
            ps = g['parents'].get(v, [])
            cand = sum(avail[q] for q in ps) % 23 if ps and all(q in avail for q in ps) else None
            forbidden = {val[v]} | ({cand} if cand is not None else set())
            pool = [z for z in range(23) if z not in forbidden]
            x = rng.choice(pool)
        if mode == 'random' and rng.random() >= ratio:
            avail[v] = val[v]
            continue
        batch[p, P.F_VALUE] = x
        chosen[int(p)] = x
        avail[v] = x
    return chosen, batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-c', type=int, default=42, help='用哪个 C 权重')
    ap.add_argument('--num-nodes', type=int, default=16)
    ap.add_argument('--max-depth', type=int, default=4)
    ap.add_argument('--gen-seed', type=int, default=1)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--source', choices=['generate', 'train'], default='generate',
                    help='train = 取训练分布内的样本；generate = 现场生成')
    ap.add_argument('--intervention', choices=['none', 'correct', 'random', 'all_wrong'],
                    default='none', help='第一次前向之前如何预填结果格')
    ap.add_argument('--prefill-ratio', type=float, default=0.5, help='random 模式下预填的比例')
    ap.add_argument('--save-vectors', action='store_true',
                    help='额外保存原始隐藏向量（float16 npz）')
    ap.add_argument('--out', default='tmp/analysis/channel/trace')
    a = ap.parse_args()

    if a.source == 'train':
        # 训练分布内的图：32 节点，取第一张最大深度等于 --max-depth 的样本
        tr = torch.load(os.path.join(P.REPO, 'experiments/data/Tr32Test128-ADD/train_data.pt'),
                        map_location='cpu', weights_only=True)
        info_all = tr['dataset_info'][:, :, 0]
        mx = info_all.max(dim=1).values
        idx = next(i for i in range(len(mx)) if int(mx[i]) == a.max_depth)
        tok_ds = tr['tokenized_dataset'][idx:idx + 1]
        info = tr['dataset_info'][idx:idx + 1]
        print(f'取训练集第 {idx} 题（最大深度 {int(mx[idx])}）')
    else:
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

    a.tag = (f'train{a.max_depth}' if a.source == 'train' else f'{a.num_nodes}nodes')
    import random as _rr
    rng = _rr.Random(a.gen_seed)
    chosen, batch = apply_prefill(batch, label, g, dv, val, a.intervention, a.prefill_ratio, rng)
    print(f'干预 {a.intervention}：预填 {len(chosen)} 个结果格')

    lit, tk2, dev = P.load_model(os.path.join(P.REPO, *P.WEIGHTS[a.seed_c]))
    rt = lit.model
    b, dp = batch.unsqueeze(0).to(dev), depths.unsqueeze(0).to(dev)
    res = (dp >= 0)

    pos_of = {v: g['res_pos'][v] for v in g['res_pos']}
    order = sorted(pos_of.values(), key=lambda p: (int(dp[0, p]), p))
    rows = []
    vec_before, vec_after, vec_read = [], [], []
    h = rt.token_embedder(b)
    L = min(int(dp.max().item()) + 1, 36)
    FN = ['SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE']
    with torch.no_grad():
        for t in range(L):
            h_prev = h
            h, preds = rt(h, True)
            dec = [preds[f].argmax(-1)[0] for f in range(4)]
            vec_before.append(h_prev[0, order].detach().to(torch.float16).cpu().numpy())
            vec_after.append(h[0, order].detach().to(torch.float16).cpu().numpy())
            vec_read.append(np.stack([dec[f][order].detach().cpu().numpy() for f in range(4)], -1))
            for p in order:
                v = g['res_var_of'][p]
                d = int(dp[0, p])
                a_, b_ = h_prev[0, p], h[0, p]
                dd = b_ - a_
                an, bn, dn = float(a_.norm()), float(b_.norm()), float(dd.norm())
                par = float((dd * a_).sum() / max(an, 1e-12))
                perp = max(dn ** 2 - par ** 2, 0.0) ** 0.5
                ps = g['parents'].get(v, [])
                avail = [q for q in ps if dv[q] == -1 or dv[q] <= t]
                row = dict(t=t, var=v, pos=p, depth=d,
                           phase=('pre' if d > t else ('at' if d == t else 'post')),
                           prefilled=int(p in chosen),
                           prefill_value=int(chosen.get(p, -1)),
                           parents_total=len(ps), parents_available=len(avail),
                           parents_desc=';'.join(f'{q}@d{dv[q]}' for q in ps),
                           parents_readable=';'.join(f'{q}={int(dec[3][g["res_pos"][q]] if q in g["res_pos"] else -1)}' for q in ps),
                           label_syntax=int(label[p, 0]), label_variable=int(label[p, 1]),
                           label_operation=int(label[p, 2]), label_value=int(label[p, 3]),
                           read_syntax=int(dec[0][p]), read_variable=int(dec[1][p]),
                           read_operation=int(dec[2][p]), read_value=int(dec[3][p]),
                           value_correct=int(int(dec[3][p]) == int(label[p, 3])),
                           norm_before=an, norm_after=bn,
                           one_minus_cos=1 - float(F.cosine_similarity(a_, b_, dim=0)),
                           norm_ratio=bn / max(an, 1e-12), rel_l2=dn / max(an, 1e-12),
                           radial=par / max(an, 1e-12), tangential=perp / max(an, 1e-12))
                for i, nm in enumerate(FN):
                    row[f'{nm.lower()}_correct'] = int(int(dec[i][p]) == int(label[p, i]))
                rows.append(row)
    os.makedirs(os.path.join(P.REPO, a.out), exist_ok=True)
    csv_p = os.path.join(P.REPO, a.out, f'trace_{a.tag}_{a.intervention}_seed{a.seed_c}.csv')
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if a.save_vectors:
        npz_p = os.path.join(P.REPO, a.out, f'vec_{a.tag}_{a.intervention}_seed{a.seed_c}.npz')
        np.savez_compressed(
            npz_p,
            h_before=np.stack(vec_before),      # [L, P, D] float16，本轮前
            h_after=np.stack(vec_after),        # [L, P, D] float16，本轮后
            readouts=np.stack(vec_read),        # [L, P, 4] int16，四字段读出
            positions=np.array(order),
            vars=np.array([g['res_var_of'][p] for p in order]),
            depths=np.array([int(dp[0, p]) for p in order]),
            labels=np.array([int(label[p, P.F_VALUE]) for p in order]))
        print(f'向量已存 -> {os.path.relpath(npz_p, P.REPO)}  '
              f'shape [L,P,D]={np.stack(vec_before).shape}')
    print(f'图: {len(g["res_pos"])} 个结果位置，最大 depth {int(dp.max())}，标签复现失败 {bad}')
    print(f'轮数 {L}，记录 {len(rows)} 行 -> {os.path.relpath(csv_p, P.REPO)}')
    # 摘要：按是否已到轮次分组
    import statistics as st
    for ph, nm in (('pre', '未到轮次'), ('at', '本轮该算'), ('post', '已算完')):
        sel = [r for r in rows if r['phase'] == ph]
        if not sel: continue
        print(f"  {nm}: n={len(sel)}  1-cos={st.mean(r['one_minus_cos'] for r in sel):.4f}  "
              f"norm_ratio={st.mean(r['norm_ratio'] for r in sel):.4f}  rel_l2={st.mean(r['rel_l2'] for r in sel):.4f}  "
              f"VALUE 读对={sum(r['value_correct'] for r in sel)}/{len(sel)}")


if __name__ == '__main__':
    main()
