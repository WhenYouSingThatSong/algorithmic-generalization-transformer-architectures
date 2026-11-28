#!/usr/bin/env python3
"""Freeze the unreached positions, sweeping how many rounds ahead may still update.

grace=0 freezes every result position with depth > t (the plain freeze).
grace=g lets positions up to g rounds ahead keep updating, i.e. freezes
depth > t + g.  If accuracy recovers at g=1 or g=2, the precomputation the model
relies on reaches that many rounds ahead.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1234,2026')
    ap.add_argument('--n-graphs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--graces', default='0,1,2,3,4')
    ap.add_argument('--out', default='tmp/analysis/channel-probe/freeze_sweep.json')
    a = ap.parse_args()

    data = torch.load(os.path.join(P.REPO, P.DATA), map_location='cpu', weights_only=True)
    tok_t, info = data['tokenized_dataset'][:a.n_graphs], data['dataset_info'][:a.n_graphs]
    label = tok_t.clone()
    batch = tok_t.clone()
    batch[..., P.F_VALUE] = torch.where(batch[..., P.F_SYNTAX] == P.SYN_VAR,
                                        torch.full_like(batch[..., P.F_VALUE], P.EMPTY),
                                        batch[..., P.F_VALUE])
    depths = info[..., 0].clone()
    valid = (batch[..., P.F_SYNTAX] != P.SYN_PAD) & (depths >= 0)

    out = {}
    for seed in [int(s) for s in a.seeds.split(',')]:
        d, ck = P.WEIGHTS[seed]
        lit, tk, dev = P.load_model(os.path.join(P.REPO, d, ck))
        rt = lit.model
        b, l, dp, v = (x.to(dev) for x in (batch, label, depths, valid))
        row = {}
        _, lg = P.run_loop(rt, b, dp)
        row['baseline'] = P.metrics_from_logits(lg, l, v)
        print(f"seed {seed} baseline: node={row['baseline']['value_node_acc']:.6f} "
              f"graph_all={row['baseline']['value_graph_all']:.6f}", flush=True)
        for g in [int(x) for x in a.graces.split(',')]:
            _, lg = P.run_loop(rt, b, dp, variant='freeze', grace=g)
            m = P.metrics_from_logits(lg, l, v)
            row[f'grace{g}'] = m
            print(f"  grace={g}: node={m['value_node_acc']:.6f} graph_all={m['value_graph_all']:.6f}",
                  flush=True)
        out[str(seed)] = row
        del lit
        torch.cuda.empty_cache()

    p = os.path.join(P.REPO, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('已写', os.path.relpath(p, P.REPO))


if __name__ == '__main__':
    main()
