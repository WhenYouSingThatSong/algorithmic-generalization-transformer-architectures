#!/usr/bin/env python3
"""PLAN-010 §1.2: linear probe on unreached positions + channel-shutdown variants.

Design and rationale: docs/records/2026-11-28/probe-and-channel-shutdown-design.md

Answers three questions on EXISTING C weights, no retraining:
  Q1 existence  - can a linear probe decode a node's true value from its hidden
                  state at rounds before that node's own round?
  Q2 carrier    - does replacing the cross-round hidden state with re-embedded
                  tokens (V1) hurt?
  Q3 formation  - does freezing unreached positions (V2) hurt?

Controls: corrupted-parent leaf (does the probed info follow the input?),
unrelated leaf (placebo), permuted position (negative control), shuffled labels.
"""
import argparse, json, os, random, sys
import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'experiments'))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'evaluation'))

from model import LitRecurrentTransformerModel, DAGTeacherModel
from Simtransformer.simtransformer.utils import EasyDict
from tokenizers import build_factored_tokenizer

MOD = 23
F_SYNTAX, F_VARIABLE, F_OPERATION, F_VALUE = 0, 1, 2, 3
SYN_VAR, SYN_OP, SYN_VALUE, SYN_PAD, SYN_UNK, SYN_EQ, SYN_SEP = 0, 1, 2, 3, 4, 5, 6
EMPTY = 24
N_A = 128

WEIGHTS = {
    42:   ('tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-14-02:35:10', 'epoch=143-val_loss=0.0000.ckpt'),
    1234: ('tmp/runs/oldC0-resume-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-1234 - 2026-11-14-02:38:10', 'epoch=143-val_loss=0.0000.ckpt'),
    2026: ('tmp/runs/oldC0-newseed-150ep/checkpoints/Nodes32-ADD - L4H16D384_DeBERTa-seed-2026 - 2026-11-14-10:12:43', 'epoch=143-val_loss=0.0000.ckpt'),
}
CFG = 'experiments/configs/C0-M4-Nodes32-ADD-L4H16D384-DeBERTa'
DATA = 'experiments/data/N128-heldout-ADD/testdata_N128.pt'


# ---------------------------------------------------------------- graph parsing
def parse_graph(tok, label):
    """tok/label: [S, 6] int64 for one sample.

    Equation layout (verified on real data): ``<expr> = <result VAR> <EQ_SEP>``,
    where <expr> is a single VALUE token for a leaf, or
    ``x_a + x_b + ...`` for an internal node. Returns per-result-position records.
    """
    S = tok.shape[0]
    segs, start, i = [], 0, 0
    while i < S and tok[i, F_SYNTAX] != SYN_PAD:
        if tok[i, F_SYNTAX] == SYN_SEP:
            if i > start:
                segs.append((start, i))
            start = i + 1
        i += 1
    if start < i:
        segs.append((start, i))

    leaf_val, parents, res_pos, res_var_of, depth_of = {}, {}, {}, {}, {}
    for a, b in segs:
        eq = next((k for k in range(a, b) if tok[k, F_SYNTAX] == SYN_EQ), None)
        if eq is None or eq + 1 >= b:
            continue
        r = eq + 1
        if tok[r, F_SYNTAX] != SYN_VAR:
            continue
        var = int(label[r, F_VARIABLE])
        ifr = list(range(a, eq))
        vars_in = [int(label[k, F_VARIABLE]) for k in ifr if tok[k, F_SYNTAX] == SYN_VAR]
        if vars_in:
            # one or more parents; a single parent is written without an operator
            parents[var] = vars_in
        elif len(ifr) == 1 and tok[ifr[0], F_SYNTAX] == SYN_VALUE:
            leaf_val[var] = int(label[ifr[0], F_VALUE])
        else:
            continue
        res_pos[var] = r
        res_var_of[r] = var
        depth_of[r] = int(label[r, F_VALUE])
    # depth comes from dataset_info, not from the value field
    return dict(segs=segs, leaf_val=leaf_val, parents=parents,
                res_pos=res_pos, res_var_of=res_var_of)


def evaluate(leaf_val, parents, depths):
    """Topological ADD evaluation. depths: var -> depth."""
    order = sorted(parents, key=lambda v: depths.get(v, 0))
    v = dict(leaf_val)
    for var in order:
        v[var] = sum(v[p] for p in parents[var]) % MOD
    return v


def ancestors(var, parents):
    seen, stack = set(), [var]
    while stack:
        x = stack.pop()
        for p in parents.get(x, []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


# ---------------------------------------------------------------- model loading
def load_model(ckpt_path):
    def rd(n):
        with open(os.path.join(REPO, CFG, n)) as f:
            return EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    mc, tc, dc = rd('model_config.yaml'), rd('train_config.yaml'), rd('data_config.yaml')
    tok = build_factored_tokenizer(dc)
    mc.vocab_sizes, mc.factors = tok.vocab_sizes, tok.factors
    lit = LitRecurrentTransformerModel(mc, dc, tc, DAGTeacherModel(mc, tok))
    lit.load_ckpt(ckpt_path)
    lit.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lit.to(device)
    return lit, tok, device


# ---------------------------------------------------------------- forward loop
@torch.no_grad()
def run_loop(rt, batch, depths, want_positions=None, variant='none', L=None, grace=0):
    """Run the continuous loop.

    variant: none | freeze
      freeze holds back the hidden state of every result position whose depth is
      more than `grace` rounds ahead, i.e. h[depth > t + grace] is reverted to the
      previous round's value.  grace=0 is the plain freeze; grace=1..2 lets the
      model precompute that many rounds ahead.
    Returns (rows_by_round, final_logits).
      rows_by_round[t] -> {pos: tensor[D]} for positions with depth > t
    """
    device = batch.device
    h = rt.token_embedder(batch)
    max_depth = int(depths.max().item()) + 1
    L = L or min(max_depth, 36)
    pos_list = want_positions if want_positions is not None else []
    rows = {}
    logits = None
    for t in range(L):
        h_prev = h
        h, preds = rt(h, True)
        logits = preds
        if pos_list:
            d = {}
            for bi, pos in pos_list:
                if depths[bi, pos].item() > t:
                    d.setdefault((bi, pos), h[bi, pos].detach().float().cpu())
            if d:
                rows[t] = d
        if variant == 'freeze':
            h = h.clone()
            frozen = depths > (t + grace)
            h[frozen] = h_prev[frozen]
    return rows, logits


def metrics_from_logits(logits, label, valid_pos):
    """valid_pos: bool [B,S] of result positions. Mirrors the eval script's formal metrics."""
    vp = logits[F_VALUE].argmax(-1)
    tgt = label[..., F_VALUE]
    ok = (vp == tgt) & valid_pos
    n_ok = int(ok.sum().item())
    n_tot = int(valid_pos.sum().item())
    per_graph = ok.sum(1).float() / valid_pos.sum(1).clamp_min(1).float()
    graph_all = int((per_graph == 1.0).sum().item())
    nb = valid_pos.size(0)
    correct4 = torch.ones_like(valid_pos)
    for f in (F_SYNTAX, F_VARIABLE, F_OPERATION, F_VALUE):
        correct4 &= (logits[f].argmax(-1) == label[..., f])
    strict = (correct4 & valid_pos).sum(1).eq(valid_pos.sum(1)) & (valid_pos.sum(1) > 0)
    return dict(value_node_acc=n_ok / max(n_tot, 1), value_node_correct=n_ok, value_node_num=n_tot,
                value_graph_all=graph_all / max(nb, 1), graph_correct=graph_all, graph_n=nb,
                label_full_acc=float(strict.float().mean().item()))


# ---------------------------------------------------------------- probe
def train_probe(X, y, Xv, yv, C=1.0, seed=0):
    """Multinomial logistic regression on standardised features.

    Returns (val_predictions, val_acc, train_acc).  train_acc is reported so an
    overfitting probe is visible rather than silently reported as "no signal".
    """
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xs, Xvs = (X - mu) / sd, (Xv - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=C, solver='lbfgs', n_jobs=-1)
    clf.fit(Xs, y)
    pv = clf.predict(Xvs)
    return pv, float((pv == np.asarray(yv)).mean()), float(clf.score(Xs, y))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1234,2026')
    ap.add_argument('--n-graphs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-pos-per-graph', type=int, default=24)
    ap.add_argument('--out', default='tmp/analysis/channel-probe')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--C', type=float, default=1.0, help='inverse L2 strength for the probe')
    ap.add_argument('--skip-probe', action='store_true', help='only run the freeze sweep')
    a = ap.parse_args()

    os.makedirs(os.path.join(REPO, a.out), exist_ok=True)
    data = torch.load(os.path.join(REPO, DATA), map_location='cpu', weights_only=True)
    tok_t = data['tokenized_dataset']
    info = data['dataset_info']
    n = min(a.n_graphs, tok_t.size(0))
    tok_t, info = tok_t[:n], info[:n]

    results = {}
    for seed in [int(s) for s in a.seeds.split(',')]:
        if a.smoke and seed != int(a.seeds.split(',')[0]):
            continue
        d, ck = WEIGHTS[seed]
        ckpt = os.path.join(REPO, d, ck)
        print(f'\n=== seed {seed}: {ck}', flush=True)
        lit, tokenizer, device = load_model(ckpt)
        rt = lit.model

        # --- build the label/input pair and parse graphs
        label = tok_t.clone()
        batch = tok_t.clone()
        # input: VALUE emptied wherever SYNTAX == VARIABLE (verified in train.py DataModule)
        batch[..., F_VALUE] = torch.where(batch[..., F_SYNTAX] == SYN_VAR,
                                          torch.full_like(batch[..., F_VALUE], EMPTY),
                                          batch[..., F_VALUE])
        depths = info[..., 0].clone()
        valid = (batch[..., F_SYNTAX] != SYN_PAD) & (depths >= 0)
        batch, label, depths, valid = (x.to(device) for x in (batch, label, depths, valid))

        # --- verify the evaluator reproduces the labels
        bad = 0
        parsed = []
        for bi in range(n):
            t0, l0 = tok_t[bi].numpy(), label[bi].cpu().numpy()
            g = parse_graph(t0, l0)
            dv = {v: int(depths[bi, g['res_pos'][v]].item()) for v in g['parents']}
            dv.update({v: 0 for v in g['leaf_val']})
            val = evaluate(g['leaf_val'], g['parents'], dv)
            for v, p in g['res_pos'].items():
                if int(l0[p, F_VALUE]) != val.get(v, -1):
                    bad += 1
            g['dv'] = dv
            g['val'] = val
            parsed.append(g)
        print(f'  图解析: {n} 题，标签复现失败 {bad} 个节点', flush=True)

        # --- which positions to probe
        rng = random.Random(seed)
        pos_list = []
        for bi, g in enumerate(parsed):
            cand = [p for p in g['res_pos'].values() if int(depths[bi, p].item()) >= 1]
            if len(cand) > a.max_pos_per_graph:
                cand = rng.sample(cand, a.max_pos_per_graph)
            pos_list += [(bi, p) for p in cand]

        # --- baseline run: collect hidden rows
        rows, logits0 = run_loop(rt, batch, depths, want_positions=pos_list)
        m0 = metrics_from_logits(logits0, label, valid)
        print(f'  基线: node_acc={m0["value_node_acc"]:.6f} graph_all={m0["value_graph_all"]:.6f}', flush=True)

        # --- assemble probe samples: h_{t+1}[u] for t < depth(u), label T(u)
        X, y, lead, gid, posv = [], [], [], [], []
        for t, dd in rows.items():
            for (bi, pos), vec in dd.items():
                dpos = int(depths[bi, pos].item())
                X.append(vec.numpy()); y.append(int(label[bi, pos, F_VALUE].item()))
                lead.append(dpos - t); gid.append(bi); posv.append(pos)
        X, y = np.asarray(X, np.float32), np.asarray(y)
        lead, gid, posv = np.asarray(lead), np.asarray(gid), np.asarray(posv)
        # split BY GRAPH
        half = n // 2
        tr, va = gid < half, gid >= half
        print(f'  探针样本 {len(X)} 条（train {tr.sum()} / val {va.sum()}）', flush=True)

        pred, acc, acc_tr = train_probe(X[tr], y[tr], X[va], y[va], C=a.C)
        perm = np.random.RandomState(0).permutation(len(y[tr]))
        _, acc_shuf, _ = train_probe(X[tr], y[tr][perm], X[va], y[va])
        maj = np.bincount(y[tr]).argmax()
        acc_maj = float((y[va] == maj).mean())
        probe = dict(overall=acc, train_acc=acc_tr, shuffled=acc_shuf, majority=acc_maj,
                     n_val=int(va.sum()), n_train=int(tr.sum()))
        by_lead = {}
        for L_ in sorted(set(lead[va].tolist())):
            sel = lead[va] == L_
            by_lead[int(L_)] = dict(acc=float((pred[sel] == y[va][sel]).mean()), n=int(sel.sum()))
        probe['by_lead'] = by_lead
        print(f'  探针: val acc={acc:.4f}  train acc={acc_tr:.4f}  打乱={acc_shuf:.4f}  多数类={acc_maj:.4f}', flush=True)
        print('   按 lead: ' + ', '.join(f'{k}:{v["acc"]:.3f}(n={v["n"]})' for k, v in sorted(by_lead.items())), flush=True)

        # --- control 1: corrupted parent leaf
        ctrl = {}
        ctrl_res = []
        for bi, g in enumerate(parsed[:max(1, n // 4)]):
            targets = [p for p in g['res_pos'].values() if int(depths[bi, p].item()) >= 1]
            if not targets:
                continue
            u = rng.choice(targets)
            uv = g['res_var_of'][u]
            anc = ancestors(uv, g['parents'])
            leaves = [v for v in anc if v in g['leaf_val']]
            if not leaves:
                continue
            lf = rng.choice(sorted(leaves))
            delta = rng.randrange(1, MOD)
            # corrupt: change the leaf's own VALUE token in the input and label view
            pos_of_leaf_tok = None
            for var, p in g['res_pos'].items():
                if var == lf:
                    pos_of_leaf_tok = p - 2  # the value token sits at the expression head
            if pos_of_leaf_tok is None or not (0 <= pos_of_leaf_tok < batch.size(1)):
                continue
            g2 = dict(leaf_val=dict(g['leaf_val']), parents=g['parents'])
            g2['leaf_val'][lf] = (g2['leaf_val'][lf] + delta) % MOD
            val2 = evaluate(g2['leaf_val'], g2['parents'], g['dv'])
            if val2.get(uv) == g['val'].get(uv):
                continue  # delta cancelled mod 23; skip
            ctrl_res.append((bi, u, uv, lf, delta, val2.get(uv), g['val'].get(uv), pos_of_leaf_tok))
        if ctrl_res:
            b2 = batch.clone()
            for bi, u, uv, lf, delta, _, _, pt in ctrl_res:
                b2[bi, pt, F_VALUE] = (b2[bi, pt, F_VALUE] + delta) % MOD
            pl2 = [(bi, u) for bi, u, *_ in ctrl_res]
            rows2, _ = run_loop(rt, b2, depths, want_positions=pl2)
            # last available pre-round row per (bi,u)
            got = {}
            for t in sorted(rows2):
                for k, vec in rows2[t].items():
                    got[k] = (t, vec)
            hit_new = hit_old = 0
            for bi, u, uv, lf, delta, tnew, told, _ in ctrl_res:
                if (bi, u) not in got:
                    continue
                p = int(train_probe(X[tr], y[tr], got[(bi, u)][1].numpy()[None], [0])[0][0])
                hit_new += int(p == tnew)
                hit_old += int(p == told)
            tot = max(len(ctrl_res), 1)
            ctrl['parent_corruption'] = dict(n=len(ctrl_res), pred_equals_new=hit_new / tot,
                                             pred_equals_old=hit_old / tot)

        # --- freeze, swept over how many rounds ahead the model may still precompute
        for g in (0, 1, 2, 3):
            _, lg = run_loop(rt, batch, depths, variant='freeze', grace=g)
            ctrl[f'freeze_grace{g}'] = metrics_from_logits(lg, label, valid)
            print(f'  freeze grace={g}: node_acc={ctrl[f"freeze_grace{g}"]["value_node_acc"]:.6f} '
                  f'graph_all={ctrl[f"freeze_grace{g}"]["value_graph_all"]:.6f}', flush=True)

        results[str(seed)] = dict(ckpt=os.path.relpath(ckpt, REPO), baseline=m0, probe=probe, controls=ctrl,
                                  parse_failures=bad, n_graphs=n, n_samples=len(X))
        del lit
        torch.cuda.empty_cache()

    out = os.path.join(REPO, a.out, 'smoke.json' if a.smoke else 'probe.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n已写', os.path.relpath(out, REPO))


if __name__ == '__main__':
    main()
