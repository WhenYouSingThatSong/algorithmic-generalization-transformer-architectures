#!/usr/bin/env python3
"""Shared helpers for the cross-round channel experiments on existing C weights.

Model loading, graph parsing, the forward loop with the freeze ablation, and the
formal metrics.  No training.  Design: see
 docs/records/2026-11-28/vector-difference-and-freeze-ablation-design.md
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
