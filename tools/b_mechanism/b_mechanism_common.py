#!/usr/bin/env python3
"""Shared utilities for the B mechanism experiments.

Design: tmp/records/2026-11-14/b-mechanism-design.md

This module is deliberately model-agnostic between A' and B; both are discrete
recurrent models.  It does not train anything.
"""
from __future__ import annotations

import copy
import hashlib
import os
import sys
from collections import defaultdict, deque

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simtransformer.simtransformer.utils import EasyDict  # noqa: E402
from model import LitRecurrentTransformerModel, DAGTeacherModel, INVALID_NUM  # noqa: E402
from train import DAGDataModule  # noqa: E402
from tokenizers import build_factored_tokenizer  # noqa: E402
import tokenizers as _tok  # noqa: E402

sys.modules.setdefault('factored_tokenizers', _tok)

MOD = None
V = None
S = None
VAR = None
OP = None
RG = None
MD = None


def cfg_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_configs(config_dir_name):
    cdir = os.path.join(ROOT, 'experiments', 'configs', config_dir_name)
    with open(os.path.join(cdir, 'model_config.yaml')) as f:
        mc = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    with open(os.path.join(cdir, 'train_config.yaml')) as f:
        tc = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    with open(os.path.join(cdir, 'data_config.yaml')) as f:
        dc = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return mc, tc, dc


def load_eval_context(config_dir_name, group_name, run_name, ckpt_base_dir,
                      ckpt_name, ds_file, batch_size=8, num_workers=0):
    """Load one discrete model + tokenizer + data module."""
    global MOD, V, S, VAR, OP, RG, MD
    mc, tc, dc = load_configs(config_dir_name)
    tok = build_factored_tokenizer(dc)
    mc.vocab_sizes = tok.vocab_sizes
    mc.factors = tok.factors
    MOD = tok.mod_val
    V = tok.factors.index('VALUE')
    S = tok.factors.index('SYNTAX')
    VAR = tok.factors.index('VARIABLE')
    OP = tok.factors.index('OPERATION')
    RG = tok.factors.index('REGRET')
    MD = tok.factors.index('MODIFICATION')

    teacher = DAGTeacherModel(mc, tok)
    lit = LitRecurrentTransformerModel(mc, dc, tc, teacher)
    ckpt = os.path.join(ckpt_base_dir, f'{group_name}-{run_name}', ckpt_name)
    lit.load_ckpt(ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit = lit.to(device).eval()

    dm = DAGDataModule(data_path=[ds_file, ds_file], batch_size=batch_size,
                       num_workers=num_workers, factored_tokenizer=tok)
    dm.setup()
    mc.data_dim = dm.data_dim
    mc.label_dim = dm.label_dim
    mc.info_dim = dm.info_dim

    return dict(lit=lit, tok=tok, teacher=teacher, mc=mc, tc=tc, dc=dc, dm=dm,
                device=device, ckpt=ckpt)


def split_batch(batch_tensor, mc):
    b = batch_tensor[..., :mc.data_dim]
    label = batch_tensor[..., mc.data_dim:mc.data_dim + mc.label_dim]
    info = batch_tensor[..., mc.data_dim + mc.label_dim:]
    return b, label, info


def tok_str(tok, x):
    return tok.value_idx2tok[int(x)]


def is_number_token(tok, x):
    return tok_str(tok, x).isdigit()


def parse_expr(info_b, q):
    """Return (parents, ops) from the info factors for node q.

    info_b is [T, 6]; columns 1: are alternating parent position / operator.
    """
    e = info_b[q, 1:].tolist()
    parents = [int(x) for x in e[::2] if x != INVALID_NUM]
    ops = [int(x) for x in e[1::2] if x != INVALID_NUM]
    return parents, ops


def apply_expr(info_b, q, vals, tok, teacher):
    """Evaluate node q's expression with values dict {parent_pos: value}."""
    parents, ops = parse_expr(info_b, q)
    if len(ops) != max(0, len(parents) - 1):
        return None
    vv = []
    for pos in parents:
        if pos not in vals or vals[pos] is None:
            return None
        vv.append(vals[pos])
    if not vv:
        return None
    val = vv[0]
    for i, op in enumerate(ops):
        name = tok.operation_idx2tok[int(op)]
        if name == 'N/A':
            continue
        val = teacher.apply_operation(name, val, vv[i + 1])
    return val % MOD


def force_aux_false(preds, teacher):
    preds[..., RG] = teacher.regret_tok2idx['FALSE']
    preds[..., MD] = teacher.modification_tok2idx['FALSE']
    return preds


def rollout(lit, teacher, start, n_rounds, input_hook=None):
    """Free discrete rollout from a token state.

    Returns:
      inputs[t]  = state fed to forward t
      outputs[t] = model argmax output after forward t
    """
    cur = start.clone()
    inputs, outputs = [], []
    with torch.no_grad():
        for t in range(n_rounds):
            if input_hook is not None:
                cur = input_hook(t, cur)
            inputs.append(cur.detach().clone())
            logits = lit.predict_step(cur)
            if isinstance(logits, tuple):
                logits = logits[1]
            preds = torch.stack([l.argmax(-1) for l in logits], dim=-1)
            force_aux_false(preds, teacher)
            outputs.append(preds.detach().clone())
            cur = preds
    return inputs, outputs


def numeric_mask(data_b, label_b, info_b, tok):
    """Non-PAD DAG nodes whose final VALUE label is numeric.

    Constant tokens (depth=-1) are excluded; leaf variable nodes (depth=0) and
    internal nodes (depth>=1) are included.
    """
    valid = data_b[:, S] != tok.syntax_tok2idx['<PAD>']
    depth = info_b[:, 0].long()
    numeric = valid & (depth >= 0) & (label_b[:, V] < tok.mod_val)
    return numeric, valid, depth


def all_numeric_correct(preds, label_b, numeric):
    if numeric.sum() == 0:
        return False
    return bool((preds[..., V][numeric] == label_b[..., V][numeric]).all().item())


def build_meta(data_b, label_b, info_b, tok, teacher):
    """Per-graph structural metadata used by both selection and runners."""
    numeric, valid, depth = numeric_mask(data_b, label_b, info_b, tok)
    T = data_b.size(0)
    children = defaultdict(list)
    parents = {}
    for q in range(T):
        if not bool(numeric[q]):
            continue
        ps, _ = parse_expr(info_b, q)
        parents[q] = ps
        for p in ps:
            if 0 <= p < T and bool(numeric[p]):
                children[p].append(q)
    nodes = [i for i in range(T) if bool(numeric[i])]
    sinks = [i for i in nodes if not children.get(i)]
    leaves = []
    for i in nodes:
        if int(depth[i]) != 0:
            continue
        ps = parents.get(i, [])
        cpos = None
        for p in ps:
            if 0 <= p < T and int(depth[p]) < 0 and tok.syntax_idx2tok[int(data_b[p, S])] == 'VALUE':
                cpos = p
                break
        if cpos is not None:
            leaves.append((i, cpos))
    return dict(numeric=numeric, valid=valid, depth=depth, children=dict(children),
                parents=parents, nodes=nodes, sinks=sinks, leaves=leaves)


def descendants(children, start):
    """All descendants of a node, breadth-first, with edge distance."""
    out = {}
    q = deque([(start, 0)])
    seen = {start}
    while q:
        x, d = q.popleft()
        for y in children.get(x, []):
            if y in seen:
                continue
            seen.add(y)
            out[y] = d + 1
            q.append((y, d + 1))
    return out


def ancestors(parents, start):
    out = {}
    q = deque([(start, 0)])
    seen = {start}
    while q:
        x, d = q.popleft()
        for y in parents.get(x, []):
            if y in seen:
                continue
            seen.add(y)
            out[y] = d + 1
            q.append((y, d + 1))
    return out


def pick_chain(meta, token_data, label_b, tok, k=4, require_q2=True, allow_multi=False):
    """Pick internal u candidates with one numeric parent r and numeric child q.

    Returns records with r/u/q1/q2/path flags.  Selection is independent of
    outcomes; q1 is the deepest/shortest available child and q2 is a child of q1.
    """
    nodes = [i for i in meta['nodes'] if int(meta['depth'][i]) >= 1]
    candidates = []
    for u in nodes:
        ps = [p for p in meta['parents'].get(u, []) if p in meta['nodes']]
        qs = [q for q in meta['children'].get(u, []) if q in meta['nodes']]
        if not ps or not qs:
            continue
        if not allow_multi and len(ps) != 1:
            continue
        # deepest parent as r
        r = max(ps, key=lambda p: (int(meta['depth'][p]), -p))
        r_true = int(label_b[r, V])
        a = int(label_b[u, V])
        q1 = max(qs, key=lambda q: (int(meta['depth'][q]), -q))
        q1_uses_r = r in meta['parents'].get(q1, [])
        q2s = [q for q in meta['children'].get(q1, []) if q in meta['nodes']]
        q2 = max(q2s, key=lambda q: (int(meta['depth'][q]), -q)) if q2s else None
        if require_q2 and q2 is None:
            continue
        q2_uses_r = q2 is not None and r in meta['parents'].get(q2, [])
        candidates.append(dict(pos=u, d=int(meta['depth'][u]), a=a,
                               r_pos=r, r_true=r_true,
                               q1=q1, q2=q2,
                               u_multi=len(ps) > 1,
                               q1_uses_r=q1_uses_r,
                               q2_uses_r=q2_uses_r,
                               primary=(not allow_multi or len(ps) == 1),
                               group='single_parent' if len(ps) == 1 else 'multi_parent'))
    # Never mix a fallback multi-parent pool into the formal primary group.
    pool = [c for c in candidates if c['primary'] and not c['q1_uses_r'] and not c['q2_uses_r']]
    # deterministic depth-stratified sample
    pool = sorted(pool, key=lambda c: (c['d'], c['pos']))
    if k and len(pool) > k:
        import numpy as np
        idxs = np.array_split(np.arange(len(pool)), k)
        pool = [pool[int(ix[0])] for ix in idxs]
    return pool


def factor_strings(row, tok, factors=('SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE')):
    out = {}
    for name in factors:
        idx = tok.factors.index(name)
        mapping = getattr(tok, name.lower() + '_idx2tok', None)
        if mapping is None:
            out[name] = str(int(row[idx]))
        else:
            out[name] = mapping.get(int(row[idx]), str(int(row[idx])))
    return out


def node_state(state, label_b, tok, positions):
    """First-four prediction/truth for tracked nodes (batch 0)."""
    out = {}
    for p in positions:
        pred4 = factor_strings(state[0, p], tok)
        true4 = factor_strings(label_b[p], tok)
        out[str(p)] = dict(pos=int(p), pred4=pred4, true4=true4,
                           first4_correct=pred4 == true4,
                           value_correct=pred4['VALUE'] == true4['VALUE'])
    return out


def graph_records(meta, label_b, tok):
    """JSON-serializable true values for tracked positions."""
    out = {}
    for p in meta['nodes']:
        out[p] = dict(pos=p, depth=int(meta['depth'][p]),
                      true=tok_str(tok, int(label_b[p, V])),
                      true4=factor_strings(label_b[p], tok),
                      parents=[x for x in meta['parents'].get(p, []) if x in meta['nodes']],
                      children=list(meta['children'].get(p, [])),
                      n_children=len(meta['children'].get(p, [])))
    return out


def checkpoint_path(base, group, run, ckpt):return os.path.join(base, f'{group}-{run}', ckpt)


def verify_manifest_identity(manifest, ds_file, ckpt_file):
    ds_sha = manifest.get('ds_sha256')
    if ds_sha and sha256_file(ds_file) != ds_sha:
        raise SystemExit('data SHA256 does not match manifest')
    wanted = []
    for k in ('model_a', 'model_b'):
        v = manifest.get(k, {})
        if v.get('ckpt_sha256'):
            wanted.append(v['ckpt_sha256'])
    got = sha256_file(ckpt_file)
    if wanted and got not in wanted:
        raise SystemExit('checkpoint SHA256 does not match manifest')
    return True


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()