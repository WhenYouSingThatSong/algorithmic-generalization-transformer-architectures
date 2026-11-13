# -*- coding: utf-8 -*-
"""
Parent-corruption 2x2（离散）：
  clean / U only / R only / U+R
用来区分 B 的恢复是“从父节点重算 (recompute)”还是“保留/改回已存值 (recorrect)”。

同时支持两种时机：
  after_u  : 从 S[d+1] 起跑（u 已经算好，4->5 / a->a+1）
  before_u : 从 S[d]   起跑（u 还没算，EMPTY->5 / EMPTY->a+1）

对每个 u：
- 选一个直接父节点 r（depth < d，label 为数字），把 r 改成 r_true + delta (mod 23)；
- 用 r 的错值重算 u，得到第三个可追踪值 a_recomp（默认 a+3）；
- 四种条件一次 rollout（batch 内复制 4 份）：
    clean: u 原值/EMPTY，r 原值
    u    : u 注入 a_inj，r 原值
    r    : u 原值/EMPTY，r 错值
    ur   : u 注入 a_inj，r 错值
- 记录注错后第 1/2/4 个 forward 的 u、r、q 读出；
- 主判读：R only 下 u 是否变成 a_recomp；U+R 下 u 是 a / a_inj / a_recomp。

运行目录：experiments/evaluation
"""
import argparse, os, sys, copy, json
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
import torch, yaml

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import numpy as np
from collections import defaultdict, Counter

from Simtransformer.simtransformer.utils import EasyDict
from model import LitRecurrentTransformerModel, DAGTeacherModel, INVALID_NUM
from train import DAGDataModule
from tokenizers import FactoredVocabTokenizer
import tokenizers as _tok
sys.modules.setdefault('factored_tokenizers', _tok)

p = argparse.ArgumentParser()
p.add_argument('--config_dir_name', required=True)
p.add_argument('--group_name', required=True)
p.add_argument('--run_name', required=True)
p.add_argument('--ckpt_base_dir', required=True)
p.add_argument('--ckpt_name', default='last.ckpt')
p.add_argument('--ds_file', required=True)
p.add_argument('--n_graphs', type=int, default=256)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--k', type=int, default=4, help='0 = all eligible u; else depth-stratified k')
p.add_argument('--start', choices=['teacher', 'own'], default='teacher')
p.add_argument('--timing', choices=['after_u', 'before_u'], default='after_u')
p.add_argument('--value_mode', choices=['a_plus_1', 'four_to_five'], default='a_plus_1')
p.add_argument('--delta', type=int, default=3, help='parent corruption delta; default gives a_recomp=a+3')
p.add_argument('--extra_steps', type=int, default=3)
p.add_argument('--tag', required=True)
p.add_argument('--out_dir', default='../../tmp/analysis/parent-2x2')
a = p.parse_args()

root = os.path.abspath('../..')
def load(name):
    with open(os.path.join(root, 'experiments/configs', a.config_dir_name, name)) as f:
        return EasyDict(yaml.load(f, Loader=yaml.FullLoader))
mc, tc, dc = load('model_config.yaml'), load('train_config.yaml'), load('data_config.yaml')
n_vars = getattr(dc, 'tokenizer_n_vars', None)
if n_vars is None:
    n_vars = max(dc.dag_config.num_nodes, dc.val_dag_config.num_nodes)
tok = FactoredVocabTokenizer(
    n_vars=n_vars,
    ops=dc.dag_config.func_vocab, mod_val=dc.dag_config.mod_val,
    max_fan_in=dc.dag_config.max_fan_in_deg)
mc.vocab_sizes = tok.vocab_sizes; mc.factors = tok.factors

ckpt = os.path.join(a.ckpt_base_dir, f'{a.group_name}-{a.run_name}', a.ckpt_name)
teacher = DAGTeacherModel(mc, tok)
lit = LitRecurrentTransformerModel(mc, dc, tc, teacher)
lit.load_ckpt(ckpt)
lit = lit.to('cuda').eval()

dm = DAGDataModule(data_path=[a.ds_file, a.ds_file], batch_size=a.batch_size,
                   num_workers=0, factored_tokenizer=tok)
dm.setup()
mc.data_dim, mc.label_dim, mc.info_dim = dm.data_dim, dm.label_dim, dm.info_dim

V = tok.factors.index('VALUE')
Rg = tok.factors.index('REGRET'); Md = tok.factors.index('MODIFICATION')
r_false = teacher.regret_tok2idx['FALSE']; m_false = teacher.modification_tok2idx['FALSE']
MOD = tok.mod_val
CONDS = ['clean', 'u', 'r', 'ur']

def tok_str(x): return tok.value_idx2tok[int(x)]
def is_num(x): return tok_str(x).isdigit()
def parse_expr(info_b, q):
    e = info_b[q, 1:].tolist()
    parents = [x for x in e[::2] if x != INVALID_NUM]
    ops = [x for x in e[1::2] if x != INVALID_NUM]
    return parents, ops
def apply_expr(info_b, q, vals):
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

def build_candidates(data, label, info, depths, own_preds=None):
    B, T = data.shape[:2]
    out = []
    for bi in range(B):
        info_b = info[bi].cpu(); label_b = label[bi].cpu(); dep_b = depths[bi].cpu()
        children = defaultdict(list)
        for q in range(T):
            if int(dep_b[q]) < 0 or not is_num(label_b[q, V]):
                continue
            parents, _ = parse_expr(info_b, q)
            for pp in parents:
                children[pp].append(q)
        us = []
        for pos in range(T):
            d = int(dep_b[pos])
            if d < 1 or not is_num(label_b[pos, V]):
                continue
            parents, _ = parse_expr(info_b, pos)
            if not parents:
                continue
            qs = [q for q in children.get(pos, []) if is_num(label_b[q, V])]
            if not qs:
                continue
            a_val = int(label_b[pos, V])
            if a.value_mode == 'four_to_five':
                if a_val != 4:
                    continue
                a_inj = 5
            else:
                a_inj = (a_val + 1) % MOD
            # pick parent r: deepest valid parent, tie smallest pos
            rcands = []
            for pp in parents:
                dr = int(dep_b[pp])
                if dr < 0 or dr >= d or not is_num(label_b[pp, V]):
                    continue
                r_true = int(label_b[pp, V])
                r_wrong = (r_true + a.delta) % MOD
                if r_wrong == r_true:
                    continue
                vals = {}
                ok = True
                for ppp in parents:
                    if ppp == pp:
                        vals[ppp] = r_wrong
                    elif is_num(label_b[ppp, V]):
                        vals[ppp] = int(label_b[ppp, V])
                    else:
                        ok = False; break
                if not ok:
                    continue
                a_recomp = apply_expr(info_b, pos, vals)
                if a_recomp is None or a_recomp == a_val or a_recomp == a_inj:
                    continue
                rcands.append(dict(r_pos=pp, r_depth=dr, r_true=r_true,
                                   r_wrong=r_wrong, a_recomp=a_recomp))
            if not rcands:
                continue
            r = sorted(rcands, key=lambda x: (-x['r_depth'], x['r_pos']))[0]
            # own-state：只保留“模型自己的 clean 轨迹里 u 已算对/还是 EMPTY，且 r 已算对”的 u
            if own_preds is not None:
                if a.timing == 'after_u':
                    base = own_preds[d, bi]
                    if int(base[pos, V]) != a_val:
                        continue
                else:
                    if d == 0:
                        continue
                    base = own_preds[d - 1, bi]
                    if tok_str(base[pos, V]) not in ('EMPTY', 'N/A'):
                        continue
                if int(base[r['r_pos'], V]) != r['r_true']:
                    continue
            us.append(dict(pos=pos, d=d, parents=parents, qs=qs,
                           a=a_val, a_inj=a_inj, **r))
        if a.k == 0 or len(us) <= a.k:
            sel = us
        else:
            us_sorted = sorted(us, key=lambda x: (x['d'], x['pos']))
            chunks = np.array_split(np.arange(len(us_sorted)), a.k)
            sel = [us_sorted[int(ch[0])] for ch in chunks]
        out.append(dict(b=bi, us=sel))
    return out

def q_expect(info_b, q, u_pos, u_val, r_pos, r_val, label_b):
    vals = {}
    for pp in parse_expr(info_b, q)[0]:
        if pp == u_pos:
            vals[pp] = u_val
        elif pp == r_pos:
            vals[pp] = r_val
        elif is_num(label_b[pp, V]):
            vals[pp] = int(label_b[pp, V])
        else:
            vals[pp] = None
    return apply_expr(info_b, q, vals)

all_records = []
done = 0; batch_idx = 0
with torch.no_grad():
    for batch in dm.val_dataloader():
        if done >= a.n_graphs:
            break
        b = lit.reduce_batch(copy.deepcopy(batch)).to('cuda')
        data = b[..., :mc.data_dim]
        label = b[..., mc.data_dim:mc.data_dim + mc.label_dim]
        info = b[..., mc.data_dim + mc.label_dim:]
        depths = info[..., 0].long()
        B, T = data.shape[:2]
        D = int(depths.max()) + 1
        n_loops = D + a.extra_steps
        if a.start == 'teacher':
            S, _, _ = lit.get_teacher_batch_with_depth(data, label, depths)
            own_preds = None
        else:
            # 模型自己的 clean 自由续算，作为 own-state 的基座
            cur_clean = data.clone()
            own = []
            with torch.no_grad():
                for loop in range(n_loops):
                    logits = lit.predict_step(cur_clean)
                    preds_clean = torch.stack([l.argmax(-1) for l in logits], dim=-1)
                    preds_clean[..., Rg] = r_false; preds_clean[..., Md] = m_false
                    own.append(preds_clean)
                    cur_clean = preds_clean
            own_preds = torch.stack(own, dim=0)  # [L,B,T,F]
            S = None
        cands = build_candidates(data, label, info, depths, own_preds=own_preds)
        max_k = max((len(c['us']) for c in cands), default=0)

        for k in range(max_k):
            items = [(c, c['us'][k]) for c in cands if k < len(c['us'])]
            if not items:
                continue
            # base index: after_u -> S[d+1] (u computed); before_u -> S[d] (u EMPTY)
            start_idx = (lambda u: u['d'] + 1) if a.timing == 'after_u' else (lambda u: u['d'])
            start_loop = (lambda u: u['d'] + 1) if a.timing == 'after_u' else (lambda u: u['d'])
            states = []; meta = []
            for c, u in items:
                if a.start == 'teacher':
                    base = S[start_idx(u), c['b']].clone()
                else:
                    idx = u['d'] if a.timing == 'after_u' else u['d'] - 1
                    base = own_preds[idx, c['b']].clone()
                bi = c['b']
                for cond in CONDS:
                    st = base.clone()
                    if cond in ('u', 'ur'):
                        st[u['pos'], V] = tok.value_tok2idx[str(u['a_inj'])]
                    if cond in ('r', 'ur'):
                        st[u['r_pos'], V] = tok.value_tok2idx[str(u['r_wrong'])]
                    st[Rg] = r_false; st[Md] = m_false
                    states.append(st)
                    meta.append((c, u, cond))
            cur = torch.stack(states).to('cuda')
            n_steps = n_loops - max(start_loop(u) for _, u, _ in meta)
            # record target loops per item and condition
            for step in range(n_steps + 1):
                logits = lit.predict_step(cur)
                preds = torch.stack([l.argmax(-1) for l in logits], dim=-1)
                preds[..., Rg] = r_false; preds[..., Md] = m_false
                for si, (c, u, cond) in enumerate(meta):
                    abs_loop = start_loop(u) + step
                    post = None
                    if a.timing == 'after_u':
                        post = {u['d'] + 1: 1, u['d'] + 2: 2, u['d'] + 4: 4}.get(abs_loop)
                    else:
                        post = {u['d']: 1, u['d'] + 1: 2, u['d'] + 3: 4}.get(abs_loop)
                    if post is None:
                        continue
                    bi = c['b']
                    uval = int(preds[si, u['pos'], V])
                    rval = int(preds[si, u['r_pos'], V])
                    us = ('true' if uval == u['a'] else
                          'inj' if uval == u['a_inj'] else
                          'recomp' if uval == u['a_recomp'] else
                          'no_value' if tok_str(uval) in ('EMPTY', 'N/A') else 'other')
                    rs = ('true' if rval == u['r_true'] else
                          'wrong' if rval == u['r_wrong'] else
                          'no_value' if tok_str(rval) in ('EMPTY', 'N/A') else 'other')
                    label_b = label[bi].cpu(); info_b = info[bi].cpu()
                    qrows = []
                    for q in u['qs']:
                        dq = int(depths[bi, q].cpu())
                        qval = int(preds[si, q, V])
                        q_true = int(label_b[q, V])
                        q_u = q_expect(info_b, q, u['pos'], u['a_inj'], u['r_pos'], u['r_true'], label_b)
                        q_r_true_u = q_expect(info_b, q, u['pos'], u['a'], u['r_pos'], u['r_wrong'], label_b)
                        q_r_recomp = q_expect(info_b, q, u['pos'], u['a_recomp'], u['r_pos'], u['r_wrong'], label_b)
                        q_ur = q_expect(info_b, q, u['pos'], u['a_inj'], u['r_pos'], u['r_wrong'], label_b)
                        match = []
                        if qval == q_true: match.append('true')
                        if q_u is not None and qval == q_u: match.append('u')
                        if q_r_true_u is not None and qval == q_r_true_u: match.append('r_true_u')
                        if q_r_recomp is not None and qval == q_r_recomp: match.append('r_recomp_u')
                        if q_ur is not None and qval == q_ur: match.append('ur')
                        qrows.append(dict(q=q, dq=dq, value=tok_str(qval),
                                          true=tok_str(q_true),
                                          q_u=None if q_u is None else tok_str(q_u),
                                          q_r_true_u=None if q_r_true_u is None else tok_str(q_r_true_u),
                                          q_r_recomp=None if q_r_recomp is None else tok_str(q_r_recomp),
                                          q_ur=None if q_ur is None else tok_str(q_ur),
                                          match=match,
                                          not_yet=(abs_loop < dq)))
                    all_records.append(dict(
                        graph_id=batch_idx * a.batch_size + bi,
                        timing=a.timing, condition=cond, post_round=post, loop=abs_loop,
                        pos=u['pos'], d=u['d'], a=u['a'], a_inj=u['a_inj'], a_recomp=u['a_recomp'],
                        r_pos=u['r_pos'], r_depth=u['r_depth'], r_true=u['r_true'], r_wrong=u['r_wrong'],
                        u_value=tok_str(uval), u_status=us,
                        r_value=tok_str(rval), r_status=rs,
                        q=qrows))
                cur = preds
        done += B; batch_idx += 1

# ---- 汇总 ----
def cnt(rows, key, val):
    return sum(1 for r in rows if r[key] == val)
summary = dict(tag=a.tag, start=a.start, timing=a.timing, value_mode=a.value_mode, k=a.k,
               n_graphs=done, n_records=len(all_records))
for post in [1, 2, 4]:
    rs = [r for r in all_records if r['post_round'] == post]
    summary[f'post{post}'] = dict(
        n=len(rs),
        u={s: cnt(rs, 'u_status', s) for s in ['true', 'inj', 'recomp', 'other', 'no_value']},
        r={s: cnt(rs, 'r_status', s) for s in ['true', 'wrong', 'other', 'no_value']},
        by_cond={c: dict(
            n=sum(1 for x in rs if x['condition'] == c),
            u={s: sum(1 for x in rs if x['condition'] == c and x['u_status'] == s)
               for s in ['true', 'inj', 'recomp', 'other', 'no_value']})
            for c in CONDS},
    )
# q match summary by condition/post
qsum = {}
for post in [1, 2, 4]:
    for c in CONDS:
        rows = [r for r in all_records if r['post_round'] == post and r['condition'] == c]
        qs = [q for r in rows for q in r['q'] if not q['not_yet']]
        qsum[f'{c}_post{post}'] = dict(
            n=len(qs),
            true=sum(1 for q in qs if 'true' in q['match']),
            u=sum(1 for q in qs if 'u' in q['match']),
            r_true_u=sum(1 for q in qs if 'r_true_u' in q['match']),
            r_recomp=sum(1 for q in qs if 'r_recomp_u' in q['match']),
            ur=sum(1 for q in qs if 'ur' in q['match']),
            other=sum(1 for q in qs if not q['match']),
        )
summary['q'] = qsum
print(json.dumps(summary, indent=2, ensure_ascii=False))
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, f'{a.tag}.json'), 'w') as f:
    json.dump(dict(summary=summary, records=all_records), f, ensure_ascii=False, indent=2)
print('saved ->', os.path.join(a.out_dir, f'{a.tag}.json'))