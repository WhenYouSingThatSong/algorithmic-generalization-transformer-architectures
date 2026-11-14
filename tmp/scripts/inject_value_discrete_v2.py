# -*- coding: utf-8 -*-
"""
离散注错 v2：按图选多个 u、一次读出所有直接子节点 q、同一绝对轮次、按可算时间计分。

当前实现：teacher-state 模式（从 clean teacher 快照 S[d+1] 注入）。
- 每张图枚举所有内部 u（depth>=1、有父节点、有至少一个数字 label 的直接子节点 q）；
- --k 0 表示全量，k=4 表示按深度分层选 4 个 u；
- 对每个 u 注入 a -> (a+1) mod 23（或 --value_mode four_to_five，只做 4->5）；
- 一次续算读出 u 的所有直接子节点 q；
- 记录注错后第 1/2/4 轮；q 在 dq 之前是“未可算”，不算错；
- 按图聚合（每张图权重相同）。

运行目录：experiments/evaluation
"""
import argparse, os, sys, copy, json
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
import torch, yaml

# [perf] 小模型 + 大量短 forward：限制 CPU 线程，减少多进程争用
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import numpy as np
from collections import defaultdict

from Simtransformer.simtransformer.utils import EasyDict
from model import LitRecurrentTransformerModel, DAGTeacherModel, INVALID_NUM
from train import DAGDataModule
from tokenizers import FactoredVocabTokenizer, build_factored_tokenizer
import tokenizers as _tok
sys.modules.setdefault('factored_tokenizers', _tok)

p = argparse.ArgumentParser()
p.add_argument('--config_dir_name', required=True)
p.add_argument('--group_name', required=True)
p.add_argument('--run_name', required=True)
p.add_argument('--ckpt_base_dir', required=True)
p.add_argument('--ckpt_name', default='last.ckpt')
p.add_argument('--ds_file', required=True)
p.add_argument('--n_graphs', type=int, default=32)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--start', choices=['teacher', 'own'], default='teacher')
p.add_argument('--k', type=int, default=0, help='0 = all eligible u; else depth-stratified k')
p.add_argument('--value_mode', choices=['a_plus_1', 'four_to_five'], default='a_plus_1')
p.add_argument('--extra_steps', type=int, default=3)
p.add_argument('--tag', required=True)
p.add_argument('--out_dir', default='../../tmp/analysis/inject-v2')
a = p.parse_args()

root = os.path.abspath('../..')
def load(name):
    with open(os.path.join(root, 'experiments/configs', a.config_dir_name, name)) as f:
        return EasyDict(yaml.load(f, Loader=yaml.FullLoader))
mc, tc, dc = load('model_config.yaml'), load('train_config.yaml'), load('data_config.yaml')
tok = build_factored_tokenizer(dc)
mc.vocab_sizes = tok.vocab_sizes; mc.factors = tok.factors

ckpt = os.path.join(a.ckpt_base_dir, f'{a.group_name}-{a.run_name}', a.ckpt_name)
teacher = DAGTeacherModel(mc, tok)
lit = LitRecurrentTransformerModel(mc, dc, tc, teacher)
lit.load_ckpt(ckpt)
lit = lit.to('cuda').eval()

dm = DAGDataModule(data_path=[a.ds_file, a.ds_file], batch_size=a.batch_size,
                   num_workers=2, factored_tokenizer=tok)
dm.setup()
mc.data_dim, mc.label_dim, mc.info_dim = dm.data_dim, dm.label_dim, dm.info_dim

V = tok.factors.index('VALUE')
Rg = tok.factors.index('REGRET'); Md = tok.factors.index('MODIFICATION')
r_false = teacher.regret_tok2idx['FALSE']; m_false = teacher.modification_tok2idx['FALSE']
MOD = tok.mod_val

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

# ---------- 读一个 batch、构造候选 ----------
def build_candidates(data, label, info, depths):
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
            us.append(dict(pos=pos, d=d, parents=parents, qs=qs))
        # depth-stratified selection
        if a.k == 0 or len(us) <= a.k:
            sel = us
        else:
            us_sorted = sorted(us, key=lambda x: (x['d'], x['pos']))
            chunks = np.array_split(np.arange(len(us_sorted)), a.k)
            sel = [us_sorted[int(ch[0])] for ch in chunks]
        out.append(dict(b=bi, us=sel, all_u=us))
    return out

# ---------- teacher-state 注入 ----------
def run_teacher(batch_cands, data, label, info, depths):
    B, T = data.shape[:2]
    D = int(depths.max()) + 1
    S, _, _ = lit.get_teacher_batch_with_depth(data, label, depths)
    n_loops = D + a.extra_steps
    records = []
    max_k = max((len(c['us']) for c in batch_cands), default=0)
    for k in range(max_k):
        # 为每个样本选第 k 个 u（没有则用第一个可用位置占位）
        state = torch.stack([S[(c['us'][k]['d'] + 1) if k < len(c['us']) else 0, c['b']].clone()
                             for c in batch_cands]).to('cuda')
        clean = state.clone()
        vals = {}
        for c in batch_cands:
            if k >= len(c['us']):
                continue
            u = c['us'][k]; bi = c['b']; p = u['pos']
            a_val = int(label[bi, p, V].cpu())
            if a.value_mode == 'four_to_five':
                if a_val != 4:
                    continue
                a_inj = 5
            else:
                a_inj = (a_val + 1) % MOD
            state[bi, p, V] = tok.value_tok2idx[str(a_inj)]
            vals[bi] = (u, a_val, a_inj)
        state[..., Rg] = r_false; state[..., Md] = m_false
        clean[..., Rg] = r_false; clean[..., Md] = m_false
        # 自由续算（注入 & clean）
        def rollout(start):
            cur = start; us = [[None] * n_loops for _ in range(B)]
            qs = [[None] * n_loops for _ in range(B)]
            with torch.no_grad():
                for loop in range(n_loops):
                    logits = lit.predict_step(cur)
                    preds = torch.stack([l.argmax(-1) for l in logits], dim=-1)
                    preds[..., Rg] = r_false; preds[..., Md] = m_false
                    for c in batch_cands:
                        if k >= len(c['us']): continue
                        u = c['us'][k]; bi = c['b']
                        us[bi][loop] = int(preds[bi, u['pos'], V])
                        qs[bi][loop] = [int(preds[bi, q, V]) for q in u['qs']]
                    cur = preds
            return us, qs
        us_inj, qs_inj = rollout(state)
        us_cln, qs_cln = rollout(clean)
        for c in batch_cands:
            if k >= len(c['us']) or c['b'] not in vals:
                continue
            u = c['us'][k]; bi = c['b']
            u0, a_val, a_inj = vals[bi]
            # 参考轮：注入发生在 loop d 之后；post round r -> loop d+r。
            # rollout 从 S[d+1] 开始，列表下标 0 就对应 loop d+1，因此下标是
            # r-1；绝对 loop（d+r）只用于记录/可算性判断，不能当下标。
            for r in [1, 2, 4]:
                idx = r - 1
                loop = u['d'] + r
                if idx >= n_loops:
                    continue
                rec = dict(graph_id=None, b=bi, pos=u['pos'], d=u['d'],
                           a=a_val, a_inj=a_inj, post_round=r, loop=loop)
                rec['u_status'] = ('recovered' if us_inj[bi][idx] == a_val else
                                   'injected' if us_inj[bi][idx] == a_inj else 'other')
                qrows = []
                for j, q in enumerate(u['qs']):
                    dq = int(depths[bi, q].cpu())
                    qval = qs_inj[bi][idx][j]
                    if loop < dq:
                        status = 'not_yet'
                    else:
                        # expected wrong value under a_inj
                        parents_q, _ = parse_expr(info[bi].cpu(), q)
                        vals_q = {}
                        for pp in parents_q:
                            if pp == u['pos']:
                                vals_q[pp] = a_inj
                            elif is_num(label[bi, pp, V].cpu()):
                                vals_q[pp] = int(label[bi, pp, V].cpu())
                            else:
                                vals_q[pp] = None
                        q_wrong = apply_expr(info[bi].cpu(), q, vals_q)
                        q_corr = int(label[bi, q, V].cpu())
                        if tok_str(qval) in ('EMPTY', 'N/A'):
                            status = 'no_value'
                        elif qval == q_corr:
                            status = 'correct'
                        elif q_wrong is not None and qval == q_wrong:
                            status = 'expected_wrong'
                        else:
                            status = 'other_wrong'
                    qrows.append(dict(q=q, dq=dq, status=status, value=tok_str(qval)))
                rec['q'] = qrows
                records.append(rec)
    return records

# ---------- own-state 注入 ----------
def run_own(batch_cands, data, label, info, depths):
    """从模型自己的 clean 轨迹取 u 算完后的状态，再注入 a->a+1。"""
    B, T = data.shape[:2]
    D = int(depths.max()) + 1
    n_loops = D + a.extra_steps
    # clean free-run：离散 token 状态从 data 开始
    cur = data.clone()
    preds_all = []
    with torch.no_grad():
        for loop in range(n_loops):
            logits = lit.predict_step(cur)
            preds = torch.stack([l.argmax(-1) for l in logits], dim=-1)
            preds[..., Rg] = r_false; preds[..., Md] = m_false
            preds_all.append(preds)
            cur = preds
    preds_all = torch.stack(preds_all, dim=0)  # [L,B,T,6]
    records = []; elig = []
    max_k = max((len(c['us']) for c in batch_cands), default=0)
    for k in range(max_k):
        state = torch.stack([
            (preds_all[c['us'][k]['d']][c['b']] if k < len(c['us']) else preds_all[0][c['b']]).clone()
            for c in batch_cands]).to('cuda')
        vals = {}
        for c in batch_cands:
            if k >= len(c['us']): continue
            u = c['us'][k]; bi = c['b']; p = u['pos']
            a_val = int(label[bi, p, V].cpu())
            if int(state[bi, p, V]) != a_val:  # own-state 下 u 必须已经算对
                continue
            if a.value_mode == 'four_to_five':
                if a_val != 4: continue
                a_inj = 5
            else:
                a_inj = (a_val + 1) % MOD
            state[bi, p, V] = tok.value_tok2idx[str(a_inj)]
            vals[bi] = (u, a_val, a_inj)
        state[..., Rg] = r_false; state[..., Md] = m_false
        for c in batch_cands:
            if k >= len(c['us']): continue
            u = c['us'][k]; bi = c['b']
            elig.append(dict(graph_id=None, b=bi, pos=u['pos'], d=u['d'],
                             eligible=(bi in vals)))
        if not vals:
            continue
        cur = state
        with torch.no_grad():
            for step in range(1, n_loops):
                logits = lit.predict_step(cur)
                preds = torch.stack([l.argmax(-1) for l in logits], dim=-1)
                preds[..., Rg] = r_false; preds[..., Md] = m_false
                if step in (1, 2, 4):
                    for bi, (u, a_val, a_inj) in vals.items():
                        loop = u['d'] + step
                        if loop >= n_loops: continue
                        rec = dict(graph_id=None, b=bi, pos=u['pos'], d=u['d'],
                                   a=a_val, a_inj=a_inj, post_round=step, loop=loop)
                        rec['u_status'] = ('recovered' if int(preds[bi, u['pos'], V]) == a_val else
                                           'injected' if int(preds[bi, u['pos'], V]) == a_inj else 'other')
                        rec['u_clean'] = int(preds_all[loop][bi, u['pos'], V])
                        qrows = []
                        for q in u['qs']:
                            dq = int(depths[bi, q].cpu())
                            qval = int(preds[bi, q, V])
                            q_clean_val = int(preds_all[loop][bi, q, V])
                            if loop < dq:
                                status = 'not_yet'; clean_status = 'not_yet'
                            else:
                                parents_q, _ = parse_expr(info[bi].cpu(), q)
                                vals_q = {}
                                for pp in parents_q:
                                    if pp == u['pos']:
                                        vals_q[pp] = a_inj
                                    elif is_num(label[bi, pp, V].cpu()):
                                        vals_q[pp] = int(label[bi, pp, V].cpu())
                                    else:
                                        vals_q[pp] = None
                                q_wrong = apply_expr(info[bi].cpu(), q, vals_q)
                                q_corr = int(label[bi, q, V].cpu())
                                def stat(v):
                                    if tok_str(v) in ('EMPTY', 'N/A'): return 'no_value'
                                    if v == q_corr: return 'correct'
                                    if q_wrong is not None and v == q_wrong: return 'expected_wrong'
                                    return 'other_wrong'
                                status = stat(qval); clean_status = stat(q_clean_val)
                            qrows.append(dict(q=q, dq=dq, status=status, value=tok_str(qval),
                                              clean_status=clean_status, clean_value=tok_str(q_clean_val)))
                        rec['q'] = qrows
                        records.append(rec)
                cur = preds
    return records, elig

# ---------- 主循环 ----------
all_records = []
own_elig = []
done = 0; batch_idx = 0
with torch.no_grad():
    for batch in dm.val_dataloader():
        if done >= a.n_graphs: break
        b = lit.reduce_batch(copy.deepcopy(batch)).to('cuda')
        data = b[..., :mc.data_dim]; label = b[..., mc.data_dim:mc.data_dim + mc.label_dim]
        info = b[..., mc.data_dim + mc.label_dim:]; depths = info[..., 0].long()
        B = data.size(0)
        cands = build_candidates(data, label, info, depths)
        if a.start == 'teacher':
            recs = run_teacher(cands, data, label, info, depths)
        else:
            recs, e = run_own(cands, data, label, info, depths)
            for x in e:
                x['graph_id'] = batch_idx * a.batch_size + x['b']
                own_elig.append(x)
        # graph ids
        for r in recs:
            r['graph_id'] = batch_idx * a.batch_size + r['b']
            all_records.append(r)
        done += B; batch_idx += 1

# ---------- 汇总（pair-level + per-graph） ----------
def rate(rs, key, val):
    return sum(1 for r in rs if r[key] == val) / len(rs) if rs else 0.0
r1 = [r for r in all_records if r['post_round'] == 1]
r2 = [r for r in all_records if r['post_round'] == 2]
r4 = [r for r in all_records if r['post_round'] == 4]
# q 状态按 post round 汇总（只算 available）
def qstat(rs):
    out = defaultdict(lambda: [0, 0])
    for r in rs:
        for q in r['q']:
            if q['status'] == 'not_yet':
                continue
            out[q['status']][0] += 1; out[q['status']][1] += 1
    return {k: v[0] for k, v in out.items()}
# per-graph: 每个图/每个 u 的 q 状态
by_graph = defaultdict(list)
for r in all_records:
    by_graph[r['graph_id']].append(r)

summary = dict(
    tag=a.tag, start=a.start, k=a.k, value_mode=a.value_mode,
    n_graphs=done, n_records=len(all_records),
    u_r1=dict((s, rate(r1, 'u_status', s)) for s in ['recovered', 'injected', 'other']),
    u_r2=dict((s, rate(r2, 'u_status', s)) for s in ['recovered', 'injected', 'other']),
    u_r4=dict((s, rate(r4, 'u_status', s)) for s in ['recovered', 'injected', 'other']),
    q_r1=qstat(r1), q_r2=qstat(r2), q_r4=qstat(r4),
    own_total_u=len(own_elig),
    own_eligible_u=sum(1 for x in own_elig if x['eligible']),
    own_total_graphs=len(set(x['graph_id'] for x in own_elig)) if own_elig else 0,
    own_eligible_graphs=len(set(x['graph_id'] for x in own_elig if x['eligible'])) if own_elig else 0,
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, f'{a.tag}.json'), 'w') as f:
    json.dump(dict(summary=summary, eligibility=own_elig, records=all_records), f,
              ensure_ascii=False, indent=2)
print('saved ->', os.path.join(a.out_dir, f'{a.tag}.json'))
