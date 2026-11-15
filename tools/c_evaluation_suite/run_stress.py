#!/usr/bin/env python3
"""Evaluate sustained u corruption or fixed noise scales; never train or pick a dose."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stress_core as S
U, C = S.U, S.C


def qualifications(sources, gold, pos, plan, clean_first, mode):
    ui, qi = pos.index(plan['u']), pos.index(plan['q'])
    required = set([plan['u'], plan['q']] + plan['parents'][plan['u']] + plan['parents'][plan['q']])
    original = sources[0][0]
    value = all(original[pos.index(p)][3] == gold[pos.index(p)][3] for p in required)
    value = value and all(clean_first[i][3] == gold[i][3] for i in (ui, qi))
    strict = all(original[i] == gold[i] for i in (ui, qi))
    # Validate time-matched original/same/wrong u donors at EVERY forcing step.
    for src in sources:
        for j, v in [(0, gold[ui][3]), (1, gold[ui][3]), (2, plan['expected_u'])]:
            target = list(gold[ui]); target[3] = v
            value = value and src[j][ui][3] == v
            strict = strict and src[j][ui] == target
    if mode == 'sustained':
        qtarget = list(gold[qi]); qtarget[3] = plan['expected_q']
        value = value and sources[0][4][qi][3] == qtarget[3]
        strict = strict and sources[0][4][qi] == qtarget
    return value, value and strict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['sustained', 'noise'], required=True)
    p.add_argument('--study-dir', type=Path, default=C.STUDY)
    p.add_argument('--data-dir', type=Path, default=C.P.STUDY/'data')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--source-ckpt', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    R = C.runtime()
    import torch
    torch.set_num_threads(a.num_threads)
    torch.set_float32_matmul_precision('medium')
    models = U.models(a.study_dir, a.data_dir, a.source_ckpt)
    cases = C.read_cases(a.study_dir/'plan/val.jsonl')
    limit = 4 if a.smoke else (128 if a.mode == 'sustained' else 64)
    if len(cases) < limit:
        raise ValueError('fixed candidate window unavailable')
    durations = S.DURATIONS if a.mode == 'sustained' else (1,)
    specs = S.branches(a.mode)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    meta = dict(status='running', protocol='C_stress_v1', mode=a.mode,
        cohort='smoke' if a.smoke else 'main', models=models,
        candidate_graphs=limit, candidate_rule=f'first {limit}, no replacement',
        horizon=S.HORIZON, durations=durations, branches=specs,
        data_sha256=C.P.sha(a.data_dir/'val.pt'), plan_sha256=C.P.sha(a.study_dir/'plan/val.jsonl'),
        code_hashes={n: C.P.sha(U.HERE/n) for n in ('run_stress.py', 'stress_core.py', 'suite_core.py')},
        dependencies=C.dependency_hashes(), num_threads=a.num_threads,
        note='Development graphs, no independent final-test claim; no decoded feedback.')
    C.P.write_json(a.out_dir/'meta.json', meta)
    with (a.out_dir/'records.jsonl').open('x', encoding='utf-8') as output, \
         (a.out_dir/'coverage.jsonl').open('x', encoding='utf-8') as coverage, torch.no_grad():
        for spec in models:
            ctx = C.context(R, spec['path'], a.data_dir/'val.pt', a.device, 1)
            lit = ctx['litmodel']; lit.eval(); count = 0
            for gid, raw in enumerate(ctx['data_module'].val_dataloader()):
                if gid >= limit:
                    break
                count += 1
                x, y, info = R.split_batch(lit.reduce_batch(raw.to(ctx['device'])), ctx['model_config'])
                base = cases[gid]
                if base['graph_id'] != gid or y[0, base['positions'], :4].tolist() != base['truth4']:
                    raise ValueError('case/graph mismatch')
                plan = U.pair_plan(base, y[0].tolist(), info[0].tolist())
                if plan['status'] != 'selected':
                    coverage.write(json.dumps(dict(arm=spec['arm'], graph_id=gid, **plan))+'\n')
                    continue
                u, q, pos = plan['u'], plan['q'], plan['positions']
                gold = y[0, pos, :4].tolist()
                names = ['original', 'u_same', 'u_wrong', 'q_same', 'q_wrong']
                xs = [x]
                for c in (plan['u_case'], plan['q_case']):
                    xs.extend([C.donor_input(x, c, 'same'), C.donor_input(x, c, f"shift_{c['delta']}")])
                times = {'hold': base['state_round'], 'late': base['D']+16}
                needed = {t+j for t in times.values() for j in range(max(durations))}
                sources = {}; h = lit.model.token_embedder(torch.cat(xs))
                for t in range(1, max(needed)+1):
                    h, _ = lit.model(h, True)
                    if t in needed:
                        sources[t] = h.clone()
                for timing, t in times.items():
                    for length in durations:
                        source_decodes = [C.decode(lit, sources[t+j])[:, pos, :4].tolist() for j in range(length)]
                        states = sources[t][0:1].repeat(len(specs), 1, 1)
                        bs = {b['name']: dict(spec=b, inputs=[], outputs=[], dose=[]) for b in specs}
                        total = length+S.HORIZON if a.mode == 'sustained' else S.HORIZON
                        seed = C.stable_seed(C.SEED, 'dose-calibration', gid, timing)
                        for step in range(1, total+1):
                            replace = S.should_replace(a.mode, step, length)
                            current = sources[t+step-1] if replace else None
                            states, doses = S.apply_inputs(states, current, specs, u, q, replace, step == 1, seed)
                            ip = C.decode(lit, states)[:, pos, :4].tolist()
                            states, ls = lit.model(states, True)
                            op = torch.stack([z.argmax(-1) for z in ls], -1)[:, pos, :4].tolist()
                            for i, b in enumerate(specs):
                                dest = bs[b['name']]
                                dest['inputs'].append(ip[i]); dest['outputs'].append(op[i]); dest['dose'].append(doses[i])
                            if a.mode == 'sustained' and step == 1:
                                # Only the first clean replay is an exact-writeback sham.
                                for suffix in ('q_clean', 'q_wrong'):
                                    if bs['free_'+suffix]['outputs'][0] != bs['original_'+suffix]['outputs'][0]:
                                        raise ValueError('first clean replay differs from exact writeback')
                        ref = 'free_q_clean' if a.mode == 'sustained' else 'free'
                        value, strict = qualifications(source_decodes, gold, pos, plan, bs[ref]['outputs'][0], a.mode)
                        cov = dict(arm=spec['arm'], graph_id=gid, timing=timing, duration=length,
                                   status='selected', value_qualified=value, strict_qualified=strict)
                        coverage.write(json.dumps(cov)+'\n')
                        rec = dict(**cov, mode=a.mode, state_round=t, D=base['D'], u=u, q=q,
                            positions=pos, truth4=gold, parents=plan['parents'], depths=plan['depths'],
                            children=base['children'], descendants=base['descendants'],
                            delta_u=plan['delta_u'], delta_q=plan['delta_q'],
                            expected_u=plan['expected_u'], expected_q=plan['expected_q'],
                            start_graph_value_all=all(row[3] == g[3] for row, g in zip(source_decodes[0][0], gold)),
                            start_graph_first4_all=source_decodes[0][0] == gold,
                            source_names=names, source_decodes=source_decodes,
                            source_edits={n: c['changes'] for n, c in [('u', plan['u_case']), ('q', plan['q_case'])]},
                            noise_seed=seed, branches=bs)
                        output.write(json.dumps(rec)+'\n'); output.flush(); coverage.flush()
                if gid % 16 == 0:
                    print(json.dumps(dict(arm=spec['arm'], graph_id=gid, mode=a.mode)), flush=True)
            if count != limit:
                raise ValueError('fixed candidate count differs')
            del ctx, lit
    meta['status'] = 'complete'; C.P.write_json(a.out_dir/'meta.json', meta)


if __name__ == '__main__':
    main()
