#!/usr/bin/env python3
"""A/B repeated assignments and two dead-slot versions, from fixed saved starts."""
import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import core as P
sys.path.insert(0, str(HERE.parent/'priority_pilots'))
from io_utils import jsonl, read_json, write_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True, help='completed discrete_reassignment_v1 run')
    p.add_argument('--mode', choices=('repeat', 'reuse'), required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--horizons', type=int, nargs='+', default=[8, 16])
    p.add_argument('--limit-graphs', type=int, help='smoke only; first fixed candidate IDs, no refill')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if any(h < 2 for h in a.horizons) or len(set(a.horizons)) != len(a.horizons):
        p.error('distinct horizons >=2 required')
    if a.limit_graphs is not None and a.limit_graphs < 1:
        p.error('limit-graphs must be positive')
    source = read_json(a.source_dir/'meta.json')
    if source['status'] != 'complete' or source['protocol'] != 'discrete_reassignment_v1':
        raise ValueError('wrong source')
    for name in ('records', 'selection'):
        if sha(a.source_dir/f'{name}.jsonl') != source[f'{name}_sha256']:
            raise ValueError('source checksum differs')
    gids = source['candidate_ids'][:a.limit_graphs] if a.limit_graphs else source['candidate_ids']
    records = {}
    for r in jsonl(a.source_dir/'records.jsonl'):
        if r['graph_id'] in gids and r['delta'] == source['deltas'][0]:
            if r['graph_id'] in records:
                raise ValueError('duplicate saved graph')
            records[r['graph_id']] = r
    sys.path.insert(0, str(HERE.parent/'b_mechanism'))
    from trajectory_io import configure_runtime
    configure_runtime()
    import torch
    import b_mechanism_common as C
    torch.set_num_threads(a.num_threads)
    spec = source['model']; ckpt = Path(C.checkpoint_path(spec['ckpt_base_dir'], spec['group_name'], spec['run_name'], spec['ckpt_name']))
    if sha(ckpt) != spec['ckpt_sha256'] or sha(source['dataset']) != source['dataset_sha256']:
        raise ValueError('checkpoint/dataset changed')
    ctx = C.load_eval_context(spec['config_dir_name'], spec['group_name'], spec['run_name'],
                             spec['ckpt_base_dir'], spec['ckpt_name'], source['dataset'], batch_size=1, num_workers=0)
    lit, tok, teacher = (ctx[k] for k in ('lit', 'tok', 'teacher'))
    if lit.nointerm or tok.mod_val != 23:
        raise ValueError('discrete mod23 only')
    lit.eval()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    deps = [Path(__file__), HERE/'core.py', HERE.parent/'priority_pilots/io_utils.py',
            HERE.parent/'b_mechanism/b_mechanism_common.py', ROOT/'experiments/model.py', ROOT/'experiments/tokenizers.py']
    meta = dict(status='running', protocol='state_reuse_discrete_v1', mode=a.mode,
        model_key=source['model_key'], display_name='A' if source['model_key']=='model_a' else 'B',
        source_meta=source, source_meta_sha256=sha(a.source_dir/'meta.json'),
        candidate_ids=gids, horizons=a.horizons, cohort='smoke' if a.limit_graphs else 'development',
        branches=P.DISCRETE_BRANCHES, code_hashes={str(x.relative_to(ROOT)): sha(x) for x in deps},
        note='Fixed source-qualified development graphs; no post-update success filtering. No capacity claim.',
        runtime=dict(torch=torch.__version__, device=str(ctx['device']), num_threads=a.num_threads))
    write_json(a.out_dir/'meta.json', meta)
    written = 0
    with (a.out_dir/'records.jsonl').open('x', encoding='utf-8') as dst, (a.out_dir/'coverage.jsonl').open('x', encoding='utf-8') as cov, torch.no_grad():
        for gid in gids:
            r = records.get(gid)
            if r is None:
                cov.write(json.dumps(dict(graph_id=gid, status='source_not_qualified'))+'\n'); continue
            nodes = r['nodes']; parents = {int(k): v for k,v in r['parents'].items()}
            constants = {int(k): v for k,v in r['constants'].items()}
            raw = r['raw_input_tokens']; base = r['completed_tokens']
            truth0, _ = P.evaluate(nodes, parents, constants)
            if any(int(C.tok_str(tok, base[p0][C.V])) != truth0[p0] or base[p0][:3] != raw[p0][:3] for p0 in nodes):
                raise ValueError('saved completed state fails independent truth/identity')
            if a.mode == 'repeat':
                suites = {name: list(P.assignment_plans(nodes, parents, constants, gid, schedule))
                          for name, schedule in P.schedules(nodes, parents, constants, gid).items()}
            else:
                seq = P.reuse_plans(nodes, parents, constants, gid)
                suites = {'two_dead_slots': seq} if seq else {}
            cov.write(json.dumps(dict(graph_id=gid, status='selected' if suites else 'no_structural_plan',
                schedules=list(suites), episodes={k:len(v) for k,v in suites.items()},
                source_state_round=r['state_round']))+'\n'); cov.flush()
            for schedule, plans in suites.items():
                for horizon in a.horizons:
                    states = torch.tensor([base for _ in P.DISCRETE_BRANCHES], device=ctx['device'], dtype=torch.long)
                    current_raw = copy.deepcopy(raw); archived = []
                    for plan in plans:
                        # Plans contain truth for metrics only. Runtime edits use constants,
                        # symbolic names, and EMPTY; never computed node answers.
                        edits = {}; target = plan['target']
                        if a.mode == 'repeat':
                            cp = plan['constant']; current_raw[cp][C.V] = tok.value_tok2idx[str(plan['new_constant'])]
                            edits[cp] = {C.V: current_raw[cp][C.V]}
                        else:
                            refs = P.expression_positions(current_raw, target, plan['old_parents'], tok.syntax_tok2idx)
                            for ref, parent in zip(refs, plan['new_parents']):
                                current_raw[ref] = list(raw[parent]); current_raw[ref][C.V] = tok.value_tok2idx['EMPTY']
                                edits[ref] = {i:v for i,v in enumerate(current_raw[ref])}
                            # Save only model-generated old results, never feed this archive back.
                            archived.append(dict(slot=target, version=plan['old_version'],
                                values={name:int(states[j,target,C.V]) for j,name in enumerate(P.DISCRETE_BRANCHES)}))
                        for j, name in enumerate(P.DISCRETE_BRANCHES):
                            if name in ('clean', 'sham'):
                                if name == 'sham': states[j,target] = states[j,target].clone()
                                continue
                            if name == 'scratch':
                                states[j] = torch.tensor(current_raw, device=states.device)
                            else:
                                for at, fields in edits.items():
                                    for field, val in fields.items(): states[j,at,field] = val
                            if name == 'clear':
                                states[j,plan['affected'],C.V] = tok.value_tok2idx['EMPTY']
                        C.force_aux_false(states, teacher)
                        initial_tokens = states.cpu().tolist()
                        # Pin only externally supplied input: constants, and revised symbolic
                        # operand rows for reuse. No result node is pinned.
                        pin = {p0:{C.V:tok.value_tok2idx[str(v)]} for p0,v in plan['constants'].items()}
                        if a.mode == 'reuse':
                            for prior in plans[:plan['episode']]:
                                refs = P.expression_positions(current_raw, prior['target'], prior['new_parents'], tok.syntax_tok2idx)
                                for ref in refs: pin[ref] = {i:v for i,v in enumerate(current_raw[ref])}
                        def hook(step, cur):
                            j = P.DISCRETE_BRANCHES.index('warm_pinned')
                            for at, fields in pin.items():
                                for field, val in fields.items(): cur[j,at,field] = val
                            return cur
                        ins, outs = C.rollout(lit, teacher, states, horizon, input_hook=hook)
                        states = outs[-1].clone()
                        # Include symbolic operators/references, so failures cannot
                        # silently be attributed to VALUE while the equation drifts.
                        positions = [p0 for p0,row in enumerate(raw) if row[C.S] != tok.syntax_tok2idx['<PAD>']]
                        def pick(x): return [x[p0][:4] for p0 in positions]
                        old4 = [list(raw[p0][:4]) for p0 in positions]
                        new4 = [list(current_raw[p0][:4]) for p0 in positions]
                        orig4 = copy.deepcopy(old4)
                        for i,p0 in enumerate(positions):
                            if p0 in plan['old_truth']: old4[i][3] = tok.value_tok2idx[str(plan['old_truth'][p0])]
                            if p0 in plan['new_truth']: new4[i][3] = tok.value_tok2idx[str(plan['new_truth'][p0])]
                            if p0 in truth0: orig4[i][3] = tok.value_tok2idx[str(truth0[p0])]
                        ips = [x.cpu().tolist() for x in ins]; ops = [x.cpu().tolist() for x in outs]
                        branches = {name:dict(initial=pick(initial_tokens[j]), inputs=[pick(x[j]) for x in ips],
                            outputs=[pick(x[j]) for x in ops],
                            truth_ref='original' if name in ('clean','sham') else 'new',
                            solve_start_round=0 if name=='scratch' else r['state_round']+(plan['episode']-1)*horizon,
                            cumulative_forward_cost=(0 if name=='scratch' else r['state_round'])+plan['episode']*horizon)
                            for j,name in enumerate(P.DISCRETE_BRANCHES)}
                        if branches['clean'] != branches['sham']:
                            raise ValueError('sham differs')
                        expected_pin = {int(k):{int(f):v for f,v in vs.items()} for k,vs in pin.items()}
                        pj = P.DISCRETE_BRANCHES.index('warm_pinned')
                        if any(x[pj][at][f] != v for x in ips for at,fs in expected_pin.items() for f,v in fs.items()):
                            raise ValueError('pin failed')
                        if any(x[j][at][f] != teacher.regret_tok2idx['FALSE'] for x in ops for j in range(len(P.DISCRETE_BRANCHES)) for at in range(len(raw)) for f in (C.RG,C.MD)):
                            raise ValueError('aux not false')
                        rec = dict(graph_id=gid, mode=a.mode, schedule=schedule, horizon=horizon, episode=plan['episode'],
                            source_state_round=r['state_round'], positions=positions, nodes=nodes, plan=plan,
                            original_truth4=orig4, old_truth4=old4, new_truth4=new4, branches=branches,
                            symbolic_edits=edits, pin=pin, raw_task=current_raw, archive=copy.deepcopy(archived),
                            regions=dict(all_nodes=nodes, target=[target], changed=plan['changed'],
                                affected=plan['affected'], unchanged=plan['unchanged'],
                                constants=sorted(constants)), initial_full_tokens=initial_tokens)
                        dst.write(json.dumps(rec)+'\n'); dst.flush(); written += 1
                print(json.dumps(dict(graph_id=gid,schedule=schedule,written=written)), flush=True)
    meta.update(status='complete', records=written, records_sha256=sha(a.out_dir/'records.jsonl'),
                coverage_sha256=sha(a.out_dir/'coverage.jsonl'))
    write_json(a.out_dir/'meta.json',meta)


if __name__ == '__main__': main()
