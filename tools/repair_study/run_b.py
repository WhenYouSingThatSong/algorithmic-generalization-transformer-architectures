#!/usr/bin/env python3
"""B experiment 1: update an already-filled node while its parents are correct.

Fixed candidate graphs from the existing manifest; no refill by outcome.
Primary after_u and complete cohorts stay separate from graph142 diagnostics.
Run once per model_a/model_b, then summarize common eligible case identities.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'b_mechanism'))
import core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model', choices=['model_a', 'model_b'], required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--candidate_start', type=int, default=32)
    p.add_argument('--candidate_count', type=int, default=64)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--include_anchor', action='store_true')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if min(a.candidate_count, a.horizon, a.batch_size) < 1 or a.candidate_start < 0:
        p.error('invalid candidate/horizon/batch arguments')
    from trajectory_io import configure_runtime
    configure_runtime()
    import torch
    import b_mechanism_common as C
    manifest = core.read_json(a.manifest)
    spec = manifest[a.model]
    ckpt = C.checkpoint_path(spec['ckpt_base_dir'], spec['group_name'], spec['run_name'], spec['ckpt_name'])
    ds_file = manifest['ds_file']
    C.verify_manifest_identity(manifest, ds_file, ckpt)
    C.validate_manifest_graphs(manifest)
    if core.sha(ckpt) != spec['ckpt_sha256']:
        raise ValueError('checkpoint does not match requested model identity')
    gids = set(range(a.candidate_start, a.candidate_start+a.candidate_count))
    if a.include_anchor:
        gids.add(142)
    graphs = {g['graph_id']: g for g in manifest['graphs'] if g['graph_id'] in gids}
    a.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', experiment='B_existing_value_v1',
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    model=spec, dataset=ds_file, dataset_sha256=core.sha(ds_file),
                    manifest_sha256=core.sha(a.manifest), script_sha256=core.sha(__file__),
                    core_sha256=core.sha(HERE/'core.py'),
                    common_sha256=core.sha(HERE.parent/'b_mechanism/b_mechanism_common.py'),
                    missing_manifest_graphs=sorted(gids-graphs.keys()))
    core.write_json(a.out_dir/'meta.json', metadata)
    ctx = C.load_eval_context(spec['config_dir_name'], spec['group_name'], spec['run_name'],
                              spec['ckpt_base_dir'], spec['ckpt_name'], ds_file,
                              batch_size=a.batch_size, num_workers=0)
    lit, tok, teacher, mc = (ctx[k] for k in ('lit', 'tok', 'teacher', 'mc'))
    written, skipped, scanned = 0, 0, []
    with (a.out_dir/'records.jsonl').open('x', encoding='utf-8') as dst, torch.no_grad():
        for batch_index, raw in enumerate(ctx['dm'].val_dataloader()):
            if batch_index*a.batch_size > max(gids):
                break
            batch = lit.reduce_batch(raw).to(ctx['device'])
            data, label, info = C.split_batch(batch, mc)
            for bi in range(data.shape[0]):
                gid = batch_index*a.batch_size+bi
                if gid not in graphs:
                    continue
                scanned.append(gid)
                mg = graphs[gid]
                meta = C.build_meta(data[bi], label[bi], info[bi], tok, teacher)
                positions = meta['nodes']
                parents = {q: [x for x in ps if x in positions] for q, ps in meta['parents'].items()}
                truth4 = label[bi, positions, :4].cpu().tolist()
                _, original = C.rollout(lit, teacher, data[bi:bi+1], mg['n_loops'])
                targets = []
                if a.candidate_start <= gid < a.candidate_start+a.candidate_count:
                    for ch in mg['chains']:
                        first_correct = next((t+1 for t, state in enumerate(original)
                                              if int(state[0,ch['pos'],C.V]) == int(label[bi,ch['pos'],C.V])), None)
                        targets.append((ch['pos'], 'just_solved', first_correct, 'fixed_candidates'))
                        complete_round = None if mg.get('T') is None else mg['T']+1
                        targets.append((ch['pos'], 'whole_graph_complete', complete_round, 'fixed_candidates'))
                if a.include_anchor and gid == 142 and mg.get('T') is not None:
                    targets.append((128, 'after_original_fault', mg['T']+3, 'diagnostic_anchor'))
                for target, timing, state_round, cohort in targets:
                    rec = dict(case_id=f'B:{gid}:{target}:{timing}', graph_id=gid,
                               weight=a.model, cohort=cohort, timing=timing, category='existing_value',
                               state_round=state_round, target_pos=target,
                               parent_positions=parents[target],
                               descendant_positions=core.descendants(parents, target),
                               node_positions=positions, truth4=truth4,
                               data_sha256=metadata['dataset_sha256'], status='eligible')
                    if state_round is None:
                        rec.update(status='ineligible', reason='no_qualifying_state_in_original_budget')
                        skipped += 1
                        dst.write(json.dumps(rec)+'\n'); dst.flush()
                        continue
                    if cohort == 'diagnostic_anchor':
                        base = original[mg['T']].clone()
                        old = int(label[bi, 26, C.V])
                        base[0, 26, C.V] = tok.value_tok2idx[str((old+1) % tok.mod_val)]
                        _, trace = C.rollout(lit, teacher, base, 2)
                        base = trace[-1]
                        rec['anchor_preparation'] = dict(injected_pos=26, from_value=old,
                                                         to_value=(old+1) % tok.mod_val, forwards=2)
                    else:
                        base = original[state_round-1].clone()
                    ps = parents[target]
                    parent_ok = bool(ps) and all(int(base[0, q, C.V]) == int(label[bi, q, C.V]) for q in ps)
                    target_ok = int(base[0, target, C.V]) == int(label[bi, target, C.V])
                    graph_ok = C.all_numeric_correct(base[0], label[bi], meta['numeric'])
                    rec['qualification'] = dict(parents_correct=parent_ok, target_correct=target_ok,
                                                graph_correct=graph_ok)
                    if not parent_ok or (cohort == 'fixed_candidates' and not target_ok) or (
                            timing == 'whole_graph_complete' and not graph_ok):
                        rec['status'] = 'ineligible'
                        skipped += 1
                    else:
                        truth_value = int(label[bi, target, C.V])
                        rec['branches'] = {}
                        for name, value in core.b_variants(truth_value, tok.mod_val, tok.value_tok2idx['EMPTY']):
                            start = base.clone()
                            if name != 'unchanged':
                                start[0, target, C.V] = value
                            _, out = C.rollout(lit, teacher, start, a.horizon)
                            rec['branches'][name] = dict(initial=start[0, positions, :4].cpu().tolist(),
                                assigned_value=None if name == 'unchanged' else value,
                                outputs=[v[0, positions, :4].cpu().tolist() for v in out])
                        if rec['branches']['correct_write']['outputs'] != rec['branches']['unchanged']['outputs'] and target_ok:
                            raise ValueError('identical-input B control diverged; investigate replay before proceeding')
                        written += 1
                    dst.write(json.dumps(rec)+'\n'); dst.flush()
                print(json.dumps(dict(graph_id=gid, eligible=written, ineligible=skipped)), flush=True)
    if set(scanned) != set(graphs):
        raise ValueError('some requested manifest graphs were not visited; inspect dataset length')
    metadata.update(status='complete', eligible_cases=written, ineligible_cases=skipped,
                    scanned_graph_ids=scanned)
    core.write_json(a.out_dir/'meta.json', metadata)


if __name__ == '__main__':
    main()
