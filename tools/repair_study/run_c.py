#!/usr/bin/env python3
"""Replay selected C natural events; replace local vectors once, then release.

The original batch composition/trimmed sequence length is retained during
replay. No decoded attribute is ever fed back. Healthy matched-lag controls
are separate records. Checkpoint/data/NPZ identities are fixed by select_c.py.
"""
import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--weight', choices=['s42_e143', 's1234_e143'], required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--limit_cases', type=int, help='smoke only: first N fixed cases, no reselection')
    p.add_argument('--device', default='cuda')
    p.add_argument('--ckpt', type=Path, help='path override only; SHA must still match original')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if a.horizon < 2:
        p.error('horizon must be >=2')
    sys.path.insert(0, str(HERE.parent/'b_mechanism'))
    from trajectory_io import configure_runtime
    configure_runtime()
    sys.path.insert(0, str(HERE.parent/'c_capability'))
    import run_capability as R
    import numpy as np
    import torch
    plan = core.read_json(a.cases)
    if plan['protocol'] != 'C_local_repair_v1':
        raise ValueError('unexpected selection protocol')
    source = plan['sources'][a.weight]
    original_meta = source['run_meta']
    ckpt = a.ckpt or Path(original_meta['weight_path'])
    if core.sha(ckpt) != original_meta['weight_sha256']:
        raise ValueError('checkpoint hash mismatch')
    cases = [c for c in plan['cases'] if c['weight'] == a.weight]
    cases.sort(key=lambda c: c['case_id'])
    if a.limit_cases is not None:
        if a.limit_cases < 1:
            p.error('limit_cases must be positive')
        cases = cases[:a.limit_cases]
    by_stratum = defaultdict(list)
    for case in cases:
        by_stratum[case['stratum']].append(case)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    (a.out_dir/'states').mkdir()
    metadata = dict(status='running', experiment='C_local_repair_v1', weight=a.weight,
                    checkpoint_sha256=core.sha(ckpt), cases_sha256=core.sha(a.cases),
                    script_sha256=core.sha(__file__), core_sha256=core.sha(HERE/'core.py'),
                    source=source, args={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                    selected_cases=len(cases))
    core.write_json(a.out_dir/'meta.json', metadata)
    written = 0
    with (a.out_dir/'records.jsonl').open('x', encoding='utf-8') as dst, torch.no_grad():
        for stratum, subset in sorted(by_stratum.items()):
            data_file = Path(plan['data_dir'])/f'{stratum}.pt'
            if any(core.sha(data_file) != c['data_sha256'] for c in subset):
                raise ValueError('dataset hash mismatch')
            cfg = dict(original_meta['args'])
            cfg.update(ckpt=ckpt, weight_name=a.weight, device=a.device)
            args = SimpleNamespace(**cfg)
            ctx = R.build_model_and_data(args, data_file)
            lit, mc, tok, device = (ctx[k] for k in ('litmodel', 'model_config', 'tokenizer', 'device'))
            if not lit.nointerm:
                raise ValueError('continuous nointerm checkpoint/config required')
            by_graph = defaultdict(list)
            for c in subset:
                by_graph[c['graph_index']].append(c)
            for batch_idx, raw in enumerate(ctx['data_module'].val_dataloader()):
                first = batch_idx*args.batch_size
                active = {g: cs for g,cs in by_graph.items() if first <= g < first+len(raw)}
                if not active:
                    continue
                batch = lit.reduce_batch(raw.to(device))
                data, label, info = R.split_batch(batch, mc)
                expected, positions, truth, needed, states = {}, {}, {}, {}, {}
                for gid, cs in active.items():
                    c = cs[0]
                    if core.sha(c['npz_path']) != c['npz_sha256']:
                        raise ValueError('source NPZ hash mismatch')
                    with np.load(c['npz_path'], allow_pickle=False) as z:
                        expected[gid] = np.stack([z['pred_'+f] for f in ('syntax','variable','operation','value')], -1)
                        positions[gid] = z['positions'].tolist()
                        truth[gid] = np.stack([z['true_'+f] for f in ('syntax','variable','operation','value')], -1).tolist()
                    if label[gid-first, positions[gid], :4].cpu().tolist() != truth[gid]:
                        raise ValueError('NPZ labels/positions differ from original dataset')
                    needed[gid] = set()
                    for c in cs:
                        needed[gid].update([c['state_round'], c['donor_round']])
                        if c['control_round'] is not None:
                            needed[gid].update([c['control_round'], c['control_round']-c['donor_gap']])
                    states[gid] = {}
                hidden = lit.model.token_embedder(data)
                for t in range(1, original_meta['rounds']+1):
                    hidden, logits = lit.model(hidden, True)
                    pred = torch.stack(lit.generate_factor_decodes(logits, temperature=0.0), -1)
                    for gid in active:
                        bi = gid-first
                        observed = pred[bi, positions[gid], :4].cpu().numpy()
                        if not np.array_equal(observed, expected[gid][t-1]):
                            core.write_json(a.out_dir/'replay_failure.json', dict(
                                graph_id=gid, stratum=stratum, round=t,
                                note='Original batch replay differs from saved predictions; stop before intervention interpretation.'))
                            raise ValueError('original replay mismatch; inspect replay_failure.json')
                        if t in needed[gid]:
                            states[gid][t] = hidden[bi:bi+1].detach().cpu().clone()
                # A branch contains the same padded length as its original batch.
                def decode(h, pos):
                    heads = [head(h) for head in lit.model.output_layers]
                    return torch.stack(lit.generate_factor_decodes(heads, temperature=0.0), -1)[0, pos, :4].cpu().tolist()
                def continue_from(h, pos):
                    initial = decode(h, pos)
                    outputs = []
                    for _ in range(a.horizon):
                        h, logits = lit.model(h, True)
                        outputs.append(torch.stack(lit.generate_factor_decodes(logits, temperature=0.0), -1)[0,pos,:4].cpu().tolist())
                    return dict(initial=initial, outputs=outputs)
                for gid, cs in active.items():
                    for case in cs:
                        modes = [('natural_error_state', case['state_round'], case['donor_round'])]
                        if case['control_round'] is not None:
                            modes.append(('healthy_matched_lag', case['control_round'], case['control_round']-case['donor_gap']))
                        for timing, recipient, donor in modes:
                            base = states[gid][recipient].to(device)
                            old = states[gid][donor].to(device)
                            pos = positions[gid]
                            target = case['target_pos']
                            parents = case['parent_positions']
                            rec = dict(case, case_id=case['case_id']+':'+timing,
                                       paired_event_id=case['case_id'], timing=timing,
                                       state_round=recipient, donor_round=donor,
                                       node_positions=pos, truth4=truth[gid],
                                       status='eligible', branches={})
                            donor_readout = decode(old, pos)
                            indices = {v: i for i,v in enumerate(pos)}
                            rec['donor_target_first4_correct'] = donor_readout[indices[target]] == truth[gid][indices[target]]
                            rec['donor_parents_first4_correct'] = all(
                                donor_readout[indices[v]] == truth[gid][indices[v]] for v in parents)
                            if a.limit_cases is not None:
                                rec['cohort'] = 'smoke_only'
                            replacements = dict(unchanged=[], sham_current_target=[target],
                                                restore_target=[target], restore_parents=parents)
                            for name, selected in replacements.items():
                                h = base.clone()
                                if selected:
                                    h[:, selected, :] = (base if name == 'sham_current_target' else old)[:, selected, :]
                                rec['branches'][name] = continue_from(h, pos)
                            baseline = rec['branches']['unchanged']['outputs']
                            overlap = min(a.horizon, len(expected[gid])-recipient)
                            if baseline[:overlap] != expected[gid][recipient:recipient+overlap].tolist():
                                raise ValueError('single-graph continuation differs from original batch; stop and inspect')
                            if rec['branches']['sham_current_target'] != rec['branches']['unchanged']:
                                raise ValueError('sham branch differs from unchanged branch')
                            state_file = f'states/{written:04d}.pt'
                            torch.save(dict(base_hidden=base.cpu(), donor_positions=[target]+parents,
                                            donor_vectors=old[:, [target]+parents, :].cpu(),
                                            state_round=recipient, donor_round=donor,
                                            dataset_graph_index=gid, original_batch_index=batch_idx,
                                            data_sha256=case['data_sha256']), a.out_dir/state_file)
                            rec.update(hidden_snapshot=state_file, replay_checked_rounds=original_meta['rounds'],
                                       continuation_checked_rounds=overlap)
                            dst.write(json.dumps(rec)+'\n'); dst.flush()
                            written += 1
                        print(json.dumps(dict(case_id=case['case_id'], written=written)), flush=True)
                del hidden, states
            del ctx, lit
    metadata.update(status='complete', written_records=written)
    core.write_json(a.out_dir/'meta.json', metadata)


if __name__ == '__main__':
    main()
