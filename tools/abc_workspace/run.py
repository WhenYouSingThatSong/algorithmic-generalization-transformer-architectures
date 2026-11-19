"""Steps 1--3 only: ABC reassignment, bounded workspace programs, diagnostic forks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P
from runtime import Runtime
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'priority_pilots'))
from io_utils import sha, read_json, write_json


def source_hashes():
    files = list(HERE.glob('*.py')) + [ROOT/'experiments'/n for n in ('model.py','tokenizers.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--mode', choices=('assignment','program'), required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--per-stratum', type=int, default=8)
    p.add_argument('--stages', type=int, default=20)
    p.add_argument('--slack', type=int, default=8)
    p.add_argument('--probe-per-stratum', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if min(a.per_stratum, a.stages) < 1 or a.slack < 2 or a.probe_per_stratum < 0:
        p.error('invalid budget')
    data = read_json(a.tasks); registry = read_json(a.models)
    if data['protocol'] != P.VERSION or registry['protocol'] != P.VERSION: raise ValueError('wrong manifest')
    cases = [c for c in data['cases'] if c['ordinal'] < a.per_stratum]
    if len(cases) != len(P.STRATA)*a.per_stratum: raise ValueError('fixed candidates missing')
    spec = next(m for m in registry['models'] if m['id'] == a.model)
    if sha(spec['path']) != spec['sha256']: raise ValueError('weight hash mismatch')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    rt = Runtime(spec, a.device)
    meta = dict(protocol=P.VERSION, status='running', mode=a.mode, model=spec,
                task_sha256=sha(a.tasks), registry_sha256=sha(a.models), code_hashes=source_hashes(),
                commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                cases=[c['id'] for c in cases], per_stratum=a.per_stratum, stages=a.stages,
                slack=a.slack, probe_per_stratum=a.probe_per_stratum, physical_nodes=32, padded_tokens=256,
                interface='C hidden feedback within/across assignment rounds; program numeric output commits at frame boundaries for ALL families; C hybrid, not latent-only',
                reset_semantics='A/B VALUE EMPTY; C whole result hidden row reinitialized from token embedding',
                snapshot_note='Only fixed diagnostic prefix snapshots; no outcome-dependent snapshot selection')
    config_dir=ROOT/'experiments/configs'/spec['config']
    meta['config_hashes']={str(x.relative_to(ROOT)):sha(x) for x in config_dir.glob('*.yaml')}
    write_json(a.out_dir/'meta.json', meta)
    records = 0
    out = (a.out_dir/'records.jsonl').open('x', encoding='utf-8')

    def emit(case, stage, branch, layout, raw, truth, trace, **extra):
        nonlocal records
        pos = layout['nodes']; terminal = [pos[i] for i in extra.pop('output_slots')]
        gold = rt.truth(raw, pos, truth)
        metrics=P.measures(trace['outputs'], gold, pos, terminal)
        metrics['task_first4_n']=sum(all(z[i]==gold[i] for z in trace['outputs'][-2:]) for i in layout['task_rows'])
        metrics['task_n']=len(layout['task_rows'])
        row = dict(case=case['id'], stratum=case['stratum'], ordinal=case['ordinal'],
                   model=a.model, family=spec['family'], mode=a.mode, stage=stage, branch=branch,
                   status='complete', nodes=pos, terminals=terminal,
                   truth4=gold[:len(layout['tokens'])], task_tokens=layout['tokens'],
                   metrics=metrics, trace=trace, **extra)
        out.write(json.dumps(row)+'\n'); out.flush(); records += 1

    def diagnose(case, stage, state, raw, layout, truth, conditional, f):
        # Fixed slots / stages, never chosen by observed failure or desired rescue.
        if case['ordinal'] >= a.probe_per_stratum or stage not in (1,4,10,20): return
        rounds = max(P.evaluate(f['parents'], f['constants'])[1])+1+a.slack
        starts = {'continue': state.clone(),
                  'refresh_task': rt.write_rows(state, raw, layout['task_rows']),
                  'reset_results': rt.reset_results(state, raw, layout['nodes']),
                  'refresh_and_reset': rt.reset_results(rt.write_rows(state, raw, layout['task_rows']), raw, layout['nodes']),
                  'scratch_same_input': rt.init(raw)}
        # Exact parent state and input permit replay; main trajectory is never replaced.
        snap = a.out_dir/f"snapshot-{case['id'].replace(':','-')}-{stage}.pt"
        rt.torch.save(dict(state=state.cpu(), raw=raw.cpu(), nodes=layout['nodes'],
                           task_rows=layout['task_rows'], model_sha256=spec['sha256']), snap)
        for name, start in starts.items():
            _, trace = rt.run(start, rounds, len(layout['tokens']))
            conditional_gold = rt.truth(raw, layout['nodes'], conditional)
            emit(case, stage, 'probe_'+name, layout, raw, truth, trace, output_slots=f['outputs'],
                 diagnostic=True, rounds=rounds, snapshot=snap.name, snapshot_sha256=sha(snap),
                 conditional_metrics=P.measures(trace['outputs'], conditional_gold, layout['nodes'],
                                                [layout['nodes'][i] for i in f['outputs']]))

    with rt.torch.no_grad():
        for case in cases:
            if a.mode == 'assignment':
                f = P.frame(case, 0); original = f['constants']; truth0, depth = P.evaluate(f['parents'], original)
                layout = P.render(f, case['names'], original); raw0 = rt.tokens(layout)
                rounds = max(depth)+1+a.slack
                base, baseline = rt.run(rt.init(raw0), rounds, len(layout['tokens']))
                base_gold = rt.truth(raw0, layout['nodes'], truth0)
                eligibility = P.measures(baseline['outputs'], base_gold, layout['nodes'],
                                         [layout['nodes'][i] for i in f['outputs']])
                # Old-task error control: discrete VALUE edit or continuous counterfactual row.
                target = P.LEAVES
                donor_inputs = list(original); donor_inputs[0] = (donor_inputs[0]+1)%23
                donor_raw = rt.tokens(P.render(f, case['names'], donor_inputs))
                fault = base.clone(); target_pos = layout['nodes'][target]
                if rt.continuous:
                    donor, _ = rt.run(rt.init(donor_raw), rounds, len(layout['tokens']))
                    fault[:,target_pos] = donor[:,target_pos]
                else:
                    fault[:,target_pos,3] = rt.tok.value_tok2idx[str((truth0[target]+1)%23)]
                for name, state in [('old_clean',base), ('old_corrupt',fault),
                                    ('old_same_write',rt.write_rows(base,raw0,[layout['constants'][0]]))]:
                    _, tr = rt.run(state, rounds, len(layout['tokens']))
                    emit(case, 0, name, layout, raw0, truth0, tr, output_slots=f['outputs'],
                         diagnostic=False, rounds=rounds, baseline=eligibility, parents=f['parents'],
                         injection_target=target_pos, intended_fault=(truth0[target]+1)%23,
                         actual_input_value=tr['inputs'][0][target_pos][3])
                states = {name:base.clone() for name in P.BRANCHES}
                control = base.clone(); sham = base.clone(); previous = list(original)
                for stage, change in enumerate(P.update_inputs(original), 1):
                    values, _ = P.evaluate(f['parents'], change['inputs'])
                    lay = P.render(f,case['names'],change['inputs']); raw = rt.tokens(lay)
                    edit_rows = [lay['constants'][i] for i in change['changed_leaves']]
                    affected = P.affected_nodes(f['parents'], change['changed_leaves'])
                    for name in P.BRANCHES:
                        if name == 'scratch': start = rt.init(raw)
                        else:
                            start = rt.write_rows(states[name],raw,edit_rows)
                            if name == 'reset_results':
                                start = rt.reset_results(start,raw,[lay['nodes'][i] for i in affected])
                        states[name], tr = rt.run(start, rounds, len(lay['tokens']))
                        emit(case,stage,name,lay,raw,values,tr,output_slots=f['outputs'],
                             diagnostic=False,rounds=rounds,baseline=eligibility,parents=f['parents'],
                             boundary_inputs=change['inputs'],previous_inputs=previous,
                             changed_task_rows=edit_rows,reset_slots=affected if name=='reset_results' else [],
                             cumulative_main_forwards=rounds*(stage+1) if name!='scratch' else rounds*stage)
                    control, tr = rt.run(control,rounds,len(layout['tokens']))
                    emit(case,stage,'unchanged',layout,raw0,truth0,tr,output_slots=f['outputs'],
                         diagnostic=False,rounds=rounds,baseline=eligibility)
                    sham = rt.write_rows(sham,raw0,edit_rows)
                    sham, tr = rt.run(sham,rounds,len(layout['tokens']))
                    emit(case,stage,'same_input_write',layout,raw0,truth0,tr,output_slots=f['outputs'],
                         diagnostic=False,rounds=rounds,baseline=eligibility)
                    diagnose(case,stage,states['carry'],raw,lay,values,values,f)
                    previous = list(change['inputs'])
            else:
                refs = P.reference_program(case,a.stages)  # scoring only
                states = {b:None for b in P.BRANCHES}; commits = {b:None for b in P.BRANCHES}
                old_raw = {}; ended = {}; costs = {b:0 for b in P.BRANCHES}
                for s in range(a.stages):
                    f = P.frame(case,s); rounds = refs[s]['local_depth']+1+a.slack
                    for name in P.BRANCHES:
                        inputs = None if name in ended else P.emitted_inputs(f,commits[name])
                        if inputs is None:
                            ended.setdefault(name,s+1)
                            out.write(json.dumps(dict(case=case['id'],stratum=case['stratum'],ordinal=case['ordinal'],
                                model=a.model,family=spec['family'],mode=a.mode,stage=s+1,branch=name,
                                status='invalid_commit',first_blocked_stage=ended[name],diagnostic=False,
                                reference=refs[s]))+'\n');out.flush();records+=1
                            continue
                        lay = P.render(f,case['names'],inputs); raw = rt.tokens(lay)
                        conditional, _ = P.evaluate(f['parents'],inputs)
                        edited = lay['task_rows'] if states[name] is None else [i for i in lay['task_rows']
                                  if not rt.torch.equal(raw[:,i],old_raw[name][:,i])]
                        if states[name] is None or name=='scratch': start=rt.init(raw)
                        else:
                            start=rt.write_rows(states[name],raw,edited)
                            if name=='reset_results':start=rt.reset_results(start,raw,lay['nodes'])
                        states[name], tr = rt.run(start,rounds,len(lay['tokens']));costs[name]+=rounds
                        commits[name]=rt.numeric_commit(states[name],[lay['nodes'][i] for i in f['outputs']])
                        conditional_gold=rt.truth(raw,lay['nodes'],conditional)
                        emit(case,s+1,name,lay,raw,refs[s]['truth'],tr,output_slots=f['outputs'],
                             diagnostic=False,rounds=rounds,parents=f['parents'],boundary_inputs=inputs,
                             actual_commit=commits[name],reference=refs[s],changed_task_rows=edited,
                             cumulative_main_forwards=costs[name],decoded_boundary_values=len(commits[name]),
                             conditional_metrics=P.measures(tr['outputs'],conditional_gold,lay['nodes'],
                                                            [lay['nodes'][i] for i in f['outputs']]))
                        if name=='carry':diagnose(case,s+1,states[name],raw,lay,refs[s]['truth'],conditional,f)
                        old_raw[name]=raw
            print(json.dumps(dict(case=case['id'],model=a.model,mode=a.mode,records=records)),flush=True)
    out.close()
    if source_hashes()!=meta['code_hashes']:raise ValueError('source changed during execution')
    if any(sha(ROOT/n)!=h for n,h in meta['config_hashes'].items()):raise ValueError('config changed during execution')
    meta.update(status='complete',records=records,actual_total_forwards=rt.forwards,
                records_sha256=sha(a.out_dir/'records.jsonl'))
    write_json(a.out_dir/'meta.json',meta)


if __name__ == '__main__': main()
