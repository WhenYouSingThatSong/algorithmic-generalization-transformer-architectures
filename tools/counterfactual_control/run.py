#!/usr/bin/env python3
"""Matched a->a / a->a+delta continuous swaps on six fixed data strata.

New paired experiment, not an append to legacy C4 percentages. Changes two
leaf constants in donor INPUTS, but swaps only one hidden row in the recipient.
No decoded state is fed back. Every structural/model rejection is retained.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import design
sys.path.insert(0, str(HERE.parent/'repair_study'))
import core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, default=Path('tmp/analysis/c0-capability-20261114/runs'))
    p.add_argument('--data_dir', type=Path, default=Path('tmp/analysis/c0-capability-20261114/data'))
    p.add_argument('--weight', choices=['s42_e143','s1234_e143'], required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--strata', nargs='+', default=['N32_shallow','N32_deep','N64_shallow','N64_deep','N128_shallow','N128_deep'])
    p.add_argument('--n_graphs', type=int, default=16, help='fixed first N per stratum; no refill')
    p.add_argument('--per_graph', type=int, default=2)
    p.add_argument('--deltas', type=int, nargs='+', default=[1,3])
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--ckpt', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if min(a.n_graphs,a.per_graph,a.horizon) < 1:
        p.error('positive sample/window arguments required')
    design.matched_changes({0:1,1:1}, a.deltas)  # validate deltas before runtime
    sys.path.insert(0, str(HERE.parent/'b_mechanism'))
    from trajectory_io import configure_runtime
    configure_runtime()
    sys.path.insert(0, str(HERE.parent/'c_capability'))
    import run_capability as R
    import torch

    source_dir = a.run_dir/a.weight
    source = core.read_json(source_dir/'run_meta.json')
    if source['status'] != 'complete':
        raise ValueError('source capability run is not complete')
    index = [json.loads(s) for s in (source_dir/'index.jsonl').read_text(encoding='utf-8').splitlines()]
    ckpt = a.ckpt or Path(source['weight_path'])
    if core.sha(ckpt) != source['weight_sha256']:
        raise ValueError('checkpoint hash mismatch')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    meta = dict(protocol='matched_counterfactual_v1', status='running',
                weight=a.weight, checkpoint_sha256=core.sha(ckpt), source_run=source,
                source_index_sha256=core.sha(source_dir/'index.jsonl'),
                code_sha256={f:core.sha(HERE/f) for f in ['run.py','design.py']},
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                note='New batch-size-1 graph extraction; donor batch and branch batch fixed across conditions. Not a replay of old C4.')
    core.write_json(a.out_dir/'meta.json', meta)
    counts = dict(graphs=0, candidates=0, eligible=0, no_candidate_graphs=0)
    with (a.out_dir/'records.jsonl').open('x',encoding='utf-8') as out, \
         (a.out_dir/'selection.jsonl').open('x',encoding='utf-8') as sel, torch.no_grad():
        for stratum in a.strata:
            data_file = a.data_dir/f'{stratum}.pt'
            data_sha = core.sha(data_file)
            ix = [r for r in index if r['stratum']==stratum]
            if not ix or any(r['data_sha256'] != data_sha for r in ix):
                raise ValueError(f'source dataset mismatch: {stratum}')
            cfg = dict(source['args'])
            cfg.update(ckpt=ckpt, weight_name=a.weight, device=a.device,
                       batch_size=1, num_workers=0, compile=False)
            ctx = R.build_model_and_data(SimpleNamespace(**cfg), data_file)
            lit, tok, mc, device = (ctx[k] for k in ['litmodel','tokenizer','model_config','device'])
            if not lit.nointerm or tok.mod_val != 23 or tok.operation_tok2idx.get('ADD') != 0:
                raise ValueError('requires continuous ADD mod23 model')
            cfg_dir = R.EXPERIMENTS/'configs'/cfg['config_dir_name']
            meta['config_sha256'] = {n:core.sha(cfg_dir/n) for n in ['model_config.yaml','train_config.yaml','data_config.yaml']}
            graph_count = 0
            for gid, raw in enumerate(ctx['data_module'].val_dataloader()):
                if gid >= a.n_graphs:
                    break
                graph_count += 1; counts['graphs'] += 1
                data, label, info = R.split_batch(lit.reduce_batch(raw.to(device)), mc)
                inp, lab, inf = data[0].cpu().tolist(), label[0].cpu().tolist(), info[0].cpu().tolist()
                pos = [i for i,row in enumerate(inf) if row[0]>=0 and 0<=lab[i][3]<23
                       and inp[i][0]!=tok.syntax_tok2idx[tok.PAD]]
                depths = {u:int(inf[u][0]) for u in pos}
                parents, leaf_constants, leaves = {}, {}, {}
                for u in pos:
                    ps = [int(v) for v in inf[u][1::2] if v != -1]
                    ops = [int(v) for v in inf[u][2::2] if v != -1]
                    if len(ops) != len(ps)-1 or any(v != 0 for v in ops):
                        raise ValueError('unsupported expression structure/operator')
                    if depths[u]==0:
                        if len(ps)!=1 or not 0<=inp[ps[0]][3]<23 or ps[0] in pos:
                            raise ValueError('leaf must reference a numeric input constant')
                        leaf_constants[u] = ps[0]; leaves[u] = inp[ps[0]][3]; parents[u] = []
                    else:
                        if any(v not in depths for v in ps):
                            raise ValueError('internal parent is not a numeric variable')
                        parents[u] = ps
                if len(set(leaf_constants.values())) != len(leaf_constants):
                    raise ValueError('shared leaf constant positions require separate construction')
                truth = design.evaluate(parents,depths,leaves)
                if any(truth[u]!=lab[u][3] for u in pos):
                    raise ValueError('recomputed DAG truth differs from labels')
                graph_key = f'{stratum}:{gid}'
                selected, coverage = design.choose_candidates(parents,depths,leaves,graph_key,
                    limit=a.per_graph,deltas=a.deltas,seed=a.seed)
                sel.write(json.dumps(dict(graph_id=graph_key,data_sha256=data_sha,nodes=coverage))+'\n'); sel.flush()
                if not selected:
                    counts['no_candidate_graphs'] += 1
                truth4 = [lab[u][:4] for u in pos]
                for candidate in selected:
                    counts['candidates'] += 1
                    u = candidate['target_pos']; ui = pos.index(u)
                    names = ['original',*candidate['changes']]
                    inputs = [data.clone()]
                    for name, edits in candidate['changes'].items():
                        x = data.clone()
                        for leaf,delta in edits.items():
                            x[0,leaf_constants[leaf],3] = (leaves[leaf]+delta)%23
                        inputs.append(x)
                    hidden = lit.model.token_embedder(torch.cat(inputs,dim=0))
                    for _ in range(candidate['state_round']):
                        hidden, logits = lit.model(hidden,True)
                    source_pred = torch.stack(lit.generate_factor_decodes(logits,temperature=0.0),-1)
                    source_initial = {name:source_pred[i,pos,:4].cpu().tolist() for i,name in enumerate(names)}
                    branches = ['unchanged','sham','same',*[f'shift_{d}' for d in a.deltas]]
                    branch_states = []
                    distance = {}
                    for name in branches:
                        h = hidden[0:1].clone()
                        if name != 'unchanged':
                            donor_i = 0 if name=='sham' else names.index(name)
                            h[:,u,:] = hidden[donor_i:donor_i+1,u,:]
                        branch_states.append(h)
                        v,base = h[0,u].float(),hidden[0,u].float()
                        distance[name] = dict(l2=float((v-base).norm()),
                            cosine=float(torch.dot(v,base)/(v.norm()*base.norm()).clamp_min(1e-12)),
                            exactly_equal=bool(torch.equal(h[0,u],hidden[0,u])))
                    # Full donor states are reference runs with their OWN truth, not local interventions.
                    refnames = ['full_'+n for n in names[1:]]
                    states = torch.cat(branch_states+[hidden[i:i+1].clone() for i in range(1,len(names))],dim=0)
                    allnames = branches+refnames
                    def decode_heads(h):
                        logits = [head(h) for head in lit.model.output_layers]
                        return torch.stack(lit.generate_factor_decodes(logits,temperature=0.0),-1)
                    initial = decode_heads(states)
                    traces = {name:dict(initial=initial[i,pos,:4].cpu().tolist(),outputs=[])
                              for i,name in enumerate(allnames)}
                    for _ in range(a.horizon):
                        states, logits = lit.model(states,True)
                        pred = torch.stack(lit.generate_factor_decodes(logits,temperature=0.0),-1)
                        for i,name in enumerate(allnames):
                            traces[name]['outputs'].append(pred[i,pos,:4].cpu().tolist())
                    if traces['sham'] != traces['unchanged']:
                        raise ValueError('identical hidden sham diverged from unchanged')
                    for name in names[1:]:
                        if traces[name]['initial'][ui] != source_initial[name][ui]:
                            raise ValueError('hidden row copy did not preserve donor immediate readout')
                    flags, qs = design.qualify(truth4,pos,parents,candidate,source_initial,
                                              traces['unchanged']['outputs'][0])
                    eligible = bool(qs) and all(flags.values())
                    counts['eligible'] += int(eligible)
                    rec = dict(candidate, case_id=f'{graph_key}:u{u}', graph_id=graph_key,
                        stratum=stratum, weight=a.weight, data_sha256=data_sha,
                        modulus=23, deltas=a.deltas, node_positions=pos,truth4=truth4,
                        parents=parents,depths=depths,leaf_constants=leaf_constants,
                        source_initial=source_initial,source_qualification=flags,
                        eligible_children=qs,status='eligible' if eligible else 'ineligible',
                        same_vector_equal=distance['same']['exactly_equal'],vector_distance=distance,
                        branches={n:traces[n] for n in branches},
                        references={n:traces[n] for n in refnames})
                    out.write(json.dumps(rec)+'\n'); out.flush()
                    print(json.dumps(dict(case_id=rec['case_id'],status=rec['status'],q=len(qs))),flush=True)
            if graph_count != a.n_graphs:
                raise ValueError(f'{stratum}: fewer than requested fixed graph count; no refill')
            del ctx, lit
    meta.update(status='complete',counts=counts)
    core.write_json(a.out_dir/'meta.json',meta)


if __name__=='__main__':
    main()
