#!/usr/bin/env python3
"""Evaluate fixed pilot endpoints on NEW validation graphs; never select best epochs."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from collections import defaultdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P
sys.path.insert(0, str(HERE.parent / 'b_mechanism'))
from trajectory_io import configure_runtime


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def endpoints(study, data_meta):
    """Require matched data, initialization, batching and optimizer settings."""
    models, configs = [], []
    for arm in P.ARMS:
        meta = P.read_json(study / arm / 'meta.json')
        if meta['status'] != 'complete' or meta['cohort'] != 'main' or meta['arm'] != arm:
            raise ValueError(f'{arm}: main training incomplete')
        if meta['source_sha256'] != P.SOURCE_SHA or meta['data_manifest'] != data_meta:
            raise ValueError('unmatched source/data across arms')
        configs.append((meta['batch_plan'], meta['config_sha256'], meta['optimizer'], meta['args']['seed']))
        cdir = P.ROOT / 'experiments/configs' / P.CONFIG
        if any(P.sha(cdir / name) != h for name, h in meta['config_sha256'].items()):
            raise ValueError('current config differs from frozen training config')
        for name in ('same_exposure', 'equal_compute'):
            saved = meta['checkpoints'][name]
            path = Path(saved['path'])
            if P.sha(path) != saved['sha256']:
                raise ValueError('checkpoint changed')
            models.append(dict(arm=arm, endpoint=name, **saved))
    if not all(c == configs[0] for c in configs):
        raise ValueError('unmatched training batch plan/config/optimizer/seed')
    for field, name in [('graph_presentations', 'same_exposure'),
                        ('optimizer_steps', 'same_exposure'),
                        ('sample_forwards', 'equal_compute'),
                        ('token_forwards', 'equal_compute'),
                        ('attention_units', 'equal_compute')]:
        if len({m[field] for m in models if m['endpoint'] == name}) != 1:
            raise ValueError(f'{name}: {field} not matched')
    return models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path, default=P.STUDY)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--source-ckpt', type=Path)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('positive batch size required')
    configure_runtime()
    sys.path.insert(0, str(HERE.parent / 'c_capability'))
    import run_capability as R
    import torch
    import numpy as np

    data_meta = P.read_json(a.study_dir / 'data/manifest.json')
    file = a.study_dir / 'data/val.pt'
    if data_meta['status'] != 'complete' or P.sha(file) != data_meta['splits']['val']['sha256']:
        raise ValueError('validation identity mismatch')
    models = endpoints(a.study_dir, data_meta)
    ckpt = a.source_ckpt or P.ROOT / P.source_spec()['weight_path']
    if P.sha(ckpt) != P.SOURCE_SHA:
        raise ValueError('wrong initial checkpoint')
    models.insert(0, dict(arm='initial', endpoint='initial', path=str(ckpt), sha256=P.SOURCE_SHA, epoch=0))
    a.out_dir.mkdir(parents=True, exist_ok=False)
    P.write_json(a.out_dir / 'meta.json', dict(status='running', checkpoints=models,
        data_sha256=P.sha(file), script_sha256=P.sha(__file__), protocol_sha256=P.sha(HERE / 'protocol.py')))
    all_curves, all_events = [], []
    # plus_2d's epoch2 is both endpoints: evaluate once, reuse identical result.
    cache, raw_dirs = {}, {}
    for spec in models:
        name = f"{spec['arm']}_{spec['endpoint']}"
        cache_key = (spec['arm'], spec['epoch'])
        if cache_key in cache:
            curves, events = cache[cache_key]
            all_curves.extend(dict(r, endpoint=spec['endpoint']) for r in curves)
            all_events.extend(dict(r, endpoint=spec['endpoint']) for r in events)
            continue
        cfg = dict(config_dir_name=P.CONFIG, ckpt=Path(spec['path']), weight_name=name,
                   device=a.device, batch_size=a.batch_size, num_workers=0, compile=False)
        ctx = R.build_model_and_data(SimpleNamespace(**cfg), file)
        lit, dm, device, tok = (ctx[k] for k in ('litmodel', 'data_module', 'device', 'tokenizer'))
        lit.eval(); curves, events = [], []
        raw_dir = a.out_dir / name; raw_dir.mkdir()
        raw_dirs[cache_key] = raw_dir
        gid = 0
        with torch.no_grad():
            for raw in dm.val_dataloader():
                raw = lit.reduce_batch(raw.to(device))
                inputs, labels, info = R.split_batch(raw, ctx['model_config'])
                ds = (info[..., 0].max(1).values + 1).cpu().tolist()
                valid = (inputs[..., 0] != tok.syntax_tok2idx[tok.PAD]) & (info[..., 0] >= 0) & (labels[..., 3] < tok.mod_val)
                positions = [torch.nonzero(m, as_tuple=True)[0] for m in valid]
                truth = [labels[i, pos, :4].cpu().numpy() for i, pos in enumerate(positions)]
                saved = [[] for _ in positions]
                hidden = lit.model.token_embedder(inputs)
                for t in range(1, max(ds) + 65):
                    hidden, logits = lit.model(hidden, True)
                    pred = torch.stack([x.argmax(-1) for x in logits], -1)
                    for i, pos in enumerate(positions):
                        if t <= ds[i] + 64:
                            saved[i].append(pred[i, pos, :4].cpu().numpy())
                for i, (seq, gold, d) in enumerate(zip(saved, truth, ds)):
                    values = np.stack(seq)
                    v = values[..., 3] == gold[None, :, 3]
                    f = (values == gold[None, :, :]).all(-1)
                    if len(gold) != 32:
                        raise ValueError('pilot validation requires 32 numeric nodes')
                    identity = dict(arm=spec['arm'], endpoint=spec['endpoint'], graph_id=gid, D=d)
                    np.savez_compressed(raw_dir / f'graph_{gid:04d}.npz', predictions=values, truth4=gold,
                                        positions=positions[i].cpu().numpy(), D=d)
                    for metric, correct in [('value', v), ('first4', f)]:
                        flags = correct.all(1).tolist()
                        event = P.post_events(flags)
                        for t in range(1, len(correct) + 1):
                            curves.append(dict(**identity, metric=metric, round=t,
                                               relative_to_D=t-d, correct=int(correct[t-1].sum()),
                                               n=len(gold), graph_all=bool(flags[t-1])))
                        events.append(dict(**identity, metric=metric, **event,
                            correct_at_D=bool(flags[d-1]),
                            ever_wrong_after_D=int((~correct[d:]).any(0).sum()),
                            final_two_correct=int(correct[-2:].all(0).sum()),
                            final_two_graph=bool(all(flags[-2:])), n=len(gold)))
                    gid += 1
        if gid != data_meta['splits']['val']['count']:
            raise ValueError('validation graph count mismatch')
        cache[cache_key] = (curves, events)
        all_curves.extend(curves); all_events.extend(events)
        print(json.dumps(dict(model=name, graphs=gid)), flush=True)
        del ctx, lit, hidden
    write_csv(a.out_dir / 'graph_curves.csv', all_curves)
    write_csv(a.out_dir / 'events.csv', all_events)
    groups = defaultdict(list)
    for r in all_curves:
        # Same graph cohort at D+offset, and fixed absolute rounds separately.
        marks = [mark for mark, t in P.evaluation_points(r['D']).items() if t == r['round']]
        for mark in marks:
            groups[(r['arm'], r['endpoint'], r['metric'], mark)].append(r)
    summary = []
    for key, rs in sorted(groups.items()):
        n = sum(r['n'] for r in rs); c = sum(r['correct'] for r in rs)
        summary.append(dict(zip(['arm','endpoint','metric','time'],key),
            graphs=len(rs), correct=c, n=n, node_acc=c/n,
            node_acc_graph_equal=sum(r['correct']/r['n'] for r in rs)/len(rs),
            graph_all=sum(r['graph_all'] for r in rs),
            graph_all_rate=sum(r['graph_all'] for r in rs)/len(rs)))
    write_csv(a.out_dir / 'summary.csv', summary)
    eg = defaultdict(list)
    for r in all_events: eg[(r['arm'],r['endpoint'],r['metric'])].append(r)
    er = []
    for key, rs in sorted(eg.items()):
        completed = [r for r in rs if r['completion'] is not None]
        observed = [r for r in completed if r['post_observed']]
        er.append(dict(zip(['arm','endpoint','metric'],key), graphs=len(rs),
            completed=len(completed), post_observed=len(observed),
            degraded=sum(r['ever_bad_after'] for r in observed),
            ever_recovered=sum(r['ever_recovered'] for r in observed),
            reerror=sum(r['reerror'] for r in observed),
            final_two_graph=sum(r['final_two_graph'] for r in rs)))
    write_csv(a.out_dir / 'event_summary.csv', er)
    paired = []
    model_index = {(m['arm'],m['endpoint']):m for m in models}
    for spec in models[1:]:
        refs = [models[0]]
        if spec['arm'] != 'base':
            refs.append(model_index[('base', spec['endpoint'])])
        for ref in refs:
            here = raw_dirs[(spec['arm'],spec['epoch'])]
            there = raw_dirs[(ref['arm'],ref['epoch'])]
            for path in sorted(here.glob('graph_*.npz')):
                with np.load(path, allow_pickle=False) as z, np.load(there / path.name, allow_pickle=False) as b:
                    if not np.array_equal(z['positions'],b['positions']) or not np.array_equal(z['truth4'],b['truth4']):
                        raise ValueError('unmatched graph in paired evaluation')
                    for mark,t in P.evaluation_points(int(z['D'])).items():
                        gold=z['truth4']; new=z['predictions'][t-1]; old=b['predictions'][t-1]
                        for metric in ('value','first4'):
                            nc=(new[:,3]==gold[:,3]) if metric=='value' else (new==gold).all(-1)
                            bc=(old[:,3]==gold[:,3]) if metric=='value' else (old==gold).all(-1)
                            paired.append(dict(arm=spec['arm'],endpoint=spec['endpoint'],reference=ref['arm'],
                                graph_id=int(path.stem.split('_')[-1]),metric=metric,time=mark,n=len(gold),
                                correct=int(nc.sum()),reference_correct=int(bc.sum()),
                                helped=int((nc & ~bc).sum()),harmed=int((~nc & bc).sum())))
    write_csv(a.out_dir / 'paired_cases.csv', paired)
    pg = defaultdict(list)
    for r in paired: pg[(r['arm'],r['endpoint'],r['reference'],r['metric'],r['time'])].append(r)
    ps = []
    for key,rs in sorted(pg.items()):
        row=dict(zip(['arm','endpoint','reference','metric','time'],key),graphs=len(rs),n=sum(r['n'] for r in rs))
        for field in ('correct','reference_correct','helped','harmed'):
            row[field]=sum(r[field] for r in rs)
            row[field+'_graph_equal']=sum(r[field]/r['n'] for r in rs)/len(rs)
        ps.append(row)
    write_csv(a.out_dir / 'paired_summary.csv', ps)
    budget_fields=['arm','endpoint','epoch','optimizer_steps','graph_presentations','sample_forwards',
                   'token_forwards','attention_units','wall_seconds','sha256']
    write_csv(a.out_dir / 'budget.csv', [{k:m.get(k) for k in budget_fields} for m in models])
    P.write_json(a.out_dir / 'meta.json', dict(status='complete', checkpoints=models,
        data_sha256=P.sha(file), script_sha256=P.sha(__file__), protocol_sha256=P.sha(HERE / 'protocol.py'),
        note='D-based fixed cohort is primary; post-completion events use explicit conditional denominators. No automatic best-checkpoint selection.'))


if __name__ == '__main__':
    main()
