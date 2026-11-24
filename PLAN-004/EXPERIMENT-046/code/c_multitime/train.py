"""C multitime local repair with layer-2 process supervision; separate run identity."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec as S
P = S.P


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--micro-batch', type=int, default=32)
    p.add_argument('--donor-batch', type=int, default=32)
    p.add_argument('--num-threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true', help='six maximum-depth steps: each mode clean/corrupt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--acceptance', type=Path, help='server verification + smoke receipt required for main')
    a = p.parse_args()
    if a.micro_batch not in (8, 16, 32) or a.donor_batch < 1 or a.num_threads < 1:
        p.error('micro-batch must be 8/16/32; donor-batch/num-threads positive')
    if a.smoke and a.resume: p.error('smoke uses a fresh output directory')
    plan = S.validate_plan(a.plan)
    code_hashes = S.hashes()
    # Import the old evaluator before Runtime changes sys.path; do not modify it.
    from evaluate import small_check
    from runtime import Runtime, load_data, dataset_graphs, setup
    setup()
    import torch
    import numpy as np
    from tensor_targets import Topology
    from multitime_core import recurrent_loss
    from process_loss import materialize
    torch.set_num_threads(a.num_threads)
    seed = P.seed(S.VERSION, 'training', 'C_s2026') % (2**32)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    source = dict(next(m for m in plan['models'] if m['id']=='C_s2026'), arm='update')
    identity = dict(recipe_id=S.VERSION, display_name=S.DISPLAY, recipe=S.RECIPE,
                    model='C_s2026', arm='update', source=source,
                    plan_sha256=P.sha(a.plan), code_hashes=code_hashes,
                    micro_batch=a.micro_batch, donor_batch=a.donor_batch,
                    cohort='smoke' if a.smoke else 'main')
    if not a.smoke:
        if a.acceptance is None: p.error('--acceptance required after server verification/smoke')
        receipt = P.read(a.acceptance)
        expected = {k: identity[k] for k in ('plan_sha256','code_hashes','micro_batch','donor_batch')}
        if receipt.get('status')!='passed' or any(receipt.get(k)!=v for k,v in expected.items()):
            raise ValueError('acceptance belongs to different code/plan/batch configuration')
    if not a.resume:
        a.out_dir.mkdir(parents=True, exist_ok=False)
    elif not a.out_dir.exists():
        raise ValueError('resume directory absent')
    rt = Runtime(source, a.device, initialize_update=True)
    x, y, info = load_data(plan['train']['path'], rt.tok)
    gs = dataset_graphs(y, info); ds = [g['D'] for g in gs]
    if len(gs)!=50000 or any(len(g['nodes'])!=32 or g['D']>7 for g in gs):
        raise ValueError('requires original 50k N32 depth<=6 data')
    topology = Topology(gs)
    del info
    heads = list(rt.model.output_layers[4].parameters()); head_ids = {id(v) for v in heads}
    opt = torch.optim.AdamW([
        dict(params=[v for v in rt.model.parameters() if id(v) not in head_ids], lr=1e-4, initial_lr=1e-4),
        dict(params=heads, lr=5e-4, initial_lr=5e-4)], betas=(.9,.999), eps=1e-8, weight_decay=.01)
    per_epoch = len(P.batches(ds, 1)); step = 0; presentations = 0
    start_epoch, start_batch, best = 1, 0, None
    if a.resume:
        meta = P.read(a.out_dir/'meta.json')
        if meta['identity']!=identity or meta['status']=='complete': raise ValueError('resume identity/status mismatch')
        ck = torch.load(a.out_dir/'last.ckpt', map_location='cpu', weights_only=False)
        if ck['identity']!=identity: raise ValueError('checkpoint identity mismatch')
        rt.model.load_state_dict(ck['state_dict'], strict=True); opt.load_state_dict(ck['optimizer_state'])
        step=ck['step']; presentations=ck['presentations']; best=ck['best']
        start_epoch=ck['next_epoch']; start_batch=ck['next_batch']
        random.setstate(ck['rng']['python']); np.random.set_state(ck['rng']['numpy'])
        torch.set_rng_state(ck['rng']['torch'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        for name in ('steps.jsonl','injections.jsonl'):
            path = a.out_dir/name
            if path.exists():
                keep=[]
                for line in path.read_text(encoding='utf-8').splitlines():
                    try: row=json.loads(line)
                    except json.JSONDecodeError: break
                    if row['step']<=step: keep.append(line)
                path.write_text(''.join(z+'\n' for z in keep), encoding='utf-8')
    else:
        P.write(a.out_dir/'meta.json', dict(status='running', identity=identity, epochs=50,
                                         batches_per_epoch=per_epoch, seed=seed,
                                         torch_version=torch.__version__, device=str(rt.device)))

    def phase_event(phase, seconds, **extra):
        with (a.out_dir/'timings.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(phase=phase, seconds=seconds, step=step, **extra))+'\n')

    def save(name, next_epoch, next_batch):
        begin=time.perf_counter()
        dest=a.out_dir/(name+'.ckpt'); tmp=dest.with_suffix('.tmp')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in rt.model.state_dict().items()},
                        optimizer_state=opt.state_dict(), identity=identity,
                        step=step, presentations=presentations, next_epoch=next_epoch,
                        next_batch=next_batch, best=best,
                        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])), tmp)
        tmp.replace(dest)
        phase_event('checkpoint', time.perf_counter()-begin, name=name)

    def check(epoch):
        nonlocal best
        path=a.out_dir/f'dev-{epoch:02d}.json'
        begin=time.perf_counter()
        # Complete receipt written AFTER both JSON and JSONL are durable.
        receipt_path=path.with_suffix('.complete.json')
        if receipt_path.exists():
            saved=P.read(receipt_path)
            if saved['json_sha256']!=P.sha(path) or saved['jsonl_sha256']!=P.sha(path.with_suffix('.jsonl')):
                raise ValueError('incomplete/changed dev artifacts')
            report=P.read(path)
        else:
            report=small_check(rt, plan, epoch, path)
            P.write(receipt_path, dict(json_sha256=P.sha(path), jsonl_sha256=P.sha(path.with_suffix('.jsonl'))))
        phase_event('dev', time.perf_counter()-begin, epoch=epoch)
        score=report['selection_score']
        if best is None or score>best['score']:
            best=dict(epoch=epoch, score=score); save('best-dev', epoch+1, 0)
        save('last', epoch+1, 0)

    if a.resume and start_batch==0 and start_epoch-1 in P.CHECKS: check(start_epoch-1)
    epoch=start_epoch-1
    with (a.out_dir/'steps.jsonl').open('a', encoding='utf-8') as log, (a.out_dir/'injections.jsonl').open('a', encoding='utf-8') as inj:
        for epoch in range(start_epoch, 51):
            begin_epoch=time.perf_counter(); rt.model.train()
            batches=P.batches(ds, epoch)
            if a.smoke:
                # All six steps use identical maximum-depth recipients for memory qualification.
                ids=next(z for z in batches if len(z)==32 and ds[z[0]]==max(ds))
                batches=[ids]*6
            for batch, ids in enumerate(batches):
                if epoch==start_epoch and batch<start_batch: continue
                if rt.device.type=='cuda': torch.cuda.reset_peak_memory_stats(rt.device)
                begin=time.perf_counter()
                settings=S.schedule(ds[ids[0]], epoch, batch,
                                    force_mode=S.MODES[batch//2] if a.smoke else None,
                                    force_enabled=bool(a.smoke and batch%2))
                if a.smoke: settings['enabled']=bool(batch%2)
                candidates, events=S.events([gs[i] for i in ids], ids, epoch, batch, settings)
                for group in opt.param_groups: group['lr']=group['initial_lr']*P.lr_factor(step, 50*per_epoch)
                opt.zero_grad(set_to_none=True); losses=[]; stats=[]; audits=[]; donor_cost=0; before_calls=rt.calls
                for off in range(0, len(ids), a.micro_batch):
                    ii=ids[off:off+a.micro_batch]
                    xx, yy=x[ii].to(rt.device), y[ii].to(rt.device)
                    top=topology.batch(ii, xx.shape[1], rt.device)
                    loss, ms, au, dc=recurrent_loss(rt, xx, yy, [gs[i] for i in ii], top,
                                                   events, settings, off, a.donor_batch)
                    weight=len(ii)/len(ids)
                    (loss*weight).backward()
                    losses.append(loss.detach()*weight)
                    stats.extend(dict(micro_offset=off, graphs=len(ii), **m) for m in ms)
                    # Restore effective-batch indices in audit, unlike internal micro coordinates.
                    audits.extend(dict(z, bi=z['bi']+off) for z in au)
                    donor_cost+=dc
                norm=torch.nn.utils.clip_grad_norm_(rt.model.parameters(), float('inf'), error_if_nonfinite=True)
                packed=materialize(dict(loss=sum(losses), grad_norm=norm, metrics=stats, injections=audits))
                if not __import__('math').isfinite(packed['loss']): raise ValueError('nonfinite loss')
                opt.step()
                if rt.device.type=='cuda': torch.cuda.synchronize(rt.device)
                step+=1; presentations+=len(ids)
                audit=packed.pop('injections')
                row=dict(epoch=epoch, batch=batch, step=step, graphs=len(ids), **settings,
                         candidates=candidates, attempts=len(events), accepted=sum(z['accepted'] for z in audit),
                         actual_value_changed=sum(z.get('actual_value_changed',False) for z in audit),
                         donor_forwards=donor_cost, recipient_forwards=rt.calls-before_calls-donor_cost,
                         train_step_seconds=time.perf_counter()-begin,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(rt.device) if rt.device.type=='cuda' else None,
                         **packed)
                log.write(json.dumps(row)+'\n'); log.flush()
                for z in audit: inj.write(json.dumps(dict(epoch=epoch, step=step, mode=settings['mode'], **z))+'\n')
                inj.flush()
                if step%100==0 or a.smoke:
                    save('last', epoch, batch+1)
                    print(json.dumps({k:v for k,v in row.items() if k!='metrics'}), flush=True)
            if S.hashes()!=code_hashes: raise ValueError('source changed during run')
            phase_event('train_epoch_segment', time.perf_counter()-begin_epoch, epoch=epoch,
                        resumed_from_batch=start_batch if epoch==start_epoch else 0)
            save('last', epoch+1, 0)
            if a.smoke: break
            if epoch in P.CHECKS:
                save(f'epoch-{epoch:02d}', epoch+1, 0); check(epoch)
        save('smoke-final' if a.smoke else 'final', epoch+1, 0)
    meta=P.read(a.out_dir/'meta.json'); meta.update(status='complete', steps=step, presentations=presentations, best=best)
    P.write(a.out_dir/'meta.json', meta)
    from dose_report import summarize_run
    summarize_run(a.out_dir)


if __name__=='__main__': main()
