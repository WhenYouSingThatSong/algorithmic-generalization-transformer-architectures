#!/usr/bin/env python3
"""Post-train fixed s2026 C0: original-depth, +D, +2D. No corruption/detach."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P
sys.path.insert(0, str(HERE.parent / 'b_mechanism'))
from trajectory_io import configure_runtime


def recurrent_loss(lit, inputs, labels, info, extra):
    """Full BPTT through solve and continuation; correct labels never fed back."""
    import torch
    depths = info[..., 0]
    ds = depths.max(dim=1).values + 1
    if not torch.all(ds == ds[0]):
        raise ValueError('batch must have identical graph D')
    d = int(ds[0])
    teachers, _, _ = lit.get_teacher_batch_with_depth(inputs, labels, depths)
    hidden = lit.model.token_embedder(inputs)
    solve = hold = total = 0
    weights = P.phase_weights(d, extra)
    for t, w in enumerate(weights, 1):
        hidden, logits = lit.model(hidden, True)
        target = teachers[P.target_index(t, d)]
        losses = [lit.criterion(logit.reshape(-1, logit.size(-1)), target[..., f].reshape(-1))
                  for f, logit in enumerate(logits) if lit.factor_loss_scales[f] != 0]
        scales = [s for s in lit.factor_loss_scales if s != 0]
        loss = sum(s * x for s, x in zip(scales, losses))
        total = total + w * loss
        if t <= d:
            solve = solve + loss.detach() / d
        else:
            hold = hold + loss.detach() / (extra * d)
    return total, float(solve), float(hold), d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=P.ARMS, required=True)
    p.add_argument('--data-dir', type=Path, default=P.STUDY / 'data')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, help='path override only; fixed source hash required')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=P.SEED)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke-batches', type=int, help='one incomplete epoch; never combine with main')
    p.add_argument('--num_threads', type=int, default=4)
    a = p.parse_args()
    if a.batch_size < 1 or a.lr <= 0 or (a.smoke_batches is not None and a.smoke_batches < 1):
        p.error('positive batch size/lr/smoke batches required')
    configure_runtime()
    sys.path.insert(0, str(HERE.parent / 'c_capability'))
    import run_capability as R
    import torch
    import numpy as np
    import random

    source = P.source_spec()
    ckpt = a.ckpt or P.ROOT / source['weight_path']
    if P.sha(ckpt) != P.SOURCE_SHA:
        raise ValueError('source checkpoint differs from fixed s2026/e143')
    data_meta = P.read_json(a.data_dir / 'manifest.json')
    if data_meta['status'] != 'complete':
        raise ValueError('data preparation incomplete')
    path = a.data_dir / 'train.pt'
    if P.sha(path) != data_meta['splits']['train']['sha256']:
        raise ValueError('training data changed')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_float32_matmul_precision('medium')
    cfg = dict(config_dir_name=P.CONFIG, ckpt=ckpt, weight_name='s2026_e143',
               device=a.device, batch_size=a.batch_size, num_workers=0, compile=False)
    ctx = R.build_model_and_data(SimpleNamespace(**cfg), path)
    lit, dm, device = ctx['litmodel'], ctx['data_module'], ctx['device']
    if not lit.nointerm or list(lit.factor_loss_scales) != [1, 1, 1, 1, 0, 0]:
        raise ValueError('expected continuous first-four-field supervision')
    # Fresh optimizer for ALL arms; this is weight-only post-training, not trainer resume.
    opt_cfg = dict(lit.train_config.AdamW_optimizer_config)
    opt_cfg['lr'] = a.lr
    optimizer = torch.optim.AdamW(lit.model.parameters(), **opt_cfg)
    batches = P.batches(dm.data_train_info[..., 0].max(1).values.tolist(), a.batch_size, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    config_dir = R.EXPERIMENTS / 'configs' / P.CONFIG
    meta = dict(status='running', protocol='C_long_pilot_v1', arm=a.arm, source_sha256=P.SOURCE_SHA,
                source_checkpoint=str(ckpt), config_dir_name=P.CONFIG, data_manifest=data_meta,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                source_code_sha256={n:P.sha(HERE / n) for n in ('train.py', 'protocol.py')},
                config_sha256={n:P.sha(config_dir / n) for n in ('model_config.yaml', 'train_config.yaml', 'data_config.yaml')},
                optimizer=opt_cfg, epochs=P.EPOCHS[a.arm], batch_plan=batches,
                cohort='smoke' if a.smoke_batches else 'main', checkpoints={})
    P.write_json(a.out_dir / 'meta.json', meta)
    extra = P.ARMS[a.arm]
    seen = sample_forwards = token_forwards = attention_units = steps = 0
    start = time.perf_counter()
    def save(name, epoch):
        dest = a.out_dir / f'{name}.ckpt'
        # Same format consumed by LitRecurrentTransformerModel.load_ckpt.
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in lit.model.state_dict().items()},
                        optimizer_state=optimizer.state_dict(), epoch=epoch, pilot_arm=a.arm,
                        source_sha256=P.SOURCE_SHA), dest)
        meta['checkpoints'][name] = dict(path=str(dest), sha256=P.sha(dest), epoch=epoch,
            optimizer_steps=steps, graph_presentations=seen, sample_forwards=sample_forwards,
            token_forwards=token_forwards, attention_units=attention_units,
            wall_seconds=time.perf_counter() - start)
        P.write_json(a.out_dir / 'meta.json', meta)
    lit.train()
    with (a.out_dir / 'steps.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, P.EPOCHS[a.arm] + 1):
            order = P.epoch_order(len(batches), epoch, a.seed)
            if a.smoke_batches:
                order = order[:a.smoke_batches]
            for bi in order:
                selected = batches[bi]
                ids = selected['ids']
                raw = torch.cat([dm.data_train[ids], dm.data_train_label[ids], dm.data_train_info[ids]], -1)
                raw = lit.reduce_batch(raw).to(device)
                inputs, labels, info = R.split_batch(raw, ctx['model_config'])
                if a.smoke_batches and steps == 0:
                    # One numerical compatibility check, only in server smoke.
                    lit.eval()
                    with torch.no_grad():
                        new0, _, _, d0 = recurrent_loss(lit, inputs, labels, info, 0)
                        old0 = lit.compute_batch_loss_nointerm(raw, log_prefix='pilot_check', log=False, max_loop=-1)
                        if not torch.allclose(new0, old0 / d0, rtol=1e-5, atol=1e-6):
                            raise ValueError('base loss does not match original continuous loss divided by D')
                    meta['base_loss_equivalence_checked'] = True
                    lit.train()
                optimizer.zero_grad(set_to_none=True)
                loss, solve, hold, d = recurrent_loss(lit, inputs, labels, info, extra)
                if d != selected['D'] or not torch.isfinite(loss):
                    raise ValueError('unexpected D or nonfinite training loss')
                loss.backward()
                # max_norm=inf measures/checks the norm without clipping finite gradients.
                grad_norm = torch.nn.utils.clip_grad_norm_(lit.model.parameters(), float('inf'), error_if_nonfinite=True)
                optimizer.step()
                b, s = inputs.shape[:2]
                rounds = (1 + extra) * d
                seen += b; sample_forwards += b * rounds
                token_forwards += b * s * rounds
                attention_units += b * s * s * rounds
                steps += 1
                row = dict(epoch=epoch, batch_id=bi, step=steps, D=d, rounds=rounds, graphs=b,
                           loss=float(loss.detach()), solve_loss=solve, hold_loss=hold,
                           grad_norm=float(grad_norm),
                           sample_forwards=sample_forwards, token_forwards=token_forwards,
                           elapsed=time.perf_counter() - start)
                log.write(json.dumps(row) + '\n'); log.flush()
                if steps % 16 == 0 or a.smoke_batches:
                    print(json.dumps(row), flush=True)
            if a.smoke_batches:
                save('smoke', epoch); break
            if epoch == 2:
                save('same_exposure', epoch)
        if not a.smoke_batches:
            save('equal_compute', P.EPOCHS[a.arm])
    meta.update(status='complete', completed_steps=steps)
    P.write_json(a.out_dir / 'meta.json', meta)


if __name__ == '__main__':
    main()
