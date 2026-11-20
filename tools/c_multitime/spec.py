"""Independent recipe; reuse the frozen dataset/weights, never mutate old runners."""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'c_update'))
import protocol as P

VERSION = 'c_multitime_process_update_v1'
DISPLAY = 'C（多时点局部纠错＋过程监督UPDATE）'
MODES = ('frontier', 'middle', 'late_budget')
RECIPE = dict(
    model='C_s2026', epochs=50, effective_batch=32,
    modes=list(MODES), mode_probability=[1/3]*3,
    batch_probability=.05, position_probability=.5,
    middle_start='max(1,D//2)', late_start='D+4',
    recovery_rounds='D+1', loss_weights=P.LOSS_WEIGHTS,
    full_bptt=True, receiver_correctness_filter=False,
    main_lr=1e-4, update_head_lr=5e-4, precision='float32',
)


def hashes():
    result = P.source_hashes()
    for path in HERE.glob('*.py'):
        result[str(path.relative_to(P.ROOT))] = P.sha(path)
    return result


def schedule(d, epoch, batch, force_mode=None, force_enabled=False):
    mode = force_mode or random.Random(P.seed(VERSION, 'mode', epoch, batch)).choice(MODES)
    if mode not in MODES:
        raise ValueError(mode)
    enabled = force_enabled or random.Random(P.seed(VERSION, 'gate', epoch, batch)).random() < .05
    start = None if mode == 'frontier' else max(1, d//2) if mode == 'middle' else d+4
    # Identical horizons for contaminated and non-contaminated batches of each mode.
    rounds = d+1 if start is None else start+d+1
    return dict(mode=mode, enabled=enabled, start=start, rounds=rounds)


def events(gs, ids, epoch, batch, settings):
    """Only structural/time eligibility. NEVER inspect model answers here."""
    candidates = 0
    out = []
    for bi, (g, gid) in enumerate(zip(gs, ids)):
        for pos in g['nodes']:
            if settings['mode'] == 'middle' and g['depths'][pos] >= settings['start']:
                continue
            candidates += 1
            rng = random.Random(P.seed(VERSION, 'position', epoch, batch, gid, pos))
            if not settings['enabled'] or rng.random() >= .5:
                continue
            delta = rng.randrange(1, 23)
            t = g['depths'][pos] if settings['mode'] == 'frontier' else settings['start']
            out.append(dict(bi=bi, gid=gid, pos=pos, round=t, delta=delta,
                            donor_round=t+1 if settings['mode']=='frontier' else t,
                            edits=P.donor_edits(g, pos, delta)))
    return candidates, out


def validate_plan(path):
    plan = P.read(path)
    if plan['protocol'] != P.VERSION or plan['status'] != 'complete':
        raise ValueError('reuse the completed c_update plan')
    if plan['train']['sha256'] != P.TRAIN_SHA or P.sha(plan['train']['path']) != P.TRAIN_SHA:
        raise ValueError('wrong training data')
    for name, digest in plan['config_hashes'].items():
        if P.sha(P.ROOT/name) != digest:
            raise ValueError('changed config: '+name)
    return plan
