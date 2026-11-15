"""Fixed small post-training comparison; stdlib-only design and accounting."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'C0-M4-Nodes32-ADD-L4H16D384-DeBERTa'
STUDY = ROOT / 'tmp/analysis/c-long-pilot-20261115'
SOURCE_ARCHIVE = ROOT / 'ai-collab/artifacts/c0-capability-s2026-e143-raw.zip'
SOURCE_SHA = '40babb206b1393d0892b047202b9d6469cd140c82f3ba82c96a466fa3c76e780'
ARMS = {'base': 0, 'plus_d': 1, 'plus_2d': 2}
EPOCHS = {'base': 6, 'plus_d': 3, 'plus_2d': 2}
SEED = 20261115


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def batches(depths, batch_size, seed=SEED):
    """Fixed memberships, homogeneous per-graph D. No drops or duplication."""
    if batch_size < 1 or not depths or min(depths) < 0:
        raise ValueError('nonempty depths and positive batch size required')
    groups = defaultdict(list)
    for i, depth in enumerate(depths):
        groups[int(depth) + 1].append(i)
    rng = random.Random(seed)
    result = []
    for d, ids in sorted(groups.items()):
        rng.shuffle(ids)
        for start in range(0, len(ids), batch_size):
            result.append(dict(D=d, ids=ids[start:start + batch_size]))
    return result


def epoch_order(n_batches, epoch, seed=SEED):
    ids = list(range(n_batches))
    random.Random(seed + epoch).shuffle(ids)
    return ids


def target_index(round_number, d):
    """Teacher array index: inputs at 0, final truth at D; never restart."""
    if round_number < 1 or d < 1:
        raise ValueError('rounds and D are one-based positive integers')
    return min(round_number, d)


def phase_weights(d, extra):
    if d < 1 or extra not in (0, 1, 2):
        raise ValueError('unsupported D/extension')
    if not extra:
        return [1 / d] * d
    # Equal solve/hold objective weight for +D and +2D, not 1:1 vs 1:2.
    return [0.5 / d] * d + [0.5 / (extra * d)] * (extra * d)


def evaluation_points(d):
    return {'D':d, '2D':2*d, '3D':3*d, **{f'D+{x}':d+x for x in (16,32,64)},
            **{f'absolute_{x}':x for x in (8,16,32,64)}}


def source_spec():
    import zipfile
    with zipfile.ZipFile(SOURCE_ARCHIVE) as z:
        name = next(n for n in z.namelist() if n.endswith('/run/run_meta.json'))
        meta = json.loads(z.read(name))
    if meta['weight_sha256'] != SOURCE_SHA or meta['status'] != 'complete':
        raise ValueError('unexpected source identity')
    return meta


def post_events(flags):
    """Whole-graph completion and subsequent failures, conditional denominator."""
    first = next((i for i, x in enumerate(flags) if x), None)
    if first is None:
        return dict(completion=None, post_observed=False, first_bad_after=None,
                    ever_bad_after=None, ever_recovered=None, reerror=None)
    post = flags[first + 1:]
    bad = next((i for i, x in enumerate(post) if not x), None)
    recovery = None if bad is None else next((i for i in range(bad + 1, len(post)) if post[i]), None)
    return dict(completion=first + 1, post_observed=bool(post),
                first_bad_after=None if bad is None else bad + 1,
                ever_bad_after=bool(bad is not None) if post else None,
                ever_recovered=bool(recovery is not None) if post else None,
                reerror=bool(recovery is not None and not all(post[recovery + 1:])) if post else None)
