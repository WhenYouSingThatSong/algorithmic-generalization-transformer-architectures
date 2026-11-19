"""CPU-only specification for the approved 50-epoch C experiments."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'c_update_v1'
CONFIG = 'C0-M4-Nodes32-ADD-L4H16D384-DeBERTa'
TRAIN_SHA = 'cee36b84b01b7c1fdbd2e32619b402e571275c534a4d685684db07a6e11de1ce'
SEED = 20261119
EPOCHS = 50
CHECKS = (1, 3, 5, *range(10, 51, 5))
LOSS_WEIGHTS = dict(state=1., locate=2., align=5., clear=.1)
STRATA = tuple(f'N{n}_{d}' for n in (32, 64, 128) for d in ('shallow', 'deep'))

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def seed(*args):
    return int(hashlib.sha256(repr((SEED, args)).encode()).hexdigest()[:15], 16)

def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'experiments'/n for n in ('model.py', 'tokenizers.py')]
    paths += [ROOT/'Simtransformer/simtransformer/model_base.py']
    paths += [ROOT/p for p in ('tools/b_mechanism/b_mechanism_common.py','tools/b_mechanism/trajectory_io.py',
                              'tools/counterfactual_control/design.py','tools/c_capability/c_common.py',
                              'experiments/generate_data.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}

def graph(labels, info):
    """Only definition/result nodes; leaves point to original constant tokens."""
    nodes = [p for p, row in enumerate(info) if row[0] >= 0 and 0 <= labels[p][3] < 23]
    depths = {p: int(info[p][0]) for p in nodes}
    parents, constants, coefs = {}, {}, {}
    for p in sorted(nodes, key=lambda p: (depths[p], p)):
        ps = [int(x) for x in info[p][1::2] if x != -1]
        ops = [int(x) for x in info[p][2::2] if x != -1]
        if depths[p] == 0:
            if len(ps) != 1 or ps[0] in depths: raise ValueError('bad leaf metadata')
            constants[p] = ps[0]; parents[p] = []; coefs[p] = {p: 1}
        else:
            if not ps or any(q not in coefs for q in ps) or len(ops) != len(ps)-1 or any(ops):
                raise ValueError('requires depth-ordered ADD DAG')
            parents[p] = ps; c = {}
            for q in ps:
                for k,v in coefs[q].items(): c[k] = (c.get(k,0)+v)%23
            coefs[p] = c
    g = dict(nodes=nodes, depths=depths, parents=parents, constants=constants,
             coefs=coefs, D=max(depths.values())+1)
    vals = truth(g, labels)
    if any(vals[p] != labels[p][3] for p in nodes): raise ValueError('independent ADD truth mismatch')
    return g

def truth(g, labels):
    vals = {}
    for p in sorted(g['nodes'], key=lambda p: (g['depths'][p], p)):
        vals[p] = labels[g['constants'][p]][3] if not g['parents'][p] else sum(vals[q] for q in g['parents'][p])%23
    return vals

def local_step(g, current, original):
    """Synchronous target from INPUT values, never recursively use new outputs."""
    out = [list(r[:4]) for r in original]; mask = [False]*len(out)
    for p in g['nodes']:
        old = current[p][3]; ps = g['parents'][p]
        if not ps: expected = original[g['constants'][p]][3]
        elif all(0 <= current[q][3] < 23 for q in ps): expected = sum(current[q][3] for q in ps)%23
        else: expected = old
        mask[p] = expected != old
        out[p][3] = expected
    return out, mask

def donor_edits(g, pos, delta):
    """One invertible leaf coefficient suffices, including leaf/sink targets."""
    for leaf, c in sorted(g['coefs'][pos].items()):
        if c%23: return {g['constants'][leaf]: delta*pow(c, -1, 23)%23}
    return None

def same_edits(g, pos):
    nz = [(k,c) for k,c in sorted(g['coefs'][pos].items()) if c%23]
    if len(nz) < 2: return {} # explicitly original-source, not different-source control
    (a,x),(b,y) = nz[:2]
    return {g['constants'][a]: 1, g['constants'][b]: -x*pow(y,-1,23)%23}

def events(graphs, ids, epoch, batch_id, force=False):
    """One coin per effective batch; each new frontier position sampled once."""
    enabled = force or random.Random(seed('batch',epoch,batch_id)).random() < .05
    out = []
    if enabled:
        for bi,(g,gid) in enumerate(zip(graphs,ids)):
            for p in g['nodes']:
                r = random.Random(seed('position',epoch,gid,p))
                if r.random() < .5:
                    delta = r.randrange(1,23)
                    out.append(dict(bi=bi,gid=gid,pos=p,round=g['depths'][p],delta=delta,
                                    donor_round=g['depths'][p]+1,edits=donor_edits(g,p,delta)))
    return enabled, out

def batches(ds, epoch):
    groups = {}
    for i,d in enumerate(ds): groups.setdefault(d,[]).append(i)
    out=[]
    for d,ids in sorted(groups.items()):
        random.Random(seed('shuffle',epoch,d)).shuffle(ids)
        out.extend(ids[i:i+32] for i in range(0,len(ids),32))
    random.Random(seed('batchorder',epoch)).shuffle(out)
    return out

def lr_factor(step, total):
    return .1+.9*.5*(1+math.cos(math.pi*min(step,total)/max(total,1)))

def alignment_side(plan, changed, reference):
    if plan == changed: return None
    return 'locate' if changed == reference else 'execute'

def horizon_flags(flags):
    return dict(first=next((i+1 for i,v in enumerate(flags) if v),None),
                stable=next((i+1 for i in range(len(flags)-1) if all(flags[i:])),None),
                reerror=any(flags[i] and not all(flags[i+1:]) for i in range(len(flags)-1)),
                last_two=len(flags)>=2 and all(flags[-2:]))

def select_target(g, split, gid):
    edges=[(p,q) for p in g['nodes'] if g['depths'][p]>0
           for q in g['nodes'] if p in g['parents'][q]]
    if not edges:return None
    u,q=min(edges,key=lambda pair:seed('target',split,gid,*pair))
    return dict(u=u,q=q,delta=1+seed('delta',split,gid)%22)

def descendants(g,u):
    found={u}
    for p in sorted(g['nodes'],key=lambda p:g['depths'][p]):
        if any(q in found for q in g['parents'][p]):found.add(p)
    return found-{u}
