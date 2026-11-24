"""Shared C corruption pilot protocol; tensor dependencies imported lazily."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
P = load_module('c_long_protocol_for_corruption', HERE.parent/'c_long_pilot/protocol.py')
DESIGN = load_module('c_donor_design_for_corruption', HERE.parent/'counterfactual_control/design.py')
STUDY = ROOT/'tmp/analysis/c-corruption-pilot-20261115'
ARMS = ('hold', 'noise', 'semantic')
EPOCHS = 3
SEED = 20261117
RATE = 0.5
HORIZON = 16

def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15], 16)

def scheduled(gid, epoch):
    return random.Random(stable_seed(SEED, 'schedule', epoch, gid)).random() < RATE

def training_state_round(d, gid):
    # Inject BEFORE the next forward, in first half of hold, leaving >=2 updates.
    if d < 2: raise ValueError('D must allow two post-injection updates')
    return d + stable_seed(SEED, 'time', gid) % max(1, d//2)

def make_case(tokens, info, gid):
    positions = [p for p, row in enumerate(info) if row[0] >= 0 and 0 <= tokens[p][3] < 23]
    depths = {p:int(info[p][0]) for p in positions}
    parents, constants, leaves = {}, {}, {}
    for p in positions:
        ps = [int(x) for x in info[p][1::2] if x != -1]
        ops = [int(x) for x in info[p][2::2] if x != -1]
        if len(ops) != len(ps)-1 or any(x != 0 for x in ops):
            raise ValueError('requires ADD DAG')
        if depths[p] == 0:
            if len(ps) != 1 or ps[0] in positions: raise ValueError('invalid leaf')
            constants[p] = ps[0]; leaves[p] = tokens[ps[0]][3]; parents[p] = []
        else:
            if any(q not in depths for q in ps): raise ValueError('invalid parent')
            parents[p] = ps
    if len(set(constants.values())) != len(constants): raise ValueError('shared constant')
    truth = DESIGN.evaluate(parents, depths, leaves)
    if any(truth[p] != tokens[p][3] for p in positions): raise ValueError('bad DAG labels')
    delta = 1 + stable_seed(SEED, 'delta', gid) % 22
    selected, coverage = DESIGN.choose_candidates(parents, depths, leaves, f'pilot:{gid}',
                                                   limit=1, deltas=(delta,), seed=SEED)
    d = max(depths.values()) + 1
    result = dict(graph_id=gid, D=d, positions=positions, truth4=[tokens[p][:4] for p in positions],
                  parents=parents, constants=constants, delta=delta, state_round=training_state_round(d,gid),
                  status='selected' if selected else 'no_structural_candidate', coverage=coverage)
    if selected:
        c = selected[0]; u=c['target_pos']; affected={u}
        for p in sorted(positions, key=lambda p:depths[p]):
            if any(q in affected for q in parents[p]): affected.add(p)
        result.update(target=u, children=c['child_positions'], descendants=sorted(affected-{u}),
                      changes=c['changes'], original_value=truth[u], shifted_value=(truth[u]+delta)%23,
                      expected_q=c['local_expected_q'][str(delta)], target_depth=depths[u],
                      fan_in=len(parents[u]))
    return result

def runtime():
    import sys
    sys.path.insert(0,str(HERE.parent/'b_mechanism'))
    from trajectory_io import configure_runtime
    configure_runtime()
    sys.path.insert(0,str(HERE.parent/'c_capability'))
    import run_capability as R
    return R

def context(R, checkpoint, dataset, device, batch_size=8):
    from types import SimpleNamespace
    return R.build_model_and_data(SimpleNamespace(config_dir_name=P.CONFIG,ckpt=Path(checkpoint),
        weight_name='corruption_pilot',device=device,batch_size=batch_size,num_workers=0,compile=False),dataset)

def decode(lit, hidden):
    import torch
    return torch.stack([head(hidden).argmax(-1) for head in lit.model.output_layers],-1)

def donor_input(inputs, case, name):
    out=inputs.clone()
    for leaf, delta in case['changes'][name].items():
        p=case['constants'][str(leaf)] if str(leaf) in case['constants'] else case['constants'][leaf]
        out[0,p,3]=(out[0,p,3]+delta)%23
    return out

def donors(lit, inputs, events, include_same=False):
    """Current-model donors, dropout OFF, no gradient; identical overhead for all arms.

    events: (batch_index, case, state_round). Return detached rows at the same
    absolute round as recipient intervention; never use donor truth as recipient label.
    """
    import torch
    xs=[]; keys=[]; ends=[]
    for bi,c,t in events:
        for name in ([f"shift_{c['delta']}",'same'] if include_same else [f"shift_{c['delta']}"]):
            xs.append(donor_input(inputs[bi:bi+1],c,name)); keys.append((bi,name)); ends.append(t)
    if not xs: return {},dict(sample_forwards=0,token_forwards=0,attention_units=0)
    training=lit.training
    try:
        lit.eval()
        with torch.no_grad():
            x=torch.cat(xs); h=lit.model.token_embedder(x); out={}
            for t in range(1,max(ends)+1):
                h,_=lit.model(h,True)
                for k,(bi,c,_) in enumerate([e for e in events for _ in range(2 if include_same else 1)]):
                    if t==ends[k]: out[keys[k]]=h[k,c['target']].detach().clone()
    finally:
        lit.train(training)
    b,s=x.shape[:2]; rounds=max(ends)
    return out,dict(sample_forwards=b*rounds,token_forwards=b*s*rounds,attention_units=b*s*s*rounds)

def intervene(hidden, bi, pos, donor, mode, seed, scale=1.0):
    """Noise norm matches hypothetical semantic edit at THIS recipient state.

    Semantic replacement stops gradient only through overwritten row. Noise is
    additive with detached magnitude/direction; all untouched rows remain BPTT.
    """
    import torch
    old=hidden[bi,pos]; radius=(donor-old.detach()).norm().detach()
    if mode=='hold': new=old
    elif mode in ('semantic','same'): new=donor.detach()
    elif mode=='noise':
        gen=torch.Generator(device='cpu').manual_seed(seed)
        z=torch.randn(old.shape,generator=gen,dtype=torch.float32).to(old.device,old.dtype)
        new=old+z/z.norm().clamp_min(1e-12)*radius*scale
    else: raise ValueError(mode)
    out=hidden.clone();out[bi,pos]=new
    return out,float(radius),float((new.detach()-old.detach()).norm())

def read_cases(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]

def check_plan(plan_dir,data_dir):
    m=P.read_json(plan_dir/'manifest.json'); data=P.read_json(data_dir/'manifest.json')
    if m['status']!='complete' or data!=m['data_manifest']: raise ValueError('plan/data mismatch')
    if m['dependency_hashes']!=dependency_hashes(): raise ValueError('implementation changed since plan preparation')
    for split in ('train','val'):
        if P.sha(data_dir/f'{split}.pt')!=data['splits'][split]['sha256']: raise ValueError('changed data')
        if P.sha(plan_dir/f'{split}.jsonl')!=m['case_sha256'][split]: raise ValueError('changed cases')
    return m


def dependency_hashes():
    files = [HERE/'pilot_core.py', HERE.parent/'counterfactual_control/design.py',
             HERE.parent/'c_long_pilot/protocol.py', ROOT/'experiments/model.py',
             HERE.parent/'c_capability/run_capability.py']
    return {str(p.relative_to(ROOT)):P.sha(p) for p in files}
