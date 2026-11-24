"""Create shared symbolic task manifest and checkpoint registry; no model execution."""
import argparse
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'priority_pilots'))
from io_utils import read_json, write_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--a-source', type=Path, default=ROOT/'tmp/analysis/priority-pilots-20261118/reassign-main-A')
    p.add_argument('--b-source', type=Path, default=ROOT/'tmp/analysis/priority-pilots-20261118/reassign-main-B')
    p.add_argument('--c-source', type=Path, default=ROOT/'tmp/analysis/state-reuse-exec-20261119/C-main/C-initial')
    a = p.parse_args()
    models = []
    for name, source in [('A', a.a_source), ('B', a.b_source)]:
        m = read_json(source/'meta.json')
        if m.get('status') != 'complete': raise ValueError('incomplete source')
        s = m['model']; path = Path(s['ckpt_base_dir'])/f"{s['group_name']}-{s['run_name']}"/s['ckpt_name']
        models.append(dict(id=name+'_s42', family=name, seed=42, config=s['config_dir_name'],
                           path=str(path.resolve()), sha256=s['ckpt_sha256']))
    m = read_json(a.c_source/'meta.json')
    if m.get('status') != 'complete': raise ValueError('incomplete C source')
    s = next(x for x in m['models'] if x['arm'] == 'initial')
    config = 'C0-M4-Nodes32-ADD-L4H16D384-DeBERTa'
    models.append(dict(id='C_s2026', family='C', seed=2026, config=config,
                       path=str(Path(s['path']).resolve()), sha256=s['sha256']))
    # Existing base C weights; no new training, no silent replacement by best/last.
    import importlib.util
    spec = importlib.util.spec_from_file_location('abc_c_weights', HERE.parent/'c_capability/c_common.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for key in ('s42_e143', 's1234_e143'):
        s = mod.WEIGHTS[key]; path = ROOT/s['base']/f"{s['group']}-{s['run']}"/s['ckpt']
        models.append(dict(id='C_s'+str(s['seed']), family='C', seed=s['seed'], config=config,
                           path=str(path), sha256=s['sha256']))
    for s in models:
        if sha(s['path']) != s['sha256']: raise ValueError('weight mismatch: '+s['id'])
    cases = P.make_cases()
    fingerprints = [P.digest(P.render(P.frame(c, 0), c['names'], P.frame(c, 0)['constants'])['tokens']) for c in cases]
    if len(set(fingerprints)) != len(fingerprints): raise ValueError('duplicate new tasks')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.out_dir/'tasks.json', dict(protocol=P.VERSION, seed=P.SEED, cases=cases,
        strata=P.STRATA, stages=20, milestones=[128,256,512], input_fingerprints=fingerprints,
        note='Fresh generated development tasks; no historical graph-isomorphism exclusion claimed. Fixed 32 slots.'))
    write_json(a.out_dir/'models.json', dict(protocol=P.VERSION, models=models))
    print(json.dumps(dict(out=str(a.out_dir), cases=len(cases), models=[m['id'] for m in models])))


if __name__ == '__main__': main()
