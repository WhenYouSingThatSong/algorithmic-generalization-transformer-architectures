#!/usr/bin/env python3
"""Freeze structural positions/deltas/times using existing new long-pilot data."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pilot_core as C

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=C.P.STUDY/'data')
    p.add_argument('--out-dir',type=Path,default=C.STUDY/'plan')
    a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    data=C.P.read_json(a.data_dir/'manifest.json')
    if data['status']!='complete': raise ValueError('data incomplete')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    meta=dict(status='building',protocol='C_corruption_pilot_v1',data_manifest=data,
              source_sha256=C.P.SOURCE_SHA,dependency_hashes=C.dependency_hashes(),seed=C.SEED,rate=C.RATE,case_sha256={},counts={})
    C.P.write_json(a.out_dir/'manifest.json',meta)
    for split in ('train','val'):
        path=a.data_dir/f'{split}.pt'
        if C.P.sha(path)!=data['splits'][split]['sha256']: raise ValueError('changed data')
        blob=torch.load(path,map_location='cpu',weights_only=False)
        cases=[C.make_case(t.tolist(),i.tolist(),gid) for gid,(t,i) in enumerate(zip(blob['tokenized_dataset'],blob['dataset_info']))]
        if len(cases)!=data['splits'][split]['count']: raise ValueError('count mismatch')
        dest=a.out_dir/f'{split}.jsonl'
        dest.write_text(''.join(json.dumps(c)+'\n' for c in cases),encoding='utf-8')
        meta['case_sha256'][split]=C.P.sha(dest)
        meta['counts'][split]=dict(graphs=len(cases),selected=sum(c['status']=='selected' for c in cases),
            scheduled_by_epoch=[sum(c['status']=='selected' and C.scheduled(c['graph_id'],e) for c in cases) for e in range(1,C.EPOCHS+1)])
    meta['status']='complete';C.P.write_json(a.out_dir/'manifest.json',meta)
    print(json.dumps(meta['counts']))
if __name__=='__main__': main()
