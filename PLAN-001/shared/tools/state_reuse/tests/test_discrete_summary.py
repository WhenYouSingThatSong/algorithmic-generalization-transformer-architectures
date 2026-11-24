import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as P
import summarize as S


class DiscreteSummaryTest(unittest.TestCase):
    def test_paired_new_truth_and_old_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=[];old=[[0,0,0,3]];new=[[0,0,0,4]]
            for arm in ('model_a','model_b'):
                path=root/arm;path.mkdir();runs.append(path)
                branches={}
                for b in P.DISCRETE_BRANCHES:
                    correct=b in ('scratch','clear') or arm=='model_b' and b in ('warm','warm_pinned')
                    output=new if correct else old
                    branches[b]=dict(inputs=[old,output],outputs=[output,output])
                row=dict(graph_id=9,schedule='one',episode=1,horizon=2,positions=[2],nodes=[2],
                    old_truth4=old,original_truth4=old,new_truth4=new,plan=dict(changed=[2]),branches=branches,
                    regions=dict(all_nodes=[2],target=[2],changed=[2],unchanged=[]))
                (path/'records.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
                (path/'coverage.jsonl').write_text(json.dumps(dict(graph_id=9,status='selected',episodes=dict(one=1)))+'\n',encoding='utf-8')
                S.write_json(path/'meta.json',dict(status='complete',protocol='state_reuse_discrete_v1',mode='repeat',
                    model_key=arm,display_name='A' if arm=='model_a' else 'B',candidate_ids=[9],horizons=[2],code_hashes={},
                    source_meta=dict(dataset_sha256='ds',manifest_sha256='plan'),
                    records_sha256=S.sha(path/'records.jsonl'),coverage_sha256=S.sha(path/'coverage.jsonl')))
            S.run(runs,root/'summary')
            with (root/'summary/paired_models.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
            target=next(r for r in rows if r['branch']=='warm' and r['region']=='target')
            self.assertEqual(target['B_minus_A'],'1')
            with (root/'summary/update_retention.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
            target=next(r for r in rows if r['arm']=='model_a' and r['branch']=='warm' and r['round_or_window']=='last_two')
            self.assertEqual(target['retain_previous_n'],'1');self.assertEqual(target['accept_new_n'],'0')


if __name__=='__main__':unittest.main()
