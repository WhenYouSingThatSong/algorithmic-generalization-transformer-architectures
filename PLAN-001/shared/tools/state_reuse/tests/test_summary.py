import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import core as P
import summarize as S


class SummaryTest(unittest.TestCase):
    def test_joint_cohort_uses_intersection_and_keeps_empty_regions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'run';src.mkdir();arms=['initial','semantic']
            rows=[];gold=[[0,0,0,4],[0,1,0,13]]
            for arm in arms:
                for timing,mode,lag in P.c_probe_grid():
                    eligible=not (arm=='semantic' and (timing,mode,lag)==('late','noise2',4))
                    rows.append(dict(arm=arm,graph_id=0,timing=timing,mode=mode,probe_lag=lag,
                        positions=[1,2],truth4=gold,u=1,q=2,horizon=2,children=[2],descendants=[2],
                        u_had_wrong_readout=True,q_had_wrong_readout=True,
                        context=dict(u_value_correct=eligible,q_value_correct=True,other_parents_value_correct=True,
                            clean_local_first4_correct=True,donor_pre_empty=True,donor_pre_identity_correct=True),
                        branches={n:dict(inputs=[gold,gold],outputs=[gold,gold]) for n in P.C_BRANCHES}))
            (src/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            (src/'coverage.jsonl').write_text(''.join(json.dumps(dict(arm=a,graph_id=0,status='selected'))+'\n' for a in arms),encoding='utf-8')
            S.write_json(src/'meta.json',dict(status='complete',protocol='C_function_probe_v1',
                models=[dict(arm=a) for a in arms],candidate_ids=[0],horizon=2,grid=P.c_probe_grid(),
                records_sha256=S.sha(src/'records.jsonl'),coverage_sha256=S.sha(src/'coverage.jsonl')))
            S.run([src],root/'out')
            meta=S.read_json(root/'out/meta.json')
            self.assertEqual(meta['common_cases'],36);self.assertEqual(meta['common_input_qualified'],35)
            with (root/'out/summary.csv').open(encoding='utf-8') as f: summary=list(csv.DictReader(f))
            empty=[r for r in summary if r['region']=='other_children']
            self.assertTrue(empty);self.assertTrue(all(r['node_n']=='0' and r['value_macro']=='' for r in empty))
            target=[r for r in summary if r['region']=='q' and r['branch']=='observed_pre']
            self.assertTrue(all(r['value_macro']=='1.0' for r in target))


if __name__=='__main__':unittest.main()
