import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SummaryTest(unittest.TestCase):
    def test_b_primary_intersection_and_graph_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=[]
            for weight in ('model_a','model_b'):
                run=root/weight;run.mkdir()
                (run/'meta.json').write_text(json.dumps({'status':'complete'}))
                records=[]
                for gid in (1,2):
                    r=dict(weight=weight,case_id=f'B:{gid}',graph_id=gid,
                           cohort='fixed_candidates',timing='just_solved',category='existing_value',
                           state_round=3,target_pos=5,parent_positions=[],descendant_positions=[],
                           node_positions=[5],truth4=[[0,0,1,4]],data_sha256='abc',
                           status='ineligible' if weight=='model_b' and gid==2 else 'eligible',
                           branches={'unchanged':{'initial':[[0,0,1,4]],'outputs':[[[0,0,1,4]],[[0,0,1,4]]]}})
                    records.append(r)
                path=run/'records.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                inputs.append(str(path))
            script=Path(__file__).resolve().parents[1]/'summarize.py'
            subprocess.run([sys.executable,str(script),'--inputs',*inputs,'--out_dir',str(root/'summary')],check=True)
            meta=json.loads((root/'summary/meta.json').read_text())
            self.assertEqual(meta['common_b_cases'],1)
            self.assertEqual(meta['included_cases'],2)
            # A late-only eligible target must not leak into the matched timing comparison.
            for name in inputs:
                path=Path(name)
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                late=[dict(r,case_id=r['case_id']+':complete',timing='whole_graph_complete',status='eligible') for r in rows]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows+late))
            subprocess.run([sys.executable,str(script),'--inputs',*inputs,'--require_timings',
                'just_solved','whole_graph_complete','--out_dir',str(root/'matched')],check=True)
            matched=json.loads((root/'matched/meta.json').read_text())
            self.assertEqual(matched['common_b_cases'],2)
            self.assertEqual(matched['included_cases'],4)

    def test_matched_timings_requires_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inputs=[]
            for weight in ('model_a','model_b'):
                run=root/weight; run.mkdir()
                (run/'meta.json').write_text(json.dumps({'status':'complete'}))
                records=[]
                for gid in (1,2):
                    timings=('just_solved','whole_graph_complete')
                    if weight=='model_b' and gid==2:
                        timings=('just_solved',)
                    for timing in timings:
                        records.append(dict(
                            weight=weight, case_id=f'B:{gid}:{timing}', graph_id=gid,
                            cohort='fixed_candidates', timing=timing,
                            category='existing_value', state_round=3, target_pos=5,
                            parent_positions=[], descendant_positions=[],
                            node_positions=[5], truth4=[[0,0,1,4]], data_sha256='abc',
                            status='eligible',
                            branches={'unchanged':{'initial':[[0,0,1,4]],
                                                   'outputs':[[[0,0,1,4]],[[0,0,1,4]]]}}))
                path=run/'records.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                inputs.append(str(path))
            script=Path(__file__).resolve().parents[1]/'summarize.py'
            subprocess.run([sys.executable,str(script),'--inputs',*inputs,'--require_timings',
                'just_solved','whole_graph_complete','--out_dir',str(root/'matched')],check=True)
            matched=json.loads((root/'matched/meta.json').read_text())
            # gid2 is complete only in model_a, so the target must be dropped for both models.
            self.assertEqual(matched['common_b_cases'],2)
            self.assertEqual(matched['included_cases'],4)


if __name__ == '__main__':
    unittest.main()
