import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import stages
import design


class StageTests(unittest.TestCase):
    def test_absolute_completed_forward_schedule_and_censoring(self):
        self.assertEqual(stages.schedule(4,8,16,64),dict(early=4,complete=8,late=24))
        self.assertEqual(stages.schedule(4,60,16,64),dict(early=4,complete=60,late=None))
        self.assertEqual(stages.schedule(4,None,16,64),dict(early=4,complete=None,late=None))

    def test_child_intersection_and_missing_stage_not_just_target_matching(self):
        def record(stage,qs):
            return dict(weight='w',graph_id='g',target_pos=1,data_sha256='d',
                status='eligible',stage=stage,eligible_children=qs,
                source_qualification={'original_target_first4':True})
        rows=[record('early',[2,3]),record('complete',[3,4]),record('late',[3,5])]
        key=('w','g',1,'d')
        self.assertEqual(stages.common_children(rows,['early','complete','late']),{key:[3]})
        self.assertEqual(stages.common_children(rows[:2],['early','complete','late']),{})
        rows[-1]['source_qualification']['original_target_first4']=False
        self.assertEqual(stages.common_children(rows,['early','complete','late'],True),{})

    def test_value_qualification_does_not_silently_require_late_variable_correct(self):
        truth=[[0,0,1,4],[0,1,1,8]]
        candidate=dict(target_pos=0,changes={'same':{},'shift_1':{}},
                       donor_truth={'same':{0:4},'shift_1':{0:5}},ready_children=[1])
        original=copy.deepcopy(truth);original[0][1]=9
        initial={n:copy.deepcopy(original) for n in ['original','same','shift_1']}
        initial['shift_1'][0][3]=5
        self.assertEqual(design.qualify(truth,[0,1],{1:[0]},candidate,initial,truth)[1],[])
        flags,qs=design.qualify(truth,[0,1],{1:[0]},candidate,initial,truth,value_only=True)
        self.assertEqual(qs,[1]);self.assertFalse(flags['original_target_first4'])

    def test_summary_uses_common_q_and_keeps_stage_identity(self):
        truth=[[0,0,1,4],[0,1,1,8],[0,2,1,9]]
        shifted=copy.deepcopy(truth);shifted[1][3]=9;shifted[2][3]=10
        rows=[]
        for stage,qs in [('early',[1,2]),('complete',[2]),('late',[1,2])]:
            rows.append(dict(weight='w',case_id='g:u0:'+stage,graph_id='g',stratum='s',
                target_pos=0,data_sha256='d',fan_in=2,depth_band='lower',depth=1,
                original_value=4,state_round=2,stage=stage,same_vector_equal=False,
                status='eligible',source_qualification={'original_target_first4':True},
                vector_distance={},eligible_children=qs,node_positions=[0,1,2],truth4=truth,
                modulus=23,deltas=[1],local_expected_q={'1':{'1':9,'2':10}},references={},
                branches={n:dict(outputs=[shifted if n=='shift_1' else truth])
                          for n in ['unchanged','sham','same','shift_1']}))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'meta.json').write_text(json.dumps(dict(status='complete',protocol='matched_counterfactual_stages_v1')))
            (p/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            subprocess.run([sys.executable,str(HERE/'summarize.py'),'--inputs',str(p/'records.jsonl'),
                '--common_stages','early','complete','late','--out_dir',str(p/'summary')],check=True,capture_output=True)
            import csv
            with (p/'summary/summary.csv').open() as f:
                result=[r for r in csv.DictReader(f) if r['view']=='all']
            self.assertEqual({r['stage'] for r in result},{'early','complete','late'})
            self.assertTrue(all(r['q_observations']=='1' and r['different_expected_count']=='1' for r in result))
