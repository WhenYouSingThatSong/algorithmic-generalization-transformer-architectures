import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pilot_core as C
from summarize import trajectory_event

class ProtocolTests(unittest.TestCase):
    def test_hold_injection_leaves_recovery_window(self):
        for d in range(2,13):
            for gid in range(64):
                t=C.training_state_round(d,gid)
                self.assertGreaterEqual(t,d)
                self.assertGreaterEqual(2*d-t,2)
    def test_schedule_not_dependent_on_arm_or_batch(self):
        ids=list(range(1024));expected={i:C.scheduled(i,2) for i in ids}
        ids.reverse()
        self.assertEqual(expected,{i:C.scheduled(i,2) for i in ids})
        self.assertTrue(400<sum(expected.values())<624)
        self.assertTrue(any(C.scheduled(i,1)!=C.scheduled(i,2) for i in ids))
        self.assertEqual({1+C.stable_seed(C.SEED,'delta',i)%22 for i in ids},set(range(1,23)))
    def test_recovery_counts_input_and_reerror(self):
        ev=trajectory_event([False,True,False,True,True],[True]*5)
        self.assertEqual((ev['first_damage'],ev['first_recovery'],ev['last_stable_start']),(0,1,3))
        self.assertTrue(ev['recovered_then_error']);self.assertTrue(ev['final_two_correct'])
        self.assertEqual(ev['wrong_rounds'],1)
    def test_no_paired_damage_is_not_recovery(self):
        ev=trajectory_event([False,False,True],[False,False,True])
        self.assertFalse(ev['damaged']);self.assertFalse(ev['ever_recovered'])
    def test_matched_source_changes_same_or_wrong_value(self):
        edits=C.DESIGN.matched_changes({1:2,2:3},(7,))
        self.assertEqual(set(edits['same']),set(edits['shift_7']))
        self.assertEqual(sum({1:2,2:3}[p]*v for p,v in edits['same'].items())%23,0)
        self.assertEqual(sum({1:2,2:3}[p]*v for p,v in edits['shift_7'].items())%23,7)

class SummaryIntegrationTests(unittest.TestCase):
    def test_summary_keeps_input_damage_separate_from_output(self):
        import csv,json,subprocess,tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ev=root/'eval';ev.mkdir()
            C.P.write_json(ev/'meta.json',dict(status='complete',cohort='smoke',models=[dict(arm='smoke')]))
            gold=[[0,1,1,4],[0,2,1,13]]
            branches={}
            for name in ('unchanged','same','semantic','noise','noise3'):
                initial=[row[:] for row in gold]
                if name=='semantic':initial[0][3]=5
                outs=[[row[:] for row in gold] for _ in range(16)]
                if name=='semantic':outs[0][1][3]=14
                branches[name]=dict(initial=initial,outputs=outs)
            r=dict(arm='smoke',graph_id=0,timing='hold',positions=[0,1],truth4=gold,target=0,
                children=[1],descendants=[1],delta=1,expected_q={'1':14},value_qualified=True,
                strict_qualified=True,branches=branches,
                distances={n:dict(reference_norm=1,actual_norm=1 if n=='semantic' else 0) for n in branches})
            (ev/'records.jsonl').write_text(json.dumps(r)+'\n',encoding='utf-8')
            with (ev/'clean_curves.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['arm','graph_id','D','round','n','value_correct','first4_correct','syntax_correct','variable_correct','operation_correct'])
                w.writeheader();w.writerow(dict(arm='smoke',graph_id=0,D=2,round=2,n=2,value_correct=2,first4_correct=2,syntax_correct=2,variable_correct=2,operation_correct=2))
            subprocess.run([sys.executable,str(C.HERE/'summarize.py'),'--eval-dir',str(ev),'--out-dir',str(root/'summary')],check=True)
            with (root/'summary/event_summary.csv').open(encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            target=next(r for r in rows if r['branch']=='semantic' and r['group']=='target' and r['metric']=='value' and r['cohort']=='common_value')
            self.assertEqual((target['damaged'],target['ever_recovered'],target['final_two_correct_among_damaged']),('1','1','1'))
            with (root/'summary/evaluation_dose.csv').open(encoding='utf-8') as f:
                dose=list(csv.DictReader(f))
            row=next(r for r in dose if r['branch']=='semantic')
            self.assertEqual((row['input_wrong'],row['first_round_correct']),('True','True'))

if __name__=='__main__':unittest.main()
