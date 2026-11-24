import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import protocol as P
sys.path.insert(0,str(HERE.parent/'priority_pilots'))
from io_utils import sha,write_json


class SummaryTest(unittest.TestCase):
    def test_invalid_commit_not_dropped_and_probe_is_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dirs=[]
            gold=[[0,0,1,4]]*32
            for model in ('A','B'):
                folder=root/model;folder.mkdir();dirs.append(folder)
                rr=[]
                for stage in (1,2):
                    ref=dict(cumulative_logical_nodes=32+(stage-1)*26,
                             output_ancestor_operations=stage*26,logical_depth=stage*3)
                    r=dict(case='w2_f2:000',stratum='w2_f2',model=model,family=model,mode='program',
                           stage=stage,branch='carry',diagnostic=False,reference=ref,status='complete')
                    if model=='A' and stage==2:r['status']='invalid_commit'
                    else:
                        outputs=[gold,gold];r['metrics']=P.measures(outputs,gold,list(range(32)),[30,31])
                        r.update(nodes=list(range(32)),truth4=gold,trace=dict(outputs=outputs))
                    rr.append(r)
                for branch,value in [('probe_continue',5),('probe_reset_results',4)]:
                    outputs=[[[0,0,1,value]]*32]*2
                    rr.append(dict(case='w2_f2:000',stratum='w2_f2',model=model,family=model,mode='program',
                         stage=1,branch=branch,diagnostic=True,status='complete',nodes=list(range(32)),truth4=gold,
                         trace=dict(outputs=outputs),metrics=P.measures(outputs,gold,list(range(32)),[30,31])))
                (folder/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr),encoding='utf8')
                write_json(folder/'meta.json',dict(protocol=P.VERSION,status='complete',mode='program',
                    model=dict(id=model),task_sha256='same',per_stratum=1,stages=2,slack=8,
                    probe_per_stratum=1,cases=['w2_f2:000'],records_sha256=sha(folder/'records.jsonl')))
            args=[sys.executable,str(HERE/'summarize.py'),'--expect-models','A','B','--out-dir',str(root/'out')]
            for d in dirs:args+=['--run-dir',str(d)]
            subprocess.run(args,check=True,capture_output=True)
            def read(name):
                with (root/'out'/name).open(encoding='utf8') as f:return list(csv.DictReader(f))
            trajectories=read('programs.csv')
            self.assertEqual(trajectories[0]['first_invalid_commit'],'2')
            self.assertEqual(trajectories[0]['final_terminal_success'],'False')
            self.assertEqual(trajectories[1]['final_terminal_success'],'True')
            end=next(r for r in read('summary.csv') if r['model']=='A' and r['stage']=='2')
            self.assertEqual(float(end['cases']),1);self.assertEqual(float(end['nodes_value_macro']),0)
            pair=read('probe_pairs.csv')[0]
            self.assertEqual(pair['repaired'],'32');self.assertEqual(pair['damaged'],'0')


if __name__=='__main__':unittest.main()
