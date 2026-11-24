import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from summarize import summarize

class SummaryTests(unittest.TestCase):
    def test_immediate_direct_repair_and_downstream_relapse(self):
        truth=[[0,0,1,4],[0,1,1,9],[0,2,1,13]]
        clean=[dict(input=truth,output=[[v[3],True,True,True,True] for v in truth]) for _ in range(5)]
        rows=[]
        for t,v in enumerate((14,13,24,13,13)):
            inp=copy.deepcopy(truth);inp[0][3]=5 if t==0 else 4
            rows.append(dict(input=inp,output=[[4,True,True,True,True],[9,True,True,True,True],[v,v==13,True,True,True]]))
        r=dict(module='pair',split='test',condition='u_wrong',status='ok',gid=0,nodes=[0,1,2],
               truth=truth,D=2,rows=rows,clean=clean,parents=[[],[],[0,1]],depths=[0,0,1],
               u=0,q=2,positions=[0],downstream=[2],hold=1,timing='early',input_qualified=True)
        out=summarize([r]);direct=next(z for z in out['counts'] if z['key'][-2:]==['direct','value'])['counts']
        self.assertEqual((direct['actual_injected'],direct['first_round_repair'],direct['damaged'],direct['stable']),(1,1,1,1))
        q=next(z for z in out['paired'] if z['role']=='q' and z['metric']=='value')
        self.assertEqual(q['mean_first_restore_round'],2);self.assertEqual(q['mean_stable_start_round'],4)
        qcounts=next(z for z in out['counts'] if z['key'][-2:]==['q','value'])['counts']
        self.assertEqual(qcounts['reerror'],1)
        self.assertEqual(q['correct_r1'],0);self.assertEqual(q['correct_r2'],1)
    def test_rejected_sources_remain_in_coverage(self):
        r=dict(module='pair',split='test',condition='u_wrong',status='source_rejected')
        out=summarize([r]);self.assertEqual(out['coverage'][0]['count'],1);self.assertEqual(out['paired'],[])

if __name__=='__main__':unittest.main()
