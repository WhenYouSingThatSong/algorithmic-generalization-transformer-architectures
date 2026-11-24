import sys,unittest,json,csv,tempfile,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import suite_core as U
from analyze_existing import joint_rows
from summarize_pair import pair_trajectory

class BasicTests(unittest.TestCase):
    def test_pair_donor_is_wrong_for_child_substitution(self):
        tokens=[[0,p,1,v,0,0] for p,v in enumerate([1,3,4,7])]+[[2,128,1,1,0,0],[2,128,1,3,0,0]]
        info=[[0,4,-1,-1,-1,-1,-1],[0,5,-1,-1,-1,-1,-1],[1,0,0,1,-1,-1,-1],[2,2,0,1,-1,-1,-1]]+[[-1]*7]*2
        case=U.C.make_case(tokens,info,11);p=U.pair_plan(case,tokens,info)
        self.assertEqual((p['u'],p['q']),(2,3))
        self.assertEqual(p['expected_q'],(p['expected_u']+3)%23)
        for name,edits in p['q_case']['changes'].items():
            leaves={0:1,1:3}
            for leaf,delta in edits.items():leaves[leaf]=(leaves[leaf]+delta)%23
            values=U.C.DESIGN.evaluate(p['parents'],p['depths'],leaves)
            self.assertEqual(values[3],7 if name=='same' else p['expected_q'])
    def test_type_groups_do_not_mix_leaf_with_sink(self):
        groups=U.node_groups([0,1,2],{0:0,1:1,2:2},{0:[],1:[0],2:[1]})
        self.assertEqual([groups[k] for k in ('leaf','internal','sink')],[[0],[1],[2]])
    def test_failed_to_finish_is_not_stable(self):
        ev=U.clean_events([False]*6,2)
        self.assertIsNone(ev['first_all']);self.assertFalse(ev['post_completion_observed']);self.assertIsNone(ev['postD_failure'])
    def test_completion_failure_recovery_and_reerror(self):
        ev=U.clean_events([False,True,False,True,False,True],2)
        self.assertEqual(ev['first_all'],2);self.assertTrue(ev['recovered_after_failure']);self.assertTrue(ev['reerror']);self.assertFalse(ev['final_two_all'])
    def test_readout_unchanged_can_still_have_child_damage(self):
        g=[[0,1,1,4],[0,2,1,13]];wrong=[g[0],[0,2,1,14]]
        r=dict(arm='hold',graph_id=0,timing='hold',target=0,positions=[0,1],truth4=g,children=[1],expected_q={'1':14},
            branches={'unchanged':dict(initial=g,outputs=[g]),'noise':dict(initial=g,outputs=[wrong])})
        row=next(x for x in joint_rows(r) if x['branch']=='noise')
        self.assertFalse(row['input_readout_changed']);self.assertEqual(row['child_harmed'],1)
    def test_parent_sum_uses_input_not_same_round_output(self):
        g=[[0,1,1,4],[0,2,1,13],[0,3,1,9]];start=[[0,1,1,5],g[1],g[2]];out=[g[0],[0,2,1,14],g[2]]
        r=dict(arm='hold',graph_id=0,timing='hold',u=0,q=1,positions=[0,1,2],truth4=g,parents={'0':[],'1':[0,2],'2':[]},
            branches={'unchanged':dict(initial=g,outputs=[g]),'u_wrong':dict(initial=start,outputs=[out])})
        row=pair_trajectory(r,'u_wrong')[0]
        self.assertEqual((row['u_output'],row['q_input_parent_sum'],row['q_output']),(4,14,14))
        self.assertTrue(row['q_matches_input_parent_readouts'])

try:import torch
except ImportError:torch=None
@unittest.skipIf(torch is None,'torch check delegated to server')
class TensorTests(unittest.TestCase):
    def test_branch_copy_touches_only_designated_rows(self):
        src=torch.arange(5*6*4,dtype=torch.float32).reshape(5,6,4)
        names=['original','u_same','u_wrong','q_same','q_wrong']
        states,norms=U.make_branches(src,1,3,names);idx=list(U.BRANCHES)
        self.assertTrue(torch.equal(states[idx.index('sham')],src[0]));self.assertEqual(norms['sham'],0)
        self.assertTrue(torch.equal(states[idx.index('both_wrong'),1],src[2,1]))
        self.assertTrue(torch.equal(states[idx.index('both_wrong'),3],src[4,3]))
        self.assertTrue(torch.equal(states[idx.index('both_wrong'),[0,2,4,5]],src[0,[0,2,4,5]]))

if __name__=='__main__':unittest.main()
