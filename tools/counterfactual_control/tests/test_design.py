import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import design


class DesignTests(unittest.TestCase):
    def graph(self,a=4):
        return {0:[],1:[],2:[0,1],3:[2,0]}, {0:0,1:0,2:1,3:2}, {0:1,1:(a-1)%23}

    def test_all_original_values_and_wraparound(self):
        for a in range(23):
            ps,ds,vs=self.graph(a)
            selected,_=design.choose_candidates(ps,ds,vs,'g')
            c=selected[0]
            self.assertEqual(c['original_value'],a)
            self.assertEqual(c['donor_truth']['same'][2],a)
            self.assertEqual(c['donor_truth']['shift_1'][2],(a+1)%23)
            self.assertEqual(c['donor_truth']['shift_3'][2],(a+3)%23)
            for edits in c['changes'].values():
                self.assertEqual(set(edits),{0,1})
                self.assertTrue(all(0<v<23 for v in edits.values()))

    def test_nonunit_path_coefficients_and_matching(self):
        edits=design.matched_changes({5:2,9:3})
        for name,change in edits.items():
            expected=0 if name=='same' else int(name.split('_')[1])
            self.assertEqual((2*change[5]+3*change[9])%23,expected)
        self.assertEqual(len({c[5] for c in edits.values()}),1)
        self.assertIsNone(design.matched_changes({5:1,9:0}))

    def test_local_q_expectation_is_not_full_donor_truth(self):
        ps,ds,vs=self.graph()
        c=design.choose_candidates(ps,ds,vs,'g')[0][0]
        self.assertEqual(c['local_expected_q']['1']['3'],6)
        self.assertNotEqual(c['donor_truth']['shift_1'][3],6)

    def test_source_qualification_and_fixed_child_denominator(self):
        ps,ds,vs=self.graph()
        c=design.choose_candidates(ps,ds,vs,'g')[0][0]
        truth=[[0,i,1,c['truth'][i]] for i in range(4)]
        initial={n:copy.deepcopy(truth) for n in ['original',*c['changes']]}
        for n in c['changes']:
            initial[n][2][3]=c['donor_values'][n]
        flags,qs=design.qualify(truth,list(range(4)),ps,c,initial,truth)
        self.assertTrue(all(flags.values())); self.assertEqual(qs,[3])
        initial['same'][2][1]=99
        self.assertEqual(design.qualify(truth,list(range(4)),ps,c,initial,truth)[1],[])

    def test_selection_independent_of_dict_order_and_records_exclusions(self):
        ps,ds,vs=self.graph()
        one=design.choose_candidates(ps,ds,vs,'g')
        two=design.choose_candidates(dict(reversed(list(ps.items()))),ds,vs,'g')
        self.assertEqual(one,two)
        self.assertEqual(one[1][-1]['reason'],'no_next_round_child')

    def test_metrics_keep_nonnumeric_in_denominator(self):
        ps,ds,vs=self.graph()
        c=design.choose_candidates(ps,ds,vs,'g')[0][0]
        truth=[[0,i,1,c['truth'][i]] for i in range(4)]
        changed=copy.deepcopy(truth); changed[3][3]=24
        r=json.loads(json.dumps(dict(c,node_positions=list(range(4)),truth4=truth,
            eligible_children=[3],modulus=23,branches={
                'unchanged':{'outputs':[truth]},'same':{'outputs':[changed]}})))
        row=design.branch_metrics(r,'same',1)[0]
        self.assertEqual((row['q_n'],row['q_nonnumeric'],row['q_harmed']),(1,1,1))


if __name__=='__main__':
    unittest.main()
