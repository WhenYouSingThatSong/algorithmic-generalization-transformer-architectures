import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core


class CoreTests(unittest.TestCase):
    def test_wrong_values_are_distinct_and_empty_is_not_a_number(self):
        self.assertEqual(core.b_variants(22, 23, 24),
                         [('unchanged',22),('correct_write',22),('empty',24),
                          ('wrong_plus_1',0),('wrong_plus_3',2)])

    def test_c_parent_status_is_previous_input_not_repaired_output(self):
        truth = [[0,0,1,4],[0,1,1,8]]
        pred = [copy.deepcopy(truth) for _ in range(5)]
        pred[1][0][3] = 5
        pred[2][1][3] = 9
        events = core.c_events(pred, truth, [10,20], {10:[],20:[10]})
        event = events[0]
        self.assertEqual(event['state_round'],3)
        self.assertEqual(event['category'],'value_parent_wrong')
        self.assertEqual(event['parent_values_before_error'],[5])
        self.assertEqual(event['donor_round'],1)

    def test_matched_control_has_same_gap_and_no_future_donor(self):
        truth = [[0,0,1,4],[0,1,1,8]]
        pred = [copy.deepcopy(truth) for _ in range(7)]
        pred[5][1][3] = 7
        pred[6][1][3] = 2
        donor,control = core.select_donor_and_control(pred,truth,1,[0],7)
        self.assertEqual((donor,control),(5,5))
        self.assertEqual(control-(7-donor),3)

    def test_variable_only_requires_value_kept_in_observed_future(self):
        truth = [[0,0,1,4],[0,1,1,8]]
        pred = [copy.deepcopy(truth) for _ in range(6)]
        pred[2][1][1] = 9
        events = core.c_events(pred,truth,[10,20],{10:[],20:[10]})
        self.assertEqual(events[0]['category'],'variable_only')
        pred[4][1][3] = 7
        self.assertEqual(core.c_events(pred,truth,[10,20],{10:[],20:[10]})[0]['category'],'value_parent_correct')

    def test_sampling_is_order_independent_and_does_not_repeat_graph(self):
        cases=[dict(weight='w',stratum='s',category='c',graph_id=i//2,case_id=str(i)) for i in range(8)]
        a=core.balanced_sample(cases,3,5)
        b=core.balanced_sample(list(reversed(cases)),3,5)
        self.assertEqual(a,b)
        self.assertEqual(len({c['graph_id'] for c in a}),3)

    def test_post_completion_search_does_not_stop_at_earlier_error(self):
        truth = [[0,0,1,4],[0,1,1,8],[0,2,1,12]]
        pred = [copy.deepcopy(truth) for _ in range(8)]
        for t in range(3):
            pred[t][2][3] = 24
        pred[1][1][3] = 9
        pred[6][1][3] = 7
        events = core.c_events(pred,truth,[10,20,30],{10:[],20:[10],30:[]})
        self.assertEqual(events[0]['graph_first_complete'],4)
        self.assertEqual(events[0]['state_round'],7)
        self.assertEqual(events[0]['phase'],'post_complete')

    def test_baseline_relative_help_harm_and_empty_roles(self):
        truth=[[0,0,1,4],[0,1,1,8]]
        bad=copy.deepcopy(truth);bad[0][3]=5
        treated=copy.deepcopy(truth);treated[1][3]=9
        rec=dict(node_positions=[10,20],truth4=truth,branches={'unchanged':{'outputs':[bad]}})
        row=core.pair_metrics(rec,{'outputs':[treated]},[10,20])[0]
        self.assertEqual((row['helped'],row['harmed'],row['n']),(1,1,2))
        self.assertEqual(core.pair_metrics(rec,{'outputs':[treated]},[])[0]['n'],0)


if __name__ == '__main__':
    unittest.main()
