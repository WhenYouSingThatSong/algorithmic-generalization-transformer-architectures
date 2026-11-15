import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import multifanin
import core


class MultiTests(unittest.TestCase):
    def test_all_fanins_selected_without_model_or_value_filter(self):
        ps={0:[],1:[],2:[],3:[0],4:[0,1],5:[0,1,2],6:[3,4,5]}
        ds={0:0,1:0,2:0,3:1,4:1,5:1,6:2}
        targets,coverage=multifanin.select_targets(ps,ds,42)
        self.assertEqual(targets,[3,4,5])
        self.assertEqual(multifanin.select_targets(dict(reversed(list(ps.items()))),ds,42),(targets,coverage))

    def test_exhaustive_values_cover_every_other_residue_once(self):
        for value in (0,4,22):
            branches=multifanin.value_branches(value,23,24,True)
            wrong=[v for name,v in branches if name.startswith('wrong_')]
            self.assertEqual(set(wrong),set(range(23))-{value})
            self.assertEqual(len(wrong),22)
        self.assertEqual(len(multifanin.value_branches(22,23,24)),7)

    def test_last_two_is_not_last_only_or_always_correct_recovery(self):
        truth=[[0,0,1,4]];wrong=[[0,0,1,5]]
        self.assertEqual(core.window_counts(truth,[truth]*3,[wrong,truth,truth],[0])['wrong_then_final_two_correct'],1)
        self.assertEqual(core.window_counts(truth,[truth]*3,[truth,wrong,truth],[0])['final_two_harmed'],1)
        self.assertEqual(core.window_counts(truth,[truth]*3,[truth]*3,[0])['wrong_then_final_two_correct'],0)
