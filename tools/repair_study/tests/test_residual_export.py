import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_b_residual as E


def rows_for(values):
    return [[[0, 0, 0, v] for v in values] for _ in range(16)]


def make_record():
    # Nodes: 0 (parent), 1 (target), 2 (persistent descendant).
    clean = rows_for([5, 5, 5])
    wrong = rows_for([5, 5, 5])
    for t in range(1, 16):
        wrong[t][2][3] = 3  # descendant becomes wrong at output round 2
    return dict(
        case_id='B:1:1:just_solved', timing='just_solved', graph_id=1,
        cohort='fixed_candidates', target_pos=1,
        parent_positions=[0], descendant_positions=[2],
        node_positions=[0, 1, 2], truth4=[[0, 0, 0, 5]] * 3,
        depths={'0': 0, '1': 1, '2': 2},
        all_parents={'0': [], '1': [0], '2': [1]},
        fan_in=1, state_round=2,
        branches={
            'unchanged': dict(initial=[[0, 0, 0, 5]] * 3, outputs=clean,
                              assigned_value=None),
            'wrong_plus_11': dict(initial=[[0, 0, 0, 5], [0, 0, 0, 16],
                                           [0, 0, 0, 5]], outputs=wrong,
                                   assigned_value=16),
        })


class ResidualExportTests(unittest.TestCase):
    def test_selects_and_exports_persistent_descendant(self):
        rec = make_record()
        selected = E.target_cases([rec])
        self.assertEqual(sorted(selected), [(1, 1)])
        rows, summaries = E.build_rows([rec], selected)
        self.assertEqual(len(summaries), 1)
        s = summaries[0]
        self.assertEqual(s['n_final_two_wrong'], 1)
        self.assertEqual(s['first_damage_round'], 2)
        self.assertEqual(s['first_persistent_round'], 2)
        self.assertEqual(s['first_persistent_nodes'], '2')
        self.assertEqual(s['target_final_two_correct_in_wrong'], True)
        node2 = [r for r in rows if r['node_pos'] == 2]
        self.assertEqual([r['wrong_output'] for r in node2[:3]], [5, 3, 3])
        self.assertEqual([r['clean_output'] for r in node2[:3]], [5, 5, 5])
        self.assertTrue(all(r['final_two_wrong'] for r in node2))

    def test_clean_early_error_is_not_counted_as_damage(self):
        rec = make_record()
        rec['branches']['unchanged']['outputs'][0][2][3] = 0
        selected = E.target_cases([rec])
        rows, summaries = E.build_rows([rec], selected)
        self.assertEqual(summaries[0]['n_final_two_wrong'], 1)
        node2 = [r for r in rows if r['node_pos'] == 2]
        self.assertFalse(node2[0]['clean_output_correct'])
        self.assertFalse(node2[0]['is_damage_round'])
        self.assertTrue(node2[1]['is_damage_round'])
        self.assertEqual(summaries[0]['first_damage_round'], 2)


if __name__ == '__main__':
    unittest.main()
