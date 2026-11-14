"""CPU checks for long-window pairing and checkpoint parameterization."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import b_long_window as L
import b_main_summary as M


def round_rec(ok):
    return {'output_nodes': {'1': {'value_correct': bool(ok),
                                   'first4_correct': bool(ok)}}}


class LongWindowTests(unittest.TestCase):
    def test_late_target_keys_are_distinct(self):
        args = {}
        a = dict(graph_id=32, target_index=0, target={'pos': 10},
                 condition='clean', timing='complete', start='late')
        b = dict(graph_id=32, target_index=1, target={'pos': 20},
                 condition='clean', timing='complete', start='late')
        self.assertNotEqual(M.clean_key(a, 'src', args),
                            M.clean_key(b, 'src', args))

    def test_checkpoint_round_is_used(self):
        clean = {'rounds': [round_rec(True) for _ in range(4)]}
        rec = {'rounds': [round_rec(x) for x in (True, True, False, True)]}
        roles = {'all_affected': {1}}
        out = L.analyze_pair(clean, rec, roles, 4, checkpoint_round=2)
        v = out[('all_affected', 1)]
        self.assertEqual(v['n_cp'], 1)
        self.assertEqual(v['post'], 1)          # round 2 (zero-based) is after cp=2
        self.assertEqual(v['damaged'], 1)
        self.assertEqual(v['first_damage'], 3)
        self.assertEqual(v['first_recovery'], 4)
        self.assertEqual(v['stable'], 0)  # last two rounds are False, True

    def test_even_median_uses_statistics(self):
        self.assertEqual(L.median_final_start([2, 4]), 3.0)


if __name__ == '__main__':
    unittest.main()