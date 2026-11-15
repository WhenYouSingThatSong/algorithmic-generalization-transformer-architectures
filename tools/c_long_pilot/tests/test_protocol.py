import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protocol as P


class ProtocolTests(unittest.TestCase):
    def test_bucket_membership_and_per_graph_D(self):
        depths = [1, 4, 6, 1, 6, 6, 4]
        bs = P.batches(depths, 2)
        self.assertEqual(sorted(i for b in bs for i in b['ids']), list(range(7)))
        for b in bs:
            self.assertTrue(all(depths[i]+1 == b['D'] for i in b['ids']))
        self.assertEqual(bs, P.batches(depths, 2))
        self.assertEqual(sorted(P.epoch_order(len(bs), 1)), list(range(len(bs))))

    def test_labels_continue_final_truth_never_restart(self):
        self.assertEqual([P.target_index(t, 3) for t in range(1, 10)], [1,2,3,3,3,3,3,3,3])

    def test_solve_hold_weight_is_not_doubled_in_longer_arm(self):
        for d in (2, 7):
            for k in (1, 2):
                w = P.phase_weights(d,k)
                self.assertEqual(len(w),(1+k)*d)
                self.assertAlmostEqual(sum(w[:d]),0.5)
                self.assertAlmostEqual(sum(w[d:]),0.5)
            self.assertAlmostEqual(sum(P.phase_weights(d,0)),1)

    def test_budget_and_exposure_are_distinct(self):
        bs = P.batches([3, 6, 6, 4, 3],2)
        per_epoch = sum(len(b['ids'])*b['D'] for b in bs)
        costs = [P.EPOCHS[a]*(1+P.ARMS[a])*per_epoch for a in P.ARMS]
        self.assertEqual(len(set(costs)),1)
        self.assertEqual([P.EPOCHS[a] for a in P.ARMS],[6,3,2])
        # At common two epochs, exposures/updates agree but loop compute differs.
        self.assertEqual([2*(1+P.ARMS[a]) for a in P.ARMS],[2,4,6])

    def test_not_completed_is_not_a_stability_success(self):
        self.assertIsNone(P.post_events([False]*4)['ever_bad_after'])
        self.assertFalse(P.post_events([False,True])['post_observed'])

    def test_completion_recovery_and_reerror_are_separate(self):
        e=P.post_events([False,True,False,True,False])
        self.assertEqual(e['completion'],2)
        self.assertEqual(e['first_bad_after'],1)
        self.assertTrue(e['ever_recovered']);self.assertTrue(e['reerror'])
        self.assertFalse(P.post_events([True]*5)['ever_bad_after'])


if __name__ == '__main__':
    unittest.main()
