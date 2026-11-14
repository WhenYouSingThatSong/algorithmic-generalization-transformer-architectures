"""CPU checks for the long-horizon summary (no model, no GPU).

Run: python -I tools/b_mechanism/tests/test_horizon_summary.py

These tests pin the four approved evaluation choices where they are easy to
get wrong: H truncation, last-two-round stable recovery, graph-equal versus
node weighting, clean-referenced damage, and the post-release offset for the
sustained control.  They do not validate model behaviour.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import b_mechanism_horizon_summary as H  # noqa: E402


def node(correct):
    return dict(value_correct=bool(correct), first4_correct=bool(correct))


def rounds_ok(pattern, pos):
    return [dict(output_nodes={str(pos): node(v)},
                 input_nodes={str(pos): node(v)}) for v in pattern]


class EventSummaryTests(unittest.TestCase):
    def summarize(self, jb, H_len, cb=None):
        cb = cb if cb is not None else [True] * len(jb)
        cache = dict(cache={('all_affected', 'value'): [(1, cb, jb)]}, n=H_len,
                     offset=0)
        return H.summarize_events(cache, H_len, stable_len=2)[
            ('all_affected', 'value')]

    def test_stable_last_two_and_only_last(self):
        cases = [
            ([True] * 3 + [False, False, True, True],
             dict(stable=1, only_last=0, ever=1)),
            ([False] + [True] * 5 + [False],
             dict(stable=0, only_last=0, ever=1)),
            ([False] * 5 + [True, True],
             dict(stable=1, only_last=0, ever=1)),
            ([False] * 6 + [True],
             dict(stable=0, only_last=1, ever=1)),
        ]
        for jb, exp in cases:
            with self.subTest(jb=jb):
                v = self.summarize(jb, len(jb))
                self.assertEqual(v['damaged'], 1)
                for key, value in exp.items():
                    self.assertEqual(v[key], value, key)

    def test_horizon_truncation_changes_stability(self):
        jb = [False, True, True, False, False, False]
        # Recovered at round 2 but the last two in-window rounds are not both
        # correct at H=2; H=3 sees [T, T] and calls it stable.
        self.assertEqual(self.summarize(jb, 2)['ever'], 1)
        self.assertEqual(self.summarize(jb, 2)['stable'], 0)
        self.assertEqual(self.summarize(jb, 3)['stable'], 1)
        self.assertEqual(self.summarize(jb, 6)['stable'], 0)
        # A recovery outside the window is not an in-window recovery.
        self.assertEqual(self.summarize(jb, 1)['ever'], 0)

    def test_damage_requires_same_round_clean_correct(self):
        v = self.summarize([False, True], 2, cb=[False, False])
        self.assertEqual(v['obs'], 0)
        self.assertEqual(v['damaged'], 0)
        self.assertEqual(v['events'], [])
        v = self.summarize([False, True], 2, cb=[True, False])
        self.assertEqual(v['obs'], 1)
        self.assertEqual(v['damaged'], 1)
        self.assertEqual(v['ever'], 1)


class RateViewTests(unittest.TestCase):
    def test_graph_equal_is_primary_and_node_weighted_is_secondary(self):
        # Graph 1: 1/1 damaged; graph 2: 1/99 damaged.
        r = H.rate_view({1: [1, 1], 2: [99, 1]}, 1, 0)
        self.assertAlmostEqual(r['node_weighted'], 2 / 100)
        self.assertAlmostEqual(r['graph_equal'], (1.0 + 1 / 99) / 2)
        self.assertEqual(r['n_zero_denominator_graphs'], 0)

    def test_zero_denominator_graphs_are_reported_not_forced(self):
        r = H.rate_view({1: [0, 0], 2: [1, 1]}, 1, 0)
        self.assertAlmostEqual(r['graph_equal'], 1.0)
        self.assertAlmostEqual(r['node_weighted'], 1.0)
        self.assertEqual(r['n_graphs_with_denominator'], 1)
        self.assertEqual(r['n_zero_denominator_graphs'], 1)


class RoleAndOffsetTests(unittest.TestCase):
    NODE_TRUTH = {'1': {'children': [3]}, '2': {'children': [1]},
                  '3': {'children': []}}

    def test_early_injected_positions_follow_the_2x2_condition(self):
        chain = {'pos': 1, 'r_pos': 2}
        base = dict(chain=chain, timing='after_u', node_truth=self.NODE_TRUTH)
        u = H.role_sets(dict(base, condition='u'))
        r = H.role_sets(dict(base, condition='r'))
        ur = H.role_sets(dict(base, condition='ur'))
        self.assertEqual(u['direct'], {1})
        self.assertEqual(u['target_u'], set())
        self.assertEqual(r['direct'], {2})
        self.assertEqual(r['target_u'], {1})
        self.assertEqual(ur['direct'], {1, 2})
        self.assertEqual(ur['target_u'], set())
        self.assertEqual(r['downstream_d1'], {3})

    def test_sustained_control_direct_is_the_forced_parent(self):
        chain = {'pos': 1, 'r_pos': 2}
        roles = H.role_sets(dict(chain=chain, mode='sustained_r',
                                 node_truth=self.NODE_TRUTH))
        self.assertEqual(roles['direct'], {2})
        self.assertEqual(roles['target_u'], {1})
        self.assertEqual(roles['downstream_d1'], {3})

    def test_sustained_horizon_starts_after_release(self):
        chain = {'pos': 1, 'r_pos': 2}
        pat_clean = [True] * 8
        pat_inj = [False, False, False, True, True, True, True, True]
        clean = dict(rounds=rounds_ok(pat_clean, 1))
        rec = dict(condition='sustained_r', mode='sustained_r',
                   hold_rounds=2, chain=chain, injection={'pos': 2},
                   node_truth=self.NODE_TRUTH,
                   rounds=rounds_ok(pat_inj, 1))
        cache = H.build_cache(rec, clean, max_rounds=4)
        self.assertEqual(cache['offset'], 2)
        self.assertEqual(cache['n'], 4)
        entries = cache['cache'][('target_u', 'value')]
        self.assertEqual([j for _p, _c, j in entries], [[False, True, True, True]])
        v = H.summarize_events(cache, 4, stable_len=2)[('target_u', 'value')]
        self.assertEqual(v['stable'], 1)
        self.assertEqual(v['ever'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
