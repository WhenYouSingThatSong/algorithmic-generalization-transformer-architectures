"""CPU tests for the existing-NPZ C analysis script.

No model, torch, GPU or real NPZ file is needed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import analyze_existing as ae  # noqa: E402


def make_graph(value_all, *, tV=2, t4=2, ref=2, var_wrong=(), value_wrong=(),
               actual_max_depth=1, weight="w", stratum="s", graph_id="g"):
    value_all = np.asarray(value_all, dtype=bool)
    R = len(value_all)
    N = 3
    true_syntax = np.zeros(N, dtype=np.int16)
    true_variable = np.array([1, 2, 3], dtype=np.int16)
    true_operation = np.ones(N, dtype=np.int16)
    true_value = np.array([0, 1, 2], dtype=np.int16)
    pred_variable = np.tile(true_variable, (R, 1))
    pred_value = np.tile(true_value, (R, 1))
    value_correct = np.repeat(value_all[:, None], N, axis=1)
    for r, node in var_wrong:
        pred_variable[r, node] = (pred_variable[r, node] + 1) % 4
    for r, node in value_wrong:
        value_correct[r, node] = False
        pred_value[r, node] = (pred_value[r, node] + 1) % 4
    first4_correct = value_correct & (pred_variable == true_variable[None, :])
    return dict(
        weight=weight, stratum=stratum, graph_id=graph_id, source_index=0,
        path="g.npz", data_sha256="x",
        pred_syntax=np.tile(true_syntax, (R, 1)),
        true_syntax=true_syntax,
        pred_variable=pred_variable, true_variable=true_variable,
        pred_operation=np.tile(true_operation, (R, 1)),
        true_operation=true_operation,
        pred_value=pred_value, true_value=true_value,
        value_correct=value_correct, first4_correct=first4_correct,
        graph_value_all=value_correct.all(axis=1),
        graph_first4_all=first4_correct.all(axis=1),
        depth=np.array([0, 1, 1], dtype=np.int16),
        is_internal=np.array([False, True, True]),
        is_sink=np.array([False, False, True]),
        actual_max_depth=actual_max_depth,
        completion_ref_round=ref, tV=tV, t4=t4, n_rounds=R,
        positions=np.array([0, 1, 2]),
    )


class EventTests(unittest.TestCase):
    def test_stable_recovered_uses_last_two_not_final(self):
        # post-tV = [True, True, False, True] -> final correct but last two are
        # [False, True], so this is NOT stable recovery.
        g = make_graph([False, True, True, True, False, True])
        e = ae.graph_events(g)
        self.assertTrue(e["value_final_correct"])
        self.assertFalse(e["value_stable_recovered"])
        self.assertEqual(e["primary_event"], "value_degraded_final_correct_unstable")

    def test_stable_recovered_requires_last_two(self):
        # post-tV = [True, False, True, True] -> last two correct.
        g = make_graph([False, True, True, False, True, True])
        e = ae.graph_events(g)
        self.assertTrue(e["value_final_correct"])
        self.assertTrue(e["value_stable_recovered"])
        self.assertEqual(e["value_first_degradation_rel_round"], 2)
        self.assertEqual(e["primary_event"], "value_degraded_recovered_stable")

    def test_variable_event_is_correct_to_wrong_transition(self):
        # At tV all variables are correct; only node 1 becomes wrong at round 4.
        g = make_graph([False, True, True, True, True, True], var_wrong=[(3, 1)])
        e = ae.graph_events(g)
        self.assertTrue(e["variable_all_at_tV"])
        self.assertEqual(e["variable_first_new_error_rel_round"], 2)
        self.assertEqual(e["variable_affected_nodes_first"], 1)
        self.assertEqual(e["variable_affected_depths_first"], "1")
        self.assertTrue(e["value_never_degraded"])
        self.assertEqual(e["primary_event"], "variable_degraded_after_tV_value_preserved")

    def test_variable_already_wrong_at_tv_is_separate(self):
        g = make_graph([False, True, True, True, True, True], var_wrong=[(1, 0)])
        e = ae.graph_events(g)
        self.assertFalse(e["variable_all_at_tV"])
        self.assertIsNone(e["variable_first_new_error_rel_round"])
        self.assertEqual(e["primary_event"], "variable_already_wrong_at_tV_value_preserved")

    def test_never_and_delayed(self):
        never = ae.graph_events(make_graph([False, False, False, False], tV=-1))
        self.assertEqual(never["primary_event"], "never_value_complete")
        delayed = ae.graph_events(make_graph([False, False, False, True, True, True], tV=4, ref=2))
        self.assertTrue(delayed["delayed"])
        self.assertEqual(delayed["primary_event"], "delayed_completion_only")

    def test_event_summary_counts(self):
        graphs = [
            make_graph([False, True, True, True, True, True], var_wrong=[(3, 1)]),
            make_graph([False, True, True, False, True, True]),
            make_graph([False, False, False, False], tV=-1),
        ]
        summary = ae.event_summary_rows(ae.event_rows(graphs))
        row = summary[0]
        self.assertEqual(row["n_graphs"], 3)
        self.assertEqual(row["n_completed"], 2)
        self.assertEqual(row["n_never_complete"], 1)
        self.assertEqual(row["n_value_degraded_recovered_stable"], 1)
        self.assertEqual(row["n_variable_degraded_value_preserved"], 1)


class TableTests(unittest.TestCase):
    def test_field_accuracy_scope_and_denominator(self):
        # At H=2 all three nodes are correct; at H=3 one internal value is wrong.
        g1 = make_graph([False, True, True, True], value_wrong=[(2, 1)])
        g2 = make_graph([False, True, True, True])
        rows = ae.field_accuracy_rows([g1, g2], horizons=[2, 3])
        at2 = {(r["field"], r["scope"]): r for r in rows if r["H"] == 2}
        self.assertAlmostEqual(at2[("value", "all")]["acc"], 1.0)
        self.assertAlmostEqual(at2[("value", "internal")]["acc"], 1.0)
        at3 = {(r["field"], r["scope"]): r for r in rows if r["H"] == 3}
        self.assertAlmostEqual(at3[("value", "all")]["acc"], 5 / 6)
        self.assertAlmostEqual(at3[("value", "internal")]["acc"], 3 / 4)
        self.assertEqual(at3[("value", "all")]["n_graphs_with_nodes"], 2)

    def test_completion_lag(self):
        g = make_graph([False, False, False, True], tV=4, ref=2)
        rows = ae.completion_lag_rows([g])
        self.assertEqual(rows[0]["completed"], 1)
        self.assertEqual(rows[0]["lag2_4"], 1)
        self.assertEqual(rows[0]["max_lag"], 2)


if __name__ == "__main__":
    unittest.main()