#!/usr/bin/env python3
"""CPU-only tests for the torch-free C capability helpers.

These tests do not load a model or generate data.  Run with:
    python -I tools/c_capability/tests/test_c_common.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import c_common


class SelectionTests(unittest.TestCase):
    def test_depth_group_and_deterministic_selection(self):
        depths = np.array([3, 4, 5, 6, 7, 8, 12, 13, 7, 6])
        shallow, n_shallow = c_common.select_indices(depths, "shallow", 4, seed=7)
        deep, n_deep = c_common.select_indices(depths, "deep", 4, seed=7)
        self.assertEqual(n_shallow, 5)
        self.assertEqual(n_deep, 4)
        self.assertEqual(len(shallow), 4)
        self.assertEqual(len(deep), 4)
        self.assertTrue(all(depths[i] <= 6 for i in shallow))
        self.assertTrue(all(7 <= depths[i] <= 12 for i in deep))
        shallow2, _ = c_common.select_indices(depths, "shallow", 4, seed=7)
        self.assertEqual(shallow, shallow2)

    def test_insufficient(self):
        selected, eligible = c_common.select_indices(
            np.array([3, 4, 12]), "deep", 4, seed=1)
        self.assertEqual(eligible, 1)
        self.assertEqual(selected, [2])


class NodeMetadataTests(unittest.TestCase):
    def test_parents_and_sink(self):
        info = np.full((6, 6), -1, dtype=np.int64)
        # numeric nodes are positions 2..5; 0 and 1 are value-token parents
        info[2, 0] = 0
        info[2, 1] = 0
        info[2, 2] = 1
        info[3, 0] = 0
        info[3, 1] = 1
        info[3, 2] = 1
        info[4, 0] = 1
        info[4, 1] = 2
        info[4, 2] = 1
        info[4, 3] = 3
        info[4, 4] = 1
        info[5, 0] = 2
        info[5, 1] = 4
        info[5, 2] = 1
        numeric = np.zeros((1, 6), dtype=bool)
        numeric[0, 2:] = True
        nodes = c_common.node_meta(info[None, ...], numeric)[0]
        self.assertEqual(nodes["positions"].tolist(), [2, 3, 4, 5])
        self.assertEqual(nodes["depth"].tolist(), [0, 0, 1, 2])
        self.assertEqual(nodes["is_leaf"].tolist(), [True, True, False, False])
        self.assertEqual(nodes["is_internal"].tolist(), [False, False, True, True])
        self.assertEqual(nodes["is_sink"].tolist(), [False, False, False, True])
        self.assertEqual(nodes["parents"][4], [2, 3])


class CompletionTests(unittest.TestCase):
    def test_first_true(self):
        self.assertEqual(c_common.first_true_index([False, False, True, False]), 3)
        self.assertIsNone(c_common.first_true_index([False, False]))

    def test_completion_drift(self):
        all_correct = np.array([False, True, True, False, True, True])
        drift = c_common.completion_drift(all_correct, 2)
        self.assertTrue(drift["observed"])
        self.assertEqual(drift["post_rounds"], 4)
        self.assertTrue(drift["ever_degraded"])
        self.assertEqual(drift["first_degradation_rel_round"], 2)
        self.assertTrue(drift["recovered"])
        self.assertTrue(drift["final_correct"])

    def test_no_post_rounds(self):
        drift = c_common.completion_drift([False, True], 2)
        self.assertEqual(drift["post_rounds"], 0)
        self.assertFalse(drift["ever_degraded"])
        self.assertIsNone(drift["final_correct"])

    def test_other_field_drift(self):
        value_all = np.array([False, True, True, True, True])
        first4_all = np.array([False, True, False, False, True])
        drift = c_common.other_field_drift(value_all, first4_all, 2)
        self.assertTrue(drift["ever_other_only_degraded"])
        self.assertEqual(drift["first_other_only_rel_round"], 1)
        self.assertTrue(drift["final_value_correct"])
        self.assertTrue(drift["final_first4_correct"])


class ConfigTests(unittest.TestCase):
    def test_generated_config(self):
        cfg = c_common.generated_dag_config(128, "shallow", 16)
        self.assertEqual(cfg["num_nodes"], 128)
        self.assertEqual(cfg["max_depth"], 6)
        self.assertEqual(cfg["num_samples"], 16)
        self.assertEqual(cfg["func_vocab"], ["ADD"])


if __name__ == "__main__":
    unittest.main()
