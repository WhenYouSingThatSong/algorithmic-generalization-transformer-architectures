#!/usr/bin/env python3
"""CPU-only checks for graph-equal vs node-weighted summary helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import summarize


class GraphEqualTests(unittest.TestCase):
    def test_graph_equal_differs_from_node_weighted(self):
        acc = {}
        # graph A: 1/1 correct; graph B: 1/3 correct
        summarize.add_node_counts(acc, np.array([True]), np.array([True]))
        summarize.add_node_counts(acc, np.array([True, False, False]),
                                  np.array([True, False, False]))
        self.assertAlmostEqual(acc['vc'] / acc['vn'], 0.5)  # node-weighted
        macro = summarize.macro_fields(acc)
        self.assertAlmostEqual(float(macro['value_node_acc_graph_equal']), 2 / 3, places=5)
        self.assertEqual(macro['n_graphs_with_nodes'], 2)
        self.assertEqual(macro['n_graphs_without_nodes'], 0)

    def test_empty_bucket_is_not_zero_score(self):
        acc = {}
        summarize.add_node_counts(acc, np.array([], dtype=bool),
                                  np.array([], dtype=bool))
        macro = summarize.macro_fields(acc)
        self.assertEqual(macro['n_graphs_with_nodes'], 0)
        self.assertEqual(macro['n_graphs_without_nodes'], 1)
        self.assertEqual(macro['value_node_acc_graph_equal'], '')
        self.assertEqual(macro['first4_node_acc_graph_equal'], '')


if __name__ == "__main__":
    unittest.main()
