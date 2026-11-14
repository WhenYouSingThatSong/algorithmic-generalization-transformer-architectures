"""Targeted CPU checks for the B execution-review fixes; no model loading."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_review_regressions import source_functions
import b_main_summary as M
import b_new_truth_summary as D


class ReviewTests(unittest.TestCase):
    def test_explicit_early_only_manifest_is_valid(self):
        validate = source_functions('b_mechanism_common.py',
                                    ['validate_manifest_graphs']).validate_manifest_graphs
        validate({'graphs': [{'graph_id': 7, 'T': None, 'n_loops': 8,
                              'common_complete': False}]})
        with self.assertRaises(SystemExit):
            validate({'graphs': [{'graph_id': 7, 'T': None, 'n_loops': 8}]})

    def test_unrelated_protected_and_shared_branches(self):
        rec = dict(condition='unrelated', target=dict(pos=4, for_u=1),
                   injection=dict(pos=4), node_truth={
                       '1': {'children': [2]}, '2': {'children': [3]},
                       '3': {'children': []}, '4': {'children': [3]}})
        roles = M.roles_for(rec)
        self.assertEqual(roles['protected_all'], {1, 2, 3})
        self.assertEqual(roles['protected_shared_descendants'], {3})
        self.assertEqual(roles['protected_exclusive'], {1, 2})
        self.assertEqual(roles['all_affected'], {3, 4})

    def test_d2_summarizes_new_truth_without_clean(self):
        rec = dict(condition='leaf_constant_scratch', graph_id=7, target_index=0,
                   target=dict(const_pos=0, var_pos=1),
                   node_truth={'1': dict(true='4', children=[2]),
                               '2': dict(true='13', children=[])},
                   new_truth={'0': 5, '1': 5, '2': 14})
        def flags(ok):
            return {str(p): dict(value_correct=ok, first4_correct=ok) for p in (0, 1, 2)}
        rec['rounds'] = [dict(output_nodes=flags(False), output_nodes_new=flags(True))
                         for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text(json.dumps(dict(records=[rec])), encoding='utf-8')
            self.assertEqual(D.summarize_files([path], tmp, horizons=(2,)), 1)
            with (Path(tmp) / 'd123_by_H.csv').open() as f:
                rows = list(csv.DictReader(f))
            nodes = [r for r in rows if r['role'] == 'all_variables' and
                     r['metric'] == 'value' and r['measure'] == 'correct']
            self.assertEqual({r['truth_ref']: r['numerator'] for r in nodes},
                             {'old': '0', 'new': '2'})
            self.assertTrue(all(r['denominator'] == '2' for r in nodes))
            self.assertEqual(D.role_sets(rec)['changed_value_variables'], {1, 2})


if __name__ == '__main__':
    unittest.main()
