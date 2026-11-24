"""Small CPU fixtures for complete evidence export and clean inclusion."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import b_compare_teacher_own as compare
import b_export_review_bundle as export


class ReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'run.jsonl'
        self.meta = {'args': {'config': 'B', 'run': 's42', 'ckpt': 'e92'}}
        self.clean = dict(graph_id=142, target_index=0, target={'pos': 26},
                          condition='clean', node_truth={'26': {'children': [128]}},
                          rounds=[{'input_nodes': {'26': {'value': 4}},
                                   'output_nodes': {'26': {'value': 4},
                                                    '128': {'value': 2}}}])
        self.injected = copy.deepcopy(self.clean)
        self.injected['condition'] = 'late_u'
        self.injected['rounds'][0]['input_nodes']['26']['value'] = 5
        self.cases = [{'graph_id': 142, 'condition': 'late_u', 'target_index': 0}]

    def write(self, records):
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        self.path.with_suffix('.meta.json').write_text(json.dumps(self.meta), encoding='utf-8')

    def test_complete_pair_retained_with_clean_after_injection(self):
        other_clean = {**self.clean, 'target_index': 1, 'target': {'pos': 70}}
        self.write([other_clean, self.injected, self.clean])
        pairs, sha = export.select_pairs(self.path, self.meta, self.cases)
        self.assertEqual(pairs[0]['clean'], self.clean)
        self.assertEqual(pairs[0]['injected'], self.injected)
        self.assertEqual(sha, export.sha256_file(self.path))

    def test_wrong_target_clean_is_not_used(self):
        self.write([self.injected, {**self.clean, 'target': {'pos': 70}}])
        with self.assertRaisesRegex(ValueError, 'missing matching clean'):
            export.select_pairs(self.path, self.meta, self.cases)

    def test_clean_comparison_is_included(self):
        self.write([self.clean, self.injected])
        records, _ = compare.add_records([self.path])
        self.assertEqual({k[2] for k in records}, {'clean', 'late_u'})

    def test_structure_keeps_third_parent_and_leaf_constant_reference(self):
        graph = export.graph_structure([[-1] * 6, [0, 0, -1, -1, -1, -1],
                                        [2, 1, 0, 4, 0, 7]])
        self.assertNotIn('0', graph)
        self.assertEqual(graph['1']['parents'], [0])
        self.assertEqual(graph['2']['parents'], [1, 4, 7])
        self.assertEqual(graph['2']['operations'], [0, 0])


if __name__ == '__main__':
    unittest.main()
