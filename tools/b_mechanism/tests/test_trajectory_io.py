"""CPU-only record equivalence and interruption checks; no GPU speed claim."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_io import StateRecorder, RecordWriter, write_json


class Tensor:
    """Instrument bulk transfer; production observer accepts torch tensors."""
    transfers = 0

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return Tensor(self.rows[key])

    def detach(self):
        return self

    def cpu(self):
        Tensor.transfers += 1
        return self

    def tolist(self):
        return copy.deepcopy(self.rows)


def old_nodes(rows, label, tok, positions):
    # Reference the pre-optimization scalar mapping and comparison rules.
    def fields(row):
        out = {}
        for name in tok.factors[:4]:
            idx = tok.factors.index(name)
            mapping = getattr(tok, name.lower() + '_idx2tok', None)
            out[name] = str(int(row[idx])) if mapping is None else mapping.get(
                int(row[idx]), str(int(row[idx])))
        return out
    result = {}
    for p in positions:
        pred4, true4 = fields(rows[p]), fields(label[p])
        result[str(p)] = dict(pos=p, pred4=pred4, true4=true4,
                              first4_correct=pred4 == true4,
                              value_correct=pred4['VALUE'] == true4['VALUE'])
    return result


class RecordingTests(unittest.TestCase):
    def test_fields_rounds_new_truth_and_bulk_transfers(self):
        tok = SimpleNamespace(factors=['SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE', 'RG', 'MD'],
                              syntax_idx2tok={0: 'VAR', 1: 'VALUE'},
                              variable_idx2tok={0: 'x0', 1: 'x1'},
                              operation_idx2tok={0: 'ADD'},
                              value_idx2tok={4: '4', 5: '5', 9: 'EMPTY'})
        label = [[0, 0, 0, 4, 0, 0], [1, 1, 0, 5, 0, 0]]
        wrong = [[0, 1, 7, 5, 1, 1], [1, 1, 0, 9, 0, 0]]
        new_label = copy.deepcopy(label)
        new_label[1][3] = 4  # includes changed constant position, with old truth retained
        Tensor.transfers = 0
        recorder = StateRecorder(tok, Tensor(label))
        new = StateRecorder(tok, Tensor(new_label))
        rows = recorder.rounds([Tensor([wrong]), Tensor([label])],
                               [Tensor([label]), Tensor([wrong])], [0, 1], 7,
                               new_recorder=new, truth_ref=True)
        expected = []
        for i, (inp, out) in enumerate(((wrong, label), (label, wrong)), 1):
            expected.append(dict(post_round=i, pre_forward_round=7,
                                 input={str(p): tok.value_idx2tok[inp[p][3]] for p in (0, 1)},
                                 output={str(p): tok.value_idx2tok[out[p][3]] for p in (0, 1)},
                                 input_nodes=old_nodes(inp, label, tok, (0, 1)),
                                 output_nodes=old_nodes(out, label, tok, (0, 1)),
                                 input_nodes_new=old_nodes(inp, new_label, tok, (0, 1)),
                                 output_nodes_new=old_nodes(out, new_label, tok, (0, 1)),
                                 truth_ref='old', truth_ref_new='new'))
        self.assertEqual(rows, expected)
        self.assertEqual(Tensor.transfers, 6)  # 2 labels + 2 inputs + 2 outputs, not per scalar
        plain = recorder.rounds([Tensor([wrong])], [Tensor([label])], [0], 3)[0]
        self.assertEqual(set(plain), {'post_round', 'pre_forward_round', 'input', 'output',
                                      'input_nodes', 'output_nodes'})

    def test_streamed_final_preserves_schema_and_counts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(out=str(Path(directory) / 'run.json'), progress_every=1)
            counts = dict(processed=1, emitted=0, skipped=0)
            writer = RecordWriter(args, counts, 1)
            record = {'graph_id': 3, 'rounds': [{'post_round': 1}], 'node_truth': {2: {'true': '4'}}}
            writer.append(record)
            self.assertEqual(len(writer.journal.read_text().splitlines()), 1)
            self.assertFalse(Path(args.out).exists())
            counts['emitted'] += 1
            writer.graph_done()
            writer.finish()
            expected = dict(args=vars(args), counts=counts, records=[record])
            self.assertEqual(json.loads(Path(args.out).read_text()), json.loads(json.dumps(expected)))
            self.assertTrue(json.loads(writer.meta.read_text())['complete'])
            with self.assertRaises(FileExistsError):
                RecordWriter(args, counts, 1)

    def test_interrupted_journal_retains_only_complete_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / 'partial.jsonl'
            journal.write_text('{"graph_id": 0}\n{"graph_id":')
            out = Path(directory) / 'recovered.json'
            count = write_json(journal, out, {'args': {}, 'counts': {}, 'partial': True},
                               allow_truncated=True)
            self.assertEqual(count, 1)
            data = json.loads(out.read_text())
            self.assertEqual(data['records'], [{'graph_id': 0}])
            self.assertTrue(data['partial'])


if __name__ == '__main__':
    unittest.main()
