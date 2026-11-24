"""CPU-only regression checks for experiment plumbing, not model performance.

Run: python -I tools/b_mechanism/tests/test_review_regressions.py

Tensor-dependent modules are not imported: selected functions (including the
actual controls hook) are compiled from their source AST with small test doubles.
The summary CLI is executed unmodified in a subprocess on synthetic records.
These checks do not replace a GPU rollout or validate the evaluation protocol.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


TOOLS = Path(__file__).resolve().parents[1]


def source_functions(filename, names, globals_=None):
    path = TOOLS / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    missing = set(names) - functions.keys()
    if missing:
        raise AssertionError(f'{filename} missing functions: {sorted(missing)}')
    module = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class EligibilityAndBudgetTests(unittest.TestCase):
    def setUp(self):
        self.common = source_functions(
            'b_mechanism_common.py',
            ['early_flag_key', 'set_early_flag', 'early_common_flag',
             'validate_manifest_graphs'],
            {'_EARLY_SUFFIX': {'before_u': 'before', 'after_u': 'after'}})

    def test_common_eligibility_includes_one_model_ineligible(self):
        for start in ('own', 'teacher'):
            for timing in ('before_u', 'after_u'):
                for a_ok, b_ok in ((True, True), (True, False),
                                   (False, True), (False, False)):
                    with self.subTest(start=start, timing=timing, a=a_ok, b=b_ok):
                        chain = {'event_id': 'g0:c0'}
                        self.common.set_early_flag(chain, start, timing, a_ok, b_ok)
                        self.assertEqual(self.common.early_common_flag(
                            chain, start, timing), a_ok and b_ok)

    def test_missing_and_inconsistent_eligibility_fail(self):
        with self.assertRaises(SystemExit):
            self.common.early_common_flag({}, 'own', 'after_u')
        chain = {'early_own_after': True, 'early_own_after_a': True,
                 'early_own_after_b': False}
        with self.assertRaises(SystemExit):
            self.common.early_common_flag(chain, 'own', 'after_u')

    def test_common_completion_outside_graph_budget_is_rejected(self):
        functions = source_functions('b_mechanism_select.py', ['find_common_T'],
            {'C': SimpleNamespace(all_numeric_correct=lambda output, *_: output)})
        # First common success is at index 10 (the eleventh forward).
        outputs = [False] * 10 + [True] * 6
        self.assertIsNone(functions.find_common_T(outputs, outputs, None, None, 8))
        self.assertEqual(functions.find_common_T(outputs, outputs, None, None, 16), 10)
        with self.assertRaises(SystemExit):
            self.common.validate_manifest_graphs(
                {'graphs': [{'graph_id': 0, 'T': 10, 'n_loops': 8}]})


class ControlScheduleTests(unittest.TestCase):
    @staticmethod
    def schedule(mode, hold, switch, horizon):
        a = SimpleNamespace(mode=mode, hold_rounds=hold, switch_at=switch)
        functions = source_functions('b_mechanism_controls.py', ['hook'], {
            'a': a, 'r_pos': 1, 'r_wrong': 7, 'r_other': 10, 'tok': None,
            'set_r_value': lambda current, tok, pos, value: value,
        })
        return [functions.hook(t, 'model_output') for t in range(horizon)]

    def test_hold_then_switch_once_then_release(self):
        self.assertEqual(self.schedule('switch_held_r', 4, 5, 8),
                         [7, 7, 7, 7, 10, 'model_output', 'model_output', 'model_output'])
        self.assertEqual(self.schedule('sustained_r', 4, 0, 8),
                         [7, 7, 7, 7, 'model_output', 'model_output',
                          'model_output', 'model_output'])

    def test_every_accepted_switch_actually_occurs_once(self):
        validate = source_functions('b_mechanism_controls.py',
                                    ['validate_control_schedule']).validate_control_schedule
        for hold in range(1, 9):
            horizon = hold + 4
            for switch in range(0, horizon + 2):
                with self.subTest(hold=hold, switch=switch):
                    if switch != hold + 1:
                        with self.assertRaises(ValueError):
                            validate('switch_held_r', hold, switch, horizon)
                    else:
                        validate('switch_held_r', hold, switch, horizon)
                        actual = self.schedule('switch_held_r', hold, switch, horizon)
                        self.assertEqual(actual.count(10), 1)
                        self.assertEqual(actual[:hold], [7] * hold)
                        self.assertEqual(actual[switch:], ['model_output'] * (horizon - switch))
        with self.assertRaises(ValueError):
            validate('switch_held_r', 4, 5, 4)
        with self.assertRaises(ValueError):
            validate('sustained_r', -1, 0, 8)
        with self.assertRaises(ValueError):
            validate('sustained_r', 4, 0, 0)


class ValueLabel:
    """Minimal indexing/clone double; no tensor arithmetic is under test."""
    def __init__(self, values):
        self.values = dict(values)

    def clone(self):
        return ValueLabel(self.values)

    def __getitem__(self, key):
        return self.values[key[0]]

    def __setitem__(self, key, value):
        self.values[key[0]] = value


class NewTruthTests(unittest.TestCase):
    def test_new_constant_leaf_and_descendant_share_new_truth(self):
        # constant[0]=4 -> x[1]=4; constant[2]=2 -> z[3]=2; y[4]=x+z=6.
        meta = {'nodes': [1, 3, 4], 'depth': {1: 0, 3: 0, 4: 1},
                'parents': {1: [0], 3: [2], 4: [1, 3]}, 'info_b': None}
        common = SimpleNamespace(V=3, apply_expr=lambda info, p, vals, *_:
                                 sum(vals[x] for x in meta['parents'][p]) % 17)
        functions = source_functions('b_mechanism_late.py',
            ['recompute_truth', 'truth_label', 'tracked_positions'], {'C': common})
        tok = SimpleNamespace(value_tok2idx={str(i): i for i in range(17)})
        old = ValueLabel({0: 4, 1: 4, 2: 2, 3: 2, 4: 6})
        new_truth = functions.recompute_truth(meta, old, tok, None, {0: 5})
        new_label = functions.truth_label(old, new_truth, tok)
        for kind in ('leaf_constant', 'leaf_constant_scratch', 'leaf_constant_pinned'):
            with self.subTest(kind=kind):
                positions = functions.tracked_positions(
                    meta, {}, {'const_pos': 0, 'var_pos': 1}, kind)
                expected = {0: 5, 1: 5, 3: 2, 4: 7}
                self.assertEqual({p: new_label[p, 3] for p in positions}, expected)
        self.assertEqual(new_label[2, 3], 2)  # untouched constant
        self.assertEqual(old.values, {0: 4, 1: 4, 2: 2, 3: 2, 4: 6})


def round_record(correct):
    return {'post_round': 1, 'output_nodes': {'1': {
        'value_correct': correct, 'first4_correct': correct,
        'pred4': {'VALUE': '4' if correct else '5'},
    }}}


def trajectory_pair(model, *, start='own', timing='after_u', recovered=False,
                    mode=None, clean_correct=True):
    common = dict(graph_id=0, start=start, timing=timing,
                  node_truth={'1': {'children': []}})
    if mode:
        common.update(mode=mode, target_index=0, chain={'pos': 1, 'r_pos': 0})
    else:
        common.update(chain_index=0, event_id='g0:c0', chain={'pos': 1, 'r_pos': 0})
    clean = dict(common, condition='clean', rounds=[round_record(clean_correct)] * 2)
    injected = dict(common, condition=mode or 'u',
                    rounds=[round_record(False), round_record(recovered)])
    # Same out string deliberately exercises start/timing separation within a run.
    return {'args': {'config': model, 'run': model, 'out': 'same.json'},
            'records': [clean, injected]}


class SummaryCLITests(unittest.TestCase):
    def invoke(self, directory, paths, suffix):
        output = directory / f'{suffix}.json'
        result = subprocess.run([sys.executable, '-I',
            str(TOOLS / 'b_mechanism_summary.py'), '--inputs', *map(str, paths),
            '--out_csv', str(directory / f'{suffix}.csv'), '--out_json', str(output)],
            capture_output=True, text=True)
        rows = json.loads(output.read_text()) if output.exists() else None
        return result, rows

    def test_models_timings_controls_and_input_order(self):
        data = [trajectory_pair('A', start=start, timing=timing)
                for start in ('own', 'teacher') for timing in ('before_u', 'after_u')]
        data += [trajectory_pair('B', recovered=True),
                 trajectory_pair('B', mode='sustained_r', recovered=True)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = []
            for i, item in enumerate(data):
                path = directory / f'input{i}.json'
                path.write_text(json.dumps(item))
                paths.append(path)
            result, rows = self.invoke(directory, paths, 'forward')
            reverse, rows_reverse = self.invoke(directory, list(reversed(paths)), 'reverse')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(reverse.returncode, 0, reverse.stderr)
            self.assertEqual(rows, rows_reverse)
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row['n_targets'], 1)
                self.assertEqual(row['value_recovery_rate'],
                                 0 if row['source'].startswith('A|') else 1)
                self.assertEqual(row['duplicate_cleans'], 0)
            self.assertEqual(sum(row['kind'] == 'control' for row in rows), 1)

    def test_conflicting_clean_cannot_silently_overwrite(self):
        data = trajectory_pair('B')
        conflict = copy.deepcopy(data['records'][0])
        conflict['rounds'][0] = round_record(False)
        data['records'].append(conflict)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / 'conflict.json'
            path.write_text(json.dumps(data))
            result, _ = self.invoke(directory, [path], 'output')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('conflicting duplicate clean', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
