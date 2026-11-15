import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stress_core as S
from summarize_stress import Totals


class StressTests(unittest.TestCase):
    def test_force_and_release_clock(self):
        self.assertEqual([S.should_replace('sustained', s, 2) for s in range(1, 6)],
                         [True, True, False, False, False])
        self.assertEqual(S.phase('sustained', 2, 2), ('forced', 2))
        self.assertEqual(S.phase('sustained', 3, 2), ('released', 1))

    def test_common_does_not_drop_a_bad_long_duration(self):
        records = [dict(graph_id=0, timing='hold', arm=a, duration=d,
                        value_qualified=not(a == 'b' and d == 4), strict_qualified=True)
                   for a in ('a', 'b') for d in (2, 4)]
        self.assertFalse(S.cohort_keys(records, {'a', 'b'}, (2, 4))[(0, 'hold')][0])
        with self.assertRaises(ValueError):
            S.cohort_keys(records[:-1], {'a', 'b'}, (2, 4))

    def test_release_events_use_last_forced_output(self):
        r = dict(mode='sustained', duration=2, branches={
            'bad': dict(inputs=[False], outputs=[False, False, True, False, True, True]),
            'clean': dict(inputs=[True], outputs=[True]*6)})
        ev = S.event_windows(r, 'bad', 'clean', lambda x: x)
        self.assertEqual(ev['full']['first_recovery'], 3)
        self.assertEqual(ev['release']['first_recovery'], 1)
        self.assertEqual(ev['release']['last_stable_start'], 3)
        self.assertTrue(ev['release']['recovered_then_error'])
        self.assertTrue(ev['release']['final_two_correct'])

    def test_directions_are_not_independent_graphs(self):
        totals = Totals()
        for gid, n, c in [(0, 1, 1), (0, 1, 0), (1, 3, 3), (1, 3, 3)]:
            totals.add(dict(graph_id=gid, n=n, correct=c))
        self.assertEqual(totals.graphs('noise_s1'), 2)
        self.assertEqual(totals.ratio('correct'), .75)
        self.assertNotEqual(totals.ratio('correct'), totals.sums['correct']/totals.sums['n'])

    def test_summary_cli_for_both_modes(self):
        # Exercises real CLI, finite release clock, counts and donor-independent cohorts.
        gold = [[0, 0, 0, 4], [0, 1, 0, 13]]
        bad = [[0, 0, 0, 5], [0, 1, 0, 14]]
        for mode in ('sustained', 'noise'):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td); run = root/'run'; run.mkdir()
                durations = [2, 4] if mode == 'sustained' else [1]
                specs = S.branches(mode); coverage = []; records = []
                for d in durations:
                    cov = dict(arm='initial', graph_id=0, timing='hold', duration=d,
                               status='selected', value_qualified=True, strict_qualified=True)
                    total = d+2 if mode == 'sustained' else 2
                    bs = {}
                    for spec in specs:
                        corrupted = spec['u_mode'] in ('wrong', 'noise') or spec['q_wrong']
                        bs[spec['name']] = dict(spec=spec,
                            inputs=[bad if corrupted else gold]+[gold]*(total-1),
                            outputs=[bad if corrupted else gold]+[gold]*(total-1),
                            dose=[dict(reference_norm=1., u_norm=float(corrupted), q_norm=0.,
                                       u_applied=corrupted, q_applied=spec['q_wrong'])]*total)
                    coverage.append(cov)
                    records.append(dict(**cov, mode=mode, state_round=3, D=3, u=0, q=1,
                        positions=[0, 1], truth4=gold, parents={'0': [], '1': [0]},
                        children=[1], descendants=[1], expected_u=5, expected_q=14,
                        start_graph_value_all=True, start_graph_first4_all=True, noise_seed=7, branches=bs))
                (run/'meta.json').write_text(json.dumps(dict(status='complete', mode=mode,
                    models=[dict(arm='initial')], durations=durations, candidate_graphs=1,
                    horizon=2, branches=specs)), encoding='utf-8')
                for name, rows in [('coverage', coverage), ('records', records)]:
                    (run/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
                result = subprocess.run([sys.executable, str(S.U.HERE/'summarize_stress.py'),
                    '--run-dir', str(run), '--out-dir', str(root/'out')], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                with (root/'out/accuracy.csv').open() as f:
                    rows = list(csv.DictReader(f))
                self.assertTrue(rows)
                self.assertTrue(all(int(r['graphs']) == 1 for r in rows))
                self.assertTrue(all(float(r['node_acc']) == int(r['correct'])/int(r['n']) for r in rows))
                if mode == 'sustained':
                    self.assertTrue(any(r['phase'] == 'released' and r['phase_round'] == '1' for r in rows))
                else:
                    self.assertTrue(any(r['condition'] == 'noise_s1' and r['trials'] == '2' for r in rows))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch checks delegated to server')
class StressTensorTests(unittest.TestCase):
    def test_q_once_time_matched_u_and_full_release(self):
        src = torch.arange(5*6*4, dtype=torch.float32).reshape(5, 6, 4)
        specs = S.branches('sustained'); idx = [s['name'] for s in specs].index('wrong_q_wrong')
        states = src[0:1].repeat(len(specs), 1, 1)
        out, _ = S.apply_inputs(states, src, specs, 1, 3, True, True, 7)
        self.assertTrue(torch.equal(out[idx, 1], src[2, 1]))
        self.assertTrue(torch.equal(out[idx, 3], src[4, 3]))
        self.assertTrue(torch.equal(out[idx, [0, 2, 4, 5]], states[idx, [0, 2, 4, 5]]))
        next_state = out+10
        out2, _ = S.apply_inputs(next_state, src+100, specs, 1, 3, True, False, 7)
        self.assertTrue(torch.equal(out2[idx, 1], src[2, 1]+100))
        self.assertTrue(torch.equal(out2[idx, 3], next_state[idx, 3]))
        released, _ = S.apply_inputs(out2, None, specs, 1, 3, False, False, 7)
        self.assertTrue(torch.equal(released, out2))

    def test_noise_direction_shared_across_scales(self):
        src = torch.arange(5*6*4, dtype=torch.float32).reshape(5, 6, 4)
        specs = S.branches('noise'); names = [s['name'] for s in specs]
        states = src[0:1].repeat(len(specs), 1, 1)
        out, dose = S.apply_inputs(states, src, specs, 1, 3, True, True, 7)
        a, b = names.index('noise_s1_r0'), names.index('noise_s3_r0')
        self.assertTrue(torch.allclose(out[b, 1]-states[b, 1], 3*(out[a, 1]-states[a, 1]), atol=1e-4))
        self.assertAlmostEqual(dose[b]['u_norm']/dose[a]['u_norm'], 3., places=5)
        self.assertTrue(torch.equal(out[a, [0, 2, 3, 4, 5]], states[a, [0, 2, 3, 4, 5]]))


if __name__ == '__main__':
    unittest.main()
