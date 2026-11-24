import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import batch
import deliver


class ExecutionTests(unittest.TestCase):
    def test_serial_job_grid(self):
        ab=batch.jobs_for('AB','smoke',Path('/tmp/out'),Path('/tmp/A'),Path('/tmp/B'))
        self.assertEqual(len(ab),8)
        self.assertEqual(sum(j['gpu'] for j in ab),4)
        self.assertTrue(all('--limit-graphs' in j['argv'] for j in ab if j['gpu']))
        c=batch.jobs_for('C','main',Path('/tmp/out'),None,None)
        self.assertEqual(len(c),6)
        self.assertTrue(all('--candidate-count' not in j['argv'] for j in c if j['gpu']))
        self.assertTrue(all('--num_threads' not in j['argv'] for j in c if not j['gpu']))

    def test_receipt_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'records.jsonl').write_text('one\n')
            batch.write_json(p/'meta.json',dict(status='complete',records_sha256=batch.sha(p/'records.jsonl')))
            batch.completed(dict(out=d))
            (p/'records.jsonl').write_text('two\n')
            with self.assertRaises(ValueError):batch.completed(dict(out=d))

    def test_report_does_not_merge_cohorts_or_rounds(self):
        row=dict(condition=json.dumps(['late','semantic',2]),region='q',branch='observed_pre',
                 cohort='common_input_qualified',round_or_window='last_two')
        self.assertTrue(deliver.choose_row(row,True))
        self.assertFalse(deliver.choose_row(dict(row,cohort='input_qualified'),True))
        self.assertFalse(deliver.choose_row(dict(row,round_or_window='16'),True))


if __name__=='__main__':unittest.main()
