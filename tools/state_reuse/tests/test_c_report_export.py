import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import export_c_report_tables as X


def rec():
    return dict(positions=[2, 6, 10], u=2, q=6, children=[6],
                descendants=[6, 10], parents={"2": [], "6": [2], "10": [6]},
                truth4=[[0, 11, 1, 3], [0, 12, 1, 4], [0, 13, 1, 7]])


class CReportExportTest(unittest.TestCase):
    def test_q_value_and_first4_are_separate(self):
        r = rec()
        good = [[[0, 12, 1, 4]] * 3, [[0, 12, 1, 4]] * 3]
        wrong_id = [[[0, 99, 1, 4]] * 3, [[0, 99, 1, 4]] * 3]
        r['branches'] = {'a': dict(outputs=good), 'b': dict(outputs=wrong_id)}
        self.assertEqual(X.q_correct(r, 'a'), (True, True))
        self.assertEqual(X.q_correct(r, 'b'), (True, False))

    def test_regions_and_equation(self):
        r = rec()
        reg = X.region_positions(r)
        self.assertEqual(reg['q'], [6])
        self.assertEqual(reg['other_children'], [])
        self.assertEqual(reg['descendants_excl_q'], [10])
        self.assertEqual(reg['outside'], [])
        self.assertEqual(X.equation(r, 10), 'x10=x6(4)=7')


if __name__ == '__main__':
    unittest.main()
