import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import stop_pilot as S
import stop_report as R
from test_hard_graphs import ArithmeticRuntime


class StopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_state_is_held_and_error_can_be_locked_in(self):
        rt=ArithmeticRuntime()
        c=S.make_case(4,8,"single_long",64)
        with torch.inference_mode():
            part=S.run_case(rt,c,True)[0]
        late={r["policy"]:r for r in part["events"] if r["timing"]=="late"}
        self.assertEqual(late["graph_stop"]["status"],"graph_already_stopped")
        self.assertTrue(late["node_stop"]["locked_at_injection"])
        self.assertFalse(late["node_stop"]["target_correct_last"])
        self.assertTrue(late["dense"]["target_correct_last"])
        scores={r["policy"]:r for r in part["scores"] if r["extra"]==16}
        for r in scores.values():
            self.assertEqual(r["value"],r["nodes"])
        self.assertLess(scores["graph_stop"]["committed_token_rounds"],scores["dense"]["committed_token_rounds"])
        self.assertLess(scores["node_stop"]["committed_token_rounds"],scores["dense"]["committed_token_rounds"])

    def test_identity_and_paired_reports(self):
        cells=[(4,8,"single_long"),S.EXTENSION]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=root/"plan.json"
            S.P.write(plan,{"models":[{"id":m} for m in S.MODELS]})
            receipt=root/"receipt.json"
            S.P.write(receipt,dict(plan_sha256=S.P.sha(plan),runs=[
                dict(model=m,final_ckpt=dict(path="unused",sha256=m)) for m in S.MODELS]))
            with patch.object(S,"RECEIPT",receipt),patch.object(R,"RECEIPT",receipt),patch.object(
                S,"Runtime",return_value=ArithmeticRuntime()),patch.object(S,"CELLS",cells),patch.object(R,"CELLS",cells):
                for m in S.MODELS:
                    with patch.object(sys,"argv",["pilot","--smoke","--model",m,"--plan",str(plan),"--out",str(root)]):
                        S.main()
                R.report(root/"smoke",True)
            with (root/"smoke"/"propagation.csv").open() as f:
                rows=list(csv.DictReader(f))
            late=[r for r in rows if r["timing"]=="late" and r["policy"]=="graph_stop"]
            self.assertTrue(all(r["accepted"]=="0" for r in late))
            self.assertTrue(all(r["attempted"]=="0" for r in late if r["cohort"]=="dense_on_same_subset"))
            with (root/"smoke"/"capability.csv").open() as f:
                for r in csv.DictReader(f):
                    self.assertEqual(r["loss_to_dense"],"0")
                    self.assertEqual(r["gain_over_dense"],"0")


if __name__=="__main__":
    unittest.main()
