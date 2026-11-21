import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.functional import one_hot

import reader_queue as Q
from test_research_queue import Solver
from version_readers import bindings, reader_probe, version_probe


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_binding_and_detectable_stale_response(self):
        branches,_=bindings(192,"tight")
        delayed,_=bindings(192,"delayed")
        for key,branch in branches.items():
            rows={r["slot"]:r for r in branch["nodes"] if r["slot"].startswith("reader")}
            other={r["slot"]:r for r in delayed[key]["nodes"] if r["slot"].startswith("reader")}
            self.assertEqual({k:(r["name"],r["truth"]) for k,r in rows.items()},
                             {k:(r["name"],r["truth"]) for k,r in other.items()})
            self.assertEqual(len(branch["tokens"]),len(delayed[key]["tokens"]))
            self.assertEqual(len({r["truth"] for r in rows.values()}),3)
        base=branches["reused","base"]["nodes"]
        first,last=[next(r["pos"] for r in base if r["slot"]==s) for s in ("reader0","reader2")]
        defs=[r["pos"] for r in base if r["slot"]=="slot"]
        class StaleSolver(Solver):
            def forward(self,h,middle=False):
                out,logits,mid=super().forward(h,middle)
                reused=h[:,defs[0],1]==h[:,defs[-1],1]
                out[reused,last,3]=out[reused,first,3]
                logits[3]=one_hot(out[...,3],25).float()
                return out,logits,mid
        rows,_=version_probe(StaleSolver(),192,"tight")
        for r in rows:
            if r["condition"]=="first_version" and r["reader"]==3:
                self.assertEqual(r["tracks_older_with_unique_correct"],int(r["arm"]=="reused"))

    def test_readback_precedes_append_and_has_controls(self):
        rows,examples=reader_probe(Solver(),192,4)
        self.assertEqual(sum(r["attempts"] for r in rows),2)
        for e in examples:
            self.assertEqual(e["status"],"accepted")
            self.assertEqual(e["damage_round"],1)
            self.assertEqual(e["append_round"],1)
            self.assertEqual(len(e["after_append"]),4)
        for r in rows:
            if r["step"]:
                self.assertEqual((r["paired_clean"],r["both_correct"],r["append_changes_old_clean"]),(1,1,0))
                self.assertEqual(r["old_at_append"],"still_wrong")

    def test_resume_and_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out=Path(tmp),train_root=Path(tmp),device="cpu")
            receipt=dict(plan_sha256="test",runs=[dict(model=m,final_ckpt={"sha256":m}) for m in Q.MODELS])
            plan=dict(models=[{"id":m} for m in Q.MODELS])
            for stage in Q.SETTINGS:
                with patch.object(Q,"Runtime",return_value=Solver()):
                    Q.run(stage,True,args,receipt,plan)
                folder=args.out/"smoke"/stage
                delivery=Q.P.read(folder/"delivery.json")
                self.assertEqual(delivery["summary_sha256"],Q.P.sha(folder/"summary.csv"))
                with patch.object(Q,"Runtime",side_effect=AssertionError("duplicate execution")):
                    Q.run(stage,True,args,receipt,plan)
                self.assertEqual(delivery,Q.P.read(folder/"delivery.json"))


if __name__=="__main__":
    unittest.main()
