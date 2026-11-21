import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from design import ARMS, COUNT, PROFILES, SCALES, TOOLS, make_case
import hard_eval as E
import report
sys.path.insert(0, str(TOOLS.parent/"experiments"))
from tokenizers import FactoredVocabTokenizer


class ArithmeticRuntime:
    """Synchronous ground-truth calculator; uses last preceding definition."""
    device = "cpu"
    tok = FactoredVocabTokenizer(128, ["ADD"], 23, 3)

    def init(self, x):
        return x.clone()

    def forward(self,h):
        from torch.nn.functional import one_hot
        out=h.clone()
        for bi,sample in enumerate(h.tolist()):
            latest, operands = {}, []
            for p,(syntax,var,op,value,*_) in enumerate(sample):
                if syntax==2:
                    operands.append(value)
                if syntax!=0:
                    continue
                if p and sample[p-1][0]==self.tok.syntax_tok2idx["="]:
                    out[bi,p,3] = sum(operands)%23 if all(v<23 for v in operands) else 24
                    latest[var] = value
                    operands=[]
                else:
                    operands.append(latest[var])
        logits=[one_hot(out[...,k],v) for k,v in enumerate(self.tok.vocab_sizes)]
        return out,logits,None


class Checks(unittest.TestCase):
    def test_grid(self):
        for w,d in SCALES:
            for gid in range(COUNT):
                counts={}
                for profile in PROFILES:
                    c=make_case(w,d,profile,gid)
                    versions=[]
                    for arm in ARMS:
                        rows=c["arms"][arm]["nodes"]
                        versions.append(max(z["version"] for z in rows))
                        self.assertEqual([z["truth"] for z in rows],
                                         [z["truth"] for z in c["arms"]["unique"]["nodes"]])
                    self.assertEqual(versions,[1,(d+3)//4,(d+1)//2])
                    self.assertEqual(c["layers"][1],w)
                    self.assertEqual(c["layers"][0],6)
                    self.assertEqual(max(c["layers"]),d)
                    counts[profile]=len(rows)
                self.assertEqual(counts["uneven"],counts["balanced"])
        c=make_case(8,20,"all_long",7)
        self.assertEqual(list(c["arms"]),["reuse4","reuse2"])
        for branch in c["arms"].values():
            self.assertEqual(len(branch["nodes"]),166)
            self.assertEqual(len(branch["tokens"]),983)
            self.assertLessEqual(len({n["name"] for n in branch["nodes"]}),128)

    def test_controlled_use(self):
        torch.set_num_threads(1)
        c=make_case(4,8,"single_long",0)
        raw=E.run_case(ArithmeticRuntime(),c,True)
        for z in raw["probes"]:
            self.assertEqual(z["status"],"accepted")
            self.assertEqual(z["first_child_damage"],1)
            self.assertEqual(z["target_first_restored"],1)
            self.assertTrue(z["joint_stable"])
        self.assertEqual([r["child_ever_correct"] for r in raw["probes"]],[False,True])
        for row in E.metrics(c,raw):
            self.assertEqual(row["nodes"],row["first4"])
        self.assertTrue(all(row["nodes"]==row["at_local_round"] for row in E.early_readouts(c,raw["traces"])))
        # A final one-round return is not sustained, and no observed damage has no recovery denominator.
        self.assertFalse(E.probe_score([[4,6]]*2,[[4,7],[4,6]],4,6,1)["joint_stable"])
        self.assertIsNone(E.probe_score([[4,6]],[[4,6]],4,6,1)["joint_stable"])

    def test_smoke_reports(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=root/"plan.json"
            E.P.write(plan,{"models":[{"id":m} for m in E.MODELS]})
            receipt=root/"receipt.json"
            E.P.write(receipt,dict(plan_sha256=E.P.sha(plan),runs=[
                dict(model=m,final_ckpt=dict(path="unused",sha256=m)) for m in E.MODELS]))
            with patch.object(E,"RECEIPT",receipt),patch.object(report,"RECEIPT",receipt),patch.object(
                E,"Runtime",return_value=ArithmeticRuntime()),patch.object(E,"SCALES",[(4,8)]),patch.object(report,"SCALES",[(4,8)]):
                for m in E.MODELS:
                    with patch.object(sys,"argv",["eval","--model",m,"--smoke","--plan",str(plan),"--out",str(root)]):
                        E.main()
                report.summarize(root/"smoke",True)
            self.assertEqual(len(list((root/"smoke").glob("*-summary.csv"))),5)
            rows=report.read_csv(root/"smoke"/"propagation-summary.csv")
            self.assertTrue(all(r["accepted"]==r["joint_stable_after_damage"] for r in rows))


if __name__=="__main__":
    unittest.main()
