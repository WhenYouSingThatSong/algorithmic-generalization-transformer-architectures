import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.functional import one_hot
import research_queue as Q
from research_probes import identity_probe, fanout_probe, program
from test_hard_graphs import ArithmeticRuntime


class Solver(ArithmeticRuntime):
    """Synchronous addition with missing references explicitly read as EMPTY."""
    def forward(self,h,middle=False):
        result=h.clone()
        eq=self.tok.syntax_tok2idx["="]
        for i,sample in enumerate(h.tolist()):
            latest={}
            terms=[]
            for j,(syntax,var,_,value,*_) in enumerate(sample):
                if syntax==2:
                    terms.append(value)
                if syntax!=0:
                    continue
                if j and sample[j-1][0]==eq:
                    result[i,j,3]=sum(terms)%23 if all(v<23 for v in terms) else 24
                    latest[var]=value
                    terms=[]
                else:
                    terms.append(latest.get(var,24))
        logits=[one_hot(result[...,k],n).float() for k,n in enumerate(self.tok.vocab_sizes)]
        mid=one_hot((h[...,3]!=result[...,3]).long(),2).float() if middle else None
        return result,logits,mid


class ResearchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_first_fill_repair_and_spoil(self):
        values=[24,2,5,3,5]+[5]*13
        case={"D":2,"arms":{"unique":{"nodes":[{"truth":5}]}}}
        rows=Q.transitions(case,{"unique":[[[v,1,1]] for v in values]})
        self.assertEqual(sum(r["first_correct"] for r in rows),1)
        self.assertEqual(sum(r["returned_to_correct"] for r in rows),1)
        self.assertEqual(sum(r["observations"] for r in rows),len(values))
        self.assertEqual(sum(r["observations"] for r in rows if r["previous"]=="wrong_number" and r["following"]=="correct"),2)

    def test_controlled_construction(self):
        for gid in range(80,96):
            a=program(gid,8,same=True)
            b=program(gid,8,same=False)
            self.assertEqual(a[1:],b[1:])
            self.assertEqual(a[0][:1]+a[0][2:],b[0][:1]+b[0][2:])
        rt=Solver()
        examples=[]
        rows=identity_probe(rt,82,4,False,examples)
        end=[r for r in rows if r["step"]==1]
        self.assertTrue(all(r["common_eligible"]==1 for r in end))
        self.assertTrue(all(r["u_reader_value_plus_one"]==1 for r in end if r["condition"]=="value_change"))
        self.assertTrue(all(r["actual_pair_correct"]==1 for r in end if r["condition"]=="unrelated_rename"))
        self.assertEqual(len(examples),2)
        for f in (1,2,4):
            rows=fanout_probe(rt,82,4,f)
            for step,role in ((1,"child"),(2,"merge"),(3,"sink")):
                seen=[r for r in rows if r["step"]==step and r["role"]==role and r["timing"]=="early"]
                self.assertTrue(all(r["specific"]==int(r["coefficient"]>0) for r in seen))
                self.assertEqual(sum(r["coefficient"] for r in seen),f)

    def test_signals_and_deliveries(self):
        rt=Solver()
        case=Q.make_case(4,8,"uneven",128)
        rows=Q.signal_rows(rt,case)
        self.assertEqual(sum(r["observations"] for r in rows),
                         len(case["arms"])*(case["D"]+15)*len(case["attrs"]))
        self.assertTrue(all(r["next_correct"] for r in rows if r["update"]==1))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            args=SimpleNamespace(out=root/"new",hard_root=root/"old",device="cpu",train_root=root)
            receipt=dict(plan_sha256="mock",runs=[dict(model=m,final_ckpt={"sha256":m}) for m in Q.MODELS])
            plan=dict(models=[{"id":m} for m in Q.MODELS])
            provenance={"models":{}}
            trace={name:[[[n["truth"],1,1] for n in branch["nodes"]] for _ in range(case["D"]+16)]
                   for name,branch in case["arms"].items()}
            for model in Q.MODELS:
                path=args.hard_root/"formal"/model
                Q.P.write(path/"complete.json",dict(model=model))
                provenance["models"][model]=dict(formal_complete_sha256=Q.P.sha(path/"complete.json"))
                with gzip.open(path/"w4-d8-uneven-128.json.gz","wt") as f:
                    json.dump(dict(case=case,result={"traces":trace}),f)
            Q.P.write(root/"provenance.json",provenance)
            def specs(stage,_):
                if stage=="identity": return [(2,4,"different",128)]
                if stage=="fanout": return [(4,4,str(f),128) for f in (1,2,4)]
                return [(4,8,"uneven",128)]
            with patch.object(Q,"Runtime",return_value=rt),patch.object(Q,"PROVENANCE",root/"provenance.json"),patch.object(Q,"specifications",specs):
                for stage in Q.STAGES:
                    Q.execute_stage(args,stage,True,receipt,plan)
                    folder=args.out/"smoke"/stage
                    delivery=Q.P.read(folder/"delivery.json")
                    self.assertEqual(delivery["contrasts_sha256"],Q.P.sha(folder/"contrasts.csv"))
                    with patch.object(Q,"compute",side_effect=AssertionError("recomputed existing case")):
                        Q.execute_stage(args,stage,True,receipt,plan)
                    self.assertEqual(delivery,Q.P.read(folder/"delivery.json"))

    def test_shared_fanout_and_queue_order(self):
        rows=[]
        for gid in (1,2):
            for f in (1,2,4):
                rows.append(dict(model="C_s42",width=4,depth=8,timing="late",gid=gid,
                    profile=str(f),role="attempt",accepted=int(gid==1 or f!=4),attempts=1))
        paired=Q.matched_fanout(rows)
        self.assertEqual([r["common_graphs"] for r in paired],[1,1,1])
        self.assertTrue(all(r["candidate_graphs"]==2 for r in paired))
        with tempfile.TemporaryDirectory() as td:
            plan=Path(td)/"plan.json"
            receipt=Path(td)/"receipt.json"
            Q.P.write(plan,{})
            Q.P.write(receipt,dict(plan_sha256=Q.P.sha(plan)))
            calls=[]
            with patch.object(Q,"RECEIPT",receipt),patch.object(sys,"argv",["queue","--plan",str(plan)]),patch.object(
                    Q,"finish_stop",side_effect=lambda a,r,s:calls.append(("stop",s))),patch.object(
                    Q,"execute_stage",side_effect=lambda a,k,s,r,p:calls.append((k,s))):
                Q.main()
            self.assertEqual(calls,[(k,s) for k in ("stop",*Q.STAGES) for s in (True,False)])


if __name__=="__main__":
    unittest.main()
