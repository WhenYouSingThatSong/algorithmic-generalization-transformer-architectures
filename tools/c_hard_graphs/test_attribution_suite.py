import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import attribution_queue as queue
from architecture_controls import Control, ARMS, example, stage_label
from attribution_cases import exposure_case, hide_key, run_graph, workspace_program
from test_research_queue import Solver
from transfer_tasks import carry_case, export, intervened, program_case, score


class Oracle(Solver):
    continuous=False

    def decode(self,h):
        return h


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_contaminated_cut_and_recovery_counts(self):
        rt=Oracle()
        c=run_graph(rt,352,4,"sources")
        rows=queue.counts(c)
        def event(timing,condition):
            return next(r for r in rows if r["metric"]=="intervention" and r["timing"]==timing
                        and r["condition"]==condition)
        good=event("just_solved","target")
        self.assertEqual([good[k] for k in ("direct_faults","direct_repaired_next","specific","stable_after_damage")],
                         [1,1,1,1])
        self.assertEqual(event("just_solved","clean")["direct_repaired_next"],0)
        later=next(r for r in c["events"] if r["timing"]=="later" and r["condition"]=="coherent")
        self.assertTrue(later["eligible"])
        vals=later["effective_values"]
        bad=[i for i,n in enumerate(c["nodes"]) if n["parents"] and vals[i]!=sum(vals[j] for j in n["parents"])%23]
        self.assertEqual(bad,c["focus"][:1])
        self.assertEqual([r["correct"] for r in rows if r["metric"]=="fresh_solution"],[4,4])
        four=[dict(good,model=m,gid=352,cell="4",eligible=int(m!="A_s42"))
              for m in ("A_s42","B_s42","C_s42","C_s42-correction")]
        row,=queue.four_way(four)
        self.assertEqual(row["common"],0)
        self.assertIsNone(row["B_s42_specific"])
        holes=[dict(good,model="B_s42",gid=352,cell="4",block=b,target_correct_next=int(b!="parent"))
               for b in ("none","parent","unrelated")]
        path,=queue.pathway_contrasts(holes)
        self.assertEqual((path["eligible"],path["parent_only_impairs"]),(1,1))

    def test_receipts_have_selected_weights(self):
        report=queue.P.read(queue.ROOT/"docs/records/2026-11-19/abc-workspace/abc-meta.json")
        models={s["meta"]["model"]["id"]:s["meta"]["model"] for s in report["sources"]
                if "model" in s["meta"]}
        for entry in queue.P.read(queue.OLD)["finals"].values():
            models[entry["model"]]=entry["identity"]["source"]
        specs=queue.model_specs({"models":list(models.values())},queue.ROOT)
        self.assertEqual(len(specs),8)
        self.assertIn("D384fast",specs["A_s42"]["config"])
        evidence=queue.fit_sources(specs)
        for seed in (42,1234):
            self.assertTrue(evidence[f"C_s{seed}-update"]["history"])
            self.assertTrue(evidence[f"C_s{seed}-correction"]["validation"])

    def test_mask_duration_and_effect(self):
        class Encoder(nn.Module):
            def forward(self,h,mask=None):
                return h.mean(dim=1,keepdim=True).expand_as(h) if mask is None else (
                    (mask.float()/mask.sum(-1,keepdim=True)) @ h)
        model=SimpleNamespace(transformer=Encoder())
        rt=SimpleNamespace(model=model)
        x=torch.tensor([[[3.],[9.],[18.]]])
        ref=model.transformer(x)
        with hide_key(rt,1):
            observed=model.transformer(x)[0,:,0].tolist()
            self.assertEqual(observed,[10.5,10,10.5])
        self.assertTrue(torch.equal(ref,model.transformer(x)))

    def test_workspace_uses_recent_definition(self):
        for n in (1,2,4):
            for gap in (0,1):
                for live in (0,4,8):
                    for spaced in (0,1):
                        a,b,peak=workspace_program(352,n,live,gap,spaced)
                        self.assertEqual(len(a["nodes"]),46)
                        self.assertEqual([(i["pos"],i["parents"],i["truth"]) for i in a["nodes"]],
                                         [(i["pos"],i["parents"],i["truth"]) for i in b["nodes"]])
                        self.assertEqual(max(i["version"] for i in b["nodes"]),n)
                        self.assertGreater(peak,0)
        out=queue.workspace(Oracle(),352,(4,8,1,1))
        self.assertTrue(all(r["unique_correct"]==r["reused_correct"]==r["nodes"] for r in out["rows"]))

    def test_transfer_reference_and_injection(self):
        a=carry_case(9,8,8)
        expected=sum((d+e)*10**i for i,(d,e) in enumerate(zip(a["input"]["left"],a["input"]["right"])))
        self.assertEqual(expected,sum(d*10**i for i,d in enumerate(a["answer"])))
        b=intervened(a)
        i=b["input"]["force_after_step"]
        self.assertEqual(a["trace"][:i],b["trace"][:i])
        self.assertNotEqual(a["trace"][i],b["trace"][i])
        self.assertNotEqual(a["answer"],b["answer"])
        programs=[program_case(352,16,n) for n in (1,2,4,16)]
        self.assertTrue(all(k["trace"]==programs[0]["trace"] for k in programs))
        for forced in (False,True):
            with tempfile.TemporaryDirectory() as d:
                export(Path(d),forced)
                scores=score(Path(d)/"answers.jsonl",forced)
                self.assertEqual(sum(r["answer_all_correct"] for r in scores),512)
                self.assertTrue(all(r["trace_correct"]==r["trace_positions"] for r in scores))

    def test_architecture_controls(self):
        c,x,pos,y,depth=example(3,"cpu")
        self.assertTrue(torch.equal(stage_label(y,depth,3,4),y))
        self.assertTrue((stage_label(y,depth,0,4)[...,3]==24).any())
        shapes={}
        for arm in ARMS:
            m=Control(arm,32,4,4)
            logits,lens=m(x,pos,dict(stage=1,node=c["u"]))
            self.assertEqual(len(logits),4)
            self.assertEqual(logits[-1][3].shape,(2,len(pos),25))
            self.assertEqual(lens[-1]-lens[0],3*len(pos) if arm=="written" else 0)
            shapes[arm]=(m.width,len(m.blocks))
        self.assertEqual(shapes["loop"],(32,1))
        self.assertEqual(shapes["fixed_compute"],(32,4))
        self.assertLess(shapes["fixed_parameter"][0],32)


if __name__=="__main__":
    unittest.main()
