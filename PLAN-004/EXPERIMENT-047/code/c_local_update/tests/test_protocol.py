import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import local_protocol as S
import rebinding as B
from scenarios import metrics, summarize
from outcomes import additional
sys.path.insert(0,str(S.P.ROOT/"experiments"))
from tokenizers import FactoredVocabTokenizer


class BindingTests(unittest.TestCase):
    def test_all_programs_match_sequential_interpreter(self):
        tok = FactoredVocabTokenizer(128,["ADD"],23,3)
        paired = solo = 0
        for f,n,count in B.CELLS:
            for gid in range(count):
                c = B.case(f,n,gid)
                if f=='liveness':
                    for rewrite in c['overwrites']:
                        self.assertGreater(rewrite['prior_uses'],0)
                        self.assertLess(rewrite['last_read'],rewrite['new'])
                truths = []
                for name,b in c["branches"].items():
                    env, out = {}, []
                    for expr in " ".join(b["tokens"]).split(" <EQ_SEP> "):
                        rhs,dest = expr.split(" = ")
                        value = (sum(env[x] for x in rhs.split(" ADD "))%23 if rhs.startswith("x_") else int(rhs))
                        env[dest] = value
                        out.append(value)
                    self.assertEqual(out,[p["truth"] for p in b["nodes"]])
                    enc = tok.encode_tokens(b["tokens"])
                    self.assertTrue(all(enc[z["pos"]][3]==24 for z in b["nodes"]))
                    self.assertLessEqual(len(enc),1024)
                    ok,scores = B.score(b,out)
                    self.assertTrue(ok)
                    self.assertEqual(scores["all"]["matches_stale"],0)
                    truths.append(out)
                if len(truths)==2:
                    paired += 1
                    self.assertEqual(*truths)
                    a,b=c["branches"].values()
                    self.assertEqual([z["parents"] for z in a["nodes"]],[z["parents"] for z in b["nodes"]])
                    self.assertEqual(len(a["tokens"]),len(b["tokens"]))
                else:
                    solo += 1
                    self.assertGreater(len(truths[0]),128)
        self.assertEqual((paired,solo),(228,4))

    def test_before_write_and_old_consumers(self):
        ops=[("a",2),("b",3),("y",("a","b")),("a",("a","b")),("a",("a","y")),("z",("a","y"))]
        b=B.render(ops,["x_"+str(i) for i in range(6)],False)
        self.assertEqual([z["truth"] for z in b["nodes"]],[2,3,5,5,10,15])
        self.assertEqual(b["nodes"][-1]["stale"],[7,10])

    def test_distinct_versions_and_same_value_collision(self):
        b=B.render([("a",2),("k",0),("a",("a","k")),("z",("a","k"))],list("abcd"),False)
        self.assertEqual(b["nodes"][-1]["stale"],[])
        c=B.case("self_read",8,1)
        self.assertEqual(max(z["version"] for z in c["branches"]["rebound"]["nodes"]),8)


class LocalTests(unittest.TestCase):
    def test_training_time_and_dose(self):
        g=dict(nodes=[1,2],depths={1:0,2:1},coefs={1:{1:1},2:{1:2}},constants={1:0})
        seen=set()
        for b in range(600):
            cfg=S.schedule(2,1,b)
            self.assertEqual(cfg,S.schedule(2,1,b))
            self.assertEqual(cfg["rounds"],3)
            if cfg["enabled"]: seen.add(cfg["mode"])
            n,events=S.events([g],[3],1,b,cfg)
            self.assertEqual(n,1 if cfg["mode"]=="middle" else 2)
            for e in events:
                self.assertLess(e["round"],3)
                if cfg["mode"]=="completed": self.assertEqual(e["round"],2)
        self.assertEqual(seen,{"middle","frontier","completed"})

    def test_classes_and_damage_denominators(self):
        g=dict(nodes=[1,2,3,4],parents={1:[],2:[1],3:[2],4:[]},constants={1:0,4:0},
               depths={1:0,2:1,3:2,4:0})
        y=[[0,0,0,4]]*5
        def row(inp,out):
            vals=[4,inp[0],inp[1],4]
            refs=[a!=b for a,b in zip(inp,vals)]
            return dict(input=[[0,0,0,v] for v in inp],output=[[v] for v in out],
                        update=[dict(reference=z,plan=z,changed=a!=b) for z,a,b in zip(refs,inp,out)])
        clean=[row([4]*4,[4]*4)]*3
        mixed=row([5,5,24,4],[4,5,5,4])
        trace=[mixed,row([4,5,5,4],[4,4,5,4]),row([4,4,5,4],[4]*4)]
        r=metrics(g,y,trace,clean,"actual_complete","mixed",1)
        self.assertTrue(r["rounds"][0]["mixed"])
        self.assertEqual(r["endpoints"]["target"]["first_round_repaired"],1)
        self.assertEqual(r["endpoints"]["target"]["damaged"],1)
        bins={(v["category"],v["scope"]):v["counts"] for v in r["metrics"]}
        self.assertGreaterEqual(bins["indirect_first_conflict","all"]["reference"],1)
        s=summarize([dict(**r,split="dev",status="ok")])
        self.assertEqual(s["mixed"][0]["counts"]["mixed_at_input"],1)

    def test_clean_use_joint_not_two_marginals(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"test.jsonl"
            rows=[]
            for gid,clean,pert in [(1,True,False),(2,False,True)]:
                for condition,ok in (("clean",clean),("perturbed",pert)):
                    rows.append(dict(module="reuse",status="ok",gid=gid,condition=condition,lag=2,u=1,new_node=2,
                                     truth=[[0,0,0,4]]*2,nodes=[1,2],u_readout=[0,0,0,4],u_first4_correct=True,
                                     rows=[dict(output=[[4],[4 if ok else 9]])]*2))
            p.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
            c=additional(p)["use_after_injection"][0]["counts"]
            self.assertEqual(c["paired_graphs"],2)
            self.assertEqual(c["value_1_consumer_0"],1)
            self.assertEqual(c["clean_consumer_capable"],1)
