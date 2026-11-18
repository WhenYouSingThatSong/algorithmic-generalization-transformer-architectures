import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import function_audit as F
import reassignment_core as P
import summarize_reassignment as W
from io_utils import write_json,sha

def row(v):
    return [1,0,0,v]

def state(u=4,q=13,p=9):
    return [row(u),row(q),row(p)]

def pair():
    return dict(arm="initial",graph_id=0,timing="hold",state_round=8,
        u=0,q=1,children=[1],positions=[0,1,2],truth4=state(),
        parents={"1":[0,2]},value_qualified=True,
        branches={"unchanged":dict(initial=state(),outputs=[state(),state()]),
                  "u_wrong":dict(initial=state(u=5),outputs=[state(q=14),state()])})

def save_source(path,rs,arms=("initial",)):
    path.mkdir()
    write_json(path/"meta.json",dict(status="complete",protocol="C_uq_pair_eval_v1",
        models=[dict(arm=a) for a in arms]))
    (path/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rs),encoding="utf-8")

class FunctionTests(unittest.TestCase):
    def test_input_output_clock_and_functional_candidate(self):
        rows=list(F.edge_rows(pair(),"u_wrong"))
        self.assertFalse(rows[0]["u_input_correct"])
        self.assertTrue(rows[0]["u_output_correct"])
        self.assertFalse(rows[0]["q_output_correct"])
        self.assertEqual(rows[0]["input_parent_sum"],14)
        self.assertTrue(rows[0]["matches_input_parent_sum"])
        self.assertTrue(rows[1]["u_input_correct"])
        self.assertEqual(rows[0]["absolute_output_round"],9)

    def test_forced_inputs_not_previous_outputs(self):
        r=pair();r.update(mode="sustained",duration=2)
        r["branches"]={
            "free_q_clean":dict(spec={"q_wrong":False},inputs=[state()]*3,outputs=[state()]*3),
            "wrong_q_clean":dict(spec={"q_wrong":False},
                inputs=[state(u=5),state(u=5,q=14),state(q=14)],
                outputs=[state(q=14),state(q=14),state()])}
        rows=list(F.edge_rows(r,"wrong_q_clean"))
        self.assertFalse(rows[1]["u_input_correct"])
        self.assertEqual(rows[1]["phase"],"forced")
        self.assertEqual(rows[2]["release_round"],1)
        del r["branches"]["wrong_q_clean"]["inputs"]
        with self.assertRaises(ValueError):list(F.edge_rows(r,"wrong_q_clean"))

    def test_paired_child_fault_reference(self):
        r=pair()
        r["branches"]["q_wrong"]=dict(initial=state(q=14),outputs=[state(q=14),state()])
        r["branches"]["both_wrong"]=dict(initial=state(u=5,q=14),outputs=[state(q=14),state()])
        rr=list(F.edge_rows(r,"both_wrong"))[0]
        self.assertEqual(rr["reference"],"q_wrong")
        self.assertFalse(rr["extra_q_damage"])

    def test_missing_values_other_parents_and_reerror(self):
        r=pair();r["branches"]["u_wrong"]["initial"]=state(u=24,p=10)
        rr=list(F.edge_rows(r,"u_wrong"))[0]
        self.assertIsNone(rr["input_parent_sum"])
        self.assertFalse(rr["other_parents_input_correct"])
        self.assertEqual(F.event([True,False,True,False,True])["stable_return_output"],5)
        self.assertTrue(F.event([True,False,True,False,True])["reerror_after_return"])

    def test_macro_denominator_and_source_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); r=pair();r["graph_id"]=1
            r["positions"].append(3);r["children"].append(3)
            r["parents"]["3"]=[0,2];r["truth4"].append(row(13))
            for b in r["branches"].values():
                b["initial"].append(row(13))
                for out in b["outputs"]:out.append(row(13))
            save_source(tmp/"a",[pair(),r]);save_source(tmp/"b",[pair()])
            F.run({"source_a":tmp/"a","source_b":tmp/"b"},tmp/"out")
            with (tmp/"out/quadrants.csv").open() as f: rows=list(csv.DictReader(f))
            rr=next(x for x in rows if x["source"]=="source_a" and x["branch"]=="u_wrong"
                and x["post_round"]=="1" and x["axis"]=="u_output_correct"
                and x["context"]=="all" and x["cohort"]=="all_recorded"
                and x["u_read_correct"]=="True" and x["q_output_correct"]=="True")
            self.assertAlmostEqual(float(rr["micro_fraction"]),1/3)
            self.assertAlmostEqual(float(rr["graph_macro_fraction"]),1/4)
            self.assertEqual(rr["edge_n"],"3")

    def test_unpaired_models_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);save_source(tmp/"src",[pair()],arms=("initial","hold"))
            with self.assertRaises(ValueError):F.run({"x":tmp/"src"},tmp/"out")

class ReassignmentTests(unittest.TestCase):
    def plan(self):
        return P.make_plan([1,2,3],{1:0,2:1,3:1},{1:[0],2:[1,1],3:[1]*23},
            {0:4},{1:4,2:8,3:0},1,0,3)

    def test_new_truth_and_cancellation(self):
        p=self.plan()
        self.assertEqual(p["new_truth"],{1:7,2:14,3:0,0:7})
        self.assertEqual(p["regions"]["changed_nodes"],[1,2])
        self.assertEqual(p["regions"]["unchanged_descendants"],[3])

    def test_no_new_answers_written(self):
        p=self.plan();base=[[0,0,0,v,0,0] for v in (4,4,8,0)]
        raw=copy.deepcopy(base)
        for i in (1,2,3):raw[i][3]=99
        ss=P.starts(base,raw,p,3,int)
        self.assertEqual(ss["clean"]["state"],base)
        self.assertEqual(ss["sham"]["state"],base)
        self.assertEqual(ss["state_error"]["state"][0][3],4)
        self.assertEqual(ss["state_error"]["state"][1][3],7)
        self.assertEqual(ss["new_warm"]["state"][0][3],7)
        self.assertEqual(ss["new_warm"]["state"][2][3],8)
        self.assertEqual(ss["new_scratch"]["state"][2][3],99)
        self.assertTrue(ss["new_warm_pinned"]["pinned"])
        self.assertEqual(ss["state_error"]["truth_ref"],"old")
        self.assertEqual(base[0][3],4)

    def test_independent_truth_rejects_wrong_label(self):
        with self.assertRaises(ValueError):
            P.make_plan([1],{1:0},{1:[0]},{0:4},{1:5},1,0,1)

    def test_selection_does_not_depend_on_node_order(self):
        args=dict(depths={1:0,2:0,3:1},parents={1:[0],2:[4],3:[1,2]},
            constants={0:4,4:9},gid=32)
        self.assertEqual(P.select_leaf([1,2,3],**args),P.select_leaf([3,2,1],**args))

    def test_real_summary_old_new_and_last_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);runs=[]
            for arm in ("model_a","model_b"):
                path=tmp/arm;path.mkdir();runs.append(path)
                old=[["VALUE","NA","NA","4"],["VARIABLE","x","ADD","4"]]
                new=copy.deepcopy(old)
                for x in new:x[3]="7"
                bs={}
                for name in P.BRANCHES:
                    isnew=name.startswith("new_")
                    output=copy.deepcopy(new if isnew else old)
                    first=copy.deepcopy(old) if name=="new_warm" else output
                    bs[name]=dict(initial=copy.deepcopy(old),inputs=[old,first],outputs=[first,output],
                        truth_ref="new" if isnew else "old",start_kind="scratch" if "scratch" in name else "completed")
                rec=dict(graph_id=32,leaf=1,constant=0,delta=3,positions=[0,1],old_truth4=old,new_truth4=new,
                    state_round=3,parents={"1":[0]},regions={"all_nodes":[1],"changed_nodes":[1],
                    "changed_descendants":[],"constant":[0]},branches=bs)
                (path/"records.jsonl").write_text(json.dumps(rec)+"\n",encoding="utf-8")
                (path/"selection.jsonl").write_text(json.dumps(dict(graph_id=32,leaf=1,constant=0,status="selected"))+"\n",encoding="utf-8")
                write_json(path/"meta.json",dict(status="complete",protocol="discrete_reassignment_v1",
                    model_key=arm,dataset_sha256="d",manifest_sha256="m",candidate_ids=[32],horizon=2,batch_size=8,
                    deltas=[3],code_hashes={},records_sha256=sha(path/"records.jsonl"),
                    selection_sha256=sha(path/"selection.jsonl")))
            W.run(runs,tmp/"out")
            with (tmp/"out/summary.csv").open() as f: rows=list(csv.DictReader(f))
            def get(name,window,region="changed_nodes"):
                return next(x for x in rows if x["arm"]=="model_a" and x["branch"]==name and x["region"]==region
                    and x["round_or_window"]==window and x["cohort"]=="common_qualified")
            self.assertEqual(get("new_warm","2")["value_n"],"1")
            self.assertEqual(get("new_warm","last_two")["value_n"],"0")
            self.assertEqual(get("state_error","2")["value_n"],"1")
            self.assertEqual(get("new_warm","2","changed_descendants")["value_macro"],"")

if __name__=="__main__":unittest.main()
