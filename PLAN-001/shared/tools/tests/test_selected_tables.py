"""CPU regression cases for selected-seed scoring and table generation."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from c_selected_eval import FROZEN, MODELS, RECEIPT, detail, read
from c_selected_tables import ARMS, NAMES, write_report

sys.path.insert(0, str(FROZEN))
from scenarios import metrics

G = {"nodes":[0,1,2,3], "parents":{0:[],1:[0],2:[1],3:[0]},
     "constants":{0:4}, "depths":{0:0,1:1,2:2,3:1}}
Y = [[0, i, 0, 7] for i in range(5)]


def row(before, after):
    updates = []
    for i, value in enumerate(before):
        ps = G["parents"][i]
        expected = (7 if not ps else before[ps[0]]) if all(before[p]<23 for p in ps) else value
        updates.append(dict(reference=value!=expected, plan=value!=expected, changed=value!=after[i],
                            value_correct=after[i]==expected))
    return dict(input=[[0,i,0,x] for i,x in enumerate(before)],
                output=[[x,x==7,True,True,True] for x in after], update=updates)


def focused_case(t, arm):
    clean = row([7]*4, [7]*4)
    values = [7]*4
    if arm in ("wrong","empty_and_wrong","parent_child_consistent_wrong"):
        values[1] = 9
    if arm in ("empty_only","empty_and_wrong"):
        values[3] = 24
    if arm == "parent_child_consistent_wrong":
        values[2] = 9
    after = [7,7,values[1],7]
    trace = [row(values, after)] + [clean]*15
    base = [clean]*16
    m = metrics(G,Y,trace,base,t,arm,1)
    d = detail(G,Y,trace,base,1)
    return dict(**m, status="ok", gid=0, start=4, target=1, fill_target=3, selected_detail=d)


class SelectedReports(unittest.TestCase):
    def test_functional_conflict_and_mixed(self):
        r = focused_case("actual_complete","empty_and_wrong")
        self.assertEqual(r["selected_detail"]["first_wrong_positions"],[2])
        self.assertEqual([c["position"] for c in r["selected_detail"]["input_changes"]],[1,3])
        # Position 2 was correct. Its temporarily wrong parent induces harmful recomputation.
        self.assertIn([1,False,True,True,True,True,False,1], r["selected_detail"]["conflicts"])
        self.assertTrue(r["rounds"][0]["jointly_correct"])
        self.assertFalse(r["selected_detail"]["mixed_all_gold_correct"])

    def test_report_end_to_end(self):
        # Use the production scorer and frozen field layout with small artificial graphs;
        # fixtures only exercise registration and aggregation, not model performance.
        receipt = read(RECEIPT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for model in MODELS:
                path = root/model
                path.mkdir()
                paths.append(path)
                final = next(x for x in receipt["runs"] if x["model"]==model)["final_ckpt"]["sha256"]
                (path/"meta.json").write_text(json.dumps(dict(status="complete",checkpoint=final,
                    source=dict(model=model,manifest=receipt["manifest_sha256"]))))
                for size in (32,64,128):
                    for band in ("shallow","deep"):
                        split = f"N{size}_{band}"
                        records=[]
                        for gid in range(32):
                            for timing in ("frontier","middle","train_end","actual_complete"):
                                arms = (*ARMS,"parent_child_consistent_wrong") if timing=="actual_complete" else ARMS[:2]
                                for arm in arms:
                                    r=dict(status="no_target") if gid else focused_case(timing,arm)
                                    records.append(dict(r,split=split,gid=gid,timing=timing,condition=arm))
                        (path/f"{split}-classes.json").write_text(json.dumps(dict(records=records)))
                        (path/f"{split}-standard-summary.json").write_text('{"counts":[]}')
                        (path/f"{split}-outcomes.json").write_text('{"natural":[]}')
                        (path/f"{split}-frozen-probe.json").write_text('{"records":[]}')
                        clean = dict(status="ok", module="clean", split=split, D=1,nodes=[0],
                                     truth=[Y[0]],rows=[dict(input=[Y[0]],output=[[7,True,True,True,True]])]*65)
                        with (path/f"{split}-standard.jsonl").open("w") as stream:
                            for gid in range(128):
                                print(json.dumps(dict(clean,gid=gid)), file=stream)
                            if size==32:
                                for arm,n in (("clean",7),("perturbed",9)):
                                    obs=dict(clean,gid=0,module="reuse",lag=1,condition=arm,new_node=0,u=0,
                                             u_readout=[0,0,0,n])
                                    print(json.dumps(obs),file=stream)
            out = root/"report"
            write_report(paths,out)
            self.assertEqual(len(list(out.glob("*.csv"))),6)
            def lookup(table, metric, arm=None):
                with (out/(NAMES[table]+".csv")).open(encoding="utf-8-sig") as f:
                    return next(r for r in csv.DictReader(f)
                                if r["指标"]==metric and (arm is None or r["条件"]==arm))
            self.assertEqual(lookup(4,"四臂均成功／候选")["分母合计"],"32")
            self.assertEqual(lookup(4,"四臂均成功／候选")["分子合计"],"1")
            self.assertEqual(lookup(4,"三类输出均符合原题","空位与错值同时出现")["合并比值"],"0.0")
            with (out/(NAMES[5]+".csv")).open(encoding="utf-8-sig") as f:
                used=[r for r in csv.DictReader(f) if r["指标"]=="目标读错且新下游对" and r["分母合计"]=="1"]
            self.assertTrue(used and all(r["合并比值"]=="1.0" for r in used))
            self.assertEqual(lookup(0,"节点数值正确","同起点干净")["分母合计"],"128")
            with self.assertRaises(FileExistsError):
                write_report(paths,out)
            name = "N32_shallow-classes.json"
            payload = read(paths[0]/name)
            legacy = read(paths[0]/name)
            del legacy["records"][0]["selected_detail"]
            (paths[0]/name).write_text(json.dumps(legacy))
            retry = root/"supplemented-report"
            with self.assertRaisesRegex(ValueError, "missing focused details"):
                write_report(paths,retry)
            self.assertFalse(retry.exists())
            extra = root/"supplements"/MODELS[0]
            extra.mkdir(parents=True)
            meta = read(paths[0]/"meta.json")
            payload.update(checkpoint=meta["checkpoint"], manifest=meta["source"]["manifest"])
            (extra/name).write_text(json.dumps(payload))
            write_report(paths,retry,extra.parent)
            for original in out.glob("*.csv"):
                self.assertEqual(original.read_bytes(), (retry/original.name).read_bytes())

if __name__ == "__main__":
    unittest.main()
