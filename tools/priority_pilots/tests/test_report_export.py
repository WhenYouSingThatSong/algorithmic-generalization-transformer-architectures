"""Regression: reporting must accept scientific differences instead of rejecting them."""
import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from io_utils import sha,write_json
from reassignment_core import BRANCHES

class ExportTests(unittest.TestCase):
    def test_outcome_differences_do_not_abort_and_counts_are_measured(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); runs=[]
            old=[["VALUE","NA","NA","4"],["VARIABLE","x1","ADD","4"],["VARIABLE","x2","ADD","8"]]
            new=copy.deepcopy(old)
            for row,v in zip(new,("7","7","14")):row[3]=v
            branches={}
            for name in BRANCHES:
                isnew=name.startswith("new_")
                initial=copy.deepcopy(old)
                if isnew:initial[0][3]="7"
                if name=="state_error":initial[1][3]="7"
                if "scratch" in name:
                    initial[1][3]=initial[2][3]="EMPTY"
                output=copy.deepcopy(new if isnew else old)
                if name=="new_warm":output[1][1]="drifted_variable"
                branches[name]=dict(initial=initial,inputs=[initial,output],outputs=[output,output],
                    truth_ref="new" if isnew else "old",pinned=name.endswith("_pinned"),
                    start_kind="scratch" if "scratch" in name else "completed")
            r=dict(graph_id=32,leaf=1,constant=0,delta=3,horizon=2,state_round=4,positions=[0,1,2],
                old_truth4=old,new_truth4=new,new_constant=7,constants={"0":4},parents={"1":[0],"2":[1,1]},
                regions=dict(all_nodes=[1,2],leaf=[1],changed_nodes=[1,2],changed_descendants=[2],
                    unchanged_nodes=[],unchanged_descendants=[],outside_descendants=[],constant=[0]),branches=branches)
            for arm in ("model_a","model_b"):
                path=root/arm;path.mkdir();runs.append(path)
                (path/"records.jsonl").write_text(json.dumps(r)+"\n",encoding="utf-8")
                (path/"selection.jsonl").write_text(json.dumps(dict(graph_id=32,leaf=1,constant=0,state_round=4,status="selected"))+"\n",encoding="utf-8")
                write_json(path/"meta.json",dict(status="complete",protocol="discrete_reassignment_v1",
                    model_key=arm,records_sha256=sha(path/"records.jsonl"),selection_sha256=sha(path/"selection.jsonl"),
                    dataset_sha256="dataset",manifest_sha256="manifest",candidate_ids=[32],horizon=2,batch_size=8,deltas=[3],code_hashes={}))
            out=root/"out"
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"export_reassignment_tables.py"),
                "--run-dir",str(runs[0]),"--run-dir",str(runs[1]),"--out-dir",str(out)],check=True,capture_output=True)
            audit=json.loads((out/"priority-reassignment-audit.json").read_text())
            self.assertFalse(audit["checks"]["pinned_free_identical"])
            self.assertFalse(audit["checks"]["value_equals_first4"])
            self.assertIn("not checked",audit["checks"]["raw_scratch_check"])
            with (out/"priority-reassignment-warm-scratch.csv").open() as f:
                self.assertTrue(all(r["cases"]=="1" for r in csv.DictReader(f)))
            with (out/"priority-reassignment-curves.csv").open() as f:
                self.assertEqual({r["round_or_window"] for r in csv.DictReader(f)},{"1","2","last_two"})

if __name__=="__main__":unittest.main()
