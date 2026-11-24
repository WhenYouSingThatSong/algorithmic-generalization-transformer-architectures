import unittest

import torch
from abc_scene_queue import archive,run_grid
from abc_scene_report import grid_observations
from test_attribution_suite import Oracle


class MatchedScenes(unittest.TestCase):
    def test_archived_recipes_match_discrete_reference(self):
        data=archive("C_s42")
        key="352-4"
        original=[r for r in data["records"] if r["case"]==key]
        old=dict(data["cases"][key],events=original)
        with torch.inference_mode():
            new=run_grid(Oracle(),352,4,original)
        a=dict(grid_observations(old,True))
        b=dict(grid_observations(new))
        self.assertEqual(len(b),22)
        self.assertEqual(set(a),set(b))
        self.assertTrue(all(r["eligible"] for r in b.values()))
        self.assertEqual(b["chain","before","target_pending_wrong"]["early_numerical_error"],1)
        self.assertTrue(b["chain","complete","later_target"]["end_joint"])

    def test_cancelled_parent_edits_and_independent_branch(self):
        with torch.inference_mode():
            new=run_grid(Oracle(),400,4)
        by_label={e["label"]:e for e in new["events"]}
        u=new["u"]
        gold=new["nodes"][u]["truth"]
        equal=by_label["complete_both_parents_cancel"]
        add=by_label["complete_both_parents_add"]
        self.assertTrue(equal["value_eligible"] and add["value_eligible"])
        self.assertTrue(all(w["actual"][3]!=w["before"][3] for w in equal["writes"]))
        self.assertEqual(equal["rows"][0][u],gold)
        self.assertNotEqual(add["rows"][0][u],gold)
        negative=by_label["complete_separate_branch"]
        self.assertTrue(negative["writes"][0]["actual"][3]!=negative["writes"][0]["before"][3])
        self.assertEqual(negative["rows"][0][u],gold)
        self.assertEqual(by_label["complete_restore_empty"]["writes"][0]["actual"][3],24)
