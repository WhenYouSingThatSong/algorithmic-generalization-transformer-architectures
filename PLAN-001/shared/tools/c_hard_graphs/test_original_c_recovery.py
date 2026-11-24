import unittest

import torch
from original_c_recovery import family, historical, measure, outcomes
from test_attribution_suite import Oracle


class RecoveryChecks(unittest.TestCase):
    def test_immutable_archived_counts(self):
        data = historical()
        for model,local,specific,stable in (("A_s42",0,32,0),("B_s42",32,10,10),
                                            ("C_s42",29,32,32),("C_s1234",17,32,32)):
            row = data[model]
            self.assertEqual((row["direct_repaired_next"],row["specific"],row["stable_after_damage"]),
                             (local,specific,stable))

    def test_graph_and_donors(self):
        for size in (32,64,128):
            for depth in (4,10):
                variants,spine = family(17,size,depth)
                base,sham,a,b = [v["nodes"] for v in variants]
                self.assertEqual((len(base),max(n["depth"] for n in base)),(size,depth))
                self.assertLessEqual(len(variants[0]["tokens"]),1024)
                for i in spine:
                    self.assertEqual(base[i]["truth"],sham[i]["truth"])
                    self.assertEqual(len({v[i]["truth"] for v in (base,a,b)}),3)
                self.assertFalse(any(spine[-1] in n["parents"] for n in base))

    def test_release_and_timing_on_arithmetic_reference(self):
        torch.set_num_threads(1)
        with torch.inference_mode():
            case = measure(Oracle(),999,32,4)
        events = {e["condition"]:e for e in case["events"]
                  if e["role"]=="internal" and e["timing"]=="graph_complete"}
        self.assertEqual(case["natural"],[])
        self.assertTrue(case["natural_opportunities"]>0)
        self.assertFalse(events["same_value"]["effective_change"])
        self.assertEqual(events["target"]["joint_stable_after_specific"],1)
        self.assertTrue(events["both"]["first_uses_wrong_parent"])
        e = events["held_parent"]
        self.assertEqual(len(e["trace"]),20)
        self.assertEqual([z["u"] for z in e["trace"][:4]],[e["parent_implied_u"]]*4)
        self.assertEqual(e["trace"][4]["u"],e["gold_target"])
        self.assertEqual(case["complete"],5)

    def test_censored_damage(self):
        c = [{"u":3,"q":5}]*4
        z = c[:3]+[{"u":3,"q":6}]
        out = outcomes(c,z,(3,5),6,0)
        self.assertEqual((out["specific_child"],out["specific_with_followup"],out["joint_stable_after_specific"]),
                         (1,0,0))
