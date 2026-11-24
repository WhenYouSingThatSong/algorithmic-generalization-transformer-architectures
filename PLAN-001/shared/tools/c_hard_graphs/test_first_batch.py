import unittest

import torch

from first_batch_probes import fanout_probe, history_probe, mark_probe, natural_probe, wrong_literal
from first_batch_report import propagation
from original_c_recovery import family
from research_probes import encode
from test_attribution_suite import Oracle


class FirstBatchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_position_and_clock_matched_donors(self):
        rt=Oracle()
        case=history_probe(rt,400,4)
        self.assertTrue(all(a["eligible"] for a in case["arms"]))
        self.assertEqual([a["trace"][0][1] for a in case["arms"]],
                         [case["reader_gold"]]*3+[case["wrong_reader"]])
        graphs,spine=family(2,32,4)
        u=spine[2]
        nodes=graphs[0]["nodes"]
        wanted=(nodes[u]["truth"]+11)%23
        h=wrong_literal(rt,encode(rt,graphs),nodes,u,wanted,8)
        self.assertEqual(int(h[0,nodes[u]["pos"],3]),wanted)

    def test_fanout_distinguishes_repair_and_consumption(self):
        for event in fanout_probe(Oracle(),400,4)["records"]:
            self.assertTrue(event["eligible"])
            row=propagation(event)
            self.assertEqual(row["target_first_round_sum"],1)
            self.assertEqual(len(event["children"]),event["fanout"])
            if event["phase"]=="complete":
                self.assertEqual(row["children_specific"],event["fanout"])
                self.assertGreater(row["descendant_wrong_number_rounds"],0)
                self.assertTrue(row["stable_after_specific"])

    def test_mixed_scene_scores_direct_conflict_and_healthy(self):
        rows=[z for z in mark_probe(Oracle(),400,4)["rows"] if z["phase"]=="complete"
              and z["condition"]=="wrong_and_blank" and z["eligible"]]
        self.assertTrue({"new","direct_conflict","healthy"}.issubset({z["category"] for z in rows}))
        self.assertTrue(all((z["score"]>.5)==(z["before"]!=z["after"]) for z in rows))

    def test_matched_natural_and_artificial_wrong(self):
        graphs,spine=family(0,32,4)
        nodes=graphs[0]["nodes"]
        u,q=spine[2:4]
        wrong=(nodes[u]["truth"]+1)%23
        child=(nodes[q]["truth"]+1)%23

        class TransientFault(Oracle):
            clock=0

            def forward(self,h,middle=False):
                out,logits,mid=super().forward(h,middle)
                self.clock+=1
                if self.clock==6:
                    out[0,nodes[u]["pos"],3]=wrong
                return out,logits,mid

        event=dict(t=6,u=u,q=q,wrong=wrong,expected=child,replay_child=nodes[q]["truth"],
                   observed_child=child,specific_rescue=True)
        old=dict(gid=0,size=32,depth=4,nodes=nodes,natural_opportunities=1,natural=[event])
        result=natural_probe(TransientFault(),old)["records"][0]
        self.assertTrue(result["paired_replay_agrees"])
        paths=result["backgrounds"]["observed"]["paths"]
        self.assertEqual(paths["natural_wrong"][0][1],child)
        self.assertEqual(paths["artificial_same_wrong"][0][1],child)
        self.assertEqual(paths["target_restored"][0][1],nodes[q]["truth"])


if __name__=="__main__":
    unittest.main()
