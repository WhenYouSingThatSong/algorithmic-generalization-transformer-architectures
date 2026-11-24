"""Run the actual focused evaluator on a known synchronous ADD executor."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checks.minimal_dev import measure
from local_protocol import P


class FocusedTests(unittest.TestCase):
    def test_same_graph_factorial_and_actual_complete(self):
        nodes=[1,3,4,5,6,7]
        g=dict(nodes=nodes, parents={1:[],3:[],4:[1],5:[3],6:[4],7:[5]},
               depths={1:0,3:0,4:1,5:1,6:2,7:2},D=3,constants={1:0,3:2},
               coefs={1:{1:1},3:{3:1},4:{1:1},5:{3:1},6:{1:1},7:{3:1}})
        y=torch.zeros(8,6,dtype=torch.long)
        y[:,3]=torch.tensor([4,4,8,8,4,8,4,8])
        x=y.unsqueeze(0).clone()
        x[0,nodes,3]=24
        class RT:
            update=True
            torch=torch
            def init(self,x):return x.clone()
            def decode(self,h):return h
            def forward(self,h,middle=False):
                old=h[0].tolist()
                target,ref=P.local_step(g,old,old)
                out=h.clone()
                out[0,:,:4]=torch.tensor(target)
                logits=[torch.nn.functional.one_hot(out[...,i],n).float() for i,n in enumerate((7,129,2,25,2,2))]
                mid=torch.nn.functional.one_hot(torch.tensor([ref]).long(),2).float()
                return out,logits,mid
        clean,r=measure(RT(),x,y,g,dict(gid=9,target=dict(u=4,q=6,delta=1)))
        self.assertEqual(clean,dict(gid=9,rounds=4,nodes=6,value_correct=6,first4_correct=6))
        by={(v["timing"],v["condition"]):v for v in r}
        self.assertEqual(by["actual_complete","clean"]["start"],3)
        for cond in ("empty_only","wrong","empty_and_wrong","parent_child_consistent_wrong"):
            self.assertEqual(by["actual_complete",cond]["status"],"ok")
        self.assertTrue(by["actual_complete","empty_and_wrong"]["rounds"][0]["mixed"])
        self.assertEqual(by["actual_complete","parent_child_consistent_wrong"]["rounds"][0]["roles"]["indirect_consistent"],1)
        self.assertEqual(by["train_end","wrong"]["status"],"ok")
