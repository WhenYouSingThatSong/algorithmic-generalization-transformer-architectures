import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import local_protocol
from local_loss import recurrent_loss
from tensor_targets import Topology
from rebinding import case, rollout
sys.path.insert(0,str(local_protocol.P.ROOT/"experiments"))
from tokenizers import FactoredVocabTokenizer


class TensorTests(unittest.TestCase):
    def test_completed_gates_individual_graphs_and_backpropagates(self):
        g = dict(nodes=[1],depths={1:0},parents={1:[]},constants={1:0})
        y = torch.zeros(2,2,6,dtype=torch.long)
        y[0,:,3]=4
        y[1,:,3]=7
        x=y.clone()
        x[:,1,3]=24
        topo = Topology([g,g]).batch([0,1],2,"cpu")
        class RT:
            tok=SimpleNamespace(syntax_tok2idx={"pad":6},PAD="pad")
            def __init__(self):
                self.model=torch.nn.Module()
                self.model.output_layers=torch.nn.ModuleList([torch.nn.Linear(8,n) for n in (7,129,2,25,2,2)])
                with torch.no_grad():
                    self.model.output_layers[3].bias[4]=50
            def init(self,x):
                return torch.zeros(*x.shape[:2],8)
            def forward(self,h,middle=True):
                v=torch.tanh(h+.1)
                logits=[head(v) for head in self.model.output_layers]
                return v,logits,logits[4]
        rt=RT()
        events=[dict(bi=i,pos=1,round=1,delta=1) for i in range(2)]
        donors={(i,1):(torch.ones(8),torch.tensor([0,0,0,9]),torch.tensor([0,0,0,9]),torch.tensor(True))
                for i in range(2)}
        with patch("local_loss.make_donors",return_value=donors):
            loss,audit=recurrent_loss(rt,x,y,[g,g],topo,events,dict(mode="completed",rounds=2),0,32)
        self.assertEqual([bool(r["accepted"]) for r in audit],[True,False])
        loss.backward()
        self.assertGreater(float(rt.model.output_layers[4].weight.grad.norm()),0)

    def test_single_prompt_one_initialization(self):
        tok=FactoredVocabTokenizer(128,["ADD"],23,3)
        class RT:
            device="cpu"
            def __init__(self):
                self.calls=self.inits=0
                self.tok=tok
            def init(self,x):
                self.inits+=1
                return x
            def forward(self,h):
                self.calls+=1
                logits=torch.nn.functional.one_hot(h[...,3],num_classes=25)
                heads=[torch.nn.functional.one_hot(h[...,i],n) for i,n in enumerate((7,129,2))]
                return h,[*heads,logits],None
        c=case("interleaved",4,0)
        rt=RT()
        (a,b),attrs=rollout(rt,list(c["branches"].values()),8)
        self.assertEqual((rt.inits,rt.calls),(1,8))
        self.assertEqual(len(a[0]),len(c["branches"]["rebound"]["nodes"]))
        self.assertEqual(a,b)
        self.assertTrue(all(attrs[0][8]["variable"]))
