"""Server-only small torch checks: intervention gradient and whole recurrence."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pilot_core as C
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'torch is required on execution server')
class TensorTests(unittest.TestCase):
    def test_noise_norm_and_gradient_path(self):
        h=torch.randn(2,3,4,requires_grad=True);donor=torch.ones(4)
        x,r,n=C.intervene(h,0,1,donor,'noise',42)
        self.assertAlmostEqual(r,n,places=5)
        x.sum().backward();self.assertTrue(torch.equal(h.grad,torch.ones_like(h)))
        y,_,n3=C.intervene(h,0,1,donor,'noise',42,3)
        self.assertAlmostEqual(n3,3*r,places=5)
        self.assertTrue(torch.allclose(y[0,1]-h[0,1],3*(x[0,1]-h[0,1])))
    def test_semantic_replacement_blocks_only_overwritten_row(self):
        h=torch.randn(2,3,4,requires_grad=True);donor=torch.ones(4,requires_grad=True)
        x,_,_=C.intervene(h,0,1,donor,'semantic',42)
        x.sum().backward();expected=torch.ones_like(h);expected[0,1]=0
        self.assertTrue(torch.equal(h.grad,expected));self.assertIsNone(donor.grad)
    def test_full_recurrence_no_events_matches_long_hold(self):
        # Import by path to avoid experiments/train.py name collision.
        train=C.load_module('corruption_train_test',C.HERE/'train.py')
        sys.path.insert(0,str(C.HERE.parent/'c_long_pilot'))
        old=C.load_module('long_train_test',C.HERE.parent/'c_long_pilot/train.py')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.e=torch.nn.Embedding(3,4);self.w=torch.nn.Linear(4,4)
                self.output_layers=torch.nn.ModuleList([torch.nn.Linear(4,3) for _ in range(6)])
            def token_embedder(self,x):return self.e(x[...,0])
            def forward(self,h,_):
                h=torch.tanh(self.w(h));return h,[head(h) for head in self.output_layers]
        class Lit:
            factor_loss_scales=[1,1,1,1,0,0]
            def __init__(self):self.model=Model();self.criterion=torch.nn.CrossEntropyLoss()
            def get_teacher_batch_with_depth(self,x,y,d):return torch.stack([x,y,y]),None,None
        lit=Lit();x=torch.zeros(1,3,6,dtype=torch.long);y=torch.ones_like(x)
        info=torch.ones(1,3,1,dtype=torch.long)
        new,*_=train.recurrent_loss(lit,x,y,info,[],{},'hold',1)
        oldloss,*_=old.recurrent_loss(lit,x,y,info,1)
        self.assertTrue(torch.allclose(new,oldloss))
        new.backward();self.assertGreater(float(lit.model.e.weight.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
