import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spec as S
P=S.P


def fixture():
    # Two leaves, two levels; repeated parent tests sum multiplicity.
    y=[[2,128,1,3,0,0],[2,128,1,4,0,0]]+[[0,i,1,v,0,0] for i,v in enumerate((3,4,7,14))]
    info=[[-1]*6,[-1]*6,[0,0,-1,-1,-1,-1],[0,1,-1,-1,-1,-1],
          [1,2,0,3,-1,-1],[2,4,0,4,-1,-1]]
    return y,P.graph(y,info)


class RecipeTests(unittest.TestCase):
    def test_horizons_and_structural_positions(self):
        _,g=fixture()
        for mode in S.MODES:
            config=S.schedule(g['D'],1,0,mode,True)
            n,ee=S.events([g]*32,list(range(32)),1,0,config)
            self.assertTrue(ee)
            self.assertEqual(n,32*(2 if mode=='middle' else 4))
            for e in ee:
                self.assertLess(e['round'],config['rounds'])
                self.assertEqual(e['donor_round'],e['round']+(mode=='frontier'))
                if mode!='frontier':
                    self.assertEqual(config['rounds']-e['round'],g['D']+1)
            off=dict(config,enabled=False)
            self.assertEqual(S.events([g]*32,list(range(32)),1,0,off),(n,[]))

    def test_same_event_selection_across_microbatches(self):
        _,g=fixture();gs=[g]*32;ids=list(range(32))
        config=S.schedule(3,1,2,'late_budget',True)
        _,ee=S.events(gs,ids,1,2,config)
        # Selection receives NO decoded states or model objects.
        pieces=[e for off in range(0,32,8) for e in ee if off<=e['bi']<off+8]
        self.assertEqual(ee,pieces)
        self.assertEqual(ee,S.events(gs,ids,1,2,config)[1])


@unittest.skipUnless(importlib.util.find_spec('torch'),'run torch parity on the execution server')
class TensorTests(unittest.TestCase):
    def test_local_and_teacher_exact(self):
        import torch
        from tensor_targets import Topology
        y,g=fixture(); yy=torch.tensor([y,y]); x=yy.clone();x[:,2:,3]=24
        top=Topology([g,g]).batch([0,1],len(y),'cpu')
        torch.manual_seed(3)
        for _ in range(20):
            current=yy[...,:4].clone()
            current[...,3]=torch.randint(0,25,current.shape[:2])
            current[...,1]=torch.randint(0,129,current.shape[:2])
            target,ref=top.local(current,yy)
            for i in range(2):
                expected,mask=P.local_step(g,current[i].tolist(),y)
                self.assertEqual(target[i].tolist(),expected)
                self.assertEqual(ref[i].tolist(),mask)
        for t in range(g['D']+1):
            expected=x.clone()
            for p in g['nodes']:
                if g['depths'][p]<=t: expected[:,p]=yy[:,p]
            self.assertTrue(torch.equal(top.teacher(x,yy,t),expected))
        current=yy[...,:4].clone();current[:,2,3]=5
        target,ref=top.local(current,yy)
        self.assertEqual(target[0,:,3].tolist(),[3,4,3,4,9,14])

    def test_loss_and_gradients_match_old_update(self):
        import torch
        from losses import update_loss
        from process_loss import process_loss,materialize
        torch.manual_seed(4)
        current=torch.zeros(3,6,4,dtype=torch.long)
        target=current.clone();ref=torch.tensor([[1,0,1,0,0,0],[0]*6,[1]*6],dtype=torch.bool)
        target[...,3]=ref.long()
        nodes=torch.ones_like(ref);nodes[:,0]=False
        valid=torch.ones_like(ref);valid[:,-1]=False
        z=[torch.randn(3,6,n,requires_grad=True) for n in (7,129,2,25,2,2)]
        mid=torch.randn(3,6,2,requires_grad=True)
        old,_=update_loss(z,mid,current,target,ref,nodes,valid)
        new,stats=process_loss(z,mid,current,target,ref,nodes,valid)
        self.assertTrue(torch.equal(old,new))
        params=z[:5]+[mid]
        a=torch.autograd.grad(old,params,retain_graph=True)
        b=torch.autograd.grad(new,params)
        for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))
        parts=sum(process_loss([v[i:i+1] for v in z],mid[i:i+1],current[i:i+1],
                               target[i:i+1],ref[i:i+1],nodes[i:i+1],valid[i:i+1])[0]/3 for i in range(3))
        self.assertTrue(torch.allclose(new,parts,atol=1e-5))
        out=materialize(dict(stats=stats,vec=target[0,0],flag=ref[0,0]))
        self.assertIsInstance(out['stats']['nodes'],int)
        self.assertIsInstance(out['flag'],bool)
        self.assertEqual(out['vec'],target[0,0].tolist())


if __name__=='__main__': unittest.main()
