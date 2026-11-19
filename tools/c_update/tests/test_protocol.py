import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import protocol as P

def fixture():
    # c0=3,c1=4; a=3,b=4,u=a+b=7,q=u+b=11.
    y=[[2,128,1,3,0,0],[2,128,1,4,0,0]]+[[0,i,1,v,0,0] for i,v in enumerate((3,4,7,11))]
    info=[[-1]*6,[-1]*6,[0,0,-1,-1,-1,-1],[0,1,-1,-1,-1,-1],
          [1,2,0,3,-1,-1],[2,4,0,3,-1,-1]]
    return y,info,P.graph(y,info)

class ProtocolTests(unittest.TestCase):
    def test_synchronous_wrong_parent(self):
        y,_,g=fixture();current=[r[:4] for r in y];current[2][3]=5
        out,mask=P.local_step(g,current,y)
        self.assertEqual([out[p][3] for p in g['nodes']],[3,4,9,11])
        self.assertEqual([mask[p] for p in g['nodes']],[True,False,True,False])
        # u is repaired next round, while q now uses PREVIOUS wrong u=9.
        out2,mask2=P.local_step(g,out,y)
        self.assertEqual([out2[p][3] for p in g['nodes']],[3,4,7,13])
    def test_unready_preserve_and_good_hold(self):
        y,_,g=fixture();cur=[r[:4] for r in y];cur[2][3]=24;cur[4][3]=5
        out,m=P.local_step(g,cur,y)
        self.assertEqual(out[4][3],5);self.assertFalse(m[4]);self.assertTrue(m[2])
        out,m=P.local_step(g,[r[:4] for r in y],y);self.assertFalse(any(m))
    def test_sources_arithmetic(self):
        y,_,g=fixture()
        for p in g['nodes']:
            for delta in range(1,23):
                edits=P.donor_edits(g,p,delta);yy=[r[:] for r in y]
                for k,v in edits.items():yy[k][3]=(yy[k][3]+v)%23
                self.assertEqual(P.truth(g,yy)[p],(y[p][3]+delta)%23)
        edits=P.same_edits(g,4);self.assertTrue(edits);yy=[r[:] for r in y]
        for k,v in edits.items():yy[k][3]=(yy[k][3]+v)%23
        self.assertEqual(P.truth(g,yy)[4],7)
    def test_frontier_and_batch_gate(self):
        _,_,g=fixture();enabled,events=P.events([g]*32,list(range(32)),1,0,force=True)
        self.assertTrue(enabled);self.assertTrue(events)
        self.assertEqual(len({(e['bi'],e['pos']) for e in events}),len(events))
        self.assertTrue(all(e['round']==g['depths'][e['pos']] and e['donor_round']==e['round']+1 for e in events))
        self.assertEqual(events,P.events([g]*32,list(range(32)),1,0,force=True)[1])
        grouped=[e for off in range(0,32,8) for e in events if off<=e['bi']<off+8]
        self.assertEqual(events,grouped)
    def test_no_dropped_graphs(self):
        ds=[3,4,5]*75;bs=P.batches(ds,3)
        self.assertEqual(sorted(i for b in bs for i in b),list(range(len(ds))))
        self.assertTrue(all(len({ds[i] for i in b})==1 for b in bs))
        self.assertEqual(P.lr_factor(0,100),1);self.assertAlmostEqual(P.lr_factor(100,100),.1)
    def test_alignment_truth_anchor(self):
        self.assertEqual(P.alignment_side(False,True,True),'locate')
        self.assertEqual(P.alignment_side(True,False,True),'execute')
        self.assertEqual(P.alignment_side(True,False,False),'locate')
        self.assertEqual(P.alignment_side(False,True,False),'execute')
        self.assertIsNone(P.alignment_side(True,True,False)) # state and locate losses still penalize both.
    def test_recovery_windows(self):
        r=P.horizon_flags([False,True,False,True,True])
        self.assertEqual((r['first'],r['stable'],r['reerror'],r['last_two']),(2,4,True,True))
        self.assertIsNone(P.horizon_flags([False,True])['stable'])

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch runs on execution server')
class TorchTests(unittest.TestCase):
    def test_micro_batch_equivalence(self):
        import torch
        from losses import update_loss
        torch.manual_seed(1)
        cur=torch.zeros(3,4,4,dtype=torch.long);target=cur.clone();ref=torch.tensor([[1,1,0,0],[0,0,0,0],[1,0,0,0]],dtype=torch.bool)
        target[...,3]=ref.long();nodes=torch.ones_like(ref);valid=nodes.clone()
        logits=[torch.randn(3,4,n,requires_grad=True) for n in (7,129,2,25,2,2)];mid=torch.randn(3,4,2,requires_grad=True)
        total,_=update_loss(logits,mid,cur,target,ref,nodes,valid)
        pieces=sum(update_loss([v[i:i+1] for v in logits],mid[i:i+1],cur[i:i+1],target[i:i+1],ref[i:i+1],nodes[i:i+1],valid[i:i+1])[0]/3 for i in range(3))
        self.assertTrue(torch.allclose(total,pieces,atol=1e-5))
        total.backward();self.assertTrue(mid.grad.abs().sum()>0);self.assertTrue(logits[3].grad.abs().sum()>0)

if __name__=='__main__':unittest.main()
