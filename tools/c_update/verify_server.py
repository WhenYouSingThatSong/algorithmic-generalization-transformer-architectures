"""One focused actual-model check before the authorized runs; not an experiment result."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--num_threads',type=int,default=4)
    a=p.parse_args();plan=P.read(a.plan)
    from runtime import Runtime,load_data,dataset_graphs
    spec=dict(next(z for z in plan['models'] if z['id']=='C_s2026'),arm='update')
    rt=Runtime(spec,a.device,initialize_update=True);torch=rt.torch
    x,y,info=load_data(plan['train']['path'],rt.tok);x=x[:2].to(rt.device);y=y[:2].to(rt.device);info=info[:2].to(rt.device)
    gs=dataset_graphs(y,info)
    with torch.no_grad():
        h=rt.init(x);expected,logits=rt.model(h,True);actual,_,mid=rt.forward(h,middle=True)
        if not torch.equal(expected,actual) or mid is None:raise ValueError('hook changed forward')
        # Independent old implementation verifies the new correction labels.
        ext,_,_=rt.lit.get_teacher_batch_with_depth(x,y,info[...,0])
        for t in range(len(ext)-1):
            target=x.clone()
            for bi,g in enumerate(gs):
                for pos in g['nodes']:
                    if g['depths'][pos]<=t:target[bi,pos]=y[bi,pos]
            if not torch.equal(target[...,:4],ext[t+1,...,:4]):raise ValueError('teacher labels differ')
        # Append a real tokenizer equation and verify original hidden rows untouched.
        from evaluate import append_compute
        n=int((x[0,:,0]!=rt.tok.syntax_tok2idx[rt.tok.PAD]).sum())
        appended,reason=append_compute(rt,x[:1],y[0],gs[0],h[:1],gs[0]['nodes'][-1])
        if reason:raise ValueError(reason)
        hh,yy,gg,pos=appended
        if not torch.equal(hh[:,:n],h[:1,:n]):raise ValueError('appending rewrote existing state')
        if int(yy[pos,1]) in x[0,:,1].tolist():raise ValueError('new variable was not unused')
    # Gradient test through TWO loops, including layer-2 classification.
    rt.model.train();rt.model.zero_grad(set_to_none=True)
    h0=rt.init(x[:1]);h1,_,_=rt.forward(h0,middle=True);h1.retain_grad()
    h2,z,mid=rt.forward(h1,middle=True)
    loss=z[3].square().mean()+mid.square().mean();loss.backward()
    if h1.grad is None or not torch.isfinite(h1.grad).all() or not h1.grad.abs().sum()>0:raise ValueError('BPTT severed')
    if rt.model.output_layers[4].weight.grad is None:raise ValueError('UPDATE head no gradient')
    rt.model.eval();rt.model.zero_grad(set_to_none=True)
    # A few original-C functional probes, reported even if no propagation.
    from functional_probe import run_probe
    report=run_probe(rt,plan,'dev',limit=2)
    P.write(a.output,dict(status='passed',checks=['original_forward_equal','teacher_labels_equal','append_preserves_old_rows','BPTT_two_rounds','UPDATE_head_gradient'],functional_probe=report))

if __name__=='__main__':main()
