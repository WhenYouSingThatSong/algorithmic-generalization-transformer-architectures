"""Truth-anchored alignment: penalize the wrong side, never imitate wrong plans."""
import torch
import torch.nn.functional as F

def mean_at(x,mask):
    # Per-graph mean, then graph mean: exactly additive across micro-batches.
    return ((x*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()

def balanced(x,reference,nodes):
    values=[];exists=[]
    for v in (False,True):
        mask=nodes & (reference==v);n=mask.sum(-1)
        values.append((x*mask).sum(-1)/n.clamp_min(1));exists.append(n>0)
    return ((values[0]+values[1])/(exists[0].long()+exists[1].long()).clamp_min(1)).mean()

def ce(logits,target):
    return F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').view_as(target)

def update_loss(logits,mid,current,target,reference,nodes,valid):
    # No gradients through label construction, argmax, or disagreement selectors.
    loc=ce(mid,reference.long())
    val=ce(logits[3],target[...,3])
    state=sum(mean_at(ce(logits[f],target[...,f]),valid) for f in range(3))
    state=state+balanced(val,reference,nodes)+mean_at(val,valid & ~nodes)
    plan=mid.detach().argmax(-1).bool()
    changed=logits[3].detach().argmax(-1)!=current[...,3]
    disagreement=(plan!=changed)&nodes
    wrong_plan=disagreement & (changed==reference)
    wrong_execution=disagreement & (plan==reference)
    # Average over ALL eligible nodes/groups, not just current errors.
    align=balanced(loc*wrong_plan+val*wrong_execution,reference,nodes)
    locate=balanced(loc,reference,nodes)
    clear=mean_at(ce(logits[4],torch.zeros_like(reference,dtype=torch.long)),nodes)
    loss=state+2*locate+5*align+.1*clear
    metrics=dict(state=float(state.detach()),locate=float(locate.detach()),align=float(align.detach()),
                 clear=float(clear.detach()),nodes=int(nodes.sum()),reference_updates=int((reference&nodes).sum()),
                 missed=int((reference&~plan&nodes).sum()),false_updates=int((~reference&plan&nodes).sum()),
                 disagreements=int(disagreement.sum()),wrong_plan=int(wrong_plan.sum()),
                 wrong_execution=int(wrong_execution.sum()))
    return loss,metrics
