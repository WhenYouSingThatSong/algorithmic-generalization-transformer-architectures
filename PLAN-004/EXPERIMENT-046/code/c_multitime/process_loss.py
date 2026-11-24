"""Same 1/2/5/.1 losses as old UPDATE, without per-loop scalar synchronizations."""
import torch
from losses import balanced, ce, mean_at


def process_loss(logits, mid, current, target, reference, nodes, valid):
    loc = ce(mid, reference.long())
    val = ce(logits[3], target[..., 3])
    state = sum(mean_at(ce(logits[f], target[..., f]), valid) for f in range(3))
    state = state + balanced(val, reference, nodes) + mean_at(val, valid & ~nodes)
    plan = mid.detach().argmax(-1).bool()
    changed = logits[3].detach().argmax(-1) != current[..., 3]
    disagree = (plan != changed) & nodes
    wrong_plan = disagree & (changed == reference)
    wrong_execution = disagree & (plan == reference)
    align = balanced(loc*wrong_plan + val*wrong_execution, reference, nodes)
    locate = balanced(loc, reference, nodes)
    clear = mean_at(ce(logits[4], torch.zeros_like(reference, dtype=torch.long)), nodes)
    loss = state + 2*locate + 5*align + .1*clear
    stats = dict(state=state.detach(), locate=locate.detach(), align=align.detach(), clear=clear.detach(),
                 nodes=nodes.sum(), reference_updates=(reference & nodes).sum(),
                 tp=(reference & plan & nodes).sum(), fp=(~reference & plan & nodes).sum(),
                 fn=(reference & ~plan & nodes).sum(), changed=(changed & nodes).sum(),
                 disagreements=disagree.sum(), wrong_plan=wrong_plan.sum(), wrong_execution=wrong_execution.sum(),
                 execution_correct=((logits[3].detach().argmax(-1)==target[..., 3]) & nodes).sum(),
                 cleared=((logits[4].detach().argmax(-1)==0) & nodes).sum())
    return loss, stats


def materialize(obj):
    """One batched device->host copy for nested scalar/vector metrics and audits."""
    tensors = []
    def collect(x):
        if isinstance(x, torch.Tensor):
            tensors.append(x.detach())
        elif isinstance(x, dict):
            for v in x.values(): collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x: collect(v)
    collect(obj)
    if not tensors: return obj
    host = torch.cat([x.reshape(-1).double() for x in tensors]).cpu()
    index = 0
    def rebuild(x):
        nonlocal index
        if isinstance(x, torch.Tensor):
            count = x.numel()
            value = host[index:index+count].to(dtype=x.dtype).reshape(x.shape).tolist()
            index += count
            return value
        if isinstance(x, dict): return {k: rebuild(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [rebuild(v) for v in x]
        return x
    return rebuild(obj)
