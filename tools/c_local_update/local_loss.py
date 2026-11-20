"""Local UPDATE supervision with three injection opportunities inside D+1 forwards."""
from collections import defaultdict

import torch
from multitime_core import make_donors
from process_loss import process_loss


def recurrent_loss(rt, x, y, graphs, topo, events, settings, offset, donor_batch):
    local = [dict(e, bi=e["bi"]-offset) for e in events if offset <= e["bi"] < offset+len(x)]
    donors = make_donors(rt, x, y, local, donor_batch)
    by_round = defaultdict(list)
    audit = []
    for e in local:
        if (e["bi"],e["pos"]) in donors:
            by_round[e["round"]].append(e)
        else:
            audit.append(dict(**e, accepted=False, status="no_invertible_source"))
    h = rt.init(x)
    previous = x[..., :4]
    valid = y[...,0] != rt.tok.syntax_tok2idx[rt.tok.PAD]
    total = x.new_zeros((), dtype=torch.float32)
    for t in range(settings["rounds"]):
        fired = by_round[t]
        if fired:
            bi = torch.tensor([e["bi"] for e in fired], device=x.device)
            pos = torch.tensor([e["pos"] for e in fired], device=x.device)
            vec, read, wanted, source_ok = (torch.stack(z) for z in zip(
                *(donors[e["bi"],e["pos"]] for e in fired)))
            before = previous[bi,pos]
            original = y[bi,pos,:4]
            graph_done = (previous[topo.bi,topo.pos,3]==y[topo.bi,topo.pos,3]).all(-1)[bi]
            eligible = graph_done if settings["mode"]=="completed" else torch.ones_like(source_ok)
            write = eligible & source_ok
            h = h.clone()
            h[bi,pos] = torch.where(write[:,None], vec, h[bi,pos])
            previous = previous.clone()
            previous[bi,pos] = torch.where(write[:,None], read, before)
            for i,e in enumerate(fired):
                audit.append(dict(**e, status="checked", source_ok=source_ok[i], eligible=eligible[i],
                                  accepted=write[i], graph_value_before=graph_done[i], before=before[i],
                                  before_value_correct=before[i,3]==original[i,3], source_readout=read[i],
                                  intended=wanted[i], actual_value_changed=write[i] & (read[i,3]!=before[i,3])))
        target, reference = topo.local(previous,y)
        h, logits, mid = rt.forward(h,middle=True)
        loss,_ = process_loss(logits,mid,previous,target,reference,topo.nodes,valid)
        total = total + loss/settings["rounds"]
        previous = torch.stack([a.argmax(-1) for a in logits[:4]],-1).detach()
    return total, audit
