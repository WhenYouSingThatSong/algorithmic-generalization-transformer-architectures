"""Full continuous BPTT; unfiltered recipients; source-validity checks only."""
import torch
from process_loss import process_loss


def make_donors(rt, x, y, events, chunk):
    """Current-weight eval sources, batched by fixed age. No CPU answer checks."""
    result = {}
    training = rt.model.training
    try:
        rt.model.eval()
        with torch.no_grad():
            for age in sorted({e['donor_round'] for e in events}):
                group = [e for e in events if e['donor_round']==age and e['edits'] is not None]
                for off in range(0, len(group), chunk):
                    ee = group[off:off+chunk]
                    bi = torch.tensor([e['bi'] for e in ee], device=x.device)
                    pos = torch.tensor([e['pos'] for e in ee], device=x.device)
                    xx = x.index_select(0, bi).clone()
                    edits = [(i, p, d) for i, e in enumerate(ee) for p, d in e['edits'].items()]
                    ii, pp, dd = (torch.tensor(z, device=x.device) for z in zip(*edits))
                    xx[ii, pp, 3] = (xx[ii, pp, 3]+dd).remainder(23)
                    h = rt.init(xx)
                    for _ in range(age): h, _, _ = rt.forward(h)
                    vectors = h[torch.arange(len(ee), device=x.device), pos].detach()
                    read = torch.stack([head(vectors).argmax(-1) for head in rt.model.output_layers[:4]], -1)
                    wanted = y[bi, pos, :4].clone()
                    wanted[:, 3] = (wanted[:, 3]+torch.tensor([e['delta'] for e in ee], device=x.device)).remainder(23)
                    passed = (read==wanted).all(-1)
                    for i, e in enumerate(ee):
                        result[e['bi'], e['pos']] = (vectors[i], read[i], wanted[i], passed[i])
    finally:
        rt.model.train(training)
    return result


def recurrent_loss(rt, x, y, gs, topology, events, settings, offset=0, donor_batch=32):
    local = [dict(e, bi=e['bi']-offset) for e in events if offset <= e['bi'] < offset+len(gs)]
    before_calls = rt.calls
    donors = make_donors(rt, x, y, local, donor_batch)
    donor_calls = rt.calls-before_calls
    h = rt.init(x)
    previous = x[..., :4].detach()
    total = x.new_zeros((), dtype=torch.float32)
    valid = y[..., 0] != rt.tok.syntax_tok2idx[rt.tok.PAD]
    stats, audit = [], []
    for t in range(settings['rounds']):
        fired = [e for e in local if e['round']==t]
        if fired:
            # Batch write, retaining gradients of rejected/unmodified recipient rows.
            ready = [e for e in fired if (e['bi'], e['pos']) in donors]
            for e in fired:
                if (e['bi'], e['pos']) not in donors:
                    audit.append(dict(**e, status='no_invertible_source', accepted=False))
            if ready:
                bi = torch.tensor([e['bi'] for e in ready], device=x.device)
                pos = torch.tensor([e['pos'] for e in ready], device=x.device)
                vectors, read, wanted, passed = (torch.stack(z) for z in zip(*(donors[e['bi'], e['pos']] for e in ready)))
                old = h[bi, pos]
                before = previous[bi, pos].clone()
                node_correct = (previous[topology.bi, topology.pos]==y[topology.bi, topology.pos, :4]).all(-1)
                graph_first4 = node_correct.all(-1)
                graph_value = (previous[..., 3].gather(1, topology.pos)==y[..., 3].gather(1, topology.pos)).all(-1)
                input_before = previous
                distance = (vectors-old.detach()).norm(dim=-1)/old.detach().norm(dim=-1).clamp_min(1e-12)
                h = h.clone()
                h[bi, pos] = torch.where(passed[:, None], vectors, old)
                previous = previous.clone()
                previous[bi, pos] = torch.where(passed[:, None], read, before)
                for i, e in enumerate(ready):
                    b, p = e['bi'], e['pos']
                    children = [q for q in gs[b]['nodes'] if p in gs[b]['parents'][q]]
                    correct_children = (input_before[b, children]==y[b, children, :4]).all(-1).sum()
                    audit.append(dict(**e, status='source_checked', accepted=passed[i],
                                      before=before[i], source_readout=read[i], intended=wanted[i],
                                      after=previous[b, p], before_first4_correct=(before[i]==y[b, p, :4]).all(),
                                      before_value_correct=before[i, 3]==y[b, p, 3],
                                      graph_first4_before=graph_first4[b], graph_value_before=graph_value[b],
                                      direct_children=len(children), correct_children_at_input=correct_children,
                                      actual_value_changed=passed[i] & (before[i, 3]!=read[i, 3]),
                                      relative_vector_change=distance[i]))
        # Only current INPUT values determine this synchronous target.
        target, reference = topology.local(previous, y)
        h, logits, mid = rt.forward(h, middle=True)
        loss, metrics = process_loss(logits, mid, previous, target, reference, topology.nodes, valid)
        total = total+loss/settings['rounds']
        stats.append(dict(round=t, **metrics))
        previous = torch.stack([z.argmax(-1) for z in logits[:4]], -1).detach()
    return total, stats, audit, donor_calls
