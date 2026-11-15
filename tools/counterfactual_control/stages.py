"""Time conventions and matched stage sampling (no torch import)."""


def schedule(early, complete, late_gap, budget):
    return dict(early=early if early<=budget else None,
                complete=complete,
                late=complete+late_gap if complete is not None and complete+late_gap<=budget else None)


def collect(lit, torch, inputs, candidate, positions, truth4, enabled=False,
            completion_budget=64, late_gap=16):
    """Source trajectories advance freely; snapshots align all donors in time."""
    early=candidate['state_round']
    budget=completion_budget if enabled else early
    hidden=lit.model.token_embedder(torch.cat(inputs,dim=0))
    saved={};complete=None
    for t in range(1,budget+1):
        hidden,logits=lit.model(hidden,True)
        pred=torch.stack(lit.generate_factor_decodes(logits,temperature=0.0),-1)
        if enabled and complete is None:
            vals=pred[0,positions,3].cpu().tolist()
            if vals==[r[3] for r in truth4]:
                complete=t
        times=schedule(early,complete,late_gap,budget) if enabled else dict(early=early)
        for stage,when in times.items():
            if when==t:
                saved[stage]=(hidden.detach().clone(),pred.detach().clone(),t)
    times=schedule(early,complete,late_gap,budget) if enabled else dict(early=early)
    return [(name,saved.get(name),when) for name,when in times.items()],complete


def strict_sources(record):
    flags=[v for k,v in record['source_qualification'].items() if k.endswith('_target_first4')]
    return bool(flags) and all(flags)


def common_children(records, required, strict=False):
    """Same target AND same q intersection, not just equal aggregate counts."""
    from collections import defaultdict
    groups=defaultdict(dict)
    for r in records:
        key=(r['weight'],r['graph_id'],r['target_pos'],r['data_sha256'])
        if r['status']=='eligible' and (not strict or strict_sources(r)):
            stage=r.get('stage','early')
            if stage in groups[key]:
                raise ValueError('duplicate stage for source target')
            groups[key][stage]=set(r['eligible_children'])
    return {key:sorted(set.intersection(*(items[s] for s in required)))
            for key,items in groups.items() if all(s in items for s in required)}
