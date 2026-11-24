"""Matched version manipulations and old/new consumers after verified propagation."""
import random
import torch

from design import P
from rebinding import render
from research_probes import encode, program, snapshots, source_labels


def bindings(gid, placement):
    rng = random.Random(P.seed("three-version-v1", gid))
    values = rng.sample(range(23), 6)
    delta = rng.choice([i for i in range(1,23)
                        if all((v+i)%23 not in values[:3] for v in values[:3])])
    names = [f"x_{i}" for i in rng.sample(range(128),32)]
    fillers = [(f"gap{i}",("k3","k5")) for i in range(6)]
    base = [(f"k{i}",v) for i,v in enumerate(values)]
    if placement not in ("tight","delayed"):
        raise ValueError(placement)
    for version in range(3):
        base.append(("slot",(f"k{version}","k3")))
        if placement == "delayed" and version == 2:
            base.extend(fillers)
        base.append((f"reader{version}",("slot","k4")))
    if placement == "tight":
        base.extend(fillers)
    slots = list(dict.fromkeys(name for name,_ in base))
    mapping = dict(zip(sorted(slots),names))
    conditions = ("base","first_version","second_version","current_version","unrelated")
    branches = {}
    for cond, leaf in zip(conditions,(None,0,1,2,5)):
        ops = base.copy()
        if leaf is not None:
            ops[leaf] = (ops[leaf][0],(values[leaf]+delta)%23)
        repeat = render(ops,[mapping[s] for s in slots],False)
        reserved = {r["name"] for r in repeat["nodes"]}
        unused = iter(n for n in names if n not in reserved)
        distinct = [r["name"] if r["version"]==1 else next(unused) for r in repeat["nodes"]]
        branches["unique",cond] = render(ops,distinct,True)
        branches["reused",cond] = repeat
    return branches,delta


def version_probe(rt,gid,placement):
    branches,delta = bindings(gid,placement)
    keys = list(branches)
    nodes = branches["unique","base"]["nodes"]
    readers = [i for i,r in enumerate(nodes) if r["slot"].startswith("reader")]
    gold = [nodes[i]["truth"] for i in readers]
    D = max(r["depth"] for r in nodes)+1
    positions = [nodes[i]["pos"] for i in readers]
    x = encode(rt,list(branches.values()))
    rows, traces = [], []
    for t,_,f in snapshots(rt,x,(D,D+4,D+16),positions):
        outputs = dict(zip(keys,f[...,3].tolist()))
        traces.append(dict(round=t,truth=gold,delta=delta,programs=[
            dict(naming=arm,condition=cond,tokens=branches[arm,cond]["tokens"],readers=outputs[arm,cond])
            for arm,cond in keys]))
        for arm,cond in keys:
            if cond == "base":
                continue
            index = {"first_version":0,"second_version":1,"current_version":2,"unrelated":None}[cond]
            for j,i in enumerate(readers):
                old = outputs[arm,"base"][j]
                result = outputs[arm,cond][j]
                y = gold[j]
                actual_truth = branches[arm,cond]["nodes"][i]["truth"]
                should_change = j==index
                if actual_truth != (y + (delta if should_change else 0))%23:
                    raise ValueError("counterfactual binding mismatch")
                unique_ok = outputs["unique","base"][j]==y and outputs["unique",cond][j]==actual_truth
                # Stronger than an accidental one-time match: both responses track
                # the older binding across its controlled change.
                stale_pair = index is not None and index < j
                tracks_stale = stale_pair and old==gold[index] and result==(gold[index]+delta)%23
                rows.append(dict(arm=arm,condition=cond,reader=j+1,extra=t-D,observed=1,
                    base_correct=int(old==y),edited_correct=int(result==actual_truth),
                    unchanged_numeric=int(0<=old<23 and old==result),
                    exact_response=int(0<=old<23 and result==(old+delta)%23),
                    numeric_base=int(0<=old<23),should_change=int(should_change),
                    intact_to_wrong=int(old==y and result!=actual_truth),
                    unique_pair_correct=int(unique_ok),stale_testable=int(stale_pair),
                    tracks_older_pair=int(tracks_stale),
                    tracks_older_with_unique_correct=int(tracks_stale and unique_ok),
                    stale_testable_with_unique_correct=int(stale_pair and unique_ok)))
    return rows,traces


def reader_probe(rt,gid,depth):
    ops,names,u,_,_ = program(gid,depth,fanout=4)
    original = render(ops,names,False)
    nodes = original["nodes"]
    q = next(i for i,n in enumerate(nodes) if n["slot"]=="q0")
    delta = 1 + P.seed("old-new-readers",gid,depth)%22
    ops_alt = ops.copy()
    ops_alt[0] = ("k0",(ops[0][1]+delta)%23)
    donor = render(ops_alt,names,False)
    extended = render(ops+[("fresh_reader",ops[q][1])],names,False)
    prefix = len(original["tokens"])
    if extended["tokens"][:prefix]!=original["tokens"] or extended["nodes"][-1]["parents"]!=nodes[q]["parents"]:
        raise ValueError("fresh reader must have the same operands and preserve the prefix")
    pos = [nodes[i]["pos"] for i in (u,q)]
    fresh = extended["nodes"][-1]["pos"]
    a,b = nodes[u]["truth"],nodes[q]["truth"]
    x,xx = encode(rt,[original,donor]),encode(rt,[extended])
    D = max(n["depth"] for n in nodes)+1
    rows, examples = [],[]
    for t,h,reads in snapshots(rt,x,(depth+1,D+4),pos):
        status = "accepted"
        if reads[0,0].tolist()!=source_labels(x[0],nodes,u):
            status = "recipient_rejected"
        elif reads[1,0].tolist()!=source_labels(x[1],donor["nodes"],u):
            status = "donor_rejected"
        meta = dict(timing="early" if t<D else "late")
        log = dict(round=t,tokens=extended["tokens"],positions=[*pos,fresh],truth=[a,b,b],
                   delta=delta,recipient=reads[0,0].tolist(),donor=reads[1,0].tolist(),before_append=[])
        examples.append(log)
        pair=h[:1].repeat(2,1,1)
        pair[1,pos[0]]=h[1,pos[0]]
        damage_round = append_round = None
        old_correct_at_append = None
        if status=="accepted":
            for step in range(1,9):
                pair,logits,_=rt.forward(pair)
                clean,changed=logits[3][:,pos].argmax(-1).tolist()
                log["before_append"].append([clean,changed])
                if damage_round is None and clean[1]==b and changed[1]==(b+delta)%23:
                    damage_round=step
                if damage_round is not None and clean[0]==changed[0]==a:
                    append_round=step
                    old_correct_at_append=changed[1]==b
                    break
            if damage_round is None:
                status="no_specific_damage"
            elif append_round is None:
                status="no_joint_source_readback_in_window"
        log.update(status=status,damage_round=damage_round,append_round=append_round)
        stratum=("already_correct" if old_correct_at_append else "still_wrong") if append_round else "unavailable"
        # The attempt list carries one denominator per timing, independent of follow-up rounds.
        rows.append(dict(**meta,status=status,old_at_append=stratum,step=0,attempts=1,
                         observed=0,paired_clean=0,both_correct=0,old_only=0,new_only=0,both_wrong=0,
                         paired_clean_upstream_correct=0,old_only_upstream_correct=0,new_only_upstream_correct=0,
                         append_changes_old_clean=0,append_changes_old_perturbed=0))
        if status!="accepted":
            continue
        extended_h=rt.init(xx).repeat(2,1,1)
        extended_h[:,:prefix]=pair
        log["after_append"]=[]
        for step in range(1,5):
            # Two original-length branches isolate the effect of introducing the new equation.
            pair,base_logits,_=rt.forward(pair)
            extended_h,logits,_=rt.forward(extended_h)
            unextended=base_logits[3][:,pos[1]].argmax(-1).tolist()
            clean,changed=logits[3][:,[*pos,fresh]].argmax(-1).tolist()
            log["after_append"].append(dict(clean=clean,changed=changed,old_without_append=unextended))
            eligible=clean[1:]==[b,b]
            old_ok,new_ok=(v==b for v in changed[1:])
            upstream_ok=eligible and clean[0]==changed[0]==a
            rows.append(dict(**meta,status=status,old_at_append=stratum,step=step,attempts=0,
                observed=1,paired_clean=int(eligible),both_correct=int(eligible and old_ok and new_ok),
                old_only=int(eligible and old_ok and not new_ok),new_only=int(eligible and new_ok and not old_ok),
                both_wrong=int(eligible and not old_ok and not new_ok),
                paired_clean_upstream_correct=int(upstream_ok),
                old_only_upstream_correct=int(upstream_ok and old_ok and not new_ok),
                new_only_upstream_correct=int(upstream_ok and new_ok and not old_ok),
                append_changes_old_clean=int(unextended[0]!=clean[1]),
                append_changes_old_perturbed=int(unextended[1]!=changed[1])))
    return rows,examples
