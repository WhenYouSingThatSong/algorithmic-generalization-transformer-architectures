"""Controlled identity/readout and fan-out/reconvergence experiments on modulo-23 graphs."""
import random

import torch

from design import P
from rebinding import render


def program(gid, depth, *, same=False, fanout=2):
    """Two balanced chains, four consumers, two pairwise sums, one final sum."""
    rng = random.Random(P.seed("mechanism-battery-v1", gid, depth))
    numbers = rng.sample(range(1, 23), 6)
    if same:
        numbers[1] = numbers[0]
    elif (numbers[1]-numbers[0])%23 in (1,22):
        numbers[1] = next(n for n in range(1,23) if n not in numbers and
                          (n-numbers[0])%23 not in (1,22))
    ops = [(f"k{i}", n) for i,n in enumerate(numbers)]
    u, v = "k0", "k1"
    for step in range(depth):
        k = f"k{rng.randrange(2,6)}"
        left, right = f"a{step}", f"b{step}"
        pair = [(left, (u,k)), (right, (v,k))]
        rng.shuffle(pair)
        ops.extend(pair)
        u, v = left, right
    order = list(range(4))
    rng.shuffle(order)
    exposed = set(order[:fanout])
    for j in range(4):
        ops.append((f"q{j}", (u if j in exposed else v, f"k{2+j}")))
    ops.extend([("m0", ("q0","q1")), ("m1", ("q2","q3")), ("sink", ("m0","m1"))])
    ids = {name:i for i,(name,_) in enumerate(ops)}
    names = [f"x_{j}" for j in rng.sample(range(128),len(ops)+2)]
    return ops, names, ids[u], ids[v], exposed


def encode(rt, branches):
    return torch.tensor([rt.tok.encode_tokens(b["tokens"]) for b in branches],device=rt.device)


def decoded(logits, positions):
    return torch.stack([z[:,positions].argmax(-1) for z in logits[:4]],-1)


def snapshots(rt, x, rounds, positions):
    h = rt.init(x)
    for t in range(1,max(rounds)+1):
        h, logits, _ = rt.forward(h)
        if t in rounds:
            yield t,h,decoded(logits,positions)


def source_labels(x, nodes, i):
    labels = x[nodes[i]["pos"],:4].tolist()
    labels[3] = nodes[i]["truth"]
    return labels


def identity_probe(rt, gid, depth, same, examples=None):
    ops,names,u,v,_ = program(gid,depth,same=same)
    branches = []
    controls = ("original","unrelated_rename","identity_rename","value_change")
    for condition in controls:
        spelling = names.copy()
        equations = ops.copy()
        if condition in ("identity_rename","unrelated_rename"):
            i,j = (u,v) if condition=="identity_rename" else (2,3)
            spelling[i],spelling[j] = spelling[j],spelling[i]
        if condition=="value_change":
            equations[0] = ("k0",(ops[0][1]+1)%23)
        branches.append(render(equations,spelling,False))
    original = branches[0]
    nodes = original["nodes"]
    # Append two previously unseen consumers, each with the same addend.
    extra = [("new_u",(ops[u][0],"k5")), ("new_v",(ops[v][0],"k5"))]
    extended = render(ops+extra,names,False)
    prefix = len(original["tokens"])
    if extended["tokens"][:prefix] != original["tokens"]:
        raise ValueError("appending consumers changed the original problem")
    gold = [n["truth"] for n in extended["nodes"][-2:]]
    positions = [nodes[i]["pos"] for i in (u,v)]
    read_pos = [n["pos"] for n in extended["nodes"][-2:]]
    x, xx = encode(rt,branches), encode(rt,[extended])
    D = max(n["depth"] for n in nodes)+1
    rows = []
    for t,h,reads in snapshots(rt,x,(D,D+8),positions):
        before=reads.tolist()
        wanted=[source_labels(x[i],b["nodes"],u) for i,b in enumerate(branches)]
        recipient_ok=(before[0][0]==wanted[0] and before[0][1]==source_labels(x[0],nodes,v))
        pair=rt.init(xx).repeat(len(controls),1,1)
        pair[:,:prefix]=h[0]
        pair[:,positions[0]]=h[:,positions[0]]
        traces=[[] for _ in controls]
        for _ in range(4):
            pair,logits,_=rt.forward(pair)
            read=logits[3][:,read_pos].argmax(-1).tolist()
            for trace,values in zip(traces,read):
                trace.append(values)
        qualified=[recipient_ok and r[0]==expected for r,expected in zip(before,wanted)]
        if examples is not None:
            examples.append(dict(round=t,u=positions[0],v=positions[1],tokens=extended["tokens"],
                                 recipient=before[0],reader_truth=gold,
                                 conditions={key:dict(donor=before[i][0],expected=wanted[i],
                                     eligible=qualified[i],reader_values=traces[i])
                                     for i,key in enumerate(controls)}))
        for i,condition in enumerate(controls):
            for step,(clean,actual) in enumerate(zip(traces[0],traces[i]),1):
                rows.append(dict(condition=condition,extra=t-D,step=step,
                    recipient_ok=int(recipient_ok),source_ok=int(before[i][0]==wanted[i]),
                    eligible=int(qualified[i]),common_eligible=int(all(qualified)),
                    u_name_correct=int(before[0][0][1]==wanted[0][1]),
                    u_value_correct=int(before[0][0][3]==wanted[0][3]),
                    u_read_as_v=int(before[0][0][1]==source_labels(x[0],nodes,v)[1]),
                    clean_pair_correct=int(clean==gold),actual_pair_correct=int(actual==gold),
                    clean_pair_spoiled=int(clean==gold and actual!=gold),
                    u_reader_specific_other=int(not same and clean[0]==gold[0] and actual[0]==gold[1]),
                    v_reader_specific_other=int(not same and clean[1]==gold[1] and actual[1]==gold[0]),
                    u_reader_value_plus_one=int(clean[0]==gold[0] and actual[0]==(gold[0]+1)%23),
                    attempts=1))
    return rows


def fanout_probe(rt,gid,depth,fanout,examples=None):
    ops,names,u,v,exposed=program(gid,depth,fanout=fanout)
    donor_ops=ops.copy()
    delta=1+P.seed("fanout-delta",gid,depth)%22
    donor_ops[0]=("k0",(ops[0][1]+delta)%23)
    branches=[render(eq,names,False) for eq in (ops,donor_ops)]
    nodes=branches[0]["nodes"]
    x=encode(rt,branches)
    p=nodes[u]["pos"]
    targets=[u] + list(range(len(nodes)-7,len(nodes)))
    coeff=[1]+[int(j in exposed) for j in range(4)]
    coeff.extend([sum(coeff[1:3]),sum(coeff[3:5]),fanout])
    roles=["source"]+["child"]*4+["merge"]*2+["sink"]
    for i,m in zip(targets,coeff):
        if branches[1]["nodes"][i]["truth"] != (nodes[i]["truth"]+m*delta)%23:
            raise ValueError("counterfactual truth mismatch")
    D=max(z["depth"] for z in nodes)+1
    records=[]
    for t,h,reads in snapshots(rt,x,(depth+1,D+4),[p]):
        rec_ok=reads[0,0].tolist()==source_labels(x[0],nodes,u)
        src_ok=reads[1,0].tolist()==source_labels(x[1],branches[1]["nodes"],u)
        base=dict(timing="early" if t<D else "late",recipient_ok=int(rec_ok),source_ok=int(src_ok))
        trace=[]
        if examples is not None:
            examples.append(dict(round=t,tokens=branches[0]["tokens"],target_positions=[nodes[i]["pos"] for i in targets],
                                 truth=[nodes[i]["truth"] for i in targets],delta=delta,coefficients=coeff,
                                 recipient=reads[0,0].tolist(),donor=reads[1,0].tolist(),values=trace))
        records.append(dict(**base,role="attempt",coefficient=-1,step=0,attempts=1,
                            accepted=int(rec_ok and src_ok),observed=0,clean_correct=0,harmed=0,specific=0))
        if not (rec_ok and src_ok):
            continue
        pair=h[:1].repeat(2,1,1)
        pair[1,p]=h[1,p]
        positions=[nodes[i]["pos"] for i in targets]
        for step in range(1,9):
            pair,logits,_=rt.forward(pair)
            clean,changed=logits[3][:,positions].argmax(-1).tolist()
            if examples is not None:
                trace.append(dict(clean=clean,changed=changed))
            for i,m,role,a,b in zip(targets,coeff,roles,clean,changed):
                y=nodes[i]["truth"]
                records.append(dict(**base,role=role,coefficient=m,step=step,attempts=0,accepted=0,
                    observed=1,clean_correct=int(a==y),harmed=int(a==y and b!=y),
                    specific=int(m>0 and a==y and b==(y+m*delta)%23)))
    return records
