"""Shared graph, corruption and information-source protocol for A, B and C."""
from contextlib import contextmanager
import random

import torch
from design import P
from rebinding import render
from research_probes import program, encode, source_labels


def exposure_case(gid, depth):
    ops,names,u,v,exposed=program(gid,depth,fanout=1)
    edited=list(ops)
    delta=1+P.seed("source",gid,depth)%22
    edited[0]=("k0",(ops[0][1]+delta)%23)
    base,donor=[render(e,names,False) for e in (ops,edited)]
    nodes=base["nodes"]
    r=next(i for i in nodes[u]["parents"] if nodes[i]["depth"]>0)
    unrelated=next(i for i in nodes[v]["parents"] if nodes[i]["depth"]>0)
    q=next(i for i,n in enumerate(nodes) if n["slot"]==f"q{next(iter(exposed))}")
    ancestors=[n for n in nodes[r]["parents"] if nodes[n]["depth"]>0]
    closure={r}
    for i,n in enumerate(nodes):
        if closure.intersection(n["parents"]):
            closure.add(i)
    gold=[n["truth"] for n in nodes]
    alt=[n["truth"] for n in donor["nodes"]]
    hybrid=[alt[i] if i in closure else y for i,y in enumerate(gold)]
    broken=[i for i,n in enumerate(nodes) if n["parents"] and
            hybrid[i]!=sum(hybrid[j] for j in n["parents"])%23]
    if broken!=[r] or not {u,q}.issubset(closure):
        raise ValueError("inconsistent cut construction")
    return dict(base=base,donor=donor,gold=gold,alt=alt,u=u,r=r,q=q,
                unrelated=unrelated,ancestor=ancestors[0],closure=sorted(closure),
                D=1+max(n["depth"] for n in nodes),delta=delta)


@contextmanager
def hide_key(rt, pos):
    """Temporarily block attention to this key for other queries in every layer."""
    def mask_key(module, args, kwargs):
        length=args[0].shape[1]
        mask=torch.ones(length,length,device=args[0].device,dtype=torch.bool)
        mask[:,pos]=False
        mask[pos,pos]=True
        if kwargs.get("mask") is not None:
            mask=kwargs["mask"].bool() & mask
        return args,dict(kwargs,mask=mask)
    hook=rt.model.transformer.register_forward_pre_hook(mask_key,with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()


def read(rt,h,positions):
    return rt.decode(h)[:,positions,:4].tolist()


def patch_state(rt, state, donor, positions):
    if rt.continuous:
        state[:,positions]=donor[:,positions]
    else:
        state[:,positions,3]=donor[:,positions,3]


def run_graph(rt,gid,depth,stage):
    c=exposure_case(gid,depth)
    nodes=c["base"]["nodes"]
    positions=[n["pos"] for n in nodes]
    u,r,q=c["u"],c["r"],c["q"]
    focus=[r,u,q,len(nodes)-1]
    times={"pending":nodes[u]["depth"],"just_solved":nodes[u]["depth"]+1,"later":c["D"]+4}
    x=encode(rt,[c["base"],c["donor"]])
    h=rt.init(x)
    snapshots={}
    clean=[]
    prior_correct=None
    natural=[]
    for t in range(1,c["D"]+17):
        h,_,_=rt.forward(h)
        clean.append(read(rt,h[:1],positions)[0])
        value=clean[-1][u][3]
        if value==c["gold"][u]:
            prior_correct=(t,h[:1,positions[u]].clone())
        elif (stage=="four_conditions" and value<23 and prior_correct and
              prior_correct[0]==t-1 and t>=nodes[u]["depth"]+1):
            branches=h[:1].repeat(2,1,1)
            if rt.continuous:
                branches[1,positions[u]]=prior_correct[1][0]
            else:
                branches[1,positions[u],3]=prior_correct[1][0,3]
            following,_,_=rt.forward(branches)
            parent_ids=nodes[q]["parents"]
            clean_others=all(clean[-1][p][3]==c["gold"][p] for p in parent_ids if p!=u)
            natural.append(dict(round=t,wrong=value,correct=c["gold"][u],others_correct=clean_others,
                expected_child=(c["gold"][q]+value-c["gold"][u])%23,
                child_observed_replayed=[a[0][3] for a in read(rt,following,[positions[q]])]))
        if t in times.values():
            snapshots[t]=h.clone()
    result=dict(nodes=nodes,gold=c["gold"],alt=c["alt"],positions=positions,focus=focus,
                structural_labels=x[0,positions,:3].tolist(),
                base_tokens=c["base"]["tokens"],D=c["D"],delta=c["delta"],clean=clean,
                natural_replay=natural,events=[])
    for timing,t in times.items():
        states=snapshots[t]
        before=read(rt,states,positions)
        def valid(arm,i):
            expected=x[arm,positions[i],:4].tolist()
            expected[3]=(c["alt"] if arm else c["gold"])[i]
            return before[arm][i]==expected
        pending=before[0][u][3]==24 and all(valid(0,i) for i in nodes[u]["parents"])
        start_ok=pending if timing=="pending" else valid(0,u)
        conditions={"clean":[],"target":[u]}
        if stage=="sources":
            conditions.update(parent=[r],parent_target=[r,u],ancestor=[c["ancestor"]],
                              constant=[],unrelated=[c["unrelated"]],coherent=c["closure"])
            # A changed but irrelevant state is needed for a real unrelated perturbation.
            # Use the unrelated node's own +delta donor, obtained by editing k1.
            dx=x[:1].clone()
            leafpos=positions[1]-2
            dx[0,leafpos,3]=(dx[0,leafpos,3]+c["delta"])%23
            unrelated_h=rt.init(dx)
            for _ in range(t):
                unrelated_h,_,_=rt.forward(unrelated_h)
        for cond,indices in conditions.items():
            for block in (("none","parent","unrelated") if stage=="pathway" else ("none",)):
                base=states[:1].clone()
                donor=states[1:]
                donor_ok=all(valid(1,i) for i in indices)
                if timing=="pending" and indices==[u]:
                    donor=snapshots[t+1][1:]
                    donor_ok=read(rt,donor,[positions[u]])[0][0]==source_labels(x[1],c["donor"]["nodes"],u)
                if cond=="unrelated":
                    donor=unrelated_h
                    ident=c["unrelated"]
                    expected=x[0,positions[ident],:4].tolist()
                    expected[3]=(c["gold"][ident]+c["delta"])%23
                    donor_ok=read(rt,donor,[positions[ident]])[0][0]==expected
                pos=[positions[i] for i in indices]
                if cond=="constant":
                    pos=[positions[0]-2]
                    donor_ok=read(rt,donor,pos)[0][0]==x[1,pos[0],:4].tolist()
                eligible=start_ok and donor_ok
                if cond=="coherent":
                    # Admit the stronger claim only when the actual starting numerical
                    # graph is correct, and every transplanted node has the intended value.
                    eligible=eligible and [z[3] for z in before[0]]==c["gold"]
                event=dict(timing=timing,condition=cond,block=block,start=t,eligible=eligible,
                           donor_round=t+int(timing=="pending" and indices==[u]),
                           start_ok=start_ok,donor_ok=donor_ok,target_input=before[0][u],
                           patch_positions=pos,trajectory=[])
                result["events"].append(event)
                if not eligible:
                    continue
                patch_state(rt,base,donor,pos)
                event["effective_input"]=read(rt,base,[positions[i] for i in focus])[0]
                if cond=="coherent":
                    event["effective_values"]=[z[3] for z in read(rt,base,positions)[0]]
                for step in range(8):
                    if step==0 and block!="none":
                        key=r if block=="parent" else c["unrelated"]
                        with hide_key(rt,positions[key]):
                            base,_,_=rt.forward(base)
                    else:
                        base,_,_=rt.forward(base)
                    event["trajectory"].append(read(rt,base,[positions[i] for i in focus])[0])
        if stage=="sources" and timing=="later":
            # Independent reference: solve both versions from their original empty state.
            fresh=rt.init(x)
            result["from_blank"]=[]
            for _ in range(c["D"]+8):
                fresh,_,_=rt.forward(fresh)
                result["from_blank"].append(read(rt,fresh,[positions[i] for i in focus]))
    return result


def workspace_program(gid, rewrites, live, gap, spaced=True):
    """Fixed arithmetic/positions count. Rename bindings; vary retained saved results."""
    rng=random.Random(P.seed("workspace-factorial-v1",gid))
    roots=rng.sample(range(1,23),6)
    ops=[(f"k{i}",v) for i,v in enumerate(roots)]
    # Eight rounds: definitions in cycling registers, single consumers, and
    # extra saved values read either immediately or at the very end.
    late=[]
    for j in range(8):
        register=j%(8//rewrites) if spaced else j//rewrites
        dest=f"r{register}"
        oper=(f"k{j%3}",f"k{3+j%3}")
        ops.append((dest,oper))
        fillers=[(f"f{j}_{k}",("k1","k5")) for k in range(2)]
        if gap:
            ops.extend(fillers)
        ops.append((f"s{j}",(dest,"k0")))
        if not gap:
            ops.extend(fillers)
        consumer=(f"out{j}",(f"s{j}","k2"))
        if j<live:
            late.append(consumer)
        else:
            ops.append(consumer)
    ops.extend(late)
    names=[f"x_{n}" for n in rng.sample(range(128),len(ops))]
    # Match the unique-name program across all reuse rates, and give each
    # repeated slot the unique arm's spelling at its first definition.
    one_name={}
    for i,(slot,_) in enumerate(ops):
        one_name.setdefault(slot,names[i])
    repeat=render(ops,list(one_name.values()),False)
    unique=render(ops,names,True)
    rows=repeat["nodes"]
    last=list(range(len(rows)))
    for i,n in enumerate(rows):
        for p in n["parents"]:
            last[p]=i
    peak=max(sum(i<=t<last[i] for i in range(len(rows))) for t in range(len(rows)))
    return unique,repeat,peak
