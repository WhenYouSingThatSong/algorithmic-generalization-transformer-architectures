"""Matched continuous-state probes. Clock t denotes the state AFTER forward t."""
import torch

from attribution_cases import exposure_case, patch_state
from original_c_recovery import family, H
from research_probes import encode, program, source_labels
from rebinding import render


def advance(rt, state, n):
    for _ in range(n):
        state, _, _ = rt.forward(state)
    return state


def rollout(rt, states, positions, rounds=H):
    rows = [[] for _ in states]
    state = torch.cat(states)
    for _ in range(rounds):
        state, logits, _ = rt.forward(state)
        for row, values in zip(rows, logits[3][:, positions].argmax(-1).tolist()):
            row.append(values)
    return rows


def wrong_literal(rt, x, nodes, u, wanted, t):
    # A repeated ancestor can contribute more than once. Solve its coefficient mod 23.
    coefficients = [0] * len(nodes)
    coefficients[u] = 1
    for i in range(u, -1, -1):
        for p in nodes[i]["parents"]:
            coefficients[p] = (coefficients[p] + coefficients[i]) % 23
    leaf = next((i for i,n in enumerate(nodes) if not n["parents"] and coefficients[i]), None)
    if leaf is None:
        return None
    pos = nodes[leaf]["pos"] - 2
    delta = (wanted - nodes[u]["truth"]) * pow(coefficients[leaf], -1, 23)
    donor = x[:1].clone()
    if int(donor[0,pos,3]) != nodes[leaf]["truth"]:
        raise ValueError("Unexpected literal position")
    donor[0,pos,3] = (nodes[leaf]["truth"] + delta) % 23
    return advance(rt, rt.init(donor), t)


def natural_probe(rt, old):
    graphs, _ = family(old["gid"], old["size"], old["depth"])
    nodes = graphs[0]["nodes"]
    if nodes != old["nodes"]:
        raise ValueError("Archived graph mismatch")
    x = encode(rt, graphs)
    selected = [e for e in old["natural"] if "replay_child" in e]
    result = dict(opportunities=old["natural_opportunities"], events=len(old["natural"]),
                  node_events=len({(e["t"],e["u"]) for e in old["natural"]}),
                  selected=len(selected), records=[])
    if not selected:
        return result
    pos = [n["pos"] for n in nodes]
    labels = [source_labels(x[0], nodes, i) for i in range(len(nodes))]
    hist, reads = [], []
    state = rt.init(x)
    for _ in range(max(e["t"] for e in selected)+1):
        state, _, _ = rt.forward(state)
        hist.append(state[:1].clone())
        reads.append(rt.decode(state[:1])[0,pos,:4].tolist())
    for e in selected:
        t,u,q = e["t"],e["u"],e["q"]
        wrong = labels[u][:3] + [e["wrong"]]
        z = dict(t=t,u=u,q=q,wrong=e["wrong"],expected_child=e["expected"],
                 gold=[labels[u][3],labels[q][3]],archived_specific=e["specific_rescue"])
        result["records"].append(z)
        current, prev = hist[t-1], hist[t-2]
        vals = reads[t-1]
        if (vals[u]!=wrong or reads[t-2][u]!=labels[u]
                or reads[t][q][3]!=e["observed_child"]):
            z["skip"]="archived_transition_not_reproduced"
            continue
        artificial = wrong_literal(rt,x,nodes,u,e["wrong"],t)
        if artificial is None or rt.decode(artificial)[0,pos[u],:4].tolist()!=wrong:
            z["skip"]="no_exact_same_clock_wrong_donor"
            continue
        # Construct an explicitly repaired background AT t. Record every historical source;
        # absence of any required source makes only that background ineligible.
        def repair(indices):
            h = current.clone()
            history = {}
            missing = []
            for i in indices:
                if vals[i] == labels[i]:
                    continue
                j = next((k for k in range(t-2,-1,-1) if reads[k][i]==labels[i]),None)
                if j is None:
                    missing.append(i)
                else:
                    h[:,pos[i]]=hist[j][:,pos[i]]
                    history[i]=j+1
            return h, history, missing
        backgrounds = {"observed": [], "parents_restored": nodes[u]["parents"],
                       "all_other_due_restored": [i for i,n in enumerate(nodes)
                                                  if i!=u and n["depth"]+1<=t]}
        z["parents"]=nodes[u]["parents"]
        z["backgrounds"]={}
        for name, indices in backgrounds.items():
            h,sources,missing=repair(indices)
            rec=dict(historical_rounds=sources,missing=missing)
            z["backgrounds"][name]=rec
            if missing:
                continue
            states=[]
            for src in (current,artificial,prev):
                arm=h.clone()
                arm[:,pos[u]]=src[:,pos[u]]
                states.append(arm)
            paths=rollout(rt,states,[pos[u],pos[q]])
            rec["paths"]=dict(zip(("natural_wrong","artificial_same_wrong","target_restored"),paths))
            rec["wrong_vector_distance"]=float((current[:,pos[u]]-artificial[:,pos[u]]).float().norm())
        z["paired_replay_agrees"]=z["backgrounds"]["observed"]["paths"]["natural_wrong"][0][1]==e["observed_child"]
    return result


def history_probe(rt, gid, depth):
    ops,names,u,_,_=program(gid,depth)
    graphs=[render(ops,names,False)]
    wrong_ops=ops.copy()
    wrong_ops[0]=(ops[0][0],(ops[0][1]+1)%23)
    graphs.append(render(wrong_ops,names,False))
    extended=render(ops+[("new_reader",(ops[u][0],"k5"))],names,False)
    base=graphs[0]
    if extended["tokens"][:len(base["tokens"])]!=base["tokens"]:
        raise ValueError("reader changed original prefix")
    x=encode(rt,graphs)
    xx=encode(rt,[extended])
    p=base["nodes"][u]["pos"]
    q=extended["nodes"][-1]["pos"]
    T=1+max(n["depth"] for n in base["nodes"])+2
    wanted=source_labels(x[0],base["nodes"],u)
    wrong=source_labels(x[1],graphs[1]["nodes"],u)
    state=rt.init(x)
    blank=None
    blank_round=None
    seq=[]
    for t in range(1,T):
        state,_,_=rt.forward(state)
        value=rt.decode(state)[0,p,:4].tolist()
        seq.append(value)
        if value==wanted[:3]+[24] and blank is None:
            blank=state[0,p].clone()
            blank_round=t
    # Produce four donor states at the SAME absolute round. 'first' is held
    # empty after each intervening forward until release.
    first=rt.init(x[:1])
    prior=[]
    for t in range(1,T):
        first,_,_=rt.forward(first)
        if blank is not None and t>=blank_round:
            first[0,p]=blank
        prior.append(rt.decode(first)[0,p,:4].tolist())
    input_states=torch.cat((state[:1],first,state[:1],state[1:]))
    input_states[2,p]=state[1,p]
    pre=rt.decode(input_states)[:,p,:4].tolist()
    donors,_,_=rt.forward(input_states)
    current=rt.decode(donors)[:,p,:4].tolist()
    ok=[seq[-2:]==[wanted,wanted],blank is not None and all(v==wanted[:3]+[24] for v in prior),
        seq[-1]==wanted and pre[2]==wrong,pre[3]==wrong]
    ok=[flag and obs==(wrong if i==3 else wanted) for i,(flag,obs) in enumerate(zip(ok,current))]
    recipient=rt.init(xx)
    recipient[:,:x.shape[1]]=donors[:1]
    states=[]
    for donor in donors:
        h=recipient.clone()
        h[0,p]=donor[p]
        states.append(h)
    traces=rollout(rt,states,[p,q],4)
    return dict(gid=gid,depth=depth,t=T,u=u,reader_pos=q,reader_gold=extended["nodes"][-1]["truth"],
                wrong_reader=(extended["nodes"][-1]["truth"]+1)%23,prior=seq,
                arms=[dict(name=name,eligible=eligible,pre=v,at_t=cur,
                           distance=float((donors[i,p]-donors[0,p]).float().norm()),trace=trace)
                      for i,(name,eligible,v,cur,trace) in enumerate(zip(
                          ("stable","first_after_empty","recovered","wrong_control"),ok,pre,current,traces))])


def fanout_probe(rt,gid,depth):
    records=[]
    for fanout in (1,2,4):
        ops,names,u,_,_=program(gid,depth,fanout=fanout)
        edit=ops.copy()
        edit[0]=(ops[0][0],(ops[0][1]+1)%23)
        base,donor=[render(v,names,False) for v in (ops,edit)]
        nodes=base["nodes"]
        x=encode(rt,[base,donor])
        positions=[n["pos"] for n in nodes]
        children=[i for i,n in enumerate(nodes) if u in n["parents"]]
        descendants={u}
        for i,n in enumerate(nodes):
            if descendants.intersection(n["parents"]):
                descendants.add(i)
        gold=[n["truth"] for n in nodes]
        alt=[n["truth"] for n in donor["nodes"]]
        due=nodes[u]["depth"]+1
        times={"before":due-1,"just_correct":due,"children_correct":due+1,
               "complete":max(n["depth"] for n in nodes)+5}
        h=rt.init(x)
        snapshots={}
        for t in range(1,max(times.values())+1):
            h,_,_=rt.forward(h)
            snapshots[t]=h
        for phase,t in times.items():
            clean=snapshots[t][:1]
            state=clean.clone()
            source=snapshots[max(t,due)][1:]
            before=rt.decode(clean)[0,positions,:4].tolist()
            p=positions[u]
            patch_state(rt,state,source,[p])
            actual=rt.decode(state)[0,p,:4].tolist()
            want=source_labels(x[0],nodes,u)
            got=source_labels(x[1],donor["nodes"],u)
            initial=want[:3]+[24] if phase=="before" else want
            eligible=before[u]==initial and actual==got and actual!=before[u]
            if phase in ("before","just_correct"):
                eligible &= all(before[j][3]==24 for j in children)
            if phase=="just_correct":
                eligible &= all(rt.decode(snapshots[j][:1,p:p+1])[0,0,3].item()!=gold[u] for j in range(1,t))
            if phase=="before":
                eligible &= all(before[j][3]==gold[j] for j in nodes[u]["parents"])
            if phase=="children_correct":
                eligible &= all(before[j][3]==gold[j] for j in children)
            if phase=="complete":
                eligible &= all(z[3]==y for z,y in zip(before,gold))
            ref,changed=rollout(rt,[clean,state],positions)
            records.append(dict(fanout=fanout,phase=phase,t=t,eligible=eligible,u=u,
                                children=children,descendants=sorted(descendants-{u}),
                                output=len(nodes)-1,before=before[u],actual=actual,
                                gold=gold,alt=alt,clean=ref,changed=changed))
    return dict(gid=gid,depth=depth,records=records)


def mark_probe(rt,gid,depth):
    c=exposure_case(gid,depth)
    x=encode(rt,[c["base"],c["donor"]])
    nodes=c["base"]["nodes"]
    u,r=c["u"],c["r"]
    pos=[n["pos"] for n in nodes]
    times={"new":nodes[u]["depth"],"just_correct":nodes[u]["depth"]+1,"complete":c["D"]+4}
    h=rt.init(x)
    cache={}
    for t in range(1,max(times.values())+1):
        h,_,_=rt.forward(h)
        cache[t]=h
    rows=[]
    for phase,t in times.items():
        clean=cache[t][:1]
        before=rt.decode(clean)[0,pos,:4].tolist()
        gold=c["gold"]
        stage_ok=(all(z[3]==y for z,y in zip(before,gold)) if phase=="complete" else
                  before[u][3]==(24 if phase=="new" else gold[u]))
        # Mixed scene: reopen an independent, previously filled internal node.
        v=c["unrelated"]
        for name,edits in {"clean":[],"wrong_target":[u],"wrong_parent":[r],
                           "wrong_and_blank":[u,v]}.items():
            arm=clean.clone()
            valid=stage_ok
            editlog=[]
            for i in edits:
                clearing=i==v
                at=nodes[v]["depth"] if clearing else max(t,nodes[i]["depth"]+1)
                src=cache[at][0 if clearing else 1]
                want=24 if clearing else c["alt"][i]
                arm[0,pos[i]]=src[pos[i]]
                actual=rt.decode(arm)[0,pos[i],:4].tolist()
                valid &= (actual==x[0,pos[i],:3].tolist()+[want] and before[i][3]!=want)
                editlog.append([i,at,before[i],actual])
            inp=rt.decode(arm)[0,pos,:4].tolist()
            _,logits,mid=rt.forward(arm,middle=True)
            scores=mid[0,pos].float().softmax(-1)[:,1].tolist()
            out=logits[3][0,pos].argmax(-1).tolist()
            start=len(rows)
            for i,n in enumerate(nodes):
                if not n["parents"]:
                    continue
                old=inp[i][3]
                parents=[inp[j][3] for j in n["parents"]]
                expected=sum(parents)%23 if all(y<23 for y in parents) else None
                category=("blocked" if expected is None else "new" if old==24 else
                          "other_symbol" if old>=23 else "direct_conflict" if old!=expected else
                          "healthy" if old==gold[i] else "consistent_wrong")
                rows.append(dict(phase=phase,condition=name,t=t,node=i,eligible=valid,stage_ok=stage_ok,
                                 category=category,target=i==u,score=scores[i],before=old,after=out[i],
                                 expected=expected,gold=gold[i],parents=parents,edits=editlog if i==u else []))
            categories={z["category"] for z in rows[start:]}
            for row in rows[start:]:
                row["actual_mixed"]={"new","direct_conflict","healthy"}.issubset(categories)
    return dict(gid=gid,depth=depth,rows=rows)
