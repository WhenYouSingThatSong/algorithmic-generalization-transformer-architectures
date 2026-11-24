"""Matched dense, whole-graph stop and per-definition stop on existing C weights."""
import argparse
import gzip
import json
from pathlib import Path
from typing import NamedTuple

import torch
from design import EXTENSION, EXTRA, P, PROFILES, SCALES, TOOLS, make_case
from hard_eval import MODELS, PLAN, RECEIPT, ROOT, probe_score
from runtime import Runtime

POLICIES = ("dense", "graph_stop", "node_stop")
OUT = ROOT / "tmp/analysis/c-stop-pilot-v1"
CELLS = [(w,d,s) for w,d in SCALES for s in PROFILES] + [EXTENSION]


class State(NamedTuple):
    hidden: torch.Tensor
    read: torch.Tensor       # last committed readout, result positions, first four fields
    locked: torch.Tensor     # position mask effective on the NEXT forward
    writes: torch.Tensor     # cumulative committed token positions


def initial(rt, x, n):
    batch, length = x.shape[:2]
    device = x.device
    return State(rt.init(x), torch.full((batch,n,4),-1,device=device),
                 torch.zeros((batch,length),device=device,dtype=torch.bool),
                 torch.zeros(batch,device=device,dtype=torch.long))


def advance(rt, state, pos, policy):
    candidate, logits, _ = rt.forward(state.hidden)
    fresh = torch.stack([z[:,pos].argmax(-1) for z in logits[:4]],-1)
    read = torch.where(state.locked[:,pos,None],state.read,fresh)
    hidden = torch.where(state.locked[:,:,None],state.hidden,candidate)
    stable = ((read == state.read).all(-1) & (read[...,3]<23) &
              (read[...,0]==rt.tok.syntax_tok2idx["VARIABLE"]) &
              (read[...,1]<rt.tok.n_vars))
    locked = state.locked | ((policy==1) & stable.all(-1))[:,None]
    locked[:,pos] |= (policy==2)[:,None] & stable
    return State(hidden,read,locked,state.writes+(~state.locked).sum(-1))


def row_counts(state, pos, label, D, t, dense_history):
    value = state.read[...,3]==label[:,3]
    four = (state.read==label).all(-1)
    records = []
    for i, name in enumerate(POLICIES):
        dense = value[0]
        writes = int(state.writes[i])
        length = state.hidden.shape[1]
        low, remainder = divmod(writes, length)
        high = low + int(remainder != 0)
        # History starts at the output of the first forward. Every policy
        # commits at least one full round before it can stop.
        lo = dense_history[low-1] == label
        hi = dense_history[high-1] == label
        records.append(dict(policy=name,extra=t-D,nodes=len(pos),value=int(value[i].sum()),
            first4=int(four[i].sum()),variable=int((state.read[i,:,1]==label[:,1]).sum()),
            graph_correct=int(value[i].all()),gain_over_dense=int((value[i]&~dense).sum()),
            loss_to_dense=int((~value[i]&dense).sum()),
            locked_nodes=int(state.locked[i,pos].sum()),
            locked_wrong=int((state.locked[i,pos]&~value[i]).sum()),
            committed_token_rounds=writes,dense_token_rounds=t*length,
            dense_budget_floor_round=low,dense_budget_ceil_round=high,
            dense_floor_value=int(lo[:,3].sum()),dense_floor_first4=int(lo.all(-1).sum()),
            dense_ceil_value=int(hi[:,3].sum()),dense_ceil_first4=int(hi.all(-1).sum())))
    return records


def intervention(rt, state, donor, donor_read, case, pos, labels, t, delta):
    u,q = case["u"],case["q"]
    a,b = int(labels[u,3]),int(labels[q,3])
    changed_label = labels[u].clone()
    changed_label[3] = (a+delta)%23
    donor_ok = bool((donor_read==changed_label).all())
    pair = State(*(v.repeat((2,)+(1,)*(v.ndim-1)) for v in state))
    policies = torch.arange(3,device=pos.device).repeat(2)
    rows = []
    for i, policy in enumerate(POLICIES):
        status = "accepted"
        if bool(state.locked[i].all()):
            status = "graph_already_stopped"
        elif not bool((state.read[i,u]==labels[u]).all()):
            status = "recipient_rejected"
        elif not donor_ok:
            status = "donor_rejected"
        else:
            pair.hidden[i+3,pos[u]] = donor
            # Store what the recipient really reads after the forced replacement.
            # Keep its prior decision to hold that position on the following round.
            pair.read[i+3,u] = donor_read
        rows.append(dict(policy=policy,start=t,status=status,locked_at_injection=bool(state.locked[i,pos[u]]),
                         child_locked_at_injection=bool(state.locked[i,pos[q]]),
                         child_correct_before=int(state.read[i,q,3])==b))
    paths=[[] for _ in range(6)]
    for _ in range(8):
        pair=advance(rt,pair,pos,policies)
        for path, values in zip(paths,pair.read[:,[u,q],3].tolist()):
            path.append(values)
    for i,row in enumerate(rows):
        if row["status"]=="accepted":
            control, changed = paths[i],paths[i+3]
            row.update(control=control,changed=changed,truth=[a,b],delta=delta,
                       target_correct_last=changed[-1][0]==a,**probe_score(control,changed,a,b,delta))
    return rows


def consumer_counts(previous_locks, state, parents, pos, labels, extra):
    # Membership depends on the policy. Compare dense on its exact same
    # positions; distinguish still-updating readers from held readers.
    used = parents >= 0
    held_input = (previous_locks[:,pos][:,parents.clamp_min(0)] & used).any(-1)
    correct = state.read[...,3] == labels[:,3]
    rows = []
    for i, policy in enumerate(POLICIES):
        for reader_held in (False, True):
            mask = held_input[i] & (previous_locks[i,pos] == reader_held)
            rows.append(dict(policy=policy,extra=extra,reader_held=reader_held,
                observed=int(mask.sum()),correct=int((correct[i]&mask).sum()),
                dense_correct_same_nodes=int((correct[0]&mask).sum()),
                gain_over_dense=int((correct[i]&~correct[0]&mask).sum()),
                loss_to_dense=int((~correct[i]&correct[0]&mask).sum())))
    return rows


def run_case(rt,c,probe):
    results=[]
    for arm,branch in c["arms"].items():
        nodes=branch["nodes"]
        pos=torch.tensor([n["pos"] for n in nodes],device=rt.device)
        x=torch.tensor(rt.tok.encode_tokens(branch["tokens"]),device=rt.device)
        labels=x[pos,:4].clone()
        labels[:,3]=torch.tensor([n["truth"] for n in nodes],device=rt.device)
        parents=torch.tensor([n["parents"] or [-1,-1] for n in nodes],device=rt.device)
        policies=torch.arange(3,device=rt.device)
        start=initial(rt,x[None].expand(3,-1,-1),len(nodes))
        D=c["D"]
        delta=1+P.seed("stop-donor",c["width"],c["depth"],c["gid"])%22
        use_donor=probe and arm=="unique"
        donor_steps={}
        if use_donor:
            dx=x.clone()
            const=nodes[0]["pos"]-2
            dx[const,3]=(dx[const,3]+delta)%23
            h=rt.init(dx[None])
            times=(nodes[c["u"]]["depth"]+1,D+4)
            for t in range(1,max(times)+1):
                h,logits,_=rt.forward(h)
                if t in times:
                    p=pos[c["u"]]
                    donor_steps[t]=(h[0,p].clone(),torch.stack([v[0,p].argmax() for v in logits[:4]]))
        scores, consumers, events, dense_history=[],[],[],[]
        for t in range(1,D+EXTRA+1):
            previous_locks=start.locked
            start=advance(rt,start,pos,policies)
            dense_history.append(start.read[0].clone())
            if t in (D,D+4,D+EXTRA):
                scores.extend(row_counts(start,pos,labels,D,t,dense_history))
                consumers.extend(consumer_counts(previous_locks,start,parents,pos,labels,t-D))
            if t in donor_steps:
                rows=intervention(rt,start,*donor_steps[t],c,pos,labels,t,delta)
                events.extend(dict(timing="middle" if t<D else "late",**r) for r in rows)
        results.append(dict(arm=arm,scores=scores,consumers=consumers,events=events))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--device",default="cuda")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    receipt=P.read(RECEIPT)
    if P.sha(args.plan)!=receipt["plan_sha256"]:
        raise ValueError("source plan mismatch")
    run=next(r for r in receipt["runs"] if r["model"]==args.model)
    spec=next(r for r in P.read(args.plan)["models"] if r["id"]==args.model)
    rt=Runtime(dict(spec,arm="update",sha256=run["final_ckpt"]["sha256"],
                    path=str(args.checkpoint or ROOT/run["final_ckpt"]["path"])),args.device)
    ids=[64] if args.smoke else list(range(32,48))
    out=args.out/("smoke" if args.smoke else "formal")/args.model
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(model=args.model,checkpoint=run["final_ckpt"]["sha256"],plan=P.sha(args.plan),ids=ids,
                  sources={s.name:P.sha(s) for s in [Path(__file__),Path(__file__).with_name("design.py"),
                    Path(__file__).with_name("hard_eval.py"),TOOLS/"c_update/runtime.py",
                    TOOLS/"c_update/protocol.py",TOOLS/"c_local_update/rebinding.py"]})
    if (out/"identity.json").exists() and P.read(out/"identity.json")!=identity:
        raise ValueError("output identity mismatch")
    P.write(out/"identity.json",identity)
    with torch.inference_mode():
        for w,d,shape in CELLS:
            for gid in ids:
                dest=out/f"{w}-{d}-{shape}-{gid}.json.gz"
                if dest.exists():
                    continue
                case=make_case(w,d,shape,gid)
                rows=run_case(rt,case,gid in ids[:4])
                tmp=dest.with_suffix(".tmp")
                with gzip.open(tmp,"wt",encoding="utf-8") as f:
                    json.dump(dict(width=w,depth=d,profile=shape,gid=gid,arms=rows),f,separators=(",",":"))
                tmp.replace(dest)
            print(args.model,w,d,shape,flush=True)
    P.write(out/"complete.json",dict(**identity,cases=len(CELLS)*len(ids)))


if __name__=="__main__":
    main()
