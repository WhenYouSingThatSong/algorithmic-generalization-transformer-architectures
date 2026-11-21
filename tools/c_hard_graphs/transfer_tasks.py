"""Exact carry-chain addition and a small sequential modulo-23 interpreter."""
import argparse
import json
import random
from pathlib import Path

from design import P


def add_trace(left,right,base=10):
    carry=0
    digits=[]
    carries=[]
    for a,b in zip(left,right,strict=True):
        carry,digit=divmod(a+b+carry,base)
        digits.append(digit)
        carries.append(carry)
    return digits+[carry],carries


def carry_case(gid,length,chain):
    rng=random.Random(P.seed("transfer-carry",gid,length))
    start=rng.randrange(length-chain+1) if chain else 0
    left,right=[],[]
    for i in range(length):
        if chain and i==start:
            total=rng.randrange(10,19)
        elif start<i<start+chain:
            total=9
        else:
            total=rng.randrange(9)
        a=rng.randint(max(0,total-9),min(9,total))
        left.append(a)
        right.append(total-a)
    answer,carries=add_trace(left,right)
    if sum(carries)!=chain:
        raise ValueError("uncontrolled carry chain")
    return dict(task="addition",width=length,difficulty=chain,gid=gid,
                input=dict(left=left,right=right),answer=answer,trace=carries)


def interpret(code, *, overwrite=None):
    values={}
    trace=[]
    for i,(dest,op,args) in enumerate(code):
        if op=="literal":
            v=args[0]
        elif op=="add":
            a,b=(values[n] for n in args)
            v=(a+b)%23
        else:
            raise ValueError("unknown instruction "+op)
        if overwrite and i==overwrite[0]:
            v=overwrite[1]
        values[dest]=v
        trace.append(v)
    return trace


def program_case(gid,length,slots):
    rng=random.Random(P.seed("transfer-program",gid,length))
    code=[(f"k{i}","literal",[v]) for i,v in enumerate(rng.sample(range(1,23),3))]
    for j in range(length):
        dest=f"x{j%slots}"
        prev=f"x{(j-1)%slots}" if j else "k0"
        code.append((dest,"add",[prev,f"k{rng.randrange(3)}"]))
        if j%3==0:
            code.append((f"saved{j}","add",[dest,"k1"]))
    for j in range(0,length,3):
        code.append((f"out{j}","add",[f"saved{j}",f"x{(length-1)%slots}"]))
    trace=interpret(code)
    return dict(task="program",width=length,difficulty=slots,gid=gid,input=code,
                answer=trace[-len(range(0,length,3)):],trace=trace)


def suite():
    for gid in range(320,352):
        for length in (8,16):
            for chain in (0,1,4,length):
                yield carry_case(gid,length,chain)
            for slots in (1,2,4,length):
                yield program_case(gid,length,slots)


def key(c):
    return tuple(c[k] for k in ("task","width","difficulty","gid"))


def intervened(c):
    """One forced carry/assignment; recompute following steps with the task oracle."""
    if c["task"]=="addition":
        left,right=c["input"]["left"],c["input"]["right"]
        pool=[i for i,v in enumerate(c["trace"][:-1]) if v] or list(range(len(left)-1))
        i=pool[c["gid"]%len(pool)]
        trace=list(c["trace"])
        answer=list(c["answer"])
        carry=trace[i]=1-trace[i]
        for j in range(i+1,len(left)):
            carry,answer[j]=divmod(left[j]+right[j]+carry,10)
            trace[j]=carry
        answer[-1]=carry
        value=trace[i]
    else:
        code=c["input"]
        candidates=[j for j,(dest,_,_) in enumerate(code) if dest.startswith("x")]
        i=candidates[c["gid"]%len(candidates)]
        value=(c["trace"][i]+1)%23
        trace=interpret(code,overwrite=(i,value))
        answer=trace[-len(c["answer"]):]
    return dict(c,input=dict(original=c["input"],force_after_step=i,new_value=value),
                answer=answer,trace=trace)


def export(folder, forced=False):
    folder.mkdir(parents=True,exist_ok=True)
    rows=[intervened(c) if forced else c for c in suite()]
    for name,fields in (("inputs",("input",)),("answers",("answer","trace"))):
        path=folder/f"{name}.jsonl"
        temp=path.with_suffix(".tmp")
        with temp.open("w",encoding="utf-8") as f:
            for c in rows:
                f.write(json.dumps({k:c[k] for k in ("task","width","difficulty","gid",*fields)})+"\n")
        temp.replace(path)
    P.write(folder/"manifest.json",{n:P.sha(folder/f"{n}.jsonl") for n in ("inputs","answers")})


def score(predictions, forced=False):
    from collections import Counter,defaultdict
    wanted={key(c):intervened(c) if forced else c for c in suite()}
    seen=set()
    groups=defaultdict(Counter)
    with predictions.open(encoding="utf-8") as f:
        for line in f:
            pred=json.loads(line)
            k=key(pred)
            if k in seen or k not in wanted:
                raise ValueError(f"duplicate or unknown case {k}")
            seen.add(k)
            g=groups[k[:3]]
            g["cases"]+=1
            for field in ("answer","trace"):
                y=wanted[k][field]
                guess=pred.get(field)
                if not isinstance(guess,list) or len(guess)!=len(y):
                    raise ValueError(f"invalid {field} for {k}")
                hits=sum(a==b for a,b in zip(y,guess))
                g[field+"_positions"]+=len(y)
                g[field+"_correct"]+=hits
                g[field+"_all_correct"]+=int(hits==len(y))
    if seen!=set(wanted):
        raise ValueError("prediction file lacks test cases")
    return [dict(task=k[0],width=k[1],difficulty=k[2],**v) for k,v in sorted(groups.items())]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--predictions",type=Path)
    p.add_argument("--forced",action="store_true")
    args=p.parse_args()
    if args.predictions:
        P.write(args.out,score(args.predictions,args.forced))
    else:
        export(args.out,args.forced)
