"""Evaluate sequential assignments supplied as one graph, with paired SSA naming."""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c_update"))
import protocol as P

MODELS = ("A_s42", "B_s42", "C_s42", "C_s1234", "C_s2026")
# Length counts successive uses of a destination, including its first definition.
CELLS = [(family, n, 8) for family in
         ("recycle", "self_read", "interleaved", "fanout", "delayed", "mixed_dag", "saved_result", "two_rebound_inputs")
         for n in (2, 4, 8)]
CELLS += [("liveness", size, 8) for size in (16,32,64)]
CELLS += [("wide", waves, 4) for waves in (8, 16, 30, 33)]


def allocated_program(size, rng):
    """Assign a fresh definition to a dead name on a generated DAG (last use known)."""
    parents = [()] * 3
    for i in range(3,size):
        window = list(range(max(0,i-8),i))
        parents.append(tuple(rng.sample(window,rng.randint(1,min(3,len(window))))))
    last = list(range(size))
    for i,ps in enumerate(parents):
        for p in ps:
            last[p] = i
    slots, end, prior, ops = [], {}, {}, []
    for i,ps in enumerate(parents):
        available = sorted(k for k,t in end.items() if prior[k]<t<i)
        slot = rng.choice(available) if available else f"r{len(end)}"
        expr = tuple(slots[p] for p in ps) if ps else rng.randrange(23)
        ops.append((slot,expr))
        slots.append(slot)
        end[slot] = last[i]
        prior[slot] = i
    return ops


def program(family, n, gid):
    rng = random.Random(P.seed("inline-rebinding-v2", family, n, gid))
    if family == "liveness":
        return allocated_program(n,rng)
    ops = [(f"k{i}", rng.randrange(1, 23)) for i in range(3)]

    def add(name, left, right):
        ops.append((name, (left, right)))

    lanes = 4 if family == "wide" else 2 if family in ("interleaved", "mixed_dag", "two_rebound_inputs") else 1
    if family == "wide":
        for wave in range(n):
            # Two banks alternate; a bank's old definitions are dead before reuse.
            for lane in range(lanes):
                base = f"s{1-wave%2}_{lane}" if wave else f"k{lane%3}"
                add(f"s{wave%2}_{lane}", base, f"k{(wave+lane+1)%3}")
        for lane in range(lanes):
            add(f"out{lane}", f"s{(n-1)%2}_{lane}", "k2")
        return ops

    for wave in range(n):
        for lane in range(lanes):
            slot = f"s{lane}"
            prev = f"carry{lane}"
            term = f"k{(wave+lane+1)%3}"
            if wave == 0:
                left = f"k{lane}"
            elif family == "self_read":
                left = slot
            elif family == "mixed_dag":
                left = rng.choice([prev, f"carry{1-lane}", f"s{1-lane}"])
            elif family == "interleaved":
                left = f"carry{1-lane}"
            else:
                left = prev
            add(slot, left, term)
            if family == "saved_result" and wave == 0:
                add("saved",slot,"k0")
            if family == "delayed":
                for k in range(3):
                    add(f"gap{k}", "k0", "k1" if k == 0 else f"gap{k-1}")
            if family == "fanout":
                add(f"fork{wave}", slot, "k0")
                add(prev, slot, f"fork{wave}")
            else:
                add(prev, slot, term)
    if family == "saved_result":
        add("late_reader","saved","s0")
    if family == "two_rebound_inputs":
        add("late_reader","s0","s1")
    for lane in range(lanes):
        add(f"out{lane}", f"carry{lane}", "k2")
    return ops


def render(ops, names, unique):
    """Resolve each RHS against preceding definitions, then bind the LHS."""
    latest, histories, spelling = {}, defaultdict(list), {}
    tokens, nodes = [], []
    for row, (dest, expr) in enumerate(ops):
        parents = [] if isinstance(expr, int) else [latest[v] for v in expr]
        if parents:
            truth = sum(nodes[i]["truth"] for i in parents) % 23
            rhs = []
            for p in parents:
                if rhs:
                    rhs.append("ADD")
                rhs.append(nodes[p]["name"])
        else:
            truth, rhs = expr, [str(expr)]
        if not unique and dest not in spelling:
            spelling[dest] = names[len(spelling)]
        name = names[row] if unique else spelling[dest]
        if tokens:
            tokens.append("<EQ_SEP>")
        tokens.extend([*rhs, "=", name])
        # A stale answer can coincide with the true one modulo 23. Count only
        # distinguishable alternatives, and keep their origin for interpretation.
        stale = set()
        for j, i in enumerate(parents):
            previous = histories[nodes[i]["slot"]][:-1]
            for old in previous:
                stale.add((truth - nodes[i]["truth"] + nodes[old]["truth"]) % 23)
        nodes.append(dict(slot=dest, name=name, version=len(histories[dest])+1,
                          pos=len(tokens)-1, truth=truth, parents=parents,
                          depth=1+max((nodes[i]["depth"] for i in parents), default=-1),
                          stale=sorted(stale-{truth})))
        latest[dest] = row
        histories[dest].append(row)
    return dict(tokens=tokens, nodes=nodes)


def case(family, n, gid):
    ops = program(family, n, gid)
    names = [f"x_{i}" for i in random.Random(P.seed("names", family, n, gid)).sample(range(128), 128)]
    branches = {"rebound": render(ops, names, False)}
    if len(ops) <= len(names):
        # Match first use names in both arms; reserve remaining names for rewrites.
        first = {z["slot"]: z["name"] for z in branches["rebound"]["nodes"]}
        spare = iter([name for name in names if name not in set(first.values())])
        renamed = [z["name"] if z["version"]==1 else next(spare) for z in branches["rebound"]["nodes"]]
        branches["unique"] = render(ops, renamed, True)
    if len(branches["rebound"]["tokens"]) > 1024:
        raise ValueError("sequence exceeds model budget")
    nodes = branches["rebound"]["nodes"]
    versions = Counter(z["slot"] for z in nodes)
    layers = Counter(z["depth"] for z in nodes)
    outgoing = defaultdict(list)
    for j,z in enumerate(nodes):
        for p in z["parents"]:
            outgoing[p].append(j)
    last_def = {}
    overwrites = []
    for j,z in enumerate(nodes):
        if z["slot"] in last_def:
            old = last_def[z["slot"]]
            uses = outgoing[old]
            overwrites.append(dict(old=old,new=j,prior_uses=len(uses),last_read=max(uses,default=None),
                                   self_read=old in z["parents"]))
        last_def[z["slot"]] = j
    return dict(family=family, size=n, gid=gid, branches=branches,
                definitions=len(nodes),names=len(versions),max_versions=max(versions.values()),
                max_layer_width=max(layers.values()),max_depth=max(layers),overwrites=overwrites)


def rollout(rt, branches, rounds):
    import torch
    x = torch.tensor([rt.tok.encode_tokens(b["tokens"]) for b in branches], device=rt.device)
    positions = [node["pos"] for node in branches[0]["nodes"]]
    traces = [[] for _ in branches]
    attributes = [{} for _ in branches]
    truth = torch.tensor([[n["truth"] for n in b["nodes"]] for b in branches],device=rt.device)
    with torch.inference_mode():
        h = rt.init(x)
        for t in range(1,rounds+1):
            h, logits, _ = rt.forward(h)
            decoded = logits[3][:, positions].argmax(-1)
            if t in (rounds-8,rounds-4,rounds):
                fields = [logits[f][:,positions].argmax(-1)==x[:,positions,f] for f in range(3)]
                correct = torch.stack([*fields, decoded==truth]).all(0)
                for record,four,var in zip(attributes,correct.tolist(),fields[1].tolist()):
                    record[t]=dict(first4=four,variable=var)
            for trace, v in zip(traces, decoded.tolist()):
                trace.append(v)
    return traces, attributes


def score(branch, values):
    nodes = branch["nodes"]
    counts = defaultdict(Counter)
    good = [v == z["truth"] for v,z in zip(values,nodes)]
    is_parent = {i for z in nodes for i in z["parents"]}
    rewritten = {z["slot"] for z in nodes if z["version"]>1}
    for i,(v,z) in enumerate(zip(values,nodes)):
        scopes = ["all"]
        if z["slot"] in rewritten:
            scopes.append("definition_v"+str(z["version"]))
        if z["parents"]:
            if any(nodes[p]["version"]>1 for p in z["parents"]):
                scopes.append("reads_rebound")
            if any(nodes[p]["slot"] in rewritten and nodes[p]["version"]==1 for p in z["parents"]):
                scopes.append("reads_first_binding")
        if i not in is_parent:
            scopes.append("sink")
        for scope in scopes:
            c = counts[scope]
            c.update(observed=1, correct=good[i], stale_eligible=bool(z["stale"]),
                     matches_stale=v in z["stale"], empty=v==24)
    return all(good), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", choices=MODELS, action="append")
    args = parser.parse_args()
    from runtime import Runtime
    plan = P.read(args.plan)
    if plan["status"]!="complete" or plan["protocol"]!=P.VERSION:
        raise ValueError("expected completed original C plan")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cases = [case(f,n,i) for f,n,count in CELLS for i in range(count)]
    P.write(args.out_dir/"cases.json", cases)
    for model in dict.fromkeys(args.model or MODELS):
        spec = next(m for m in plan["models"] if m["id"] == model)
        if spec.get("arm") in ("correction","update"):
            raise ValueError("expected original weights")
        rt = Runtime(spec,args.device)
        records, totals, pairs = [], defaultdict(Counter), defaultdict(Counter)
        differences = defaultdict(Counter)
        for c in cases:
            arms = c["branches"]
            d = max(z["depth"] for z in arms["rebound"]["nodes"])+1
            observations = (d,d+4,d+8)
            traces, attributes = rollout(rt,list(arms.values()),observations[-1])
            result = {}
            for (key,b),trace,attr in zip(arms.items(),traces,attributes):
                records.append(dict(family=c["family"],size=c["size"],gid=c["gid"],arm=key,D=d,rounds=trace,attributes=attr))
                for t in observations:
                    ok, groups = score(b,trace[t-1])
                    result[key,t] = ok
                    groups["all"].update(first4_correct=sum(attr[t]["first4"]), variable_correct=sum(attr[t]["variable"]))
                    for role,cnt in groups.items():
                        totals[c["family"],c["size"],key,t-d,role].update(cnt)
            if "unique" in arms:
                reference = traces[list(arms).index("unique")]
                replay = traces[list(arms).index("rebound")]
                nodes = arms["rebound"]["nodes"]
                for t in observations:
                    first_failure = None
                    for i,z in enumerate(nodes):
                        a,b = reference[t-1][i], replay[t-1][i]
                        role = ("uses_rebound" if any(nodes[p]["version"]>1 for p in z["parents"])
                                else "other")
                        bucket = differences[c["family"],c["size"],t-d,role,z["version"]]
                        bucket[f"unique_{int(a==z['truth'])}_rebound_{int(b==z['truth'])}"] += 1
                        if first_failure is None and a==z["truth"] and b!=z["truth"]:
                            first_failure = i
                    records.append(dict(family=c["family"],size=c["size"],gid=c["gid"],round=t,
                                        first_control_correct_rebound_wrong=first_failure))
            for t in observations:
                group = pairs[c["family"],c["size"],t-d]
                if "unique" in arms:
                    group[f"unique_{int(result['unique',t])}_rebound_{int(result['rebound',t])}"] += 1
                else:
                    group["unpaired_graphs"] += 1
                    group["unpaired_correct"] += int(result["rebound",t])
        def encode(groups):
            return [dict(key=k,counts=dict(v)) for k,v in groups.items()]
        P.write(args.out_dir/(model+".json"), dict(model=spec,plan_sha256=P.sha(args.plan),
                cases_sha256=P.sha(args.out_dir/"cases.json"), counts=encode(totals),pairs=encode(pairs),node_pairs=encode(differences),records=records))
        del rt


if __name__ == "__main__":
    main()
