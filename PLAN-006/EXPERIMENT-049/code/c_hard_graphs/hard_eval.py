"""Run paired hard graphs on the two selected post-trained continuous models."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from design import ARMS, COUNT, EXTENSION, EXTRA, P, PROBES, PROFILES, SCALES, TOOLS, make_case
from runtime import Runtime

ROOT = TOOLS.parent
RECEIPT = ROOT / "docs/records/2026-11-21/c-local-update-finals.json"
PLAN = ROOT / "tmp/analysis/c-update-20261119/data/plan.json"
OUT = ROOT / "tmp/analysis/c-hard-graphs-v1"
MODELS = ("C_s42", "C_s1234")


def gather(rt, h, x, pos):
    import torch
    h, logits, _ = rt.forward(h)
    f = torch.stack([head[:, pos].argmax(-1) for head in logits[:4]], -1)
    structural = f[..., :3] == x[:, pos, :3]
    # One device transfer per round; VALUE indices include EMPTY and N/A.
    obs = torch.stack([f[..., 3], structural.all(-1), structural[..., 1]], -1)
    return h, obs.tolist()


def probe_score(clean, changed, a, b, delta):
    # Each row contains the (u, q) VALUE readouts AFTER that forward.
    events = [i for i, (c, z) in enumerate(zip(clean, changed))
              if c[1] == b and z[1] == (b + delta) % 23]
    result = dict(target_first_restored=next((i for i, z in enumerate(changed, 1) if z[0] == a), None),
                  child_specific_damage=bool(events), first_child_damage=None, joint_stable=None)
    if events:
        result["first_child_damage"] = events[0] + 1
        result["joint_stable"] = any(all(tuple(z) == (a, b) for z in changed[j:])
                                     for j in range(events[0]+1, len(changed)-1))
    return result


def run_case(rt, case, do_probe):
    import torch
    arms = case["arms"]
    primary = next(iter(arms.values()))
    nodes = primary["nodes"]
    pos = [n["pos"] for n in nodes]
    D, u, q = case["D"], case["u"], case["q"]
    delta = 1 + P.seed("hard-donor", case["gid"], case["width"], case["depth"]) % 22
    tokens = [branch["tokens"] for branch in arms.values()]
    if do_probe:
        donor = list(primary["tokens"])
        donor[pos[0]-2] = str((nodes[0]["truth"] + delta) % 23)
        tokens.append(donor)
    x = torch.tensor([rt.tok.encode_tokens(z) for z in tokens], device=rt.device)
    h = rt.init(x)
    traces = {arm: [] for arm in arms}
    starts = {nodes[u]["depth"]+1, D+4}
    shots = {}
    for t in range(1, D + EXTRA + 1):
        h, obs = gather(rt, h, x, pos)
        for arm, values in zip(arms, obs):
            traces[arm].append(values)
        if do_probe and t in starts:
            shots[t] = (h[:1].clone(), h[-1:, pos[u]].clone(), obs[-1][u])
    probes = []
    a, b = nodes[u]["truth"], nodes[q]["truth"]
    for t, (base, alternate, read) in shots.items():
        before = traces["unique"][t-1]
        clean = [[z[u][0], z[q][0]] for z in traces["unique"][t:t+8]]
        row = dict(start=t, timing="middle" if t < D else "late_budget",
                   recipient=before[u], donor=read, delta=delta, child_before=before[q][0],
                   child_ever_correct=any(z[q][0]==b for z in traces["unique"][:t]),
                   graph_value_all=all(v[0]==n["truth"] for v,n in zip(before,nodes)),
                   clean=clean)
        probes.append(row)
        if before[u][:2] != [a, 1]:
            row["status"] = "recipient_not_correct"
        elif read[:2] != [(a+delta) % 23, 1]:
            row["status"] = "donor_not_correct"
        else:
            row["status"] = "accepted"
            # Matched batch-shape control: replay both untouched and patched
            # states together. Natural baseline above retains its own trajectory.
            pair = base.repeat(2, 1, 1)
            pair[1, pos[u]] = alternate[0]
            replay, changed = [], []
            for _ in range(8):
                pair, out = gather(rt, pair, x[:1].expand(2,-1,-1), [pos[u],pos[q]])
                replay.append([v[0] for v in out[0]])
                changed.append([v[0] for v in out[1]])
            row.update(control=replay, changed=changed, **probe_score(replay,changed,a,b,delta))
    return dict(traces=traces, probes=probes)


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def metrics(case, raw):
    nodes = next(iter(case["arms"].values()))["nodes"]
    roles = defaultdict(list)
    for i, (lane, d) in enumerate(case["attrs"]):
        roles["all"].append(i)
        if lane < 0:
            continue
        length = case["lengths"][lane]
        if d == length:
            roles["terminal"].append(i)
            roles["terminal_at_max_depth" if length == case["depth"] else "terminal_earlier"].append(i)
    for arm, trace in raw["traces"].items():
        for role, ix in roles.items():
            for t in (case["D"], case["D"]+4, case["D"]+8, case["D"]+16):
                correct = first4 = var = lost = gained = ready = 0
                stale_candidates = stale_hits = 0
                for i in ix:
                    truth = nodes[i]["truth"]
                    value, struct, variable = trace[t-1][i]
                    ok = value == truth
                    at_d = trace[case["D"]-1][i][0] == truth
                    correct += ok
                    first4 += ok and struct
                    var += variable
                    ready += at_d
                    lost += at_d and not ok
                    gained += not at_d and ok
                    old = case["arms"][arm]["nodes"][i]["stale"]
                    stale_candidates += bool(old)
                    stale_hits += value in old
                yield dict(arm=arm, role=role, extra=t-case["D"], nodes=len(ix),
                           value=correct, first4=first4, variable=var, correct_at_D=ready,
                           gained=gained, lost=lost, all_correct=int(correct==len(ix)),
                           stale_candidates=stale_candidates, matches_stale=stale_hits)


def early_readouts(case, traces):
    """Compare the actual dense trajectory with reading each node at depth+1."""
    D = case["D"]
    for arm, trace in traces.items():
        for role in ("all", "early_terminal"):
            total = early = final = either = spoiled = fixed = drifted = 0
            for i, node in enumerate(case["arms"][arm]["nodes"]):
                lane, d = case["attrs"][i]
                if role == "early_terminal" and not (lane >= 0 and d == case["lengths"][lane] < case["depth"]):
                    continue
                y = node["truth"]
                values = [row[i][0] for row in trace]
                was, now = values[d] == y, values[D+EXTRA-1] == y
                total += 1
                early += was
                final += now
                either += y in values[d:]
                spoiled += was and not now
                fixed += not was and now
                drifted += was and any(v != y for v in values[d+1:])
            if total:
                yield dict(arm=arm,role=role,nodes=total,at_local_round=early,at_last=final,
                           correct_at_least_once=either,early_correct_late_wrong=spoiled,
                           early_wrong_late_correct=fixed,ever_lost_early_correct=drifted)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    receipt = P.read(RECEIPT)
    run = next(r for r in receipt["runs"] if r["model"] == args.model)
    ckpt = args.checkpoint or ROOT / run["final_ckpt"]["path"]
    if P.sha(args.plan) != receipt["plan_sha256"]:
        raise ValueError("unexpected source plan")
    spec = next(m for m in P.read(args.plan)["models"] if m["id"] == args.model)
    rt = Runtime(dict(spec, arm="update", path=str(ckpt), sha256=run["final_ckpt"]["sha256"]),args.device)
    count = 1 if args.smoke else COUNT
    out = args.out / ("smoke" if args.smoke else "formal") / args.model
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(model=args.model, checkpoint=run["final_ckpt"]["sha256"], plan=P.sha(args.plan),
                    sources={s.name:P.sha(s) for s in (Path(__file__), Path(__file__).with_name("design.py"),
                       TOOLS/"c_local_update/rebinding.py", TOOLS/"c_update/runtime.py", TOOLS/"c_update/protocol.py")},
                    graphs_per_cell=count, probe_first=min(count,PROBES))
    meta = out/"identity.json"
    if meta.exists() and P.read(meta)!=identity:
        raise ValueError(f"different evaluation already at {out}")
    P.write(meta,identity)
    scores, probes, pairs, early = [], [], [], []
    import torch
    with torch.inference_mode():
        for w, d in [*SCALES, EXTENSION[:2]]:
            for profile in ((EXTENSION[2],) if (w,d)==EXTENSION[:2] else PROFILES):
                for gid in range(count):
                    case = make_case(w,d,profile,gid)
                    key = dict(width=w,depth=d,profile=profile,gid=gid)
                    dest = out/f"w{w}-d{d}-{profile}-{gid}.json.gz"
                    if dest.exists():
                        with gzip.open(dest,"rt",encoding="utf-8") as f:
                            raw=json.load(f)["result"]
                    else:
                        raw = run_case(rt,case,gid<PROBES and "unique" in case["arms"])
                        tmp = dest.with_suffix(".tmp")
                        with gzip.open(tmp,"wt",encoding="utf-8") as f:
                            json.dump(dict(case=case,result=raw),f,separators=(",",":"))
                        tmp.replace(dest)
                    scores.extend(dict(**key,**s) for s in metrics(case,raw))
                    early.extend(dict(**key,**s) for s in early_readouts(case,raw["traces"]))
                    for offset in ((0,4,8,16) if "unique" in case["arms"] else ()):
                        idx = case["D"]+offset-1
                        truth = [n["truth"] for n in case["arms"]["unique"]["nodes"]]
                        for arm in ARMS[1:]:
                            ref = raw["traces"]["unique"][idx]
                            alt = raw["traces"][arm][idx]
                            bins = [0]*4
                            for y,a,b in zip(truth,ref,alt):
                                bins[2*int(a[0]==y)+int(b[0]==y)] += 1
                            pairs.append(dict(**key,arm=arm,extra=offset,both_wrong=bins[0],
                                reuse_only_correct=bins[1],unique_only_correct=bins[2],both_correct=bins[3]))
                    for z in raw["probes"]:
                        probes.append(dict(**key,**{k:z.get(k) for k in ("timing","start","status",
                            "child_before","child_ever_correct","graph_value_all","target_first_restored",
                            "child_specific_damage","first_child_damage","joint_stable")}))
                print(args.model,w,d,profile,"complete",flush=True)
    write_csv(out/"graph-counts.csv",scores)
    write_csv(out/"paired-naming.csv",pairs)
    write_csv(out/"upstream-events.csv",probes)
    write_csv(out/"early-readouts.csv",early)
    P.write(out/"complete.json",dict(**identity,cases=(len(SCALES)*len(PROFILES)+1)*count,
                                     attempted_events=len(probes)))


if __name__ == "__main__":
    main()
