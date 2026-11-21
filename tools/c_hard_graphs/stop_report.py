"""Summarize a complete fixed-cohort pilot; do not pool models or naming arms."""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from stop_pilot import CELLS, EXTENSION, MODELS, OUT, P, POLICIES, RECEIPT
from hard_eval import write_csv


def tally_events(acc, event, comparable):
    for group,keep in (("all_attempts",True),("dense_and_policy_accepted",comparable)):
        if not keep:
            continue
        c=acc[group]
        c["attempted"]+=1
        c[event["status"]]+=1
        if event["status"]!="accepted":
            continue
        c["held_target"]+=event["locked_at_injection"]
        c["held_child"]+=event["child_locked_at_injection"]
        c["target_correct_next"]+=event["target_first_restored"]==1
        c["target_correct_last"]+=event["target_correct_last"]
        c["specific_damage"]+=event["child_specific_damage"]
        c["joint_stable_after_damage"]+=event["joint_stable"] is True


def report(root, smoke):
    ids=[64] if smoke else list(range(32,48))
    finals={r["model"]:r["final_ckpt"]["sha256"] for r in P.read(RECEIPT)["runs"]}
    capability=defaultdict(Counter)
    consumers=defaultdict(Counter)
    propagation=defaultdict(lambda:defaultdict(Counter))
    source=None
    fit_rows=[]
    for m in MODELS:
        folder=root/m
        meta=P.read(folder/"complete.json")
        if (meta["model"]!=m or meta["checkpoint"]!=finals[m] or meta["ids"]!=ids or
                meta["cases"]!=len(CELLS)*len(ids) or meta["plan"]!=P.read(RECEIPT)["plan_sha256"]):
            raise ValueError(f"unexpected cohort at {folder}")
        if source is not None and source!=meta["sources"]:
            raise ValueError("incompatible source revisions")
        source=meta["sources"]
        for w,d,profile in CELLS:
            for gid in ids:
                with gzip.open(folder/f"{w}-{d}-{profile}-{gid}.json.gz","rt",encoding="utf-8") as f:
                    case=json.load(f)
                if (case["width"],case["depth"],case["profile"],case["gid"])!=(w,d,profile,gid):
                    raise ValueError("graph key mismatch")
                arms=["reuse4","reuse2"] if (w,d,profile)==EXTENSION else ["unique","reuse4","reuse2"]
                if [arm["arm"] for arm in case["arms"]]!=arms:
                    raise ValueError("missing or duplicated naming condition")
                for part in case["arms"]:
                    name=part["arm"]
                    expected=[(t,p) for t in (0,4,16) for p in POLICIES]
                    if [(r["extra"],r["policy"]) for r in part["scores"]]!=expected:
                        raise ValueError("incomplete scores")
                    for r in part["scores"]:
                        key=(m,w,d,profile,name,r["extra"],r["policy"])
                        c=capability[key]
                        c.update({k:v for k,v in r.items() if k not in ("extra","policy")})
                        c["graphs"]+=1
                        c["sum_graph_value_fraction"]+=r["value"]/r["nodes"]
                        c["sum_graph_four_fraction"]+=r["first4"]/r["nodes"]
                        if r["extra"]==0 and r["policy"]=="dense":
                            fit_rows.append(dict(model=m,width=w,depth=d,profile=profile,gid=gid,
                                                 naming=name,nodes=r["nodes"],value=r["value"],first4=r["first4"]))
                    ev=part["events"]
                    for r in part["consumers"]:
                        key=(m,w,d,profile,name,r["extra"],r["policy"],r["reader_held"])
                        consumers[key].update({k:v for k,v in r.items() if k not in ("extra","policy","reader_held")})
                    wanted=[(t,p) for t in ("middle","late") for p in POLICIES] if name=="unique" and gid in ids[:4] else []
                    if [(r["timing"],r["policy"]) for r in ev]!=wanted:
                        raise ValueError("incomplete intervention attempts")
                    for r in ev:
                        dense=next(e for e in ev if e["policy"]=="dense" and e["timing"]==r["timing"])
                        shared=r["status"]==dense["status"]=="accepted"
                        key=(m,w,d,profile,r["timing"],r["policy"])
                        tally_events(propagation[key],r,shared)
                        # Paired dense outcomes must use exactly this policy's accepted subset.
                        if shared:
                            pair=propagation[key]["dense_on_same_subset"]
                            tally_events({"all_attempts":pair},dense,False)
    cap=[]
    for key,c in capability.items():
        row=dict(zip(("model","width","depth","profile","naming","extra","policy"),key))
        sg=c.pop("sum_graph_value_fraction")
        sf=c.pop("sum_graph_four_fraction")
        low=c.pop("dense_budget_floor_round")
        high=c.pop("dense_budget_ceil_round")
        cap.append(dict(**row,**c,graph_mean_value=sg/c["graphs"],graph_mean_first4=sf/c["graphs"],
                        mean_dense_budget_floor_round=low/c["graphs"],
                        mean_dense_budget_ceil_round=high/c["graphs"]))
    events=[]
    columns=("attempted","accepted","graph_already_stopped","recipient_rejected","donor_rejected",
             "held_target","held_child","target_correct_next","target_correct_last","specific_damage","joint_stable_after_damage")
    for key,buckets in propagation.items():
        row=dict(zip(("model","width","depth","profile","timing","policy"),key))
        for group in ("all_attempts","dense_and_policy_accepted","dense_on_same_subset"):
            events.append(dict(**row,cohort=group,**{k:buckets[group][k] for k in columns}))
    write_csv(root/"capability.csv",cap)
    write_csv(root/"propagation.csv",events)
    write_csv(root/"dense-fit.csv",fit_rows)
    cols=("model","width","depth","profile","naming","extra","policy","reader_held")
    write_csv(root/"consumers.csv",(dict(**dict(zip(cols,key)),**v) for key,v in consumers.items()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    report(args.out/("smoke" if args.smoke else "formal"),args.smoke)
