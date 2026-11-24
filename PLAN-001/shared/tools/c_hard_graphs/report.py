"""Summarize fixed cohorts; retain empty conditional denominators."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from design import COUNT, PROFILES, PROBES, SCALES
from hard_eval import MODELS, OUT, P, RECEIPT, write_csv


def groups(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    for key, values in result.items():
        yield dict(zip(keys,key)), values


def read_csv(p):
    with p.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def summarize(folder, smoke=False):
    outputs = defaultdict(list)
    finals = {r["model"]:r["final_ckpt"]["sha256"] for r in P.read(RECEIPT)["runs"]}
    sources = None
    for model in MODELS:
        path = folder / model
        meta = P.read(path/"complete.json")
        expected = 1 if smoke else COUNT
        if (meta["model"]!=model or meta["checkpoint"]!=finals[model] or
                meta["graphs_per_cell"]!=expected or meta["cases"]!=(len(SCALES)*len(PROFILES)+1)*expected):
            raise ValueError(f"unexpected run at {path}")
        if sources is not None and sources != meta["sources"]:
            raise ValueError("model evaluations used different code")
        sources = meta["sources"]
        counts = read_csv(path/"graph-counts.csv")
        dims = ["width","depth","profile","arm","role","extra"]
        paired = defaultdict(dict)
        for row in counts:
            if row["profile"] in ("uneven","balanced") and row["role"]=="all":
                key = tuple(row[k] for k in ("width","depth","arm","extra","gid"))
                paired[key][row["profile"]] = row
        diffs = []
        for (w,d,arm,extra,gid), cells in paired.items():
            a,b = cells["uneven"],cells["balanced"]
            if a["nodes"]!=b["nodes"]:
                raise ValueError("profile comparison node mismatch")
            n = int(a["nodes"])
            diffs.append(dict(width=w,depth=d,arm=arm,extra=extra,gid=gid,
                value=(int(a["value"])-int(b["value"]))/n,
                first4=(int(a["first4"])-int(b["first4"]))/n))
        for key, rows in groups(diffs,["width","depth","arm","extra"]):
            outputs["profile-differences"].append(dict(model=model,**key,graphs=len(rows),
                uneven_minus_balanced_value=mean(r["value"] for r in rows),
                uneven_minus_balanced_first4=mean(r["first4"] for r in rows)))
        for key, rows in groups(counts,dims):
            totals={k:sum(int(r[k]) for r in rows) for k in ("nodes","value","first4","variable",
                "correct_at_D","gained","lost","all_correct","stale_candidates","matches_stale")}
            outputs["capability"].append(dict(model=model,**key,graphs=len(rows),**totals,
                graph_mean_value=mean(int(r["value"])/int(r["nodes"]) for r in rows),
                graph_mean_first4=mean(int(r["first4"])/int(r["nodes"]) for r in rows)))
        for name,keys in (("early-readouts",dims[:5]),("paired-naming",dims[:4]+["extra"])):
            for key,rows in groups(read_csv(path/(name+".csv")),keys):
                cols = [k for k in rows[0] if k not in keys+["gid"]]
                totals = {k:sum(int(r[k]) for r in rows) for k in cols}
                outputs[name].append(dict(model=model,**key,graphs=len(rows),**totals))
        for key,rows in groups(read_csv(path/"upstream-events.csv"),dims[:3]+["timing"]):
            if len(rows)!=min(expected,PROBES):
                raise ValueError("incomplete intervention cohort")
            for label in ("all","previously_correct_child","never_correct_child"):
                cohort = [r for r in rows if label=="all" or
                          (label=="previously_correct_child")==(r["child_ever_correct"]=="True")]
                accepted=[r for r in cohort if r["status"]=="accepted"]
                damaged=[r for r in accepted if r["child_specific_damage"]=="True"]
                outputs["propagation"].append(dict(model=model,**key,cohort=label,attempted=len(cohort),
                    recipient_rejected=sum(r["status"]=="recipient_not_correct" for r in cohort),
                    donor_rejected=sum(r["status"]=="donor_not_correct" for r in cohort),
                    accepted=len(accepted),correct_target_first_round=sum(r["target_first_restored"]=="1" for r in accepted),
                    child_specific_damage=len(damaged),joint_stable_after_damage=sum(r["joint_stable"]=="True" for r in damaged)))
    for name, records in outputs.items():
        write_csv(folder/(name+"-summary.csv"), records)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    summarize(args.out/("smoke" if args.smoke else "formal"),args.smoke)
