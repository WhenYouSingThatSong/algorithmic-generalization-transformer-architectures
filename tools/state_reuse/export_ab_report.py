#!/usr/bin/env python3
"""Export compact AB state-reuse review tables from completed batch summaries."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "priority_pilots"))
from io_utils import jsonl, read_json, sha, write_json


def out_csv(path):
    f = path.open("w", newline="", encoding="utf-8")
    return f, csv.DictWriter(f, fieldnames=None, extrasaction="ignore", lineterminator="\n")


def write_rows(path, rows):
    f = path.open("w", newline="", encoding="utf-8")
    try:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    finally:
        f.close()


def condition(x):
    return json.loads(x["condition"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-dir", type=Path, required=True, help="AB-main batch directory")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    stage_rows = []
    retention = defaultdict(lambda: dict(changed=0, retain=0, accept=0, protected=0, damage=0))
    event_rows = []
    coverage = {}
    sources = {}
    for mode in ("repeat", "reuse"):
        sdir = a.main_dir / f"{mode}-summary"
        meta = read_json(sdir / "meta.json")
        if meta.get("status") != "complete" or meta.get("protocol") != "state_reuse_summary_v1":
            raise ValueError("incomplete summary: " + str(sdir))
        sources[mode] = meta["sources"]
        cov_rows = list(csv.DictReader((sdir / "coverage.csv").open(encoding="utf-8-sig")))
        coverage[mode] = Counter(r["status"] for r in cov_rows)
        with (sdir / "summary.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["cohort"] != "common_source_qualified" or r["round_or_window"] != "last_two":
                    continue
                if r["region"] not in ("target", "changed", "unchanged", "constants", "all_nodes"):
                    continue
                if r["branch"] not in ("warm", "clear", "scratch", "warm_pinned", "clean", "sham"):
                    continue
                c = condition(r)
                stage_rows.append(dict(mode=mode, model=r["model"], arm=r["arm"],
                    schedule=c[0], horizon=c[1], episode=c[2], branch=r["branch"], region=r["region"],
                    node_n=r["node_n"], value_n=r["value_n"], first4_n=r["first4_n"],
                    value_macro=r["value_macro"], first4_macro=r["first4_macro"]))
        with (sdir / "update_retention.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["round_or_window"] != "last_two":
                    continue
                c = condition(r)
                key = (mode, r["arm"], c[0], c[1], c[2], r["branch"])
                z = retention[key]
                z["changed"] += int(r["changed_n"])
                z["retain"] += int(r["retain_previous_n"])
                z["accept"] += int(r["accept_new_n"])
                z["protected"] += int(r["clean_correct_n"])
                z["damage"] += int(r["extra_damage_n"])
        ev = defaultdict(lambda: dict(first=0, stable=0, reerror=0, graphs=0))
        with (sdir / "events.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["region"] not in ("target", "changed", "unchanged", "constants", "all_nodes"):
                    continue
                c = condition(r)
                key = (mode, r["arm"], c[0], c[1], c[2], r["branch"], r["region"])
                e = ev[key]
                e["graphs"] += 1
                e["first"] += bool(r["first_all_correct"])
                e["stable"] += bool(r["suffix_all_correct_from"])
                e["reerror"] += r["correct_then_wrong"] == "True"
        for k, v in sorted(ev.items()):
            event_rows.append(dict(mode=k[0], arm=k[1], model="A" if k[1] == "model_a" else "B",
                schedule=k[2], horizon=k[3], episode=k[4], branch=k[5], region=k[6],
                graphs=v["graphs"], first_all_correct_graphs=v["first"],
                stable_suffix_graphs=v["stable"], correct_then_wrong_graphs=v["reerror"]))
    write_rows(a.out_dir / "ab-stage-summary.csv", stage_rows)
    write_rows(a.out_dir / "ab-retention-summary.csv",
        [dict(mode=k[0], arm=k[1], model="A" if k[1] == "model_a" else "B",
              schedule=k[2], horizon=k[3], episode=k[4], branch=k[5],
              changed_n=v["changed"], retain_previous_n=v["retain"],
              accept_new_n=v["accept"], protected_n=v["protected"], extra_damage_n=v["damage"])
         for k, v in sorted(retention.items())])
    # Interface summary from the actual reuse records.
    iface = []
    for arm in ("A", "B"):
        for r in jsonl(a.main_dir / f"reuse-{arm}" / "records.jsonl"):
            if r["horizon"] != 16:
                continue
            edits = {int(k) for k in r["symbolic_edits"]}
            iface.append(dict(arm=arm, graph_id=r["graph_id"], episode=r["episode"],
                target=r["plan"]["target"], reclaimed=json.dumps(r["plan"]["reclaimed"]),
                old_parents=json.dumps(r["plan"]["old_parents"]),
                new_parents=json.dumps(r["plan"]["new_parents"]),
                symbolic_edit_rows=len(edits), token_length=len(r["raw_task"]),
                archive_entries=len(r["archive"]),
                consumes_recycled_version=r["plan"]["consumes_recycled_version"]))
    write_rows(a.out_dir / "ab-reuse-interface.csv", iface)
    write_rows(a.out_dir / "ab-event-summary.csv", event_rows)
    receipt_rows = []
    for pth in sorted(a.main_dir.glob("*.receipt.json")):
        d = json.loads(pth.read_text())
        receipt_rows.append(dict(receipt=pth.name, status=d.get("status"),
            exit_code=d.get("exit_code"), elapsed_seconds=d.get("elapsed_seconds"),
            meta_sha256=d.get("meta_sha256")))
    write_rows(a.out_dir / "ab-resource.csv", receipt_rows)
    forward_rows = []
    for mode in ("repeat", "reuse"):
        for arm in ("A", "B"):
            rs = list(jsonl(a.main_dir / f"{mode}-{arm}" / "records.jsonl"))
            for branch in ("warm", "clear", "scratch", "warm_pinned", "clean", "sham"):
                vals = [r["branches"][branch]["cumulative_forward_cost"] for r in rs]
                forward_rows.append(dict(mode=mode, arm=arm, branch=branch, records=len(vals),
                    mean=round(sum(vals) / len(vals), 6), minimum=min(vals), maximum=max(vals)))
    write_rows(a.out_dir / "ab-forward-cost.csv", forward_rows)
    audit = dict(protocol="ab_state_reuse_report_v1", main_dir=str(a.main_dir),
        sources=sources, coverage=coverage,
        counts=dict(stage_rows=len(stage_rows), retention_rows=len(retention),
                    interface_rows=len(iface), event_rows=len(event_rows), resource_rows=len(receipt_rows),
                    forward_rows=len(forward_rows)),
        code_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))},
        files={p.name: sha(p) for p in sorted(a.out_dir.iterdir()) if p.is_file()},
        note="Derived from completed repeat/reuse summaries; no model run. H8/H16 and episodes remain separate.")
    write_json(a.out_dir / "ab-report-audit.json", audit)
    print(json.dumps(audit["counts"]))


if __name__ == "__main__":
    main()
