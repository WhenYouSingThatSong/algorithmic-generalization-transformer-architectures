#!/usr/bin/env python3
"""Derive the small W tables from the two complete reassignment records.

This is a reporting-only reader.  It does not load a model and does not alter
the run directories.  It re-checks the structural invariants that the runner
enforces, then writes per-case, last-two, per-round, constant-drift and
unchanged-node-damage tables plus an audit summary.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import CSV, jsonl, read_json, sha, write_json
from reassignment_core import BRANCHES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEY_ROUNDS = (1, 2, 4, 8, 16)
REGIONS = ("all_nodes", "leaf", "changed_nodes", "changed_descendants",
           "unchanged_nodes", "unchanged_descendants",
           "outside_descendants", "constant")
REQUIRED_REGIONS = ("leaf", "changed_descendants", "unchanged_nodes", "constant")
START_KINDS = {name: ("scratch" if "scratch" in name else "completed") for name in BRANCHES}


def key(r):
    return r["graph_id"], r["leaf"], r["constant"], r["delta"]


def fraction(n, d):
    return n / d if d else None


def load_run(path, model_key):
    path = Path(path)
    meta = read_json(path / "meta.json")
    if meta.get("status") != "complete" or meta.get("protocol") != "discrete_reassignment_v1":
        raise ValueError("run is not a complete discrete_reassignment_v1 run: " + str(path))
    if meta.get("model_key") != model_key:
        raise ValueError("unexpected model_key: " + str(path))
    if sha(path / "records.jsonl") != meta["records_sha256"]:
        raise ValueError("records checksum mismatch: " + str(path))
    if sha(path / "selection.jsonl") != meta["selection_sha256"]:
        raise ValueError("selection checksum mismatch: " + str(path))
    selected = {}
    for row in jsonl(path / "selection.jsonl"):
        gid = row["graph_id"]
        if gid in selected:
            raise ValueError("duplicate selection row")
        selected[gid] = row
    if set(selected) != set(meta["candidate_ids"]):
        raise ValueError("selection does not cover every candidate")
    records = {}
    for rec in jsonl(path / "records.jsonl"):
        k = key(rec)
        if k in records:
            raise ValueError("duplicate record")
        if set(rec["branches"]) != set(BRANCHES):
            raise ValueError("wrong branch set")
        if rec["horizon"] != meta["horizon"]:
            raise ValueError("record horizon does not match meta")
        if len(rec["positions"]) != len(set(rec["positions"])):
            raise ValueError("duplicate positions")
        for name, branch in rec["branches"].items():
            if branch["truth_ref"] != ("new" if name.startswith("new_") else "old"):
                raise ValueError("wrong truth reference")
            if branch["start_kind"] != START_KINDS[name] or branch["pinned"] != name.endswith("_pinned"):
                raise ValueError("wrong start kind/pinned flag")
            if len(branch["initial"]) != len(rec["positions"]):
                raise ValueError("bad initial length")
            if len(branch["inputs"]) != meta["horizon"] or len(branch["outputs"]) != meta["horizon"]:
                raise ValueError("incomplete horizon")
            for name2, other in rec["branches"].items():
                if name2 <= name:
                    continue
                if name == "clean" and name2 == "sham" and branch != other:
                    raise ValueError("clean/sham differ")
        records[k] = rec
    expected = {(gid, s["leaf"], s["constant"], d)
                for gid, s in selected.items() if s["status"] == "selected"
                for d in meta["deltas"]}
    if set(records) != expected:
        raise ValueError("selected targets/deltas do not match records")
    return meta, selected, records


def start_checks(records):
    counts = Counter()
    for rec in records.values():
        pos = rec["positions"]
        li, ci = pos.index(rec["leaf"]), pos.index(rec["constant"])
        clean = rec["branches"]["clean"]["initial"]
        for name in ("state_error", "new_warm"):
            cur = rec["branches"][name]["initial"]
            changed = [(i, f) for i in range(len(pos)) for f in range(4)
                       if clean[i][f] != cur[i][f]]
            want = [(li, 3)] if name == "state_error" else [(ci, 3)]
            if changed != want:
                raise ValueError("unexpected start difference: " + name)
            counts[name] += 1
        for name in ("new_warm_pinned", "new_scratch_pinned"):
            if any(row[ci][3] != str(rec["new_constant"])
                   for row in rec["branches"][name]["inputs"]):
                raise ValueError("pinned constant input changed")
    return dict(state_error_leaf_value_only_cases=counts["state_error"],
                new_warm_constant_value_only_cases=counts["new_warm"],
                unexpected_start_differences=0)


def trajectories_equal(records, left, right):
    for rec in records.values():
        a, b = rec["branches"][left], rec["branches"][right]
        for field in ("initial", "inputs", "outputs"):
            if a[field] != b[field]:
                return False
    return True


def last_two_counts(rec, branch_name, positions):
    out = rec["branches"][branch_name]["outputs"][-2:]
    goal = rec["new_truth4"] if branch_name.startswith("new_") else rec["old_truth4"]
    old = rec["old_truth4"]
    new = rec["new_truth4"]
    ix = [rec["positions"].index(p) for p in positions]
    return dict(
        n=len(ix),
        value_n=sum(all(row[i][3] == goal[i][3] for row in out) for i in ix),
        first4_n=sum(all(row[i] == goal[i] for row in out) for i in ix),
        old_n=sum(all(row[i][3] == old[i][3] for row in out) for i in ix),
        new_n=sum(all(row[i][3] == new[i][3] for row in out) for i in ix),
    )


def write_coverage(out, metas, selected, common):
    rows = CSV(out / "priority-reassignment-coverage.csv")
    try:
        a, b = selected["model_a"], selected["model_b"]
        for gid in metas["model_a"]["candidate_ids"]:
            sa, sb = a[gid], b[gid]
            rows.add(dict(graph_id=gid, a_status=sa["status"], b_status=sb["status"],
                          common_selected=sa["status"] == sb["status"] == "selected",
                          leaf_a=sa.get("leaf"), leaf_b=sb.get("leaf"),
                          constant_a=sa.get("constant"), constant_b=sb.get("constant"),
                          state_round_a=sa.get("state_round"), state_round_b=sb.get("state_round")))
    finally:
        rows.close()


def write_last_two(out, records, common):
    rows = CSV(out / "priority-reassignment-last-two.csv")
    try:
        for arm in ("model_a", "model_b"):
            for delta in metas["model_a"]["deltas"]:
                for branch in BRANCHES:
                    for region in REGIONS:
                        vals = []
                        for k, rec in records[arm].items():
                            if k[3] != delta or k not in common:
                                continue
                            vals.append(last_two_counts(rec, branch, rec["regions"][region]))
                        n = sum(v["n"] for v in vals)
                        observed = [v for v in vals if v["n"]]
                        rows.add(dict(
                            arm=arm, delta=delta, branch=branch, region=region,
                            cohort="common_qualified", graph_n=len(vals), observed_graph_n=len(observed),
                            node_n=n,
                            value_n=sum(v["value_n"] for v in observed),
                            first4_n=sum(v["first4_n"] for v in observed),
                            old_n=sum(v["old_n"] for v in observed),
                            new_n=sum(v["new_n"] for v in observed),
                            value_micro=fraction(sum(v["value_n"] for v in observed), n),
                            value_macro=(sum(v["value_n"] / v["n"] for v in observed) / len(observed)
                                         if observed else None),
                        ))
    finally:
        rows.close()


def round_counts(rec, branch_name, positions, step):
    goal = rec["new_truth4"] if branch_name.startswith("new_") else rec["old_truth4"]
    old = rec["old_truth4"]
    new = rec["new_truth4"]
    ix = [rec["positions"].index(p) for p in positions]
    row = rec["branches"][branch_name]["outputs"][step - 1]
    return dict(n=len(ix),
                value_n=sum(row[i][3] == goal[i][3] for i in ix),
                first4_n=sum(row[i] == goal[i] for i in ix),
                old_n=sum(row[i][3] == old[i][3] for i in ix),
                new_n=sum(row[i][3] == new[i][3] for i in ix))


def write_curves(out, records, common):
    rows = CSV(out / "priority-reassignment-curves.csv")
    try:
        for arm in ("model_a", "model_b"):
            for delta in metas["model_a"]["deltas"]:
                for branch in BRANCHES:
                    for region in REQUIRED_REGIONS:
                        for step, window in [(r, str(r)) for r in KEY_ROUNDS if r <= metas[arm]["horizon"]] + [(None, "last_two")]:
                            vals = []
                            for k, rec in records[arm].items():
                                if k[3] != delta or k not in common:
                                    continue
                                if step is None:
                                    vals.append(last_two_counts(rec, branch, rec["regions"][region]))
                                else:
                                    vals.append(round_counts(rec, branch, rec["regions"][region], step))
                            n = sum(v["n"] for v in vals)
                            observed = [v for v in vals if v["n"]]
                            rows.add(dict(
                                arm=arm, delta=delta, branch=branch, region=region,
                                round_or_window=window, cohort="common_qualified",
                                node_n=n,
                                value_n=sum(v["value_n"] for v in observed),
                                first4_n=sum(v["first4_n"] for v in observed),
                                old_n=sum(v["old_n"] for v in observed),
                                new_n=sum(v["new_n"] for v in observed),
                                value_micro=fraction(sum(v["value_n"] for v in observed), n),
                                value_macro=(sum(v["value_n"] / v["n"] for v in observed) / len(observed)
                                             if observed else None),
                            ))
    finally:
        rows.close()


def write_constant_drift(out, records):
    rows = CSV(out / "priority-reassignment-constant-drift.csv")
    try:
        for arm in ("model_a", "model_b"):
            for delta in metas["model_a"]["deltas"]:
                for branch in BRANCHES:
                    recs = [r for k, r in records[arm].items() if k[3] == delta]
                    in_bad = out_bad = cases_in = cases_out = 0
                    for rec in recs:
                        ci = rec["positions"].index(rec["constant"])
                        goal = str(rec["new_constant"]) if branch.startswith("new_") else str(rec["constants"][str(rec["constant"])])
                        bad_in = bad_out = False
                        for t in range(rec["horizon"]):
                            if rec["branches"][branch]["inputs"][t][ci][3] != goal:
                                in_bad += 1
                                bad_in = True
                            if rec["branches"][branch]["outputs"][t][ci][3] != goal:
                                out_bad += 1
                                bad_out = True
                        cases_in += bad_in
                        cases_out += bad_out
                    rows.add(dict(arm=arm, delta=delta, branch=branch,
                                  cases=len(recs), case_rounds=len(recs) * metas["model_a"]["horizon"],
                                  input_drift_rounds=in_bad, output_drift_rounds=out_bad,
                                  cases_with_input_drift=cases_in, cases_with_output_drift=cases_out))
    finally:
        rows.close()


def write_damage(out, records):
    rows = CSV(out / "priority-reassignment-damage.csv")
    try:
        for arm in ("model_a", "model_b"):
            for delta in metas["model_a"]["deltas"]:
                for branch in BRANCHES:
                    recs = [r for k, r in records[arm].items() if k[3] == delta]
                    by_round = [0] * metas["model_a"]["horizon"]
                    last_any = last_both = n = 0
                    for rec in recs:
                        ix = [rec["positions"].index(p) for p in rec["regions"]["unchanged_nodes"]]
                        old = rec["old_truth4"]
                        clean = rec["branches"]["clean"]["outputs"]
                        cur = rec["branches"][branch]["outputs"]
                        n += len(ix)
                        for t in range(metas["model_a"]["horizon"]):
                            by_round[t] += sum(clean[t][i][3] == old[i][3] and cur[t][i][3] != old[i][3] for i in ix)
                        clean_both = [i for i in ix if all(clean[t][i][3] == old[i][3] for t in (-2, -1))]
                        last_any += sum(any(cur[t][i][3] != old[i][3] for t in (-2, -1)) for i in clean_both)
                        last_both += sum(all(cur[t][i][3] != old[i][3] for t in (-2, -1)) for i in clean_both)
                    peak = max(by_round) if by_round else 0
                    rows.add(dict(arm=arm, delta=delta, branch=branch, unchanged_node_n=n,
                                  last_two_any_damage_n=last_any, last_two_both_damage_n=last_both,
                                  peak_round_damage_n=peak, peak_round=by_round.index(peak) + 1 if peak else None))
    finally:
        rows.close()


def write_warm_vs_scratch(out, records):
    rows = CSV(out / "priority-reassignment-warm-scratch.csv")
    try:
        for arm in ("model_a", "model_b"):
            for delta in metas["model_a"]["deltas"]:
                for region in ("leaf", "changed_descendants", "changed_nodes"):
                    counts = Counter()
                    warm_n = scratch_n = n = 0
                    cases = 0
                    for k, rec in records[arm].items():
                        if k[3] != delta:
                            continue
                        cases += 1
                        new = rec["new_truth4"]
                        ix = [rec["positions"].index(p) for p in rec["regions"][region]]
                        w = sum(all(row[i][3] == new[i][3] for row in rec["branches"]["new_warm"]["outputs"][-2:]) for i in ix)
                        s = sum(all(row[i][3] == new[i][3] for row in rec["branches"]["new_scratch"]["outputs"][-2:]) for i in ix)
                        warm_n += w
                        scratch_n += s
                        n += len(ix)
                        counts["equal" if w == s else "warm>scratch" if w > s else "scratch>warm"] += 1
                    rows.add(dict(arm=arm, delta=delta, region=region, cases=cases, node_n=n,
                                  warm_new_n=warm_n, scratch_new_n=scratch_n,
                                  equal_cases=counts["equal"], warm_better_cases=counts["warm>scratch"],
                                  scratch_better_cases=counts["scratch>warm"]))
    finally:
        rows.close()


def normalize_csv_newlines(out):
    """Keep the committed small tables LF-only; csv.writer defaults to CRLF."""
    for path in sorted(Path(out).glob("priority-reassignment-*.csv")):
        data = path.read_bytes()
        path.write_bytes(data.replace(b"\r\n", b"\n"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if len(args.run_dir) != 2:
        p.error("exactly two run directories (model_a and model_b) are required")
    loaded = {}
    for path in args.run_dir:
        meta = read_json(Path(path) / "meta.json")
        loaded[meta["model_key"]] = load_run(path, meta["model_key"])
    if set(loaded) != {"model_a", "model_b"}:
        raise ValueError("model_a and model_b are both required")
    global metas, records
    metas = {arm: loaded[arm][0] for arm in loaded}
    selected = {arm: loaded[arm][1] for arm in loaded}
    records = {arm: loaded[arm][2] for arm in loaded}
    for field in ("dataset_sha256", "manifest_sha256", "candidate_ids", "horizon", "batch_size", "deltas", "code_hashes"):
        if metas["model_a"][field] != metas["model_b"][field]:
            raise ValueError("unmatched protocol: " + field)
    if metas["model_a"]["horizon"] < 2:
        raise ValueError("last-two reporting requires at least two outputs")
    common = set(records["model_a"]) & set(records["model_b"])
    if len(common) != len(records["model_a"]) or len(common) != len(records["model_b"]):
        raise ValueError("A/B record sets differ")
    for k in common:
        a, b = records["model_a"][k], records["model_b"][k]
        for field in ("positions", "old_truth4", "new_truth4", "regions", "state_round", "parents"):
            if a[field] != b[field]:
                raise ValueError("unmatched target/truth: " + field)
    start_diffs = start_checks(records["model_a"])
    start_diffs_b = start_checks(records["model_b"])
    if start_diffs != start_diffs_b:
        raise ValueError("A/B start checks differ")
    pinned_free_identical = all(trajectories_equal(records[arm], left, right)
        for arm in records for left, right in
        (("new_warm", "new_warm_pinned"), ("new_scratch", "new_scratch_pinned")))
    # These are outcomes, not validity conditions. Other first-four fields can drift.
    value_equals_first4 = True
    for arm in records:
        for rec in records[arm].values():
            for branch in BRANCHES:
                for region in REGIONS:
                    for step in range(1, metas[arm]["horizon"] + 1):
                        got = round_counts(rec, branch, rec["regions"][region], step)
                        if got["value_n"] != got["first4_n"]:
                            value_equals_first4 = False
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_coverage(args.out_dir, metas, selected, common)
    write_last_two(args.out_dir, records, common)
    write_curves(args.out_dir, records, common)
    write_constant_drift(args.out_dir, records)
    write_damage(args.out_dir, records)
    write_warm_vs_scratch(args.out_dir, records)
    normalize_csv_newlines(args.out_dir)
    state_rounds = Counter(rec["state_round"] for rec in records["model_a"].values())
    audit = dict(
        protocol="reassignment_report_v1",
        source={"model_a": {k: metas["model_a"][k] for k in
                            ("records_sha256", "selection_sha256", "dataset_sha256", "manifest_sha256")},
                "model_b": {k: metas["model_b"][k] for k in
                            ("records_sha256", "selection_sha256", "dataset_sha256", "manifest_sha256")}},
        checks=dict(
            model_a_records=len(records["model_a"]), model_b_records=len(records["model_b"]),
            common_cases=len(common), common_graphs=len({k[0] for k in common}),
            candidate_graphs=metas["model_a"]["candidate_ids"],
            selected_a=sum(s["status"] == "selected" for s in selected["model_a"].values()),
            selected_b=sum(s["status"] == "selected" for s in selected["model_b"].values()),
            horizon=metas["model_a"]["horizon"], deltas=metas["model_a"]["deltas"],
            clean_sham_identical=True, pinned_free_identical=pinned_free_identical,
            pinned_constant_input=True, start_checks=start_diffs,
            value_equals_first4=value_equals_first4,
            raw_scratch_check="not checked by this exporter; needs separate tokenizer-based evidence",
            aux_fields="not present in first-four trajectories; verify feedback code and raw tokens separately",
        ),
        state_round_counts=dict(sorted(state_rounds.items())),
        code_hashes=metas["model_a"]["code_hashes"],
        code_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))},
        note="Counts are decoded VALUE correctness only. Post-round 0 is the intervention start; "
             "last_two requires both of the final two recorded rounds.",
    )
    write_json(args.out_dir / "priority-reassignment-audit.json", audit)


if __name__ == "__main__":
    main()
