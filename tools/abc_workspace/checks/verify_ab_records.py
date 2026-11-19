#!/usr/bin/env python3
"""Independent CPU checks for completed AB ABC-workspace records.

The checker reads records and manifests only; it does not load a model or rerun a
GPU job.  It keeps two different truths separate:

* the reference program truth (``protocol.reference_program``), which is the
  oracle for the original 20-window task; and
* the actual-input conditional truth, obtained by evaluating the same frame
  equations on the boundary commits the model actually produced.

Those two truths are allowed to disagree.  A record can solve its actual-input
window while the original program is still wrong, so neither is substituted for
the other in the checks or in ``conditional_metrics``.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/abc_workspace"))
sys.path.insert(0, str(ROOT / "tools/priority_pilots"))
sys.path.insert(0, str(ROOT / "tools/b_mechanism"))
import protocol as P
from io_utils import sha
from b_mechanism_common import load_configs, build_factored_tokenizer

AP = argparse.ArgumentParser()
AP.add_argument("--bundle", type=Path,
                default=ROOT / "tmp/analysis/abc-workspace-20261119/AB-main")
AP.add_argument("--data", type=Path,
                default=ROOT / "tmp/analysis/abc-workspace-20261119/data")
AP.add_argument("--report", type=Path,
                help="optional JSON output path for a small verification record")
ARGS = AP.parse_args()
BASE = ARGS.bundle.resolve()
DATA = ARGS.data.resolve()
PLAN_COMMIT = json.loads((BASE / "plan.json").read_text())["commit"]
_, _, DC = load_configs("B-M6-Nodes32-ADD-L4H16D384-DeBERTa")
TOK = build_factored_tokenizer(DC)
EMPTY = TOK.value_tok2idx["EMPTY"]
errors = []


def check(cond, msg):
    if not cond:
        errors.append(msg)
        if len(errors) <= 25:
            print("FAIL:", msg)


def dec(tid):
    s = TOK.value_idx2tok[int(tid)]
    return int(s) if s.isdigit() else s


def expected_metrics(lay, outputs, gold, trace):
    out = trace["outputs"]
    m = P.measures(out, gold, lay["nodes"], [lay["nodes"][i] for i in outputs])
    m["task_first4_n"] = sum(
        all(row[p] == gold[p] for row in out[-2:]) for p in lay["task_rows"]
    )
    m["task_n"] = len(lay["task_rows"])
    return m


def compare_metrics(got, want, label):
    check(set(got) == set(want), f"{label}: metric keys differ")
    for k in want:
        check(got.get(k) == want[k], f"{label}: metric {k} got {got.get(k)} want {want[k]}")


def parse_inputs(case, raw_tokens):
    f = P.frame(case, 0)
    base = P.render(f, case["names"], f["constants"])
    return f, [int(raw_tokens[p]) for p in base["constants"]]


def raw_rows(lay):
    return TOK.encode_tokens(lay["tokens"])


def set_value(row, value):
    row = list(row)
    row[3] = TOK.value_tok2idx[str(value)]
    return row


def truth_with_values(gold, lay, values):
    out = [list(r) for r in gold]
    for i, v in enumerate(values):
        out[lay["nodes"][i]] = set_value(out[lay["nodes"][i]], v)
    return out


def verify_common(rec, mode, case):
    label = f"{rec['model']} {mode} {rec['case']} s{rec['stage']} {rec['branch']}"
    if mode == "assignment":
        f, parsed = parse_inputs(case, rec["task_tokens"])
        lay = P.render(f, case["names"], parsed)
        if "parents" in rec:
            check(rec["parents"] == f["parents"], f"{label}: parents mismatch")
        vals, depths = P.evaluate(f["parents"], parsed)
        for i, v in enumerate(vals):
            check(dec(rec["truth4"][lay["nodes"][i]][3]) == v,
                  f"{label}: independent truth node {i} mismatch")
        check(rec.get("rounds") == max(depths) + 1 + 8, f"{label}: rounds mismatch")
    else:
        f = P.frame(case, rec["stage"] - 1)
        if "boundary_inputs" in rec:
            parsed = list(rec["boundary_inputs"])
        else:
            base = P.render(f, case["names"], f["constants"])
            parsed = [int(rec["task_tokens"][p]) for p in base["constants"]]
        lay = P.render(f, case["names"], parsed)
        if "parents" in rec:
            check(rec["parents"] == f["parents"], f"{label}: parents mismatch")
        ref = P.reference_program(case, rec["_stages"])[rec["stage"] - 1]
        if "reference" in rec:
            check(rec["reference"] == ref, f"{label}: reference program record mismatch")
        check(rec.get("rounds") == ref["local_depth"] + 1 + 8, f"{label}: rounds mismatch")
        for i, v in enumerate(ref["truth"]):
            check(dec(rec["truth4"][lay["nodes"][i]][3]) == v,
                  f"{label}: program gold node {i} mismatch")
        cond_vals, _ = P.evaluate(f["parents"], parsed)
        cond = truth_with_values(rec["truth4"], lay, cond_vals)
        want_cond = P.measures(rec["trace"]["outputs"], cond, lay["nodes"],
                               [lay["nodes"][i] for i in f["outputs"]])
        compare_metrics(rec["conditional_metrics"], want_cond, label + " conditional")
    check(rec["task_tokens"] == lay["tokens"], f"{label}: token layout/order mismatch")
    check(len(lay["tokens"]) <= 256, f"{label}: sequence longer than 256")
    check(len(lay["nodes"]) == 32, f"{label}: not 32 result nodes")
    rr = raw_rows(lay)
    check(len(rr) == len(lay["tokens"]), f"{label}: tokenizer row count mismatch")
    for row in rr:
        check(row[4:] == [0, 0], f"{label}: raw auxiliary fields not FALSE")
    for i, p in enumerate(lay["nodes"]):
        check(rr[p][3] == EMPTY, f"{label}: raw result row {i} is not EMPTY")
    for i, p in enumerate(lay["constants"]):
        check(int(TOK.value_idx2tok[rr[p][3]]) == parsed[i], f"{label}: constant token mismatch")
        check(int(rec["task_tokens"][p]) == parsed[i], f"{label}: rendered constant mismatch")
    want = expected_metrics(lay, f["outputs"], rec["truth4"], rec["trace"])
    compare_metrics(rec["metrics"], want, label)
    return f, lay, parsed, rr


def verify_snapshot(rec, mode, model):
    snap = BASE / f"{mode}-{model}" / rec["snapshot"]
    check(snap.is_file(), f"{model} {rec['case']} s{rec['stage']}: snapshot missing")
    check(sha(snap) == rec["snapshot_sha256"],
          f"{model} {rec['case']} s{rec['stage']}: snapshot hash mismatch")


def verify_probes(recs, mode, model, case):
    rows = [r for r in recs if r["case"] == case["id"]]
    by = {(r["stage"], r["branch"]): r for r in rows}
    if mode == "program":
        for r in rows:
            r["_stages"] = META[(model, "program")]["stages"]
    stages = sorted({r["stage"] for r in rows if r["branch"] == "probe_continue"})
    if not stages:
        return
    for stage in stages:
        carry = by[(stage, "carry")]
        f, lay, _, rr = verify_common(carry, mode, case)
        carry_final = carry["trace"]["outputs"][-1]
        for name in P.PROBES:
            r = by[(stage, "probe_" + name)]
            check(r.get("diagnostic") is True, f"{model} {case['id']} s{stage}: probe not diagnostic")
            verify_snapshot(r, mode, model)
            verify_common(r, mode, case)
            want_start = [list(x) for x in carry_final]
            if name in ("refresh_task", "refresh_and_reset"):
                for p in lay["task_rows"]:
                    want_start[p] = rr[p][:4]
            if name in ("reset_results", "refresh_and_reset"):
                for p in lay["nodes"]:
                    want_start[p][3] = EMPTY
            if name == "scratch_same_input":
                want_start = [row[:4] for row in rr]
            check(r["trace"]["inputs"][0] == want_start,
                  f"{model} {case['id']} s{stage} probe {name}: start state")


def verify_assignment(model, recs):
    by = {(r["case"], r["stage"], r["branch"]): r for r in recs}
    cases = sorted({r["case"] for r in recs})
    check(len(cases) == 5 * META[(model, "assignment")]["per_stratum"],
          f"{model}: assignment case count {len(cases)}")
    for case_id in cases:
        case = CASES[case_id]
        old = {}
        for b in ("old_clean", "old_corrupt", "old_same_write"):
            r = by[(case_id, 0, b)]
            f, lay, inputs, rr = verify_common(r, "assignment", case)
            old[b] = r
            if b == "old_clean":
                # baseline is the separate pre-branch run; only its qualification flag
                # is used later, while old_clean is a continuation from that state.
                check(isinstance(r["baseline"].get("nodes_first4_all"), bool),
                      f"{model} {case_id}: baseline not a metrics record")
                base_final = r["trace"]["outputs"][-1]
                f0 = f
            elif b == "old_same_write":
                check(r["trace"]["outputs"] == old["old_clean"]["trace"]["outputs"],
                      f"{model} {case_id}: same-value write changed old_clean trajectory")
            elif b == "old_corrupt":
                target = lay["nodes"][P.LEAVES]
                old_truth, _ = P.evaluate(f["parents"], f["constants"])
                intended = (old_truth[P.LEAVES] + 1) % P.MOD
                check(r["injection_target"] == target, f"{model} {case_id}: injection target")
                check(r["intended_fault"] == intended, f"{model} {case_id}: intended fault")
                check(dec(r["actual_input_value"]) == intended, f"{model} {case_id}: actual injected value")
                check(dec(r["trace"]["inputs"][0][target][3]) == intended,
                      f"{model} {case_id}: corruption not visible in first input")
        plans = P.update_inputs(f0["constants"])
        previous = None
        for stage, plan in enumerate(plans, 1):
            for branch in P.BRANCHES:
                r = by[(case_id, stage, branch)]
                f, lay, inputs, rr = verify_common(r, "assignment", case)
                check(inputs == plan["inputs"], f"{model} {case_id} s{stage}: branch inputs")
                expected_edit = [lay["constants"][i] for i in plan["changed_leaves"]]
                check(r["changed_task_rows"] == expected_edit,
                      f"{model} {case_id} s{stage} {branch}: changed rows")
                check(r["boundary_inputs"] == plan["inputs"],
                      f"{model} {case_id} s{stage}: boundary inputs")
                if previous is not None:
                    check(r["previous_inputs"] == previous,
                          f"{model} {case_id} s{stage} {branch}: previous inputs")
                affected = P.affected_nodes(f["parents"], plan["changed_leaves"])
                prior = base_final if stage == 1 else by[(case_id, stage - 1, branch)]["trace"]["outputs"][-1]
                want_start = [list(x) for x in prior]
                for p in expected_edit:
                    want_start[p] = rr[p][:4]
                if branch == "reset_results":
                    check(r["reset_slots"] == affected,
                          f"{model} {case_id} s{stage}: reset slots")
                    for i in affected:
                        want_start[lay["nodes"][i]][3] = EMPTY
                if branch == "scratch":
                    want_start = [row[:4] for row in rr]
                check(r["trace"]["inputs"][0] == want_start,
                      f"{model} {case_id} s{stage} {branch}: start state")
                if branch == "scratch":
                    check(all(r["trace"]["inputs"][0][p][3] == EMPTY for p in lay["nodes"]),
                          f"{model} {case_id} s{stage}: scratch node not EMPTY")
            for branch in ("unchanged", "same_input_write"):
                r = by[(case_id, stage, branch)]
                verify_common(r, "assignment", case)
                check(r["trace"]["outputs"] == old["old_clean"]["trace"]["outputs"],
                      f"{model} {case_id} s{stage}: {branch} control drifted")
            previous = plan["inputs"]
        verify_probes(recs, "assignment", model, case)
    return len(cases)


def verify_program(model, recs):
    by = {(r["case"], r["stage"], r["branch"]): r for r in recs}
    cases = sorted({r["case"] for r in recs})
    stages = META[(model, "program")]["stages"]
    for case_id in cases:
        case = CASES[case_id]
        refs = P.reference_program(case, stages)
        prev_layout = {}
        prev_commit = {}
        forward_sum = defaultdict(int)
        for stage in range(1, stages + 1):
            f = P.frame(case, stage - 1)
            for branch in P.BRANCHES:
                r = by[(case_id, stage, branch)]
                ref = refs[stage - 1]
                forward_sum[branch] += ref["local_depth"] + 1 + 8
                if r["status"] != "complete":
                    check(r["status"] == "invalid_commit",
                          f"{model} {case_id} s{stage} {branch}: bad status")
                    check(r.get("first_blocked_stage") == stage,
                          f"{model} {case_id} s{stage} {branch}: blocked stage")
                    continue
                r["_stages"] = stages
                f2, lay, parsed, rr = verify_common(r, "program", case)
                check(parsed == r["boundary_inputs"], f"{model} {case_id} s{stage}: boundary parse")
                if stage == 1:
                    check(r["boundary_inputs"] == f["constants"],
                          f"{model} {case_id} s{stage} {branch}: first boundary != constants")
                else:
                    check(r["boundary_inputs"][:f["width"]] == prev_commit[branch],
                          f"{model} {case_id} s{stage} {branch}: boundary != actual commit")
                    check(r["boundary_inputs"][f["width"]:] == f["constants"][f["width"]:],
                          f"{model} {case_id} s{stage} {branch}: exogenous inputs changed")
                commit = r["actual_commit"]
                check(len(commit) == f["width"],
                      f"{model} {case_id} s{stage} {branch}: commit length")
                check(r["decoded_boundary_values"] == len(commit),
                      f"{model} {case_id} s{stage} {branch}: commit count")
                for j, slot in enumerate(f["outputs"]):
                    p = lay["nodes"][slot]
                    check(commit[j] == dec(r["trace"]["outputs"][-1][p][3]),
                          f"{model} {case_id} s{stage} {branch}: commit {j} not final model output")
                check(r["cumulative_main_forwards"] == forward_sum[branch],
                      f"{model} {case_id} s{stage} {branch}: cumulative forwards")
                if stage == 1:
                    want_edit = lay["task_rows"]
                else:
                    old = prev_layout[branch]
                    want_edit = [p for p in lay["task_rows"] if rr[p][:4] != old[p][:4]]
                check(r["changed_task_rows"] == want_edit,
                      f"{model} {case_id} s{stage} {branch}: edited task rows")
                prev_layout[branch] = rr
                prev_commit[branch] = commit
            if stage in (1, 4, 10, 20) and (case_id, stage, "probe_continue") in by:
                verify_probes(recs, "program", model, case)
    return len(cases)


tasks = json.loads((DATA / "tasks.json").read_text())
CASES = {c["id"]: c for c in tasks["cases"]}
check(len(tasks["cases"]) == 80, "tasks: not 80 cases")
check(Counter(c["stratum"] for c in tasks["cases"]) == Counter({k: 16 for k in P.STRATA}),
      "tasks: stratum counts")
check(len({c["id"] for c in tasks["cases"]}) == 80, "tasks: duplicate case ids")
check(len(tasks["input_fingerprints"]) == 80 and len(set(tasks["input_fingerprints"])) == 80,
      "tasks: fingerprints not unique")
for idx, c in enumerate(tasks["cases"]):
    f = P.frame(c, 0)
    fp = P.digest(P.render(f, c["names"], f["constants"])["tokens"])
    check(fp == tasks["input_fingerprints"][idx], "tasks: fingerprint mismatch")

META = {}
ALL = {}
for model in ("A_s42", "B_s42"):
    for mode in ("assignment", "program"):
        folder = BASE / f"{mode}-{model}"
        m = json.loads((folder / "meta.json").read_text())
        check(m["status"] == "complete" and m["protocol"] == P.VERSION,
              f"{model} {mode}: meta incomplete")
        check(m["task_sha256"] == sha(DATA / "tasks.json"), f"{model} {mode}: task hash")
        check(m["registry_sha256"] == sha(DATA / "models.json"), f"{model} {mode}: registry hash")
        check(m["commit"] == PLAN_COMMIT,
              f"{model} {mode}: commit {m['commit']} != plan {PLAN_COMMIT}")
        check(m["records_sha256"] == sha(folder / "records.jsonl"),
              f"{model} {mode}: records hash")
        check(m["physical_nodes"] == 32 and m["padded_tokens"] == 256,
              f"{model} {mode}: fixed size")
        check(m["cases"] == [f"{s}:{i:03d}" for s in P.STRATA for i in range(m["per_stratum"])],
              f"{model} {mode}: cases not fixed first")
        recs = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()
                if line]
        check(len(recs) == m["records"], f"{model} {mode}: record count")
        cases_n = 5 * m["per_stratum"]
        probe_cases = 5 * m["probe_per_stratum"]
        if mode == "assignment":
            want_records = cases_n * (3 + 4 * 5) + probe_cases * 2 * 5
        else:
            probe_stages = sum(s in (1, 4, 10, 20) for s in range(1, m["stages"] + 1))
            want_records = cases_n * (m["stages"] * 3) + probe_cases * probe_stages * 5
        check(len(recs) == want_records,
              f"{model} {mode}: records {len(recs)} != expected {want_records}")
        META[(model, mode)] = m
        ALL[(model, mode)] = recs
        total = sum(len(r["trace"]["inputs"]) for r in recs if "trace" in r)
        if mode == "assignment":
            total += sum(len(r["trace"]["inputs"]) for r in recs
                         if r["stage"] == 0 and r["branch"] == "old_clean")
        check(total == m["actual_total_forwards"],
              f"{model} {mode}: forward accounting {total} != {m['actual_total_forwards']}")
        if mode == "assignment":
            verify_assignment(model, recs)
        else:
            verify_program(model, recs)

def csv_rows(name):
    with (BASE / "summary" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(s):
    if s == "" or s is None:
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        return int(s)
    except ValueError:
        return float(s)


# Coverage and common starts.
start_good = defaultdict(dict)
for model in ("A_s42", "B_s42"):
    for r in ALL[(model, "assignment")]:
        if r["branch"] == "old_clean":
            start_good[model][r["case"]] = r["baseline"]["nodes_first4_all"]
common = set.intersection(*(set(k for k, v in d.items() if v) for d in start_good.values()))
cov = {(r["mode"], r["model"]): r for r in csv_rows("coverage.csv")}
for model in ("A_s42", "B_s42"):
    for mode in ("assignment", "program"):
        r = cov[(mode, model)]
        candidates = 5 * META[(model, mode)]["per_stratum"]
        check(int(r["candidates"]) == candidates, f"coverage {model} {mode}: candidates")
        check(r["compare_models"] == "A_s42|B_s42", f"coverage {model} {mode}: compare set")
        if mode == "assignment":
            check(int(r["initial_first4_qualified"]) == sum(start_good[model].values()),
                  f"coverage {model} assignment: qualified")
            check(int(r["common_initial_first4"]) == len(common),
                  f"coverage {model} assignment: common")
        else:
            check(r["initial_first4_qualified"] == "" and r["common_initial_first4"] == "",
                  f"coverage {model} program: unexpected assignment columns")

# Rebuild summary groups exactly from the records.
groups = defaultdict(lambda: defaultdict(float))
for model in ("A_s42", "B_s42"):
    for mode in ("assignment", "program"):
        m = META[(model, mode)]
        for rec in ALL[(model, mode)]:
            complete = rec["status"] == "complete"
            ms = rec.get("metrics", {})
            cm = rec.get("conditional_metrics", {})
            cohorts = ["all_candidates"]
            if mode == "assignment" and rec["case"] in common:
                cohorts.append("common_initial_first4")
            for cohort in cohorts:
                key = (mode, model, rec["stratum"], rec["stage"], rec["branch"], cohort)
                g = groups[key]
                g["cases"] += 1
                g["observed"] += complete
                for reg in ("nodes", "terminal"):
                    n = ms.get(reg + "_n", P.NODES if reg == "nodes" else P.STRATA[rec["stratum"]][0])
                    g[reg + "_n"] += n
                    for metric in ("value", "first4"):
                        mnum = ms.get(reg + "_" + metric + "_n", 0)
                        g[reg + "_" + metric + "_n"] += mnum
                        g[reg + "_" + metric + "_macro_sum"] += mnum / n
                        g[reg + "_" + metric + "_all"] += ms.get(reg + "_" + metric + "_all", False)
                        g[reg + "_" + metric + "_final_all"] += ms.get(reg + "_" + metric + "_final_all", False)
                        g[reg + "_" + metric + "_final_macro_sum"] += ms.get(
                            reg + "_" + metric + "_final_n", 0) / n
                if cm:
                    g["conditional_observed"] += 1
                    g["conditional_nodes_value_macro_sum"] += cm["nodes_value_n"] / cm["nodes_n"]
                    g["conditional_terminal_all"] += cm["terminal_value_all"]
                if complete:
                    g["task_observed"] += 1
                    g["task_first4_macro_sum"] += ms.get("task_first4_n", 0) / max(ms.get("task_n", 1), 1)
summary_rows = {(r["mode"], r["model"], r["stratum"], int(r["stage"]), r["branch"], r["cohort"]): r
                for r in csv_rows("summary.csv")}
check(len(summary_rows) == len(groups), f"summary: rows {len(summary_rows)} != groups {len(groups)}")
for key, g in groups.items():
    r = summary_rows.get(key)
    check(r is not None, f"summary: missing group {key}")
    if r is None:
        continue
    check(num(r["cases"]) == g["cases"], f"summary {key}: cases")
    check(num(r["observed"]) == g["observed"], f"summary {key}: observed")
    for reg in ("nodes", "terminal"):
        for metric in ("value", "first4"):
            col = reg + "_" + metric
            check(num(r[col + "_n"]) == g[col + "_n"], f"summary {key}: {col}_n")
            check(abs(num(r[col + "_macro"]) - g[col + "_macro_sum"] / g["cases"]) < 1e-12,
                  f"summary {key}: {col}_macro")
            check(num(r["task_observed"]) == g.get("task_observed", 0), f"summary {key}: task_observed")
    check(num(r["conditional_observed"]) == g.get("conditional_observed", 0),
          f"summary {key}: conditional_observed")
    if g.get("conditional_observed"):
        check(abs(num(r["conditional_nodes_value_macro"]) -
                  g["conditional_nodes_value_macro_sum"] / g["conditional_observed"]) < 1e-12,
              f"summary {key}: conditional macro")
    if g.get("task_observed"):
        check(abs(num(r["task_first4_macro"]) -
                  g["task_first4_macro_sum"] / g["task_observed"]) < 1e-12,
              f"summary {key}: task macro")

# Program trajectory table.
prog_groups = defaultdict(list)
for model in ("A_s42", "B_s42"):
    for rec in ALL[(model, "program")]:
        if not rec["diagnostic"]:
            prog_groups[(model, rec["case"], rec["branch"])].append(rec)
prog_rows = {(r["model"], r["case"], r["branch"]): r for r in csv_rows("programs.csv")}
check(len(prog_rows) == len(prog_groups), f"programs: rows {len(prog_rows)} != groups {len(prog_groups)}")
for (model, case_id, branch), rr in prog_groups.items():
    rr.sort(key=lambda z: z["stage"])
    r = prog_rows[(model, case_id, branch)]
    check(int(r["stages"]) == META[(model, "program")]["stages"],
          f"programs {model} {case_id} {branch}: stages")
    first_bad = next((z["stage"] for z in rr if not z.get("metrics", {}).get(
        "terminal_value_final_all", False)), None)
    last = rr[-1]
    check(num(r["first_terminal_failure"]) == first_bad,
          f"programs {model} {case_id} {branch}: first failure")
    check(num(r["all_stage_terminal_success"]) == (first_bad is None),
          f"programs {model} {case_id} {branch}: all stage success")
    check(num(r["final_terminal_success"]) ==
          last.get("metrics", {}).get("terminal_value_final_all", False),
          f"programs {model} {case_id} {branch}: final success")
    check(num(r["last_two_terminal_success"]) ==
          last.get("metrics", {}).get("terminal_value_all", False),
          f"programs {model} {case_id} {branch}: last-two success")
    check(num(r["first_invalid_commit"]) ==
          next((z["stage"] for z in rr if z["status"] == "invalid_commit"), None),
          f"programs {model} {case_id} {branch}: invalid commit")
    check(num(r["final_cumulative_logical_nodes"]) == last["reference"]["cumulative_logical_nodes"],
          f"programs {model} {case_id} {branch}: cumulative nodes")
    check(num(r["final_required_ancestor_operations"]) ==
          last["reference"]["output_ancestor_operations"],
          f"programs {model} {case_id} {branch}: ancestor operations")
    check(num(r["final_logical_depth"]) == last["reference"]["logical_depth"],
          f"programs {model} {case_id} {branch}: logical depth")
    for limit in (128, 256, 512):
        at = next((z for z in rr if z["reference"]["cumulative_logical_nodes"] >= limit), None)
        check(num(r[f"at_least_{limit}_actual_nodes"]) ==
              (at["reference"]["cumulative_logical_nodes"] if at else None),
              f"programs {model} {case_id} {branch}: threshold {limit} nodes")
        check(num(r[f"at_least_{limit}_terminal_success"]) ==
              (at.get("metrics", {}).get("terminal_value_final_all", False) if at else None),
              f"programs {model} {case_id} {branch}: threshold {limit} success")

# Paired diagnostic intersections.
probes = {}
for model in ("A_s42", "B_s42"):
    for mode in ("assignment", "program"):
        for rec in ALL[(model, mode)]:
            if rec["diagnostic"]:
                correct = {i for i in rec["nodes"]
                           if all(x[i][3] == rec["truth4"][i][3] for x in rec["trace"]["outputs"][-2:])}
                probes[(mode, model, rec["case"], rec["stage"], rec["branch"])] = (correct, len(rec["nodes"]))
pair_rows = {(r["mode"], r["model"], r["case"], int(r["stage"]), r["branch"]): r
             for r in csv_rows("probe_pairs.csv")}
want_pairs = sum(1 for key in probes if key[-1] != "probe_continue")
check(len(pair_rows) == want_pairs, f"probe_pairs: rows {len(pair_rows)} != {want_pairs}")
for key, (got, total) in probes.items():
    if key[-1] == "probe_continue":
        continue
    ref = probes[(key[0], key[1], key[2], key[3], "probe_continue")][0]
    r = pair_rows.get(key)
    check(r is not None, f"probe_pairs: missing {key}")
    if r is None:
        continue
    check(num(r["nodes"]) == total, f"probe_pairs {key}: nodes")
    check(num(r["reference_correct"]) == len(ref), f"probe_pairs {key}: reference")
    check(num(r["branch_correct"]) == len(got), f"probe_pairs {key}: branch")
    check(num(r["repaired"]) == len(got - ref), f"probe_pairs {key}: repaired")
    check(num(r["damaged"]) == len(ref - got), f"probe_pairs {key}: damaged")

check(TOK.mod_val == 23 and TOK.n_vars >= 128, "tokenizer: mod/vars")
check(P.emitted_inputs(P.frame(CASES["w2_f2:000"], 1), [None, None]) is None,
      "invalid commit was not rejected at the boundary")
try:
    P.evaluate(P.frame(CASES["w2_f2:000"], 1)["parents"], [None] * 6)
    check(False, "evaluate accepted None inputs")
except ValueError:
    pass

if errors:
    print(f"\n{len(errors)} FAILURES")
    sys.exit(1)
result = dict(
    status="ok",
    errors=0,
    bundle=str(BASE),
    plan_commit=PLAN_COMMIT,
    tasks_sha256=sha(DATA / "tasks.json"),
    models_sha256=sha(DATA / "models.json"),
    tokens_mod=TOK.mod_val,
    tokenizer_n_vars=TOK.n_vars,
    records={f"{m}/{mode}": META[(m, mode)]["records"]
             for m in ("A_s42", "B_s42") for mode in ("assignment", "program")},
    node_values_checked=sum(len(r.get("nodes", [])) for recs in ALL.values() for r in recs),
    reference_records=sum("reference" in r for recs in ALL.values() for r in recs),
    conditional_records=sum("conditional_metrics" in r for recs in ALL.values() for r in recs),
    probe_records=len(probes),
    common_initial_first4=len(common),
    summary_groups=len(summary_rows),
    program_rows=len(prog_rows),
    probe_pair_rows=len(pair_rows),
)
if ARGS.report:
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, sort_keys=True))