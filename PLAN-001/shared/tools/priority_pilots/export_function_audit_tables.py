#!/usr/bin/env python3
"""Re-export C function-audit tables from the existing CPU audit.

The executable source (function_audit.py) only writes the large per-edge and
quadrant tables.  This exporter adds the small committed tables, the explicit
input/output joint counts, and the focal-q event table.  It never loads a
model, never writes to the source run directories, and never rewrites the
first submitted tables.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import CSV, jsonl, read_json, write_json

DISPLAY = {
    "initial": "C（原始）",
    "hold": "C（保持后训练）",
    "noise": "C（噪声后训练）",
    "semantic": "C（语义后训练）",
}
PHENOMENON = {
    "phenomenon_1": "现象①：u输入仍错、u输出已对、q输出错",
    "phenomenon_2": "现象②：u输入/输出正确、q输入错、q输出错（controlled）",
    "phenomenon_3": "现象③：u输入/输出与q输入正确、其余父与参照正确、q输出错（controlled）",
}
ORDER_SOURCE = {"pair": 0, "sustained": 1, "noise": 2}
ORDER_ARM = {"initial": 0, "hold": 1, "noise": 2, "semantic": 3}
ORDER_TIMING = {"hold": 0, "late": 1}
COMPARISONS = [
    ("pair", "unchanged", "u_wrong"),
    ("pair", "q_wrong", "both_wrong"),
    ("pair", "q_same", "both_same"),
    ("sustained", "free_q_clean", "wrong_q_clean"),
    ("sustained", "free_q_wrong", "wrong_q_wrong"),
    ("sustained", "same_q_clean", "wrong_q_clean"),
    ("sustained", "same_q_wrong", "wrong_q_wrong"),
    ("noise", "free", "wrong"),
]
for _b in [f"noise_s{s:g}_r{r}" for s in (0.5, 1, 2, 3, 4, 6) for r in (0, 1)]:
    COMPARISONS.append(("noise", "free", _b))
COMPARISON_SET = {(s, a) for s, a, _ in COMPARISONS} | {(s, b) for s, _, b in COMPARISONS}

QUADRANT_BRANCHES = {
    "pair": {"u_wrong", "both_wrong", "q_wrong", "unchanged"},
    "sustained": {"wrong_q_clean", "wrong_q_wrong", "free_q_clean", "free_q_wrong"},
}
QUADRANT_ROUNDS = {1, 2, 3, 4, 5, 6, 8, 10, 16, 18, 20}
QUADRANT_CONTEXTS = {
    "all",
    "other_parents_and_reference_correct",
    "controlled_q_input_correct",
    "controlled_q_input_wrong",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def branch_specs(meta):
    """Normalize pair's tag-list branch metadata and stress's spec-list."""
    out = {}
    for source, item in meta["sources"].items():
        raw = item["source_meta"]["branches"]
        specs = {}
        if isinstance(raw, dict):
            for name, tags in raw.items():
                tags = set(tags)
                u_mode = ("wrong" if "u_wrong" in tags else
                          "same" if "u_same" in tags else
                          "original" if "u_original" in tags else "free")
                specs[name] = dict(name=name, u_mode=u_mode,
                                   q_wrong=("q_wrong" in tags))
        else:
            for spec in raw:
                specs[spec["name"]] = dict(spec)
        out[source] = specs
    return out


def quadrant_branch(source, branch):
    if source == "noise":
        return branch in ("wrong", "free", "same") or branch.startswith("noise_s")
    return branch in QUADRANT_BRANCHES[source]


def edge_joint(row):
    """Return the named intersections for one per-edge/round row."""
    ui = row["u_input_correct"] == "True"
    uo = row["u_output_correct"] == "True"
    qi = row["q_input_correct"] == "True"
    qo = row["q_output_correct"] == "True"
    ctrl = (row["other_parents_input_correct"] == "True" and
            row["q_reference_correct"] == "True")
    return dict(
        u_input_ok=ui, u_output_ok=uo, q_input_ok=qi, q_output_ok=qo,
        controlled=ctrl,
        u_in_out_ok=ui and uo,
        all_readouts_ok=ui and uo and qi,
        u_out_ok_q_out_wrong=uo and not qo,
        u_in_ok_q_out_wrong=ui and not qo,
        u_in_out_ok_q_out_wrong=ui and uo and not qo,
        q_in_ok_q_out_wrong=qi and not qo,
        controlled_u_out_ok_q_out_wrong=ctrl and uo and not qo,
        controlled_u_out_ok_q_out_wrong_q_in_ok=ctrl and uo and not qo and qi,
        controlled_u_out_ok_q_out_wrong_q_in_bad=ctrl and uo and not qo and not qi,
        controlled_u_in_out_ok_q_out_wrong=ctrl and ui and uo and not qo,
        controlled_phenomenon_3=ctrl and ui and uo and qi and not qo,
        q_output_nonnumeric=row["q_output_numeric"] != "True",
    )


def event_flags(flags):
    """Output flags start at post_round 1; injection instant is not included."""
    bad = next((i for i, good in enumerate(flags) if not good), None)
    if bad is None:
        return dict(ever_wrong=False, returned=False, stable_return=False,
                    reerror=False, first_wrong=None, first_return=None)
    back = next((i for i in range(bad + 1, len(flags)) if flags[i]), None)
    stable = next((i for i in range(bad + 1, len(flags)) if all(flags[i:])), None)
    return dict(ever_wrong=True, returned=back is not None,
                stable_return=stable is not None,
                reerror=back is not None and not all(flags[back:]),
                first_wrong=bad + 1, first_return=None if back is None else back + 1)


def example_key(row):
    return (ORDER_SOURCE[row["source"]], ORDER_ARM[row["arm"]], int(row["graph_id"]),
            ORDER_TIMING[row["timing"]], int(row["duration"]), row["branch"],
            int(row["q"]), int(row["post_round"]))


def read_source_record(records, source, arm, graph_id, timing, duration):
    key = (source, arm, graph_id, timing, duration)
    if key in records:
        return records[key]
    raise ValueError("selected example record not found: " + str(key))


def build_expression(record, node, parents, truth, pos, inp):
    ps = parents.get(str(node), parents.get(node))
    if not ps:
        return f"x{truth[pos.index(node)][1]} = {truth[pos.index(node)][3]} (constant)"
    return (f"x{truth[pos.index(node)][1]} = " +
            " + ".join(f"x{truth[pos.index(p)][1]}({truth[pos.index(p)][3]})" for p in ps) +
            f" mod23 = {truth[pos.index(node)][3]}")


def build_examples(audit_dir, meta, specs, out_dir):
    """Select focal-q examples with a fixed order and distinct graphs."""
    candidates = defaultdict(list)
    with (audit_dir / "edge_rounds.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row["source"]
            spec = specs[source][row["branch"]]
            if (row["focal_q"] != "True" or row["common_value_qualified"] != "True"
                    or spec["u_mode"] not in ("wrong", "noise")):
                continue
            ui = row["u_input_correct"] == "True"
            uo = row["u_output_correct"] == "True"
            qi = row["q_input_correct"] == "True"
            qo = row["q_output_correct"] == "True"
            ctrl = (row["other_parents_input_correct"] == "True" and
                    row["q_reference_correct"] == "True")
            tag = None
            if (not ui) and uo and (not qo):
                tag = "phenomenon_1"
            elif ui and uo and (not qi) and (not qo) and ctrl:
                tag = "phenomenon_2"
            elif ui and uo and qi and (not qo) and ctrl:
                tag = "phenomenon_3"
            if tag:
                candidates[(source, tag)].append(row)

    chosen = {}
    for key, rows in candidates.items():
        # Keep source-record order for selection (deterministic); sort only the
        # two selected cases for presentation.
        picked, graphs, edges = [], set(), set()
        for row in rows:
            edge = (row["arm"], row["timing"], row["duration"],
                    row["branch"], row["q"])
            if row["graph_id"] in graphs or edge in edges:
                continue
            graphs.add(row["graph_id"])
            edges.add(edge)
            picked.append(row)
            if len(picked) == 2:
                break
        chosen[key] = sorted(picked, key=example_key)

    records = {}
    needed = defaultdict(set)
    for rows in chosen.values():
        for row in rows:
            needed[row["source"]].add((row["arm"], row["graph_id"],
                                       row["timing"], row["duration"]))
    for source, targets in needed.items():
        path = Path(meta["sources"][source]["path"]) / "records.jsonl"
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                r = json.loads(line)
                target = (r["arm"], str(r["graph_id"]), r["timing"],
                          str(r.get("duration") or 1))
                if target in targets:
                    records[(source, *target)] = r

    out_rows = []
    for (source, tag), rows in sorted(
            chosen.items(), key=lambda kv: (ORDER_SOURCE[kv[0][0]], kv[0][1])):
        for s in rows:
            record = read_source_record(records, source, s["arm"], s["graph_id"],
                                        s["timing"], s["duration"])
            pos, truth, parents = record["positions"], record["truth4"], record["parents"]
            branch = record["branches"][s["branch"]]
            step = int(s["post_round"]) - 1
            if "inputs" in branch:
                inp = branch["inputs"][step]
            else:
                inp = ([branch["initial"]] + branch["outputs"][:-1])[step]
            outv = branch["outputs"][step]
            u, q = int(record["u"]), int(s["q"])
            def values(node):
                ps = parents.get(str(node), parents.get(node)) or []
                return "; ".join(
                    f"x{truth[pos.index(p)][1]}(truth={truth[pos.index(p)][3]},input={inp[pos.index(p)][3]})"
                    for p in ps) or "(constant)"
            out_rows.append(dict(
                phenomenon=tag, phenomenon_label=PHENOMENON[tag], source=source,
                arm=s["arm"], model_display=DISPLAY[s["arm"]],
                graph_id=int(s["graph_id"]), timing=s["timing"],
                duration=int(s["duration"]), branch=s["branch"],
                post_round=int(s["post_round"]), phase=s["phase"],
                release_round=s["release_round"] or None,
                u=u, q=q, u_expr=build_expression(record, u, parents, truth, pos, inp),
                q_expr=build_expression(record, q, parents, truth, pos, inp),
                u_truth=truth[pos.index(u)][3], u_input=inp[pos.index(u)][3],
                u_output=outv[pos.index(u)][3],
                q_truth=truth[pos.index(q)][3], q_input=inp[pos.index(q)][3],
                q_output=outv[pos.index(q)][3],
                u_parent_values=values(u), q_parent_values=values(q),
                input_parent_sum=s["input_parent_sum"], reference=s["reference"]))
    return out_rows


def write_quadrants(audit_dir, out_dir):
    rows = []
    with (audit_dir / "quadrants.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        if not quadrant_branch(r["source"], r["branch"]):
            continue
        if r["cohort"] not in ("all_recorded", "common_value"):
            continue
        if r["context"] not in QUADRANT_CONTEXTS or int(r["post_round"]) not in QUADRANT_ROUNDS:
            continue
        groups[(r["source"], r["branch"], r["timing"], r["duration"],
                r["post_round"], r["axis"], r["context"], r["cohort"])].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        cells = {}
        for r in rs:
            k = (r["u_read_correct"] == "True", r["q_output_correct"] == "True")
            cells.setdefault(k, [0, []])
            cells[k][0] += int(r["cell_n"])
            cells[k][1].append(float(r["graph_macro_fraction"]))
        n = sum(v[0] for v in cells.values())
        def cell(t, q):
            v = cells.get((t, q), [0, []])
            return v[0], (sum(v[1]) / len(v[1]) if v[1] else "")
        tt, ttm = cell(True, True); tf, tfm = cell(True, False)
        ft, ftm = cell(False, True); ff, ffm = cell(False, False)
        out.append(dict(source=key[0], branch=key[1], timing=key[2], duration=key[3],
                        post_round=key[4], axis=key[5], context=key[6], cohort=key[7],
                        models=len({r["arm"] for r in rs}), edge_n=n,
                        TT=tt, TF=tf, FT=ft, FF=ff,
                        micro_TT=tt / n, micro_TF=tf / n,
                        micro_FT=ft / n, micro_FF=ff / n,
                        macro_TT=ttm, macro_TF=tfm, macro_FT=ftm, macro_FF=ffm))
    write_csv(out_dir / "c-function-audit-quadrants.csv", out)


def write_csv(path, rows):
    if not rows:
        raise ValueError("refusing to write empty table: " + str(path))
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", type=Path, required=True,
                   help="directory written by function_audit.py")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="new directory for the report tables; must not exist")
    a = p.parse_args()
    meta = read_json(a.audit_dir / "meta.json")
    if meta.get("status") != "complete" or meta.get("protocol") != "C_function_audit_v1":
        raise ValueError("incomplete or unsupported audit")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    specs = branch_specs(meta)

    source_stats = defaultdict(lambda: defaultdict(int))
    situation = defaultdict(lambda: defaultdict(int))
    joint = defaultdict(lambda: defaultdict(int))
    q_store = {}
    focal_seqs = defaultdict(list)

    with (a.audit_dir / "edge_rounds.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source, arm, branch = row["source"], row["arm"], row["branch"]
            spec = specs[source][branch]
            flags = edge_joint(row)
            edge_sets = ("all", "focal") if row["focal_q"] == "True" else ("all",)
            for edge_set in edge_sets:
                jkey = (source, arm, row["timing"], row["duration"], branch,
                        int(row["post_round"]), edge_set)
                for k, v in flags.items():
                    if isinstance(v, bool):
                        joint[jkey][k] += int(v)
                joint[jkey]["edges"] += 1
                joint[jkey]["common_value_edges"] += int(row["common_value_qualified"] == "True")
            if row["focal_q"] == "True":
                focal_seqs[(source, arm, row["timing"], row["duration"], branch,
                            row["graph_id"], row["q"])].append(
                    (int(row["post_round"]), row["q_output_correct"] == "True"))
            s = source_stats[source]
            s["edge_rounds"] += 1
            s["controlled_edges"] += int(flags["controlled"])
            s["missing_parent_sum"] += int(row["input_parent_sum"] in ("", "None"))
            s["other_parents_wrong"] += int(row["other_parents_input_correct"] != "True")
            s["reference_wrong"] += int(row["q_reference_correct"] != "True")
            s["q_nonnumeric"] += int(row["q_output_numeric"] != "True")
            if spec["u_mode"] in ("wrong", "noise"):
                c = situation[(source, arm, row["timing"], row["duration"], branch)]
                c["intervention_edge_rounds"] += 1
                if (not flags["u_input_ok"]) and flags["u_output_ok"] and not flags["q_output_ok"]:
                    c["phenomenon_1"] += 1
                if flags["controlled"] and flags["u_out_ok_q_out_wrong"]:
                    c["controlled_tf"] += 1
                    c["controlled_tf_q_input_ok"] += int(flags["q_input_ok"])
                    c["controlled_tf_q_input_wrong"] += int(not flags["q_input_ok"])
                if flags["controlled"] and flags["u_in_out_ok_q_out_wrong"] and not flags["q_input_ok"]:
                    c["phenomenon_2"] += 1
                if flags["controlled_phenomenon_3"]:
                    c["phenomenon_3"] += 1
            if (source, branch) in COMPARISON_SET:
                key = (source, arm, row["graph_id"], row["timing"], row["duration"],
                       row["q"], int(row["post_round"]))
                q_store[(branch, *key)] = flags["q_output_ok"]

    # source summary
    summary = []
    for source, v in meta["sources"].items():
        sm = v["source_meta"]
        coverage_path = Path(v["path"]) / "coverage.jsonl"
        cov = [r for r in jsonl(coverage_path)] if coverage_path.exists() else []
        s = source_stats[source]
        summary.append(dict(
            source=source, candidate_graphs=sm["candidate_graphs"],
            coverage_rows=len(cov),
            coverage_selected=sum(r.get("status") == "selected" for r in cov),
            coverage_nonselected=sum(r.get("status") != "selected" for r in cov),
            records=v["records"], paired_graph_timings=v["paired_graph_timings"],
            common_graph_timings=v["common_graph_timings"],
            value_qualified_records=sum(bool(r.get("value_qualified")) for r in cov
                                        if r.get("status") == "selected"),
            horizon=sm.get("horizon"),
            durations=";".join(map(str, sm.get("durations", [1]))),
            edge_round_rows=s["edge_rounds"], controlled_edge_rounds=s["controlled_edges"],
            missing_parent_sum=s["missing_parent_sum"],
            other_parents_wrong=s["other_parents_wrong"],
            reference_wrong=s["reference_wrong"], q_nonnumeric=s["q_nonnumeric"],
            meta_sha256=v["meta_sha256"], records_sha256=v["records_sha256"],
            coverage_sha256=v["coverage_sha256"]))
    write_csv(a.out_dir / "c-function-audit-source-summary.csv", summary)

    # phenomena counts
    rows = []
    for key in sorted(situation):
        row = dict(zip(("source", "arm", "timing", "duration", "branch"), key))
        row["model_display"] = DISPLAY[key[1]]
        row.update({k: situation[key].get(k, 0) for k in
                    ("intervention_edge_rounds", "phenomenon_1", "phenomenon_2",
                     "phenomenon_3", "controlled_tf", "controlled_tf_q_input_ok",
                     "controlled_tf_q_input_wrong")})
        rows.append(row)
    write_csv(a.out_dir / "c-function-audit-situation-counts.csv", rows)

    # joint counts
    fields = ("edges", "common_value_edges", "controlled", "u_input_ok", "u_output_ok",
              "q_input_ok", "q_output_ok", "u_in_out_ok", "all_readouts_ok",
              "u_out_ok_q_out_wrong", "u_in_ok_q_out_wrong", "u_in_out_ok_q_out_wrong",
              "q_in_ok_q_out_wrong", "controlled_u_out_ok_q_out_wrong",
              "controlled_u_out_ok_q_out_wrong_q_in_ok",
              "controlled_u_out_ok_q_out_wrong_q_in_bad",
              "controlled_u_in_out_ok_q_out_wrong", "controlled_phenomenon_3",
              "q_output_nonnumeric")
    rows = []
    for key in sorted(joint):
        row = dict(zip(("source", "arm", "timing", "duration", "branch",
                        "post_round", "edge_set"), key))
        row["model_display"] = DISPLAY[key[1]]
        row["u_mode"] = specs[key[0]][key[4]]["u_mode"]
        row["q_direct_injection"] = bool(specs[key[0]][key[4]]["q_wrong"])
        row.update({k: joint[key].get(k, 0) for k in fields})
        rows.append(row)
    write_csv(a.out_dir / "c-function-audit-joint.csv", rows)

    # matched branch comparisons
    rows = []
    for source, ref, target in COMPARISONS:
        tmp = defaultdict(lambda: defaultdict(int))
        for key, val in q_store.items():
            branch, src, arm, gid, timing, duration, q, post = key
            if branch != ref or src != source:
                continue
            other = (target, src, arm, gid, timing, duration, q, post)
            if other not in q_store:
                continue
            val2 = q_store[other]
            c = tmp[(arm, timing, duration, post)]
            c["n"] += 1
            c["harm"] += int(val and not val2)
            c["help"] += int((not val) and val2)
            c["both_bad"] += int((not val) and (not val2))
            c["both_ok"] += int(val and val2)
        for key, c in sorted(tmp.items()):
            rows.append(dict(source=source, reference=ref, target=target,
                             arm=key[0], model_display=DISPLAY[key[0]],
                             timing=key[1], duration=key[2], post_round=key[3],
                             n=c["n"], harm=c["harm"], help=c["help"],
                             both_bad=c["both_bad"], both_ok=c["both_ok"]))
    write_csv(a.out_dir / "c-function-audit-matched.csv", rows)

    # events, all child edges (same semantics as the first submission)
    ev = defaultdict(lambda: defaultdict(int))
    with (a.audit_dir / "events.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = ev[(r["source"], r["arm"], r["branch"])]
            c["q_edges"] += 1
            c["ever_wrong"] += int(bool(r["first_wrong_output"]))
            c["returned"] += int(bool(r["first_return_output"]))
            c["stable_return"] += int(bool(r["stable_return_output"]))
            c["reerror_after_return"] += int(r["reerror_after_return"] == "True")
            c["target_read_correct_child_wrong_rounds"] += int(
                r["target_read_correct_child_wrong_rounds"])
    rows = []
    for key in sorted(ev):
        row = dict(zip(("source", "arm", "branch"), key))
        row["model_display"] = DISPLAY[key[1]]
        row.update({k: ev[key].get(k, 0) for k in
                    ("q_edges", "ever_wrong", "returned", "stable_return",
                     "reerror_after_return", "target_read_correct_child_wrong_rounds")})
        rows.append(row)
    write_csv(a.out_dir / "c-function-audit-events.csv", rows)

    # focal-q events, explicitly separated from all-child events
    focal = defaultdict(lambda: defaultdict(int))
    for key, seq in focal_seqs.items():
        source, arm, timing, duration, branch, gid, q = key
        spec = specs[source][branch]
        seq.sort()
        flags = [x[1] for x in seq]
        e = event_flags(flags)
        c = focal[(source, arm, timing, duration, branch)]
        c["focal_edges"] += 1
        c["window_rounds"] = max(c["window_rounds"], len(seq))
        c["direct_q_injection_edges"] += int(spec["q_wrong"])
        c["direct_u_injection_edges"] += int(spec["u_mode"] in ("wrong", "noise"))
        if spec["q_wrong"]:
            c["direct_q_first_output_correct"] += int(flags[0])
            c["direct_q_first_output_wrong"] += int(not flags[0])
            c["direct_q_ever_wrong"] += int(e["ever_wrong"])
            c["direct_q_returned_after_ever_wrong"] += int(e["returned"])
            c["direct_q_reerror_after_return"] += int(e["reerror"])
        c["ever_wrong"] += int(e["ever_wrong"])
        c["returned_after_ever_wrong"] += int(e["returned"])
        c["stable_return"] += int(e["stable_return"])
        c["reerror_after_return"] += int(e["reerror"])
        if e["first_wrong"] is not None:
            c["first_wrong_min"] = (e["first_wrong"] if "first_wrong_min" not in c
                                    else min(c["first_wrong_min"], e["first_wrong"]))
            c["first_wrong_max"] = max(c.get("first_wrong_max", e["first_wrong"]),
                                       e["first_wrong"])
        if e["first_return"] is not None:
            c["first_return_min"] = (e["first_return"] if "first_return_min" not in c
                                     else min(c["first_return_min"], e["first_return"]))
            c["first_return_max"] = max(c.get("first_return_max", e["first_return"]),
                                        e["first_return"])
    # target-read-correct/child-wrong uses u_output, so it needs the per-edge table.
    target_counts = defaultdict(int)
    with (a.audit_dir / "edge_rounds.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["focal_q"] != "True":
                continue
            k = (row["source"], row["arm"], row["timing"], row["duration"], row["branch"])
            if row["u_output_correct"] == "True" and row["q_output_correct"] != "True":
                target_counts[k] += 1
    rows = []
    for key in sorted(focal):
        c = focal[key]
        row = dict(zip(("source", "arm", "timing", "duration", "branch"), key))
        row["model_display"] = DISPLAY[key[1]]
        row["u_mode"] = specs[key[0]][key[4]]["u_mode"]
        row["q_direct_injection"] = bool(specs[key[0]][key[4]]["q_wrong"])
        row.update(dict(
            focal_edges=c["focal_edges"],
            window_rounds=c["window_rounds"],
            direct_q_injection_edges=c["direct_q_injection_edges"],
            direct_u_injection_edges=c["direct_u_injection_edges"],
            direct_q_first_output_correct=c["direct_q_first_output_correct"],
            direct_q_first_output_wrong=c["direct_q_first_output_wrong"],
            direct_q_ever_wrong=c["direct_q_ever_wrong"],
            direct_q_returned_after_ever_wrong=c["direct_q_returned_after_ever_wrong"],
            direct_q_reerror_after_return=c["direct_q_reerror_after_return"],
            ever_wrong=c["ever_wrong"],
            never_wrong=c["focal_edges"] - c["ever_wrong"],
            returned_after_ever_wrong=c["returned_after_ever_wrong"],
            never_returned=c["ever_wrong"] - c["returned_after_ever_wrong"],
            stable_return=c["stable_return"],
            reerror_after_return=c["reerror_after_return"],
            first_wrong_min=c.get("first_wrong_min"),
            first_wrong_max=c.get("first_wrong_max"),
            first_return_min=c.get("first_return_min"),
            first_return_max=c.get("first_return_max"),
            target_read_correct_child_wrong_rounds=target_counts[key]))
        rows.append(row)
    write_csv(a.out_dir / "c-function-audit-events-focal.csv", rows)

    # quadrants are pooled; keep the exact old filter/pooling semantics
    write_quadrants(a.audit_dir, a.out_dir)

    # examples
    ex_rows = build_examples(a.audit_dir, meta, specs, a.out_dir)
    write_csv(a.out_dir / "c-function-audit-examples.csv", ex_rows)

    # audit
    def pooled_joint(source, branch, timing, duration, post_round, edge_set):
        keys = [k for k in joint
                if k[0] == source and k[2] == timing and k[3] == str(duration)
                and k[4] == branch and k[5] == post_round and k[6] == edge_set]
        if len(keys) != len(DISPLAY):
            raise ValueError("joint pooling did not find four versions")
        fields = ("edges", "controlled", "u_in_out_ok_q_out_wrong",
                  "controlled_u_out_ok_q_out_wrong",
                  "controlled_u_out_ok_q_out_wrong_q_in_ok",
                  "controlled_u_out_ok_q_out_wrong_q_in_bad",
                  "controlled_u_in_out_ok_q_out_wrong",
                  "controlled_phenomenon_3")
        return {f: sum(joint[k].get(f, 0) for k in keys) for f in fields}
    checks = dict(
        pair_u_wrong_hold_r2=pooled_joint("pair", "u_wrong", "hold", 1, 2, "all"),
        sustained_wrong_q_clean_hold_L2_R1=pooled_joint(
            "sustained", "wrong_q_clean", "hold", 2, 3, "all"))
    raw = {n: dict(path=str(a.audit_dir / n), sha256=sha(a.audit_dir / n))
           for n in ("edge_rounds.csv", "quadrants.csv", "events.csv",
                     "coverage.csv", "meta.json") if (a.audit_dir / n).exists()}
    exports = []
    for path in sorted(a.out_dir.glob("c-function-audit-*.csv")):
        exports.append(dict(path=str(path), sha256=sha(path)))
    write_json(a.out_dir / "c-function-audit-audit.json", dict(
        status="audit_summary", protocol="C_function_audit_v2",
        report="docs/records/2026-11-18/priority-function-audit.md",
        code_commit="3454e07",
        source_audit=str(a.audit_dir),
        display_names=DISPLAY,
        phenomena=PHENOMENON,
        sources={s: dict(path=v["path"], meta_sha256=v["meta_sha256"],
                         records_sha256=v["records_sha256"],
                         coverage_sha256=v["coverage_sha256"],
                         records=v["records"],
                         paired_graph_timings=v["paired_graph_timings"],
                         common_graph_timings=v["common_graph_timings"])
                 for s, v in meta["sources"].items()},
        raw=raw,
        code={n: sha(Path(__file__).with_name(n)) for n in
              ("function_audit.py", "io_utils.py", "export_function_audit_tables.py")},
        exports=exports,
        checks=checks,
        notes=["No model was loaded; this is a CPU re-export of existing records.",
               "Joint counts are direct intersections from edge_rounds.csv; checks pool the four stored versions.",
               "Focal events use focal q only and start at post_round 1; no injection instant.",
               "The first submitted CSVs are not overwritten."]))
    print(json.dumps({"out_dir": str(a.out_dir), "tables": len(exports)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
