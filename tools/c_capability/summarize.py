#!/usr/bin/env python3
"""Summarize C0 capability runs into CSVs, examples and a short report.

Design: docs/records/2026-11-14/c0-capability-exploration-plan.md

This script is torch-free.  It reads the per-graph npz files written by
run_capability.py and the index.jsonl alongside them.  It does not rerun a
model or change any checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from . import c_common
except ImportError:  # direct script execution
    import c_common

import numpy as np

NODE_TYPES = ("leaf", "internal", "sink")


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(x) for x in value.split(",") if x.strip()}))
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon required")
    return horizons


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def ratio(num: float, den: float) -> float | None:
    return (num / den) if den else None


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def depth_bucket(depth: int) -> str:
    if depth <= 0:
        return "0"
    if depth <= 3:
        return "1-3"
    if depth <= 6:
        return "4-6"
    if depth <= 9:
        return "7-9"
    if depth <= 12:
        return "10-12"
    return "13+"


def load_run_meta(weight_dir: Path) -> dict[str, Any]:
    return c_common.read_json(weight_dir / "run_meta.json")


def iter_graphs(weight_dir: Path):
    index_path = weight_dir / "index.jsonl"
    for record in c_common.read_jsonl(index_path):
        graph_path = weight_dir / record["path"]
        with np.load(graph_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        yield record, arrays


def load_all(run_root: Path):
    runs = {}
    for meta_path in sorted(run_root.glob("*/run_meta.json")):
        weight_dir = meta_path.parent
        weight = weight_dir.name
        meta = load_run_meta(weight_dir)
        graphs = list(iter_graphs(weight_dir))
        runs[weight] = dict(meta=meta, graphs=graphs, dir=weight_dir)
    if not runs:
        raise FileNotFoundError(f"no run_meta.json under {run_root}")
    return runs



def add_node_counts(acc, values, first4):
    """One graph per call: macro and micro must use the same eligible nodes."""
    n = int(values.size)
    vc, fc = int(values.sum()), int(first4.sum())
    for key, value in (('vc', vc), ('vn', n), ('f4c', fc), ('f4n', n)):
        acc[key] = acc.get(key, 0) + value
    if n:
        acc['vg'] = acc.get('vg', 0.0) + vc / n
        acc['fg'] = acc.get('fg', 0.0) + fc / n
        acc['support'] = acc.get('support', 0) + 1
    else:
        acc['zero_support'] = acc.get('zero_support', 0) + 1


def macro_fields(acc):
    support = acc.get('support', 0)
    return dict(value_node_acc_graph_equal=c_common.fmt_float(ratio(acc.get('vg', 0), support)),
                first4_node_acc_graph_equal=c_common.fmt_float(ratio(acc.get('fg', 0), support)),
                n_graphs_with_nodes=support, n_graphs_without_nodes=acc.get('zero_support', 0))


def summarize(run_root: Path, horizons: tuple[int, ...]):
    runs = load_all(run_root)

    # round_acc[(weight, stratum)][round] = counts
    round_acc: dict[tuple[str, str], dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: dict(vc=0, vn=0, f4c=0, f4n=0,
                                        gv=0, gf=0, n=0)))
    type_acc: dict[tuple[str, str, int, str], dict[str, int]] = defaultdict(
        lambda: dict(vc=0, vn=0, f4c=0, f4n=0))
    depth_acc: dict[tuple[str, str, int, str], dict[str, int]] = defaultdict(
        lambda: dict(vc=0, vn=0, f4c=0, f4n=0))
    graph_rows: list[dict[str, Any]] = []
    completion_tV: dict[tuple[str, str], list[int]] = defaultdict(list)
    completion_t4: dict[tuple[str, str], list[int]] = defaultdict(list)
    n_graphs: dict[tuple[str, str], int] = defaultdict(int)
    drift_rows: list[dict[str, Any]] = []
    relative_acc = defaultdict(lambda: dict(n=0, correct=0, retained=0, wrong_nodes=0, total_nodes=0))
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for weight, run in runs.items():
        for record, arrays in run["graphs"]:
            stratum = record["stratum"]
            key = (weight, stratum)
            n_graphs[key] += 1
            value_correct = arrays["value_correct"].astype(bool)
            first4_correct = arrays["first4_correct"].astype(bool)
            graph_value_all = arrays["graph_value_all"].astype(bool)
            graph_first4_all = arrays["graph_first4_all"].astype(bool)
            rounds, n_nodes = value_correct.shape
            tV = int(arrays["tV"])
            t4 = int(arrays["t4"])
            tV = None if tV < 0 else tV
            t4 = None if t4 < 0 else t4
            if tV is not None:
                completion_tV[key].append(tV)
            if t4 is not None:
                completion_t4[key].append(t4)

            for round_idx in range(rounds):
                acc = round_acc[key][round_idx]
                add_node_counts(acc, value_correct[round_idx], first4_correct[round_idx])
                acc["gv"] += int(graph_value_all[round_idx])
                acc["gf"] += int(graph_first4_all[round_idx])
                acc["n"] += 1

            masks = dict(
                leaf=arrays["is_leaf"].astype(bool),
                internal=arrays["is_internal"].astype(bool),
                sink=arrays["is_sink"].astype(bool),
            )
            depths = arrays["depth"].astype(int)
            for H in horizons:
                if H > rounds:
                    continue
                idx = H - 1
                for node_type, mask in masks.items():
                    cell = type_acc[(weight, stratum, H, node_type)]
                    add_node_counts(cell, value_correct[idx][mask], first4_correct[idx][mask])
                # Visit each bucket once PER GRAPH, including empty buckets.
                buckets = np.asarray([depth_bucket(int(d)) for d in depths])
                for bucket in ('0', '1-3', '4-6', '7-9', '10-12', '13+'):
                    cell = depth_acc[(weight, stratum, H, bucket)]
                    mask = buckets == bucket
                    add_node_counts(cell, value_correct[idx][mask], first4_correct[idx][mask])

            value_drift = c_common.completion_drift(graph_value_all, tV)
            first4_drift = c_common.completion_drift(graph_first4_all, t4)
            other_fields = np.stack([
                arrays['pred_' + f] == arrays['true_' + f][None, :]
                for f in ('syntax', 'variable', 'operation')], axis=-1)
            other = c_common.other_field_drift(graph_value_all, graph_first4_all, tV,
                                               other_fields_correct=other_fields)
            for metric, completion, series, nodes_correct in (
                    ('value', tV, graph_value_all, value_correct),
                    ('first4', t4, graph_first4_all, first4_correct)):
                if completion is None:
                    continue
                retained = True
                for absolute in range(completion, rounds):
                    lag = absolute - completion + 1
                    cell = relative_acc[(weight, stratum, metric, lag)]
                    retained = retained and bool(series[absolute])
                    cell['n'] += 1
                    cell['correct'] += int(series[absolute])
                    cell['retained'] += int(retained)
                    cell['wrong_nodes'] += int((~nodes_correct[absolute]).sum())
                    cell['total_nodes'] += n_nodes
            drift = dict(
                weight=weight, stratum=stratum, graph_id=record["graph_id"],
                tV=tV, t4=t4,
                value_post_rounds=value_drift["post_rounds"],
                value_ever_degraded=value_drift["ever_degraded"],
                value_first_degradation_rel_round=value_drift["first_degradation_rel_round"],
                value_recovered=value_drift["recovered"],
                value_final_correct=value_drift["final_correct"],
                first4_post_rounds=first4_drift["post_rounds"],
                first4_ever_degraded=first4_drift["ever_degraded"],
                first4_first_degradation_rel_round=first4_drift["first_degradation_rel_round"],
                first4_recovered=first4_drift["recovered"],
                first4_final_correct=first4_drift["final_correct"],
                other_only_ever=other["ever_other_only_degraded"],
                other_only_first_rel_round=other["first_other_only_rel_round"],
                other_only_final_value_correct=other["final_value_correct"],
                other_only_final_first4_correct=other["final_first4_correct"],
            )
            for prefix, result in (('value', value_drift), ('first4', first4_drift)):
                for field in ('post_observed', 'ever_recovered', 'first_recovery_rel_round',
                              'recovery_then_error', 'degradation_episodes',
                              'last_correct_run_rel_round', 'stable_recovered'):
                    drift[prefix + '_' + field] = result[field]
            drift['other_error_at_value_completion'] = other['other_error_at_value_completion']
            drift['other_only_error_ever'] = other['ever_other_only_error']
            drift_rows.append(drift)

            base = dict(weight=weight, stratum=stratum, graph_id=record["graph_id"],
                        tV=tV, t4=t4,
                        actual_max_depth=record.get("actual_max_depth"),
                        completion_ref_round=record.get("completion_ref_round"),
                        n_nodes=record.get("n_nodes"))
            if tV is None:
                examples["never_value_correct"].append(base)
            if tV is not None and record.get("completion_ref_round") is not None \
                    and tV > int(record["completion_ref_round"]) + 4:
                examples["delayed_value_correct"].append(base)
            if value_drift["ever_degraded"]:
                examples["value_drift"].append({**base, **value_drift})
            if other["ever_other_only_degraded"]:
                examples["other_field_drift"].append({**base, **other})
            if value_drift["ever_degraded"] and value_drift["recovered"]:
                examples["value_recovered"].append({**base, **value_drift})
            if value_drift["post_observed"] and not value_drift["ever_degraded"]:
                examples["stable_after_tV"].append({**base, **value_drift})

    # ---- curves ----
    curve_rows = []
    for (weight, stratum), by_round in sorted(round_acc.items()):
        for round_idx in sorted(by_round):
            acc = by_round[round_idx]
            curve_rows.append(dict(
                weight=weight, stratum=stratum, round=round_idx + 1,
                value_node_correct=acc["vc"], value_node_total=acc["vn"],
                value_node_acc=c_common.fmt_float(ratio(acc["vc"], acc["vn"])),
                first4_node_correct=acc["f4c"], first4_node_total=acc["f4n"],
                first4_node_acc=c_common.fmt_float(ratio(acc["f4c"], acc["f4n"])),
                **macro_fields(acc),
                graph_value_all=acc["gv"], graph_first4_all=acc["gf"],
                n_graphs=acc["n"],
                graph_value_all_rate=c_common.fmt_float(ratio(acc["gv"], acc["n"])),
                graph_first4_all_rate=c_common.fmt_float(ratio(acc["gf"], acc["n"])),
            ))

    # ---- capability by H ----
    cap_rows = []
    for weight, stratum in sorted(n_graphs):
        for H in horizons:
            acc = round_acc[(weight, stratum)].get(H - 1)
            if acc is None:
                cap_rows.append(dict(weight=weight, stratum=stratum, H=H,
                                     n_graphs=n_graphs[(weight, stratum)]))
                continue
            cap_rows.append(dict(
                weight=weight, stratum=stratum, H=H,
                value_node_correct=acc["vc"], value_node_total=acc["vn"],
                value_node_acc=c_common.fmt_float(ratio(acc["vc"], acc["vn"])),
                first4_node_correct=acc["f4c"], first4_node_total=acc["f4n"],
                first4_node_acc=c_common.fmt_float(ratio(acc["f4c"], acc["f4n"])),
                **macro_fields(acc),
                graph_value_all=acc["gv"], graph_first4_all=acc["gf"],
                n_graphs=acc["n"],
                graph_value_all_rate=c_common.fmt_float(ratio(acc["gv"], acc["n"])),
                graph_first4_all_rate=c_common.fmt_float(ratio(acc["gf"], acc["n"])),
            ))

    # ---- node type / depth ----
    type_rows = []
    for (weight, stratum, H, node_type), acc in sorted(type_acc.items()):
        type_rows.append(dict(
            weight=weight, stratum=stratum, H=H, node_type=node_type,
            value_node_correct=acc["vc"], value_node_total=acc["vn"],
            value_node_acc=c_common.fmt_float(ratio(acc["vc"], acc["vn"])),
            first4_node_correct=acc["f4c"], first4_node_total=acc["f4n"],
            first4_node_acc=c_common.fmt_float(ratio(acc["f4c"], acc["f4n"])),
                **macro_fields(acc),
        ))
    depth_rows = []
    for (weight, stratum, H, bucket), acc in sorted(depth_acc.items()):
        depth_rows.append(dict(
            weight=weight, stratum=stratum, H=H, depth_bucket=bucket,
            value_node_correct=acc["vc"], value_node_total=acc["vn"],
            value_node_acc=c_common.fmt_float(ratio(acc["vc"], acc["vn"])),
            first4_node_correct=acc["f4c"], first4_node_total=acc["f4n"],
            first4_node_acc=c_common.fmt_float(ratio(acc["f4c"], acc["f4n"])),
                **macro_fields(acc),
        ))

    # ---- completion ----
    completion_rows = []
    for weight, stratum in sorted(n_graphs):
        n = n_graphs[(weight, stratum)]
        tvs = completion_tV[(weight, stratum)]
        t4s = completion_t4[(weight, stratum)]
        row = dict(
            weight=weight, stratum=stratum, n_graphs=n,
            value_completed=len(tvs), value_completed_rate=c_common.fmt_float(ratio(len(tvs), n)),
            tV_mean=c_common.fmt_float(mean(tvs)), tV_median=c_common.fmt_float(median(tvs)),
            first4_completed=len(t4s), first4_completed_rate=c_common.fmt_float(ratio(len(t4s), n)),
            t4_mean=c_common.fmt_float(mean(t4s)), t4_median=c_common.fmt_float(median(t4s)),
        )
        for H in horizons:
            row[f"value_completed_by_H{H}"] = sum(1 for t in tvs if t <= H)
            row[f"first4_completed_by_H{H}"] = sum(1 for t in t4s if t <= H)
        completion_rows.append(row)

    # ---- drift aggregate ----
    drift_rows_sorted = sorted(drift_rows, key=lambda r: (r["weight"], r["stratum"], r["graph_id"]))
    drift_agg = []
    for weight, stratum in sorted(n_graphs):
        rows = [r for r in drift_rows_sorted
                if r["weight"] == weight and r["stratum"] == stratum]
        tV_rows = [r for r in rows if r["tV"] is not None]
        t4_rows = [r for r in rows if r["t4"] is not None]
        value_deg = [r for r in tV_rows if r["value_ever_degraded"]]
        first4_deg = [r for r in t4_rows if r["first4_ever_degraded"]]
        other_only = [r for r in tV_rows if r["other_only_ever"]]
        coverage_fields = {}
        for prefix, completed in (('value', tV_rows), ('first4', t4_rows)):
            observed = [r for r in completed if r[prefix + '_post_observed']]
            coverage_fields['n_' + prefix + '_post_observed'] = len(observed)
            coverage_fields['n_' + prefix + '_no_post_observation'] = len(completed) - len(observed)
            for field in ('ever_recovered', 'recovery_then_error', 'stable_recovered'):
                coverage_fields['n_' + prefix + '_' + field] = sum(bool(r[prefix + '_' + field]) for r in observed)
        coverage_fields['n_other_error_at_value_completion'] = sum(bool(r['other_error_at_value_completion']) for r in tV_rows)
        coverage_fields['n_other_only_error_ever'] = sum(bool(r['other_only_error_ever']) for r in tV_rows)
        drift_agg.append(dict(
            weight=weight, stratum=stratum, n_graphs=n_graphs[(weight, stratum)],
            **coverage_fields,
            n_value_completed=len(tV_rows),
            n_value_ever_degraded=len(value_deg),
            n_value_recovered=sum(1 for r in value_deg if r["value_recovered"]),
            n_value_final_correct=sum(1 for r in tV_rows if r["value_final_correct"]),
            value_first_deg_mean=c_common.fmt_float(mean([
                r["value_first_degradation_rel_round"] for r in value_deg
                if r["value_first_degradation_rel_round"] is not None])),
            value_first_deg_median=c_common.fmt_float(median([
                r["value_first_degradation_rel_round"] for r in value_deg
                if r["value_first_degradation_rel_round"] is not None])),
            n_first4_completed=len(t4_rows),
            n_first4_ever_degraded=len(first4_deg),
            n_first4_recovered=sum(1 for r in first4_deg if r["first4_recovered"]),
            n_first4_final_correct=sum(1 for r in t4_rows if r["first4_final_correct"]),
            first4_first_deg_mean=c_common.fmt_float(mean([
                r["first4_first_degradation_rel_round"] for r in first4_deg
                if r["first4_first_degradation_rel_round"] is not None])),
            first4_first_deg_median=c_common.fmt_float(median([
                r["first4_first_degradation_rel_round"] for r in first4_deg
                if r["first4_first_degradation_rel_round"] is not None])),
            n_other_only_ever=len(other_only),
            n_other_only_final_value_correct=sum(
                1 for r in other_only if r["other_only_final_value_correct"]),
        ))

    relative_rows = []
    for (weight, stratum, metric, lag), cell in sorted(relative_acc.items()):
        completed = (completion_tV if metric == 'value' else completion_t4)[(weight, stratum)]
        relative_rows.append(dict(weight=weight, stratum=stratum, metric=metric,
                                  relative_round=lag, n_completed=len(completed),
                                  n_observable=cell['n'], n_current_correct=cell['correct'],
                                  n_never_degraded=cell['retained'],
                                  current_correct_rate=cell['correct'] / cell['n'],
                                  retention_rate=cell['retained'] / cell['n'],
                                  wrong_nodes=cell['wrong_nodes'], total_nodes=cell['total_nodes']))
    return dict(
        relative_rows=relative_rows, runs=runs, round_acc=round_acc, n_graphs=n_graphs,
        curve_rows=curve_rows, cap_rows=cap_rows,
        type_rows=type_rows, depth_rows=depth_rows,
        completion_rows=completion_rows, drift_agg=drift_agg,
        drift_rows=drift_rows_sorted, examples=examples,
    )


def write_report(path: Path, result: dict, horizons: tuple[int, ...]) -> None:
    lines = [
        "# C0 capability and free-running exploration",
        "",
        "> Generated by `tools/c_capability/summarize.py`.  This is an exploratory report;",
        "> 64 rounds is a fixed observation cap, not a final evaluation protocol.",
        "",
        "## Capability at selected horizons (graph-equal)",
        "",
        "| weight | stratum | H | VALUE node | first-4 node | graph VALUE all | graph first-4 all | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["cap_rows"]:
        if row.get("H") not in (16, 64):
            continue
        lines.append(
            f"| {row['weight']} | {row['stratum']} | {row['H']} | "
            f"{row.get('value_node_acc_graph_equal', '')} | {row.get('first4_node_acc_graph_equal', '')} | "
            f"{row.get('graph_value_all_rate', '')} | {row.get('graph_first4_all_rate', '')} | "
            f"{row.get('n_graphs', '')} |")
    lines += ["", "## Completion", "",
              "| weight | stratum | n | VALUE completed | tV median | first-4 completed | t4 median |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["completion_rows"]:
        lines.append(
            f"| {row['weight']} | {row['stratum']} | {row['n_graphs']} | "
            f"{row['value_completed']} | {row['tV_median']} | "
            f"{row['first4_completed']} | {row['t4_median']} |")
    lines += ["", "## Post-completion drift (recovered below = final correct after degradation)", "",
              "| weight | stratum | VALUE completed | ever degraded | recovered | final correct | first-4 ever degraded | other-only ever |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["drift_agg"]:
        lines.append(
            f"| {row['weight']} | {row['stratum']} | {row['n_value_completed']} | "
            f"{row['n_value_ever_degraded']} | {row['n_value_recovered']} | "
            f"{row['n_value_final_correct']} | {row['n_first4_ever_degraded']} | "
            f"{row['n_other_only_ever']} |")
    lines += ["", "## Files", "",
              "- `curves.csv`: per-round node and whole-graph curves.",
              "- `capability_by_H.csv`: H=2/4/8/16/32/64 slices.",
              "- `by_node_type.csv` / `by_depth.csv`: decomposition.",
              "- `completion.csv`: tV/t4 completion counts and times.",
              "- `drift.csv` / `drift_by_graph.csv`: ever/final/stable recovery and recurrence; no post-observation is separate.",
              "- `post_completion_curves.csv`: completion-relative curves with the observable denominator at every lag.",
              "- CSV *_acc are node-weighted; *_acc_graph_equal are graph-equal. internal means non-leaf (includes sinks).",
              "- other_only_error denotes error presence; other_only_ever denotes a correct-to-wrong transition in a first-three field.",
              "- `examples.json`: representative graph IDs.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=c_common.DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run-dir>/summary")
    parser.add_argument("--horizons", type=parse_horizons, default=c_common.HORIZONS)
    parser.add_argument("--overwrite-ok", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.run_dir / "summary")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite_ok:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite-ok")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = summarize(args.run_dir, args.horizons)

    write_csv(out_dir / "curves.csv", result["curve_rows"], [
        "weight", "stratum", "round", "value_node_correct", "value_node_total",
        "value_node_acc", "first4_node_correct", "first4_node_total",
        "first4_node_acc", "value_node_acc_graph_equal", "first4_node_acc_graph_equal",
        "n_graphs_with_nodes", "n_graphs_without_nodes", "graph_value_all", "graph_first4_all", "n_graphs",
        "graph_value_all_rate", "graph_first4_all_rate"])
    write_csv(out_dir / "capability_by_H.csv", result["cap_rows"], [
        "weight", "stratum", "H", "value_node_correct", "value_node_total",
        "value_node_acc", "first4_node_correct", "first4_node_total",
        "first4_node_acc", "value_node_acc_graph_equal", "first4_node_acc_graph_equal",
        "n_graphs_with_nodes", "n_graphs_without_nodes", "graph_value_all", "graph_first4_all", "n_graphs",
        "graph_value_all_rate", "graph_first4_all_rate"])
    write_csv(out_dir / "by_node_type.csv", result["type_rows"], [
        "weight", "stratum", "H", "node_type", "value_node_correct",
        "value_node_total", "value_node_acc", "first4_node_correct",
        "first4_node_total", "first4_node_acc", "value_node_acc_graph_equal",
        "first4_node_acc_graph_equal", "n_graphs_with_nodes", "n_graphs_without_nodes"])
    write_csv(out_dir / "by_depth.csv", result["depth_rows"], [
        "weight", "stratum", "H", "depth_bucket", "value_node_correct",
        "value_node_total", "value_node_acc", "first4_node_correct",
        "first4_node_total", "first4_node_acc", "value_node_acc_graph_equal",
        "first4_node_acc_graph_equal", "n_graphs_with_nodes", "n_graphs_without_nodes"])

    completion_fields = ["weight", "stratum", "n_graphs", "value_completed",
                         "value_completed_rate", "tV_mean", "tV_median",
                         "first4_completed", "first4_completed_rate", "t4_mean",
                         "t4_median"]
    for H in args.horizons:
        completion_fields += [f"value_completed_by_H{H}", f"first4_completed_by_H{H}"]
    write_csv(out_dir / "completion.csv", result["completion_rows"], completion_fields)

    # All rows have the same schema; retain every explicit recovery/coverage field.
    write_csv(out_dir / "drift.csv", result["drift_agg"],
              list(result['drift_agg'][0]) if result['drift_agg'] else [])
    write_csv(out_dir / "drift_by_graph.csv", result["drift_rows"],
              list(result['drift_rows'][0]) if result['drift_rows'] else [])
    write_csv(out_dir / "post_completion_curves.csv", result['relative_rows'],
              list(result['relative_rows'][0]) if result['relative_rows'] else
              ['weight', 'stratum', 'metric', 'relative_round', 'n_observable'])

    examples = {key: values[:3] for key, values in result["examples"].items()}
    c_common.write_json(out_dir / "examples.json", examples)
    c_common.write_json(out_dir / "summary.json", dict(
        run_dir=str(args.run_dir), horizons=list(args.horizons),
        n_graphs={f"{w}/{s}": n for (w, s), n in sorted(result["n_graphs"].items())},
        weights=sorted(result["runs"]),
        files=["curves.csv", "capability_by_H.csv", "by_node_type.csv",
               "by_depth.csv", "completion.csv", "drift.csv",
               "drift_by_graph.csv", "post_completion_curves.csv", "examples.json"],
    ))
    write_report(out_dir / "report.md", result, args.horizons)
    print(f"wrote summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
