#!/usr/bin/env python3
"""CPU-only analysis of the existing C0 capability NPZ traces.

This script reads the per-graph ``.npz`` files written by
``run_capability.py``.  It never loads a model and never writes into the run
directory.  All output goes to a separate analysis directory.

Definitions used here (also written into ``analysis_meta.json``):

* full node set: every numeric node saved by the runner, including the six
  leaf variables;
* internal: non-leaf numeric nodes (includes sinks);
* sink: numeric nodes with no numeric children, as saved by the runner;
* VALUE node correctness: ``pred_value == true_value``;
* VARIABLE node correctness: ``pred_variable == true_variable``;
* value event: first post-completion round where the graph-level VALUE
  all-correct flag is false;
* variable event: first post-completion round with a correct-to-wrong
  VARIABLE transition at any node (not merely a node that was already wrong
  at tV);
* stable recovery: the last two post-completion observations are both
  correct (same definition as ``c_common.completion_drift``).

Run with, for example::

    python -I -B tools/c_capability/analyze_existing.py \
      --run-dir tmp/analysis/c0-capability-20261114/runs \
      --out-dir tmp/analysis/c0-capability-20261114/analysis

The run is CPU-only and deterministic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from . import c_common
except ImportError:  # direct script execution
    import c_common  # type: ignore[no-redef]

FACTORS = ("syntax", "variable", "operation", "value")
DEFAULT_HORIZONS = (8, 16, 32, 64)
SCOPES = ("all", "internal", "sink")


# ---------------------------------------------------------------------------
# Loading and small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    return c_common.sha256_file(path)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_weights(run_dir: Path) -> list[str]:
    weights = []
    for meta in sorted(run_dir.glob("*/run_meta.json")):
        weights.append(meta.parent.name)
    if not weights:
        raise FileNotFoundError(f"no */run_meta.json under {run_dir}")
    return weights


def load_graphs(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load index.jsonl and the needed arrays from each graph npz.

    The index is the authoritative identity map: weight, stratum, graph_id,
    source_index and npz path.  The arrays are kept in memory; the full set is
    small enough for CPU analysis (tens to low hundreds of MB).
    """
    graphs: list[dict[str, Any]] = []
    run_meta: dict[str, Any] = {}
    for weight in discover_weights(run_dir):
        weight_dir = run_dir / weight
        meta_path = weight_dir / "run_meta.json"
        run_meta[weight] = _json_load(meta_path)
        index_path = weight_dir / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        lines = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            row = json.loads(line)
            npz_path = weight_dir / row["path"]
            with np.load(npz_path) as z:
                arrays = {name: z[name] for name in z.files}
            required = {
                "pred_variable", "true_variable", "pred_value", "true_value",
                "pred_syntax", "true_syntax", "pred_operation", "true_operation",
                "value_correct", "first4_correct", "graph_value_all",
                "graph_first4_all", "depth", "is_internal", "is_sink",
                "tV", "t4", "completion_ref_round", "n_rounds",
                "actual_max_depth", "positions",
            }
            missing = required - set(arrays)
            if missing:
                raise KeyError(f"{npz_path}: missing keys {sorted(missing)}")
            arrays["weight"] = row["weight"]
            arrays["stratum"] = row["stratum"]
            arrays["graph_id"] = row["graph_id"]
            arrays["source_index"] = row.get("source_index")
            arrays["path"] = row["path"]
            arrays["data_sha256"] = row.get("data_sha256")
            graphs.append(arrays)
    return graphs, run_meta


def _mask(g: dict[str, Any], scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(g["depth"].shape[0], dtype=bool)
    if scope == "internal":
        return g["is_internal"].astype(bool)
    if scope == "sink":
        return g["is_sink"].astype(bool)
    raise ValueError(scope)


def _mean_median(values: Iterable[float]) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    return float(sum(vals) / len(vals)), float(statistics.median(vals))


def _depth_list(values: np.ndarray) -> str:
    return " ".join(str(int(x)) for x in sorted(set(values.tolist())))


def _position_list(values: np.ndarray) -> str:
    return " ".join(str(int(x)) for x in values.tolist())


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def field_accuracy_rows(graphs: list[dict[str, Any]], horizons: Iterable[int]) -> list[dict[str, Any]]:
    """Aggregate field correctness by weight/stratum/H/field/scope.

    Both denominators are kept:
    - ``acc`` is node-weighted (micro): total correct / total nodes.
    - ``acc_graph_equal`` is the mean of per-graph accuracy over graphs with
      at least one node in the scope (macro).  For all/internal with fixed N
      and six leaves these coincide; sink/depth buckets can differ because
      the number of nodes per graph varies.
    """
    agg: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for g in graphs:
        for H in horizons:
            i = H - 1
            for field in FACTORS:
                ok = g[f"pred_{field}"][i] == g[f"true_{field}"]
                for scope in SCOPES:
                    key = (g["weight"], g["stratum"], H, field, scope)
                    row = agg.setdefault(key, dict(
                        weight=g["weight"], stratum=g["stratum"], H=H,
                        field=field, scope=scope, correct=0, total=0,
                        n_graphs_with_nodes=0, n_graphs_without_nodes=0,
                        graph_acc_sum=0.0, graph_acc_n=0,
                    ))
                    mask = _mask(g, scope)
                    n = int(mask.sum())
                    if n:
                        correct_g = int(ok[mask].sum())
                        row["correct"] += correct_g
                        row["total"] += n
                        row["n_graphs_with_nodes"] += 1
                        row["graph_acc_sum"] += correct_g / n
                        row["graph_acc_n"] += 1
                    else:
                        row["n_graphs_without_nodes"] += 1
    rows = []
    for key in sorted(agg):
        row = agg[key]
        row["acc"] = (row["correct"] / row["total"]) if row["total"] else None
        row["acc_graph_equal"] = (row["graph_acc_sum"] / row["graph_acc_n"]) if row["graph_acc_n"] else None
        del row["graph_acc_sum"], row["graph_acc_n"]
        rows.append(row)
    return rows


def field_accuracy_wide(graphs: list[dict[str, Any]], horizons: Iterable[int]) -> list[dict[str, Any]]:
    """One row per weight/stratum/H with the rates used in the report.

    Rates are blank when the scope has no nodes.  Support is in
    ``field_accuracy_long.csv``; this wide file mirrors the earlier auxiliary
    ``field_acc_by_H.csv`` while making its origin reproducible.
    """
    long = field_accuracy_rows(graphs, horizons)
    bucket: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in long:
        key = (r["weight"], r["stratum"], r["H"])
        row = bucket.setdefault(key, {"weight": r["weight"], "stratum": r["stratum"], "H": r["H"]})
        row[f'{r["field"]}_{r["scope"]}'] = r["acc"]
        row[f'{r["field"]}_{r["scope"]}_graph_equal'] = r["acc_graph_equal"]
    fields = ["weight", "stratum", "H"]
    for scope in SCOPES:
        for field in FACTORS:
            fields.append(f"{field}_{scope}")
            fields.append(f"{field}_{scope}_graph_equal")
    return [{k: bucket[key].get(k) for k in fields}
            for key in sorted(bucket)]


def completion_lag_rows(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list[int]] = defaultdict(list)
    for g in graphs:
        tV = int(g["tV"])
        if tV <= 0:
            continue
        lag = tV - int(g["completion_ref_round"])
        by[(g["weight"], g["stratum"])].append(lag)
    rows: list[dict[str, Any]] = []
    for (weight, stratum), lags in sorted(by.items()):
        c = Counter(lags)
        rows.append(dict(
            weight=weight, stratum=stratum, completed=len(lags),
            lag0=c.get(0, 0), lag1=c.get(1, 0),
            lag2_4=sum(v for k, v in c.items() if 2 <= k <= 4),
            lag_gt4=sum(v for k, v in c.items() if k > 4),
            max_lag=max(lags) if lags else None,
            mean_lag=(sum(lags) / len(lags)) if lags else None,
            median_lag=statistics.median(lags) if lags else None,
        ))
    return rows


def completion_lag_distribution_rows(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, int], int] = Counter()
    for g in graphs:
        tV = int(g["tV"])
        if tV <= 0:
            continue
        counts[(g["weight"], g["stratum"], tV - int(g["completion_ref_round"]))] += 1
    return [dict(weight=w, stratum=s, lag=lag, count=n)
            for (w, s, lag), n in sorted(counts.items())]


def value_error_rows(graphs: list[dict[str, Any]], H: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Errors among graphs that had completed by H (``tV <= H``).

    This is the ``completed_by_H`` queue.  For H=64 it is the same set used in
    the formal report; for a future H=16 it excludes graphs that completed
    later, rather than silently using the final 64-round completion set.
    """
    counts_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    by_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_depth: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    pred_dist: dict[tuple[str, str], Counter] = defaultdict(Counter)
    diff_dist: dict[tuple[str, str], Counter] = defaultdict(Counter)
    non_residue: Counter = Counter()
    for g in graphs:
        tV = int(g["tV"])
        if tV <= 0 or tV > H:
            continue
        i = H - 1
        bad = ~g["value_correct"][i]
        n_wrong = int(bad.sum())
        by_counts[(g["weight"], g["stratum"])].append(n_wrong)
        for d in np.unique(g["depth"]):
            m = (g["depth"] == d) & bad
            n_all = int((g["depth"] == d).sum())
            by_depth[(g["weight"], g["stratum"], int(d))][0] += int(m.sum())
            by_depth[(g["weight"], g["stratum"], int(d))][1] += n_all
        p = g["pred_value"][i][bad].astype(np.int64)
        t = g["true_value"][bad].astype(np.int64)
        for pv in p[p < c_common.MOD_VAL]:
            pred_dist[(g["weight"], g["stratum"])][int(pv)] += 1
        for pv, tv in zip(p, t):
            if pv < c_common.MOD_VAL and tv < c_common.MOD_VAL:
                diff_dist[(g["weight"], g["stratum"])][int((pv - tv) % c_common.MOD_VAL)] += 1
            else:
                non_residue[(g["weight"], g["stratum"])] += 1
    for (w, s), vals in sorted(by_counts.items()):
        c = Counter(vals)
        counts_rows.append(dict(
            weight=w, stratum=s, H=H, queue="completed_by_H", completed=len(vals),
            zero=c.get(0, 0),
            one_two=sum(v for k, v in c.items() if 1 <= k <= 2),
            three_five=sum(v for k, v in c.items() if 3 <= k <= 5),
            gt_five=sum(v for k, v in c.items() if k > 5),
            max_wrong=max(vals) if vals else None,
        ))
    for (w, s, d), (err, total) in sorted(by_depth.items()):
        depth_rows.append(dict(weight=w, stratum=s, H=H, queue="completed_by_H",
                               depth=d, errors=err, total=total,
                               rate=(err / total) if total else None))
    for (w, s), c in sorted(pred_dist.items()):
        for value, n in sorted(c.items()):
            pred_rows.append(dict(weight=w, stratum=s, H=H, queue="completed_by_H",
                                  kind="pred_value", value=value, count=n,
                                  n_non_residue=non_residue[(w, s)]))
    for (w, s), c in sorted(diff_dist.items()):
        for value, n in sorted(c.items()):
            pred_rows.append(dict(weight=w, stratum=s, H=H, queue="completed_by_H",
                                  kind="mod_diff", value=value, count=n,
                                  n_non_residue=non_residue[(w, s)]))
    return counts_rows, depth_rows, pred_rows


# ---------------------------------------------------------------------------
# Event table
# ---------------------------------------------------------------------------


def graph_events(g: dict[str, Any]) -> dict[str, Any]:
    """Per-graph completion/drift event record.

    The output is deliberately conservative.  A graph with tV at the last
    observed round has no post-completion observation; its final flag is not
    used as evidence of stability.
    """
    tV = int(g["tV"])
    t4 = int(g["t4"])
    R = int(g["n_rounds"])
    ref = int(g["completion_ref_round"])
    value_all = g["graph_value_all"].astype(bool)
    var_ok = g["pred_variable"] == g["true_variable"][None, :]
    row: dict[str, Any] = dict(
        weight=g["weight"], stratum=g["stratum"], graph_id=g["graph_id"],
        source_index=g["source_index"], actual_max_depth=int(g["actual_max_depth"]),
        completion_ref_round=ref, tV=tV if tV > 0 else None,
        t4=t4 if t4 > 0 else None, n_rounds=R,
        completed=(tV > 0),
        delayed=(tV > 0 and tV > ref),
        post_rounds=(R - tV) if tV > 0 else 0,
        value_post_observed=(tV > 0 and tV < R),
        value_never_degraded=None,
        value_ever_degraded=None,
        value_first_degradation_rel_round=None,
        value_first_recovery_rel_round=None,
        value_affected_nodes_first=None,
        value_affected_depths_first=None,
        value_affected_positions_first=None,
        value_degradation_episodes=None,
        value_final_correct=None,
        value_last_correct_run_rel_round=None,
        value_stable_recovered=None,
        value_recovery_then_error=None,
        value_preserved=None,
        variable_all_at_tV=bool(var_ok[tV - 1].all()) if tV > 0 else None,
        variable_first_new_error_rel_round=None,
        variable_affected_nodes_first=None,
        variable_affected_depths_first=None,
        variable_affected_positions_first=None,
        variable_affected_recovered_rel_round=None,
        variable_all_recovered_after_error=False,
        variable_final_all=bool(var_ok[-1].all()),
        variable_preserved=None,
        value_and_variable_preserved=None,
        value_minus_variable_first_rel=None,
        primary_event=None,
    )
    if tV <= 0:
        row["primary_event"] = "never_value_complete"
        return row

    drift = c_common.completion_drift(value_all, tV)
    row.update(
        value_post_observed=bool(drift["post_observed"]),
        value_never_degraded=(bool(drift["post_observed"]) and not bool(drift["ever_degraded"])),
        value_ever_degraded=bool(drift["ever_degraded"]),
        value_first_degradation_rel_round=drift["first_degradation_rel_round"],
        value_first_recovery_rel_round=drift["first_recovery_rel_round"],
        value_degradation_episodes=drift["degradation_episodes"],
        value_final_correct=drift["final_correct"],
        value_last_correct_run_rel_round=drift["last_correct_run_rel_round"],
        value_stable_recovered=drift["stable_recovered"],
        value_recovery_then_error=drift["recovery_then_error"],
        value_preserved=(bool(drift["post_observed"]) and not bool(drift["ever_degraded"])),
    )
    first_value_rel = drift["first_degradation_rel_round"]
    if first_value_rel is not None:
        i = tV + int(first_value_rel) - 1
        bad = ~g["value_correct"][i]
        row["value_affected_nodes_first"] = int(bad.sum())
        row["value_affected_depths_first"] = _depth_list(g["depth"][bad])
        row["value_affected_positions_first"] = _position_list(g["positions"][bad])

    # VARIABLE: correct-to-wrong transition after tV, not merely "not all
    # correct at tV" (that case is recorded separately in variable_all_at_tV).
    if tV < R:
        first_var_rel = None
        affected: np.ndarray | None = None
        for rel in range(1, R - tV + 1):
            prev = var_ok[tV + rel - 2]
            cur = var_ok[tV + rel - 1]
            transition = prev & ~cur
            if transition.any():
                first_var_rel = rel
                affected = transition
                row["variable_first_new_error_rel_round"] = rel
                row["variable_affected_nodes_first"] = int(transition.sum())
                row["variable_affected_depths_first"] = _depth_list(g["depth"][transition])
                row["variable_affected_positions_first"] = _position_list(g["positions"][transition])
                break
        if first_var_rel is not None and affected is not None:
            for rel in range(first_var_rel + 1, R - tV + 1):
                if var_ok[tV + rel - 1][affected].all():
                    row["variable_affected_recovered_rel_round"] = rel
                    break
            row["variable_all_recovered_after_error"] = any(
                var_ok[tV + rel - 1].all()
                for rel in range(first_var_rel + 1, R - tV + 1)
            )

    if row["value_post_observed"]:
        # Independent preserve definitions; these are intentionally NOT the
        # mutually exclusive primary_event category.
        row["variable_preserved"] = bool(
            row["variable_all_at_tV"] and row["variable_first_new_error_rel_round"] is None
        )
        row["value_and_variable_preserved"] = bool(
            row["value_preserved"] and row["variable_preserved"]
        )
        if (row["value_first_degradation_rel_round"] is not None
                and row["variable_first_new_error_rel_round"] is not None):
            row["value_minus_variable_first_rel"] = (
                int(row["value_first_degradation_rel_round"])
                - int(row["variable_first_new_error_rel_round"])
            )

    # Primary event is mutually exclusive and prioritises a VALUE failure,
    # because that is the capability-level event.  Flags above keep the
    # overlapping VARIABLE information.
    if not row["value_post_observed"]:
        row["primary_event"] = "value_no_post_observation"
    elif row["value_ever_degraded"]:
        if row["value_final_correct"] and row["value_stable_recovered"]:
            row["primary_event"] = "value_degraded_recovered_stable"
        elif row["value_final_correct"]:
            row["primary_event"] = "value_degraded_final_correct_unstable"
        else:
            row["primary_event"] = "value_degraded_final_wrong"
    elif row["variable_all_at_tV"] is False:
        row["primary_event"] = "variable_already_wrong_at_tV_value_preserved"
    elif row["variable_first_new_error_rel_round"] is not None:
        row["primary_event"] = "variable_degraded_after_tV_value_preserved"
    elif row["delayed"]:
        row["primary_event"] = "delayed_completion_only"
    else:
        row["primary_event"] = "value_and_variable_preserved"
    return row


def event_rows(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [graph_events(g) for g in graphs]


def event_summary_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by[(e["weight"], e["stratum"])].append(e)
    rows: list[dict[str, Any]] = []
    for (w, s), es in sorted(by.items()):
        completed = [e for e in es if e["completed"]]
        post = [e for e in completed if e["value_post_observed"]]
        val_deg = [e for e in post if e["value_ever_degraded"]]
        var_deg = [e for e in post if e["variable_first_new_error_rel_round"] is not None]
        val_first = [e["value_first_degradation_rel_round"] for e in val_deg]
        var_first = [e["variable_first_new_error_rel_round"] for e in var_deg]
        val_mean, val_median = _mean_median(val_first)
        var_mean, var_median = _mean_median(var_first)
        primary = Counter(e["primary_event"] for e in es)

        # Independent preserve definitions; these overlap with primary_event
        # only by construction and are not derived from it.
        value_preserved = [e for e in post if e["value_preserved"] is True]
        variable_preserved = [e for e in post if e["variable_preserved"] is True]
        both_preserved = [e for e in post if e["value_and_variable_preserved"] is True]

        both_events = [
            e for e in post
            if e["value_first_degradation_rel_round"] is not None
            and e["variable_first_new_error_rel_round"] is not None
        ]
        deltas = [e["value_minus_variable_first_rel"] for e in both_events]
        delta_mean, delta_median = _mean_median(deltas)
        rows.append(dict(
            weight=w, stratum=s, n_graphs=len(es), n_completed=len(completed),
            n_never_complete=primary.get("never_value_complete", 0),
            n_delayed=sum(1 for e in completed if e["delayed"]),
            n_value_no_post_observation=sum(1 for e in completed if not e["value_post_observed"]),
            n_value_never_degraded=len(value_preserved),
            n_value_ever_degraded=len(val_deg),
            n_value_degraded_final_wrong=primary.get("value_degraded_final_wrong", 0),
            n_value_degraded_final_correct_unstable=primary.get("value_degraded_final_correct_unstable", 0),
            n_value_degraded_recovered_stable=primary.get("value_degraded_recovered_stable", 0),
            n_variable_all_at_tV_false=sum(1 for e in completed if e["variable_all_at_tV"] is False),
            n_variable_degraded_after_tV=len(var_deg),
            n_value_preserved=len(value_preserved),
            n_variable_preserved=len(variable_preserved),
            n_value_and_variable_preserved=len(both_preserved),
            n_primary_value_and_variable_preserved=primary.get("value_and_variable_preserved", 0),
            n_variable_degraded_value_preserved=sum(
                1 for e in post
                if e["variable_first_new_error_rel_round"] is not None
                and e["value_preserved"] is True
            ),
            n_primary_variable_degraded_value_preserved=primary.get(
                "variable_degraded_after_tV_value_preserved", 0
            ),
            n_variable_already_wrong_value_preserved=sum(
                1 for e in post
                if e["variable_all_at_tV"] is False and e["value_preserved"] is True
            ),
            n_primary_variable_already_wrong_value_preserved=primary.get(
                "variable_already_wrong_at_tV_value_preserved", 0
            ),
            n_delayed_completion_only=primary.get("delayed_completion_only", 0),
            n_both_events=len(both_events),
            n_variable_first=sum(1 for d in deltas if d > 0),
            n_same_round=sum(1 for d in deltas if d == 0),
            n_value_first=sum(1 for d in deltas if d < 0),
            n_variable_only=sum(
                1 for e in post
                if e["variable_first_new_error_rel_round"] is not None
                and e["value_first_degradation_rel_round"] is None
            ),
            n_value_only=sum(
                1 for e in post
                if e["variable_first_new_error_rel_round"] is None
                and e["value_first_degradation_rel_round"] is not None
            ),
            n_neither_event=sum(
                1 for e in post
                if e["variable_first_new_error_rel_round"] is None
                and e["value_first_degradation_rel_round"] is None
            ),
            value_first_deg_mean_rel=val_mean,
            value_first_deg_median_rel=val_median,
            variable_first_deg_mean_rel=var_mean,
            variable_first_deg_median_rel=var_median,
            value_minus_variable_mean_rel=delta_mean,
            value_minus_variable_median_rel=delta_median,
        ))
    return rows


def pick_examples(events: list[dict[str, Any]], max_per_event: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Round-robin examples by weight so both seeds are represented."""
    keys = [
        "never_value_complete", "delayed_completion", "delayed_completion_only",
        "value_no_post_observation",
        "variable_degraded_after_tV_value_preserved",
        "variable_already_wrong_at_tV_value_preserved",
        "value_degraded_final_wrong",
        "value_degraded_final_correct_unstable",
        "value_degraded_recovered_stable",
        "value_and_variable_preserved",
    ]
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in keys}
    for e in events:
        k = e["primary_event"]
        if k in buckets:
            buckets[k].append(e)
    # A delayed graph can also have a later VALUE/VARIABLE event; keep a
    # separate view so completion latency is not hidden by the primary-event
    # priority.
    buckets["delayed_completion"] = [e for e in events if e["completed"] and e["delayed"]]
    buckets["value_no_post_observation"] = [e for e in events if e["completed"] and not e["value_post_observed"]]
    out: dict[str, list[dict[str, Any]]] = {}
    for k in keys:
        candidates = buckets[k]
        by_weight: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in candidates:
            by_weight[e["weight"]].append(e)
        selected: list[dict[str, Any]] = []
        weight_names = sorted(by_weight)
        idx = 0
        while len(selected) < max_per_event and any(by_weight.values()):
            w = weight_names[idx % len(weight_names)]
            if by_weight[w]:
                selected.append(by_weight[w].pop(0))
            idx += 1
            if idx > 1000:
                break
        out[k] = selected
    return out


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=names)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: "" if row.get(k) is None else row.get(k) for k in names})


def build_meta(run_dir: Path, out_dir: Path, graphs: list[dict[str, Any]],
               run_meta: dict[str, Any], horizons: list[int], error_H: int) -> dict[str, Any]:
    strata: dict[str, dict[str, Any]] = {}
    for g in graphs:
        key = g["stratum"]
        if key not in strata:
            strata[key] = dict(stratum=key, n_graphs_total=0, n_graphs_per_weight=Counter(),
                               data_sha256=set(),
                               actual_depth_min=None, actual_depth_max=None)
        row = strata[key]
        row["n_graphs_total"] += 1
        row["n_graphs_per_weight"][g["weight"]] += 1
        row["data_sha256"].add(g.get("data_sha256"))
        d = int(g["actual_max_depth"])
        row["actual_depth_min"] = d if row["actual_depth_min"] is None else min(row["actual_depth_min"], d)
        row["actual_depth_max"] = d if row["actual_depth_max"] is None else max(row["actual_depth_max"], d)
    for row in strata.values():
        row["data_sha256"] = sorted(x for x in row["data_sha256"] if x)
        row["n_graphs_per_weight"] = dict(sorted(row["n_graphs_per_weight"].items()))
    runs = {}
    for weight, meta in sorted(run_meta.items()):
        weight_dir = run_dir / weight
        index_path = weight_dir / "index.jsonl"
        meta_path = weight_dir / "run_meta.json"
        runs[weight] = dict(
            weight=weight,
            weight_sha256=meta.get("weight_sha256"),
            weight_path=meta.get("weight_path"),
            run_meta_sha256=sha256_file(meta_path),
            index_sha256=sha256_file(index_path),
            graphs_written=meta.get("graphs_written"),
            rounds=meta.get("rounds"),
            status=meta.get("status"),
        )
    return dict(
        input_run_dir=str(run_dir),
        output_dir=str(out_dir),
        analysis_script="tools/c_capability/analyze_existing.py",
        analysis_script_sha256=sha256_file(Path(__file__)),
        horizons=list(horizons),
        error_H=error_H,
        runs=runs,
        strata=strata,
        definitions=dict(
            node_scope=dict(
                all="all numeric nodes, including the six leaf variables",
                internal="non-leaf numeric nodes (is_internal), includes sinks",
                sink="numeric nodes with no numeric children (is_sink)",
            ),
            value_correct="pred_value == true_value at a numeric node",
            variable_correct="pred_variable == true_variable at a numeric node",
            field_accuracy_micro="correct / total nodes (node-weighted)",
            field_accuracy_graph_equal="mean of per-graph accuracy over graphs with nodes in the scope; all/internal coincide with micro when N and leaf count are fixed, sink/depth buckets need not",
            value_event="first post-tV round where graph_value_all is false",
            variable_event="first post-tV round with a correct-to-wrong VARIABLE transition at any node; nodes already wrong at tV are not counted as a new variable event",
            stable_recovered="last two post-completion observations are both correct (c_common.completion_drift semantics), not just the final round",
            final_correct="the last observed round is correct, only defined for completed graphs with post-completion observations",
            delayed_completion="tV > actual_max_depth + 1",
            completion_lag="tV - (actual_max_depth + 1)",
            value_error_queue="completed_by_H: graphs with tV <= H; for H=64 this equals the formal completion set",
            preserved_independent="value_preserved and variable_preserved are independent booleans, not primary_event categories; variable_preserved requires VARIABLE all-correct at tV and no later correct-to-wrong transition",
            event_order="value_minus_variable_first_rel = VALUE first-event rel round minus VARIABLE first-event rel round; positive means VARIABLE first, zero same round, negative VALUE first; only graphs with both events enter the delta",
            positions="first affected token positions are stored as space-separated reduced-sequence positions; use source_index/positions to trace the original graph",
            primary_event="mutually exclusive priority: VALUE failure first, then VARIABLE, then delayed, then preserved; only for case classification, not for independent preserve totals",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only analysis of existing C capability NPZ traces")
    parser.add_argument("--run-dir", default=str(c_common.DEFAULT_RUN_DIR))
    parser.add_argument("--out-dir", default=str(c_common.ROOT / "tmp/analysis/c0-capability-20261114/analysis"))
    parser.add_argument("--horizons", default="8,16,32,64",
                        help="comma-separated horizons for field/error slices")
    parser.add_argument("--error-H", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--overwrite-ok", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if max(horizons) > 64:
        raise ValueError("horizons must be <= 64 for this 64-round run")
    if args.error_H not in horizons:
        raise ValueError("--error-H must be one of --horizons")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite_ok:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite-ok to replace outputs")

    graphs, run_meta = load_graphs(run_dir)
    if not graphs:
        raise RuntimeError(f"no graphs loaded from {run_dir}")
    for g in graphs:
        if int(g["n_rounds"]) < max(horizons):
            raise ValueError(f"{g['path']}: n_rounds < requested horizon")

    out_dir.mkdir(parents=True, exist_ok=True)
    field_long = field_accuracy_rows(graphs, horizons)
    field_wide = field_accuracy_wide(graphs, horizons)
    write_csv(out_dir / "field_accuracy_long.csv", field_long)
    write_csv(out_dir / "field_acc_by_H.csv", field_wide)
    write_csv(out_dir / "completion_lag.csv", completion_lag_rows(graphs))
    write_csv(out_dir / "completion_lag_distribution.csv", completion_lag_distribution_rows(graphs))
    err_counts, err_depth, err_pred = value_error_rows(graphs, args.error_H)
    write_csv(out_dir / "value_error_counts.csv", err_counts)
    write_csv(out_dir / "value_error_by_depth.csv", err_depth)
    write_csv(out_dir / "value_wrong_predictions.csv", err_pred)
    events = event_rows(graphs)
    write_csv(out_dir / "events_by_graph.csv", events)
    write_csv(out_dir / "event_summary.csv", event_summary_rows(events))
    examples = pick_examples(events, max_per_event=args.max_examples)
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = build_meta(run_dir, out_dir, graphs, run_meta, horizons, args.error_H)
    (out_dir / "analysis_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(graphs)} graphs to {out_dir}")
    print(f"events: {len(events)}; field rows: {len(field_long)}; error rows: {len(err_counts)}")


if __name__ == "__main__":
    main()