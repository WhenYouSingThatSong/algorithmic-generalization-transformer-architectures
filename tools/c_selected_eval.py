"""Supplement the frozen normal-window evaluation for seeds 42 and 1234."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "tools/c_local_update"
RECEIPT = ROOT / "docs/records/2026-11-21/c-local-update-finals.json"
MODELS = ("C_s42", "C_s1234")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def detail(g, labels, trace, clean, target):
    nodes = g["nodes"]
    first = trace[0]["input"]
    # Include every changed input position, including the separate fill target.
    changes = [dict(position=p, before=old[3], after=new[3], truth=labels[p][3])
               for p, old, new in zip(nodes, clean[0]["input"], first)
               if old[3] != new[3]]
    cells = Counter()
    for step, row in enumerate(trace, 1):
        for i, p in enumerate(nodes):
            old = row["input"][i][3]
            update = row["update"][i]
            if not (0 <= old < 23 and update["reference"]):
                continue
            # Under the local rule, an unready parent preserves the old value,
            # so reference=True on numeric input is exactly a direct conflict.
            cells[(step, p == target, old == labels[p][3], bool(update["plan"]),
                   bool(update["changed"]), bool(update["value_correct"]),
                   row["output"][i][0] == labels[p][3])] += 1
    first_row = trace[0]
    roles = {}
    by_position = dict(zip(nodes, first))
    for i, p in enumerate(nodes):
        old = first[i][3]
        gold = labels[p][3]
        if any(not 0 <= by_position[k][3] < 23 for k in g["parents"][p]):
            continue
        update = first_row["update"][i]
        if old == 24:
            role = "fill"
        elif 0 <= old < 23 and update["reference"]:
            role = "conflict"
        elif old == gold:
            role = "healthy"
        else:
            continue
        counts = roles.setdefault(role, Counter())
        counts["observed"] += 1
        counts["gold_correct"] += first_row["output"][i][0] == gold
    mixed = all(k in roles for k in ("fill", "conflict", "healthy"))
    joint = all(v["gold_correct"] == v["observed"] for v in roles.values()) if mixed else None
    return dict(input_changes=changes, conflicts=[[*key, n] for key, n in sorted(cells.items())],
                first_round_roles=roles, mixed_all_gold_correct=joint)


def run(args):
    sys.path.insert(0, str(FROZEN))
    import assess
    receipt = read(RECEIPT)
    model = next(r for r in receipt["runs"] if r["model"] == args.model)
    if assess.P.sha(args.checkpoint) != model["final_ckpt"]["sha256"]:
        raise ValueError("checkpoint differs from selected final")
    original = assess.metrics

    def extended(g, y, trace, clean, timing, condition, target):
        result = original(g, y, trace, clean, timing, condition, target)
        result["selected_detail"] = detail(g, y, trace, clean, target)
        return result

    # Extend returned summaries at the scoring boundary; frozen rollout and
    # model-selection checks still run in assess.main() without file changes.
    assess.metrics = extended
    previous = sys.argv
    try:
        sys.argv = [str(FROZEN / "assess.py"), "--plan", str(args.plan),
                    "--manifest", str(args.manifest), "--checkpoint", str(args.checkpoint),
                    "--out-dir", str(args.out_dir), "--device", args.device]
        assess.main()
    finally:
        assess.metrics = original
        sys.argv = previous


def combine(paths):
    receipt = read(RECEIPT)
    finals = {r["model"]: r["final_ckpt"]["sha256"] for r in receipt["runs"]}
    seeds = {}
    for path in paths:
        meta = read(path / "meta.json")
        model = meta["source"]["model"]
        if (model not in MODELS or model in seeds or meta["status"] != "complete"
                or meta["checkpoint"] != finals[model]
                or meta["source"]["manifest"] != receipt["manifest_sha256"]):
            raise ValueError(f"unexpected seed, status or identity at {path}")
        results = {}
        for size in (32, 64, 128):
            for depth in ("shallow", "deep"):
                split = f"N{size}_{depth}"
                classes = read(path / f"{split}-classes.json")
                rows = classes["records"]
                ok = [r for r in rows if r["status"] == "ok"]
                item = dict(classes=classes["summary"],
                            baseline_and_interventions=read(path / f"{split}-standard-summary.json"),
                            natural_and_use=read(path / f"{split}-outcomes.json"),
                            detail_coverage=[sum("selected_detail" in r for r in ok), len(ok)],
                            per_graph_records=str(path / f"{split}-classes.json"))
                if size != 64:
                    item["frozen_probe"] = read(path / f"{split}-frozen-probe.json")
                results[split] = item
        seeds[model] = dict(checkpoint=meta["checkpoint"], splits=results)
    if set(seeds) != set(MODELS):
        raise ValueError("both selected seeds required")
    return dict(models=seeds, conflict_columns=["round_after_intervention", "is_target",
                "old_globally_correct", "plan_update", "value_changed", "locally_correct",
                "globally_correct", "node_round_count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("assess")
    run_parser.add_argument("--model", required=True, choices=MODELS)
    for key in ("plan", "manifest", "checkpoint", "out-dir"):
        run_parser.add_argument("--" + key, type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    group = commands.add_parser("combine")
    group.add_argument("--results", type=Path, required=True, nargs=2)
    group.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assess":
        run(args)
    else:
        result = combine(args.results)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False)


if __name__ == "__main__":
    main()
