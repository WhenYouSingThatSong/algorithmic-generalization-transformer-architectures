"""Frozen shared data, training schedule and observation rules for the short-window recipe."""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "c_multitime"
sys.path.insert(0, str(SHARED))
import spec as legacy
P = legacy.P

VERSION = "c_local_update_normal_window_v1"
MODELS = ("C_s42", "C_s1234", "C_s2026")
PROTOCOL = dict(version=VERSION, models=list(MODELS), epochs=50, batch_size=32,
                modes=["frontier", "middle", "completed"], batch_gate=.05, candidate_probability=.5,
                rounds="D+1", middle="max(1,D//2)", weights=P.LOSS_WEIGHTS,
                final_epoch=50, recovery_horizon=16, completion_search="first VALUE all by D+16",
                dev_cases=32, test_cases_per_stratum=32, mixed="actual new+conflict+healthy in same input",
                primary=["new_calculation", "direct_conflict", "consistent_correct", "indirect_consistent"],
                secondary=["blocked", "non_numeric_other"], original_plan_protocol=P.VERSION)


def hashes():
    result = legacy.hashes()
    for p in sorted(HERE.glob("*.py")):
        result[str(p.relative_to(P.ROOT))] = P.sha(p)
    return result


def schedule(d, epoch, batch):
    def draw(kind):
        return random.Random(P.seed(VERSION, kind, epoch, batch)).random()
    mode = PROTOCOL["modes"][int(draw("mode") * 3)]
    return dict(mode=mode, enabled=draw("gate") < .05,
                start=None if mode=="frontier" else (d if mode=="completed" else max(1, d//2)), rounds=d+1)


def events(gs, ids, epoch, batch, settings):
    if settings["enabled"]:
        # legacy sampler uses the fixed start for any non-frontier mode.
        return legacy.events(gs, ids, epoch, batch, settings)
    eligible = sum(sum(v < settings["start"] for v in g["depths"].values())
                   if settings["mode"]=="middle" else len(g["nodes"]) for g in gs)
    return eligible, []


def build_manifest(path):
    plan = legacy.validate_plan(path)
    specs = {m["id"]: m for m in plan["models"]}
    models = {m: specs[m]["sha256"] for m in MODELS}
    splits = {}
    for key in ("dev", *P.STRATA):
        s = plan["splits"][key]
        if P.sha(s["path"]) != s["sha256"] or len(s["targets"]) != 128 or s["count"] != 128:
            raise ValueError("incomplete or changed split: " + key)
        n = PROTOCOL["dev_cases" if key=="dev" else "test_cases_per_stratum"]
        splits[key] = dict(sha256=s["sha256"], cases=[
            dict(gid=i, target=s["targets"][i]) for i in range(n)])
    return dict(protocol=PROTOCOL, plan_sha256=P.sha(path), train_sha256=P.TRAIN_SHA,
                splits=splits, sources=models, code_hashes=hashes())


def validate(plan_path, manifest_path):
    if P.read(manifest_path) != build_manifest(plan_path):
        raise ValueError("manifest, code or inputs changed")
    return P.read(plan_path)


def classify(old, expected, ready, truth):
    if not ready:
        return "blocked"
    if old == 24:
        return "new_calculation"
    if not 0 <= old < 23:
        return "non_numeric_other"
    if old != expected:
        return "direct_conflict"
    return "consistent_correct" if old == truth else "indirect_consistent"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
