"""Export training loss and standard-round dev accuracy for a completed normal-window run."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_protocol as S


def history(run):
    meta = S.P.read(run / "meta.json")
    if meta["status"] != "complete" or meta["identity"]["version"] != S.VERSION:
        raise ValueError(f"Expected a completed normal-window run: {run}")
    steps = 0
    epochs = []
    for epoch in range(1, meta["epochs"] + 1):
        path = run / f"epoch-{epoch:02d}.jsonl"
        graphs = batches = 0
        total = 0.0
        last_loss = None
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                loss = r["loss"]
                if (r["epoch"] != epoch or r["step"] != steps + 1 or r["batch"] != batches
                        or r["graphs"] <= 0 or not math.isfinite(loss)):
                    raise ValueError(f"Inconsistent training record at {path}, batch {batches}")
                steps += 1
                batches += 1
                graphs += r["graphs"]
                total += loss * r["graphs"]
                last_loss = loss
        if not graphs:
            raise ValueError(f"Empty training log: {path}")
        row = dict(epoch=epoch, graphs=graphs, batches=batches, last_step=steps,
                   graph_weighted_mean_training_loss=total / graphs, last_batch_training_loss=last_loss)
        if epoch in S.P.CHECKS:
            dev = S.P.read(run / f"dev-{epoch:02d}.json")
            (standard,) = [d for d in dev["clean"] if d["split"] == "dev" and d["horizon"] == "D"]
            row["dev_standard"] = {k: standard[k] for k in (
                "graphs", "mean_value", "mean_first4", "graph_value_all")}
        epochs.append(row)
    if steps != meta["steps"]:
        raise ValueError(f"Expected {meta['steps']} batches, found {steps}")
    return dict(identity=meta["identity"], final_sha256=meta["final_sha256"], epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    S.P.write(args.output, history(args.run))
