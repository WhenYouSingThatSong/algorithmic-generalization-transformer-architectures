"""Fixed eight-graph dev evaluation using the formal UPDATE evaluator."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_protocol as S
from assess import focused
from scenarios import summarize

P = S.P


def measure(rt, x, y, g, case):
    from evaluate import baseline

    nodes = g["nodes"]
    rounds = g["D"] + 1
    rows, _ = baseline(rt, x, y, g, rounds, set())
    final = rows[-1]["output"]
    if len(final) != len(nodes) or not nodes:
        raise ValueError("missing clean node observations")
    gold = y[nodes, 3].tolist()
    clean = dict(gid=case["gid"], rounds=rounds, nodes=len(nodes),
                 value_correct=sum(out[0] == truth for out, truth in zip(final, gold)),
                 first4_correct=sum(bool(out[1]) for out in final))
    records = list(focused(rt, x, y, g, "dev", case["gid"], case["target"]))
    return clean, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=S.MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, help="optional checkpoint from this registered run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from runtime import Runtime, setup
    setup()
    import numpy as np
    import torch
    from evaluate import load_split

    plan = S.validate(args.plan, args.manifest)
    manifest = P.read(args.manifest)
    cases = manifest["splits"]["dev"]["cases"][:8]
    if [c["gid"] for c in cases] != list(range(8)):
        raise ValueError("expected registered dev graphs 0 through 7")
    torch.set_num_threads(4)
    seed = P.seed(S.VERSION, args.model, "training") % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    src = next(m for m in plan["models"] if m["id"] == args.model)
    spec = dict(src, arm="update")
    epoch = 0
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        identity = ck["identity"]
        expected = dict(version=S.VERSION, model=args.model, source=src["sha256"],
                        manifest=P.sha(args.manifest))
        if any(identity.get(k) != v for k,v in expected.items()):
            raise ValueError("checkpoint and registered source differ")
        epoch = ck["epoch"]
        del ck
        spec.update(path=str(args.checkpoint), sha256=P.sha(args.checkpoint))
    rt = Runtime(spec, args.device, initialize_update=args.checkpoint is None)
    rt.model.eval()
    x,y,graphs = load_split(rt, plan["splits"]["dev"])
    clean, records = [], []
    with torch.inference_mode():
        for case in cases:
            i = case["gid"]
            observed, branches = measure(rt, x[i:i+1].to(rt.device), y[i].to(rt.device), graphs[i], case)
            clean.append(observed)
            records.extend(branches)
    n = sum(r["nodes"] for r in clean)
    result = dict(scope="dev_only", epoch=epoch, model=args.model, weights_sha256=spec["sha256"],
                  initialized_update=args.checkpoint is None, manifest_sha256=P.sha(args.manifest),
                  evaluator_sha256=P.sha(__file__), clean_graphs=clean, nodes=n,
                  value_accuracy=sum(r["value_correct"] for r in clean)/n,
                  first4_accuracy=sum(r["first4_correct"] for r in clean)/n,
                  update_and_repair=summarize(records), records=records)
    P.write(args.output, result)
    print(f"dev: {len(clean)} graphs, {n} nodes, VALUE {result['value_accuracy']:.3f}, "
          f"first4 {result['first4_accuracy']:.3f}, epoch {epoch}")


if __name__ == "__main__":
    main()
