#!/usr/bin/env python3
"""Verify the scratch start at the original token level.

The report exporter only sees the first four recorded factors.  This script
reconstructs the six-factor start tensor with the actual tokenizer and checks
that ``new_scratch`` differs from the raw input only at the one changed
constant VALUE.  It does not load a model or touch GPU.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import jsonl, read_json, sha, write_json
import reassignment_core as P

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def tokenizer_for(config_dir_name):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments"))
    import yaml
    from Simtransformer.simtransformer.utils import EasyDict
    from tokenizers import build_factored_tokenizer

    cdir = ROOT / "experiments" / "configs" / config_dir_name
    with (cdir / "data_config.yaml").open() as f:
        dc = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return build_factored_tokenizer(dc)


def audit_run(path, tok, model_key):
    path = Path(path)
    meta = read_json(path / "meta.json")
    if meta.get("status") != "complete" or meta.get("protocol") != "discrete_reassignment_v1":
        raise ValueError("incomplete run: " + str(path))
    if meta.get("model_key") != model_key:
        raise ValueError("wrong model_key: " + str(path))
    factors = tok.factors
    V = factors.index("VALUE")
    RG = factors.index("REGRET")
    MD = factors.index("MODIFICATION")
    maps = [getattr(tok, f.lower() + "_idx2tok") for f in factors[:4]]
    false_id = tok.regret_tok2idx["FALSE"]
    cases = positions = cells = changed_cells = unexpected = aux_bad = record_bad = 0
    for rec in jsonl(path / "records.jsonl"):
        positions += len(rec["positions"])
        raw = rec["raw_input_tokens"]
        nodes = rec["nodes"]
        depths = {int(k): int(v) for k, v in rec["depths"].items()}
        parents = {int(k): [int(x) for x in v] for k, v in rec["parents"].items()}
        constants = {int(k): int(v) for k, v in rec["constants"].items()}
        old = {p: int(rec["old_truth4"][i][3]) for i, p in enumerate(rec["positions"])}
        plan = P.make_plan(nodes, depths, parents, constants, old,
                           rec["leaf"], rec["constant"], rec["delta"])
        starts = P.starts(rec["completed_tokens"], raw, plan, V,
                          lambda v: tok.value_tok2idx[str(v)])
        branch = starts["new_scratch"]
        if branch["truth_ref"] != "new" or branch["start_kind"] != "scratch" or branch["pinned"]:
            raise ValueError("unexpected scratch branch metadata")
        expected = branch["state"]
        if len(expected) != len(raw):
            raise ValueError("raw/reconstructed start length differs")
        for p in range(len(raw)):
            cells += len(factors)
            for f in range(len(factors)):
                if p == rec["constant"] and f == V:
                    changed_cells += 1
                    if expected[p][f] != tok.value_tok2idx[str(plan["new_constant"])]:
                        unexpected += 1
                elif expected[p][f] != raw[p][f]:
                    unexpected += 1
                if f in (RG, MD) and raw[p][f] != false_id:
                    aux_bad += 1
        # The saved first-four record must agree with the reconstructed tensor.
        init = rec["branches"]["new_scratch"]["initial"]
        got = [[maps[f][expected[p][f]] for f in range(4)]
               for p in rec["positions"]]
        if got != init:
            record_bad += 1
        cases += 1
    return dict(model_key=model_key, records=cases, positions_checked=positions,
                six_factor_cells_checked=cells, expected_constant_value_cells=changed_cells,
                unexpected_six_factor_differences=unexpected,
                raw_aux_nonfalse_cells=aux_bad,
                recorded_initial_mismatch_cases=record_bad,
                records_sha256=meta["records_sha256"],
                selection_sha256=meta["selection_sha256"],
                config_dir_name=meta["model"]["config_dir_name"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if len(args.run_dir) != 2:
        p.error("exactly two run directories are required")
    metas = [read_json(Path(d) / "meta.json") for d in args.run_dir]
    if {m.get("model_key") for m in metas} != {"model_a", "model_b"}:
        raise ValueError("model_a and model_b are both required")
    toks = [tokenizer_for(m["model"]["config_dir_name"]) for m in metas]
    signatures = {(tuple(t.vocab_sizes), t.mod_val, tuple(t.factors), t.n_vars) for t in toks}
    if len(signatures) != 1:
        raise ValueError("A/B tokenizers differ")
    audits = [audit_run(d, t, m["model_key"]) for d, t, m in zip(args.run_dir, toks, metas)]
    total = dict(
        protocol="scratch_start_audit_v1",
        tokenizer=dict(config_dir_name=[m["model"]["config_dir_name"] for m in metas],
                       factors=toks[0].factors, mod_val=toks[0].mod_val,
                       n_vars=toks[0].n_vars, vocab_sizes=list(toks[0].vocab_sizes)),
        checks=audits,
        totals=dict(records=sum(a["records"] for a in audits),
                    positions=sum(a["positions_checked"] for a in audits),
                    six_factor_cells=sum(a["six_factor_cells_checked"] for a in audits),
                    expected_constant_value_cells=sum(a["expected_constant_value_cells"] for a in audits),
                    unexpected_differences=sum(a["unexpected_six_factor_differences"] for a in audits),
                    raw_aux_nonfalse_cells=sum(a["raw_aux_nonfalse_cells"] for a in audits),
                    recorded_initial_mismatch_cases=sum(a["recorded_initial_mismatch_cases"] for a in audits)),
        code_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))},
        note="This checks the six-factor start tensor reconstructed by reassignment_core.starts; "
             "it does not load or run the model.",
    )
    if total["totals"]["unexpected_differences"] or total["totals"]["raw_aux_nonfalse_cells"] or total["totals"]["recorded_initial_mismatch_cases"]:
        raise ValueError("scratch start audit failed: " + repr(total["totals"]))
    write_json(args.out, total)
    print(repr(total["totals"]))


if __name__ == "__main__":
    main()
