#!/usr/bin/env python3
"""Build the six fixed C0 capability strata.

Design: docs/records/2026-11-14/c0-capability-exploration-plan.md

The builder is deliberately separate from the model runner:
  * existing pools are filtered by actual max depth and subsampled with a fixed
    seed; no model output is consulted;
  * missing strata can be generated with the official DAG generator, using the
    full 128-name variable pool;
  * each selected stratum is saved with its source indices, hashes, depth stats
    and generator config so the runner can record data identity.

This script was written for GPT review and has not been run in this session.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from . import c_common
except ImportError:  # direct script execution
    import c_common

ROOT = c_common.ROOT
EXPERIMENTS = ROOT / "experiments"
for path in (str(ROOT), str(EXPERIMENTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch

from Simtransformer.simtransformer.utils import EasyDict
from tokenizers import FactoredVocabTokenizer
from generate_data import generate_dataset


def load_source(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "tokenized_dataset" not in data or "dataset_info" not in data:
        raise ValueError(f"source data missing tokenized_dataset/dataset_info: {path}")
    return data


def subset_data(data: dict, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return data["tokenized_dataset"][idx], data["dataset_info"][idx]


def make_meta(stratum: str, spec: dict, cfg: dict, selection_seed: int,
              selected: list[int], eligible: int, source: Path | None,
              generated: bool, generation_seed: int | None,
              generation_config: dict | None = None) -> dict:
    meta = dict(
        stratum=stratum,
        n_nodes=cfg["n_nodes"],
        depth_group=cfg["depth_group"],
        selection_seed=int(selection_seed),
        selected_count=len(selected),
        eligible_count=int(eligible),
        source_indices=[int(i) for i in selected],
        source=str(source) if source is not None else None,
        source_sha256=c_common.sha256_file(source) if source is not None and source.exists() else None,
        generated=bool(generated),
        generation_seed=generation_seed,
        generation_config=(dict(generation_config) if generated and generation_config is not None else None),
        tokenizer_n_vars=c_common.N_VARS,
        vocab_sizes=[7, c_common.N_VARS + 1, 2, 25, 2, 2],
        spec=spec,
    )
    return meta


def build_existing(stratum: str, cfg: dict, count: int, selection_seed: int,
                   pad_idx: int) -> dict | None:
    source = cfg.get("source")
    if source is None or not Path(source).exists():
        return None
    data = load_source(Path(source))
    depths = c_common.max_depth_per_graph(data["dataset_info"])
    selected, eligible = c_common.select_indices(
        depths, cfg["depth_group"], count, selection_seed)
    if len(selected) < count:
        return dict(shortfall=True, eligible=eligible, selected=len(selected),
                    source=str(source))
    tokens, info = subset_data(data, selected)
    spec = c_common.spec_from_tensors(tokens, info, pad_idx)
    meta = make_meta(stratum, spec, cfg, selection_seed, selected, eligible,
                     Path(source), generated=False, generation_seed=None)
    return dict(shortfall=False, tokens=tokens, info=info, spec=spec, meta=meta,
                source=str(source), eligible=eligible, selected=selected)


def build_generated(stratum: str, cfg: dict, count: int, selection_seed: int,
                    generation_seed: int, generation_samples: int,
                    pad_idx: int) -> dict:
    random.seed(generation_seed)
    np.random.seed(generation_seed)
    torch.manual_seed(generation_seed)

    tokenizer = FactoredVocabTokenizer(
        n_vars=c_common.N_VARS,
        ops=["ADD"],
        mod_val=c_common.MOD_VAL,
        max_fan_in=c_common.MAX_FAN_IN,
    )
    dag_config = EasyDict(c_common.generated_dag_config(
        cfg["n_nodes"], cfg["depth_group"], generation_samples))
    tokens, info = generate_dataset(dag_config, tokenizer)

    depths = c_common.max_depth_per_graph(info)
    selected, eligible = c_common.select_indices(
        depths, cfg["depth_group"], count, selection_seed)
    if len(selected) < count:
        raise RuntimeError(
            f"{stratum}: generated {generation_samples} graphs but only "
            f"{eligible} satisfy depth_group={cfg['depth_group']}; "
            f"increase --generation-samples")

    tokens, info = subset_data(
        {"tokenized_dataset": tokens, "dataset_info": info}, selected)
    spec = c_common.spec_from_tensors(tokens, info, pad_idx)
    meta = make_meta(stratum, spec, cfg, selection_seed, selected, eligible,
                     source=None, generated=True, generation_seed=generation_seed,
                     generation_config=dict(dag_config))
    return dict(shortfall=False, tokens=tokens, info=info, spec=spec, meta=meta,
                source="generated", eligible=eligible, selected=selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=c_common.DEFAULT_DATA_DIR)
    parser.add_argument("--count", type=int, default=64,
                        help="graphs per stratum (default: 64)")
    parser.add_argument("--selection-seed", type=int, default=20261114,
                        help="seed for fixed graph selection; never uses model outputs")
    parser.add_argument("--generation-seed", type=int, default=20261114,
                        help="seed for official-generator fallback")
    parser.add_argument("--generation-samples", type=int, default=256,
                        help="candidate graphs when generating a missing stratum")
    parser.add_argument("--strata", type=str, default="all",
                        help="comma-separated stratum names, or 'all'")
    parser.add_argument("--allow-generate", action="store_true",
                        help="generate missing/insufficient strata with the official generator")
    parser.add_argument("--dry-run", action="store_true",
                        help="report source eligibility without writing data")
    parser.add_argument("--overwrite-ok", action="store_true")
    args = parser.parse_args()

    names = list(c_common.STRATA) if args.strata == "all" else [
        s.strip() for s in args.strata.split(",") if s.strip()
    ]
    unknown = [s for s in names if s not in c_common.STRATA]
    if unknown:
        parser.error(f"unknown strata: {unknown}")

    out_dir = args.out_dir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite_ok:
            parser.error(f"{manifest_path} exists; pass --overwrite-ok to replace")

    tokenizer = FactoredVocabTokenizer(
        n_vars=c_common.N_VARS, ops=["ADD"],
        mod_val=c_common.MOD_VAL, max_fan_in=c_common.MAX_FAN_IN)
    pad_idx = tokenizer.syntax_tok2idx[tokenizer.PAD]

    manifest = dict(
        design="docs/records/2026-11-14/c0-capability-exploration-plan.md",
        count=args.count,
        selection_seed=args.selection_seed,
        generation_seed=args.generation_seed,
        generation_samples=args.generation_samples,
        tokenizer_n_vars=c_common.N_VARS,
        vocab_sizes=list(tokenizer.vocab_sizes),
        strata={},
    )

    for name in names:
        cfg = c_common.STRATA[name]
        result = build_existing(name, cfg, args.count, args.selection_seed, pad_idx)
        if result is not None and not result.get("shortfall"):
            pass
        else:
            if result is not None and result.get("shortfall"):
                print(f"[{name}] existing pool short: {result['selected']}/{args.count} "
                      f"(eligible={result['eligible']}); generation required")
            elif cfg.get("source") is not None:
                print(f"[{name}] source missing: {cfg['source']}; generation required")
            else:
                print(f"[{name}] no existing source; generation required")
            if args.dry_run or not args.allow_generate:
                if not args.dry_run:
                    raise RuntimeError(
                        f"{name} needs generation; pass --allow-generate or use --dry-run")
                manifest["strata"][name] = dict(
                    status="needs_generation",
                    source=str(cfg.get("source")) if cfg.get("source") else None,
                    existing=(result or {}),
                )
                continue
            result = build_generated(
                name, cfg, args.count, args.selection_seed, args.generation_seed,
                args.generation_samples, pad_idx)

        if args.dry_run:
            manifest["strata"][name] = dict(
                status="ready",
                source=result["source"],
                eligible=result["eligible"],
                selected=len(result["selected"]),
                spec=result["spec"],
            )
            print(f"[{name}] ready: selected {len(result['selected'])}/{args.count} "
                  f"from {result['source']} (eligible={result['eligible']})")
            continue

        out_file = out_dir / f"{name}.pt"
        torch.save({
            "tokenized_dataset": result["tokens"],
            "dataset_info": result["info"],
            "spec": result["spec"],
            "meta": result["meta"],
        }, out_file)
        file_sha = c_common.sha256_file(out_file)
        entry = dict(result["meta"])
        entry.update(file=str(out_file), sha256=file_sha,
                     status="generated" if result["meta"]["generated"] else "existing")
        manifest["strata"][name] = entry
        print(f"[{name}] wrote {out_file} ({len(result['selected'])} graphs, "
              f"depth {result['spec']['min_depth']}-{result['spec']['max_depth']}, "
              f"sha256={file_sha[:12]}...)")

    if args.dry_run:
        print(json_dumps(manifest))
        return 0

    c_common.write_json(out_dir / "manifest.json", manifest)
    print(f"wrote {out_dir / 'manifest.json'}")
    return 0


def json_dumps(value) -> str:
    import json
    return json.dumps(value, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
