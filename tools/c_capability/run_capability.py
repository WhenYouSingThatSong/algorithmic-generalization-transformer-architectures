#!/usr/bin/env python3
"""Run one fixed C0 weight over the six capability strata.

Design: docs/records/2026-11-14/c0-capability-exploration-plan.md

Semantics:
  * start from the original token input (variable VALUE = EMPTY);
  * for every round, feed the model's continuous hidden state back;
  * decode only for observation, never write decoded fields back;
  * record every numeric node's first four predicted fields and fixed truth;
  * run 64 rounds by default and save one npz per graph.

This runner was written for GPT review and has not been executed in this
session.  It does not touch training runs or existing checkpoints.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# trajectory_io must be imported before torch so --num_threads is applied
# before any torch/OpenMP initialization.
TOOLS_DIR = Path(__file__).resolve().parents[1]
B_DIR = TOOLS_DIR / "b_mechanism"
if str(B_DIR) not in sys.path:
    sys.path.insert(0, str(B_DIR))
from trajectory_io import add_runtime_arguments, configure_runtime  # noqa: E402

configure_runtime()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

try:
    from . import c_common
except ImportError:  # direct script execution
    import c_common

ROOT = c_common.ROOT
EXPERIMENTS = ROOT / "experiments"
for path in (str(ROOT), str(EXPERIMENTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Simtransformer.simtransformer.utils import EasyDict  # noqa: E402
from model import DAGTeacherModel, LitRecurrentTransformerModel  # noqa: E402
from tokenizers import build_factored_tokenizer  # noqa: E402
from train import DAGDataModule  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.set_defaults(progress_every=8)
    parser.add_argument("--data-dir", type=Path, default=c_common.DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=c_common.DEFAULT_RUN_DIR)
    parser.add_argument("--weight", type=str, default="s42_e143",
                        help="name in c_common.WEIGHTS, or a custom name with --ckpt")
    parser.add_argument("--weight-name", type=str, default=None,
                        help="output name when --ckpt is supplied")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="explicit checkpoint path; overrides --weight")
    parser.add_argument("--ckpt-base-dir", type=Path, default=None,
                        help="override checkpoint base dir for WEIGHTS")
    parser.add_argument("--group-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--ckpt-name", type=str, default=None)
    parser.add_argument("--config-dir-name", type=str, default=c_common.CONFIG_DIR_NAME)
    parser.add_argument("--strata", type=str, default="all",
                        help="comma-separated stratum names, or 'all'")
    parser.add_argument("--rounds", type=int, default=64,
                        help="free-running rounds per graph (default: 64)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the recurrent model (eval speed only)")
    parser.add_argument("--limit-graphs", type=int, default=None,
                        help="write at most N graphs per stratum (smoke/tests only)")
    parser.add_argument("--overwrite-ok", action="store_true",
                        help="allow writing into an output dir with an existing index.jsonl")
    return parser.parse_args()


def load_configs(config_dir_name: str):
    cdir = EXPERIMENTS / "configs" / config_dir_name
    with open(cdir / "model_config.yaml") as handle:
        model_config = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    with open(cdir / "train_config.yaml") as handle:
        train_config = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    with open(cdir / "data_config.yaml") as handle:
        data_config = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    return model_config, train_config, data_config


def resolve_checkpoint(args: argparse.Namespace) -> tuple[str, Path, dict]:
    if args.ckpt is not None:
        name = args.weight_name or "custom"
        path = args.ckpt
        return name, path, dict(seed=None, group=None, run=None, ckpt=str(path),
                                base=None, sha256=None)
    if args.weight not in c_common.WEIGHTS:
        raise SystemExit(f"unknown weight {args.weight!r}; choose one of "
                         f"{sorted(c_common.WEIGHTS)} or pass --ckpt")
    spec = c_common.WEIGHTS[args.weight]
    base = Path(args.ckpt_base_dir) if args.ckpt_base_dir else c_common.ROOT / spec["base"]
    group = args.group_name or spec["group"]
    run = args.run_name or spec["run"]
    ckpt = args.ckpt_name or spec["ckpt"]
    path = base / f"{group}-{run}" / ckpt
    return args.weight, path, dict(spec, base=str(base), group=group, run=run, ckpt=ckpt)


def split_batch(batch: torch.Tensor, model_config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = batch[..., :model_config.data_dim]
    label = batch[..., model_config.data_dim:model_config.data_dim + model_config.label_dim]
    info = batch[..., model_config.data_dim + model_config.label_dim:]
    return data, label, info


def build_model_and_data(args, stratum_file: Path):
    model_config, train_config, data_config = load_configs(args.config_dir_name)
    tokenizer = build_factored_tokenizer(data_config)
    model_config.vocab_sizes = tokenizer.vocab_sizes
    model_config.factors = tokenizer.factors

    data_module = DAGDataModule(
        data_path=[str(stratum_file), str(stratum_file)],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        factored_tokenizer=tokenizer,
    )
    data_module.setup()
    model_config.data_dim = data_module.data_dim
    model_config.label_dim = data_module.label_dim
    model_config.info_dim = data_module.info_dim

    teacher = DAGTeacherModel(model_config, tokenizer)
    litmodel = LitRecurrentTransformerModel(model_config, data_config, train_config, teacher)
    name, ckpt_path, weight_meta = resolve_checkpoint(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    litmodel.load_ckpt(str(ckpt_path))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    litmodel.to(device).eval()
    if args.compile:
        print("compiling recurrent model for eval ...")
        litmodel.model = torch.compile(litmodel.model)
    return dict(litmodel=litmodel, tokenizer=tokenizer, data_module=data_module,
                model_config=model_config, device=device, name=name,
                ckpt_path=ckpt_path, weight_meta=weight_meta)


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    names = list(c_common.STRATA) if args.strata == "all" else [
        s.strip() for s in args.strata.split(",") if s.strip()
    ]
    unknown = [s for s in names if s not in c_common.STRATA]
    if unknown:
        raise SystemExit(f"unknown strata: {unknown}")

    started = time.perf_counter()
    # Resolve weight identity once; build_model_and_data repeats it per stratum
    # but the path/hash is stable and cheap to verify.
    _, ckpt_path, weight_meta = resolve_checkpoint(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    weight_sha = c_common.sha256_file(ckpt_path)

    # One output root per weight; each stratum gets its own graph directory.
    first = True
    run_meta_path = None
    index_path = None
    index_append = index_close = None
    written_total = 0
    strata_done = []

    for name in names:
        stratum_file = args.data_dir / f"{name}.pt"
        if not stratum_file.exists():
            raise FileNotFoundError(
                f"stratum file missing: {stratum_file}; run build_data.py first")
        data_blob = torch.load(stratum_file, map_location="cpu", weights_only=False)
        data_meta = data_blob.get("meta", {})
        data_sha = c_common.sha256_file(stratum_file)

        ctx = build_model_and_data(args, stratum_file)
        litmodel = ctx["litmodel"]
        tokenizer = ctx["tokenizer"]
        data_module = ctx["data_module"]
        mc = ctx["model_config"]
        device = ctx["device"]
        weight_name = ctx["name"]
        ckpt_path = ctx["ckpt_path"]
        weight_meta = ctx["weight_meta"]

        out_root = args.out_dir / weight_name
        if first:
            out_root.mkdir(parents=True, exist_ok=True)
            run_meta_path = out_root / "run_meta.json"
            index_path = out_root / "index.jsonl"
            if index_path.exists() and not args.overwrite_ok:
                raise FileExistsError(f"{index_path} exists; use a new --out-dir or --overwrite-ok")
            index_append, index_close = c_common.jsonl_writer(index_path)
            c_common.write_json(run_meta_path, dict(
                status="running", weight=weight_name, weight_meta=weight_meta,
                weight_path=str(ckpt_path), weight_sha256=weight_sha,
                config_dir_name=args.config_dir_name,
                args={k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
                tokenizer_n_vars=getattr(tokenizer, "n_vars", c_common.N_VARS),
                vocab_sizes=list(tokenizer.vocab_sizes), factors=list(tokenizer.factors),
                rounds=args.rounds, strata=names, started_at=time.time(),
            ))
            first = False

        pad_idx = tokenizer.syntax_tok2idx[tokenizer.PAD]
        syntax_idx = tokenizer.factors.index("SYNTAX")
        value_idx = tokenizer.factors.index("VALUE")
        loader = data_module.val_dataloader()
        graph_index = 0
        written_this = 0
        for batch in loader:
            if args.limit_graphs is not None and written_this >= args.limit_graphs:
                break
            batch = batch.to(device)
            batch = litmodel.reduce_batch(batch)
            data, label, info = split_batch(batch, mc)

            valid = data[..., syntax_idx] != pad_idx
            numeric = valid & (info[..., 0] >= 0) & (label[..., value_idx] < tokenizer.mod_val)
            numeric_np = numeric.detach().cpu().numpy()
            info_np = info.detach().cpu().numpy()
            data_np = data.detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            nodes = c_common.node_meta(info_np, numeric_np)
            batch_size = data.shape[0]

            histories = []
            for node in nodes:
                k = len(node["positions"])
                histories.append(np.empty((args.rounds, k, 4), dtype=np.int16))

            hidden = litmodel.model.token_embedder(data)
            with torch.no_grad():
                for round_idx in range(args.rounds):
                    hidden, factor_preds = litmodel.model(hidden, True)
                    decodes = litmodel.generate_factor_decodes(factor_preds, temperature=0.0)
                    preds = torch.stack(decodes, dim=-1)  # [B, T, 6]
                    preds_np = preds.detach().cpu().numpy()
                    for graph_idx in range(batch_size):
                        positions = nodes[graph_idx]["positions"]
                        histories[graph_idx][round_idx] = preds_np[graph_idx, positions, :4]

            for graph_idx in range(batch_size):
                if args.limit_graphs is not None and written_this >= args.limit_graphs:
                    break
                node = nodes[graph_idx]
                positions = node["positions"]
                truth4 = label_np[graph_idx, positions, :4].astype(np.int16)
                hist = histories[graph_idx]
                value_correct = hist[:, :, value_idx] == truth4[:, value_idx]
                first4_correct = np.all(hist == truth4[None, :, :], axis=-1)
                graph_value_all = np.all(value_correct, axis=1)
                graph_first4_all = np.all(first4_correct, axis=1)
                tV = c_common.first_true_index(graph_value_all)
                t4 = c_common.first_true_index(graph_first4_all)
                actual_max_depth = int(info_np[graph_idx, :, 0].max())
                seq_len = int((data_np[graph_idx, :, 0] != pad_idx).sum())

                graph_id = f"{name}:{graph_index:04d}"
                graph_path = out_root / "graphs" / name / f"{graph_index:04d}.npz"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    graph_path,
                    pred_syntax=hist[:, :, 0], pred_variable=hist[:, :, 1],
                    pred_operation=hist[:, :, 2], pred_value=hist[:, :, 3],
                    true_syntax=truth4[:, 0], true_variable=truth4[:, 1],
                    true_operation=truth4[:, 2], true_value=truth4[:, 3],
                    positions=positions.astype(np.int32),
                    depth=node["depth"],
                    is_leaf=node["is_leaf"], is_internal=node["is_internal"],
                    is_sink=node["is_sink"],
                    value_correct=value_correct, first4_correct=first4_correct,
                    graph_value_all=graph_value_all,
                    graph_first4_all=graph_first4_all,
                    tV=np.int32(-1 if tV is None else tV),
                    t4=np.int32(-1 if t4 is None else t4),
                    actual_max_depth=np.int32(actual_max_depth),
                    completion_ref_round=np.int32(actual_max_depth + 1),
                    n_rounds=np.int32(args.rounds),
                )

                source_indices = data_meta.get("source_indices")
                source_index = (int(source_indices[graph_index])
                                if source_indices is not None and graph_index < len(source_indices)
                                else None)
                index_append(dict(
                    graph_id=graph_id, stratum=name, weight=weight_name,
                    seed=weight_meta.get("seed"), source_index=source_index,
                    path=str(graph_path.relative_to(out_root)),
                    data_file=str(stratum_file), data_sha256=data_sha,
                    n_nodes=int(len(positions)), actual_max_depth=actual_max_depth,
                    completion_ref_round=actual_max_depth + 1, seq_len=seq_len,
                    tV=tV, t4=t4, n_rounds=args.rounds,
                    leaf_count=int(node["is_leaf"].sum()),
                    internal_count=int(node["is_internal"].sum()),
                    sink_count=int(node["is_sink"].sum()),
                ))
                graph_index += 1
                written_this += 1
                written_total += 1
                if args.progress_every and written_total % args.progress_every == 0:
                    print(f"[{weight_name}/{name}] {written_this} graphs written "
                          f"(tV={tV}, t4={t4}, depth={actual_max_depth})", flush=True)

        strata_done.append(dict(stratum=name, graphs=written_this,
                                data_file=str(stratum_file), data_sha256=data_sha,
                                spec=data_blob.get("spec"), meta=data_meta))
        print(f"[{weight_name}] stratum {name} done: {written_this} graphs", flush=True)

    if index_close is not None:
        index_close()
    if run_meta_path is not None:
        meta = c_common.read_json(run_meta_path)
        meta.update(status="complete", finished_at=time.time(),
                    elapsed_s=time.perf_counter() - started,
                    graphs_written=written_total, strata_done=strata_done)
        c_common.write_json(run_meta_path, meta)
    print(f"done: {written_total} graphs in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
