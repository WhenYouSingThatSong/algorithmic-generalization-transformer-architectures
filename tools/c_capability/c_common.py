"""Torch-free shared helpers for the C0 capability exploration.

Design: docs/records/2026-11-14/c0-capability-exploration-plan.md

This module intentionally does not import torch or model code, so the summary
and the data-selection logic can be tested on CPU without loading a checkpoint.
The runner imports torch separately after configuring thread limits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR_NAME = "C0-M4-Nodes32-ADD-L4H16D384-DeBERTa"
DEFAULT_DATA_DIR = ROOT / "tmp/analysis/c0-capability-20261114/data"
DEFAULT_RUN_DIR = ROOT / "tmp/analysis/c0-capability-20261114/runs"

MOD_VAL = 23
N_VARS = 128
MAX_FAN_IN = 3
HORIZONS = (2, 4, 8, 16, 32, 64)

# Factor order is fixed by FactoredVocabTokenizer.  These indices are used only
# for raw arrays; the runner still reads the tokenizer's own factor indices.
SYNTAX = 0
VARIABLE = 1
OPERATION = 2
VALUE = 3
REGRET = 4
MODIFICATION = 5

# Six fixed strata from the approved design.  Existing pools are used when they
# have enough eligible graphs; otherwise build_data.py can generate a fresh
# pool with the official generator, using the same 128-name variable pool.
STRATA: dict[str, dict[str, Any]] = {
    "N32_shallow": dict(
        n_nodes=32,
        depth_group="shallow",
        source=ROOT / "tmp/analysis/matched-depth/ADD/testdata_N32.pt",
        source_kind="existing",
    ),
    "N32_deep": dict(
        n_nodes=32,
        depth_group="deep",
        source=ROOT / "tmp/analysis/heldout-seed42/ADD/testdata_N32.pt",
        source_kind="existing",
    ),
    "N64_shallow": dict(
        n_nodes=64,
        depth_group="shallow",
        source=ROOT / "experiments/evaluation/val_datasets/Tr32Test128/ADD/testdata_N64.pt",
        source_kind="existing_or_generate",
    ),
    "N64_deep": dict(
        n_nodes=64,
        depth_group="deep",
        source=ROOT / "experiments/evaluation/val_datasets/Tr32Test128/ADD/testdata_N64.pt",
        source_kind="existing",
    ),
    "N128_shallow": dict(
        n_nodes=128,
        depth_group="shallow",
        source=None,
        source_kind="generate",
    ),
    "N128_deep": dict(
        n_nodes=128,
        depth_group="deep",
        source=ROOT / "tmp/analysis/heldout-seed42/ADD/testdata_N128.pt",
        source_kind="existing",
    ),
}

# Fixed weights requested by the approved plan.  Keep the actual path and hash
# in the run metadata; do not silently fall back to another epoch.
WEIGHTS: dict[str, dict[str, Any]] = {
    "s42_e143": dict(
        seed=42,
        group="Nodes32-ADD - L4H16D384_DeBERTa",
        run="seed-42 - 2026-11-14-02:35:10",
        ckpt="epoch=143-val_loss=0.0000.ckpt",
        base="tmp/runs/oldC0-resume-150ep/checkpoints",
        sha256="4a00d6a21a7ed6ec9730daccf27f6077af45d2f2d7ff7376386bf0aa8903a798",
    ),
    "s1234_e143": dict(
        seed=1234,
        group="Nodes32-ADD - L4H16D384_DeBERTa",
        run="seed-1234 - 2026-11-14-02:38:10",
        ckpt="epoch=143-val_loss=0.0000.ckpt",
        base="tmp/runs/oldC0-resume-150ep/checkpoints",
        sha256="0bca3a2b91e82d70d5808859d3a907e0b8a32e0a852ba483c208f955d7d70475",
    ),
}


def as_numpy(value: Any) -> np.ndarray:
    """Convert a tensor/array without requiring torch in this module."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def max_depth_per_graph(info: Any) -> np.ndarray:
    """Actual maximum dependency depth per graph from dataset_info[:, :, 0]."""
    info = as_numpy(info)
    return info[:, :, 0].max(axis=1).astype(np.int32)


def nonpad_lengths(tokens: Any, pad_syntax_idx: int) -> np.ndarray:
    tokens = as_numpy(tokens)
    return (tokens[:, :, 0] != pad_syntax_idx).sum(axis=1).astype(np.int32)


def spec_from_tensors(tokens: Any, info: Any, pad_syntax_idx: int) -> dict[str, Any]:
    depths = max_depth_per_graph(info)
    nonpad = nonpad_lengths(tokens, pad_syntax_idx)
    return dict(
        min_depth=int(depths.min()),
        mean_depth=float(depths.mean()),
        max_depth=int(depths.max()),
        min_non_pad_len=int(nonpad.min()),
        mean_non_pad_len=float(nonpad.mean()),
        max_non_pad_len=int(nonpad.max()),
        orig_seq_len=int(as_numpy(tokens).shape[1]),
    )


def depth_group_mask(depths: Any, depth_group: str) -> np.ndarray:
    depths = as_numpy(depths)
    if depth_group == "shallow":
        return depths <= 6
    if depth_group == "deep":
        return (depths >= 7) & (depths <= 12)
    raise ValueError(f"unknown depth_group: {depth_group!r}")


def select_indices(depths: Any, depth_group: str, count: int,
                   seed: int) -> tuple[list[int], int]:
    """Deterministic subset selection independent of model outputs.

    Returns (selected original indices, number of eligible graphs).  The
    selected indices are sorted after the seeded permutation for a stable file
    order; no correctness signal is used.
    """
    if count < 1:
        raise ValueError("count must be positive")
    depths = as_numpy(depths)
    eligible = np.flatnonzero(depth_group_mask(depths, depth_group))
    rng = np.random.default_rng(seed)
    order = rng.permutation(eligible)
    selected = np.sort(order[:count]).astype(int).tolist()
    return selected, int(len(eligible))


def parse_parents(info_row: Any) -> list[int]:
    """Parents of one node from dataset_info[:, 1:] (position/op pairs)."""
    row = as_numpy(info_row)
    parents: list[int] = []
    # columns 1: are alternating parent position / operator id
    for j in range(1, len(row) - 1, 2):
        parent = int(row[j])
        if parent >= 0:
            parents.append(parent)
    return parents


def node_meta(info: Any, numeric: Any) -> list[dict[str, Any]]:
    """Per-graph node metadata for the numeric node set.

    numeric: bool [B, T] mask.  Returns one dict per graph with positions in
    the reduced sequence, depth, leaf/internal/sink flags, and parsed parents.
    """
    info = as_numpy(info)
    numeric = as_numpy(numeric).astype(bool)
    result: list[dict[str, Any]] = []
    for graph_idx in range(info.shape[0]):
        positions = np.flatnonzero(numeric[graph_idx]).astype(np.int32)
        depth = info[graph_idx, positions, 0].astype(np.int16)
        children: dict[int, set[int]] = {int(p): set() for p in positions}
        parents: dict[int, list[int]] = {}
        for pos in positions:
            p = int(pos)
            parsed = parse_parents(info[graph_idx, p])
            parents[p] = parsed
            for parent in parsed:
                if parent in children:
                    children[parent].add(p)
        is_sink = np.asarray([len(children[int(p)]) == 0 for p in positions], dtype=bool)
        result.append(dict(
            positions=positions,
            depth=depth,
            is_leaf=(depth == 0),
            is_internal=(depth >= 1),
            is_sink=is_sink,
            parents=parents,
        ))
    return result


def first_true_index(values: Any) -> int | None:
    """1-based first True index, or None if the array has no True."""
    values = as_numpy(values).astype(bool)
    idx = np.flatnonzero(values)
    if len(idx) == 0:
        return None
    return int(idx[0]) + 1


def completion_drift(all_correct: Any, completion_round: int | None) -> dict[str, Any]:
    """Post-completion behavior of one graph-level correctness series.

    completion_round is 1-based; observations after the completion round are
    all_correct[completion_round:].  "recovered" is None when there was no
    degradation to recover from.
    """
    all_correct = as_numpy(all_correct).astype(bool)
    if completion_round is None or completion_round < 1 or completion_round > len(all_correct):
        return dict(observed=False, post_rounds=0, ever_degraded=False,
                    first_degradation_rel_round=None, recovered=None,
                    final_correct=None)
    post = all_correct[completion_round:]
    if len(post) == 0:
        return dict(observed=True, post_rounds=0, ever_degraded=False,
                    first_degradation_rel_round=None, recovered=None,
                    final_correct=None)
    degraded = ~post
    ever = bool(degraded.any())
    first_rel = int(np.flatnonzero(degraded)[0]) + 1 if ever else None
    return dict(
        observed=True,
        post_rounds=int(len(post)),
        ever_degraded=ever,
        first_degradation_rel_round=first_rel,
        recovered=(bool(post[-1]) if ever else None),
        final_correct=bool(post[-1]),
    )


def other_field_drift(value_all: Any, first4_all: Any,
                      value_completion_round: int | None) -> dict[str, Any]:
    """After VALUE completion, whether first-four fields alone degrade."""
    value_all = as_numpy(value_all).astype(bool)
    first4_all = as_numpy(first4_all).astype(bool)
    if value_completion_round is None or value_completion_round < 1:
        return dict(observed=False, post_rounds=0, ever_other_only_degraded=False,
                    first_other_only_rel_round=None, final_value_correct=None,
                    final_first4_correct=None)
    post_value = value_all[value_completion_round:]
    post_first4 = first4_all[value_completion_round:]
    if len(post_value) == 0:
        return dict(observed=True, post_rounds=0, ever_other_only_degraded=False,
                    first_other_only_rel_round=None, final_value_correct=None,
                    final_first4_correct=None)
    other_only = post_value & ~post_first4
    ever = bool(other_only.any())
    first_rel = int(np.flatnonzero(other_only)[0]) + 1 if ever else None
    return dict(
        observed=True,
        post_rounds=int(len(post_value)),
        ever_other_only_degraded=ever,
        first_other_only_rel_round=first_rel,
        final_value_correct=bool(post_value[-1]),
        final_first4_correct=bool(post_first4[-1]),
    )


def weight_path(name: str) -> Path:
    spec = WEIGHTS[name]
    return ROOT / spec["base"] / f'{spec["group"]}-{spec["run"]}' / spec["ckpt"]


def generated_dag_config(n_nodes: int, depth_group: str, num_samples: int,
                         max_length: int = 1024) -> dict[str, Any]:
    """Official-generator config for missing shallow strata."""
    max_depth = 6 if depth_group == "shallow" else 12
    return dict(
        data_file_name=None,
        max_comp_depth=max_depth,
        num_nodes=int(n_nodes),
        max_depth=max_depth,
        func_vocab=["ADD"],
        mod_val=MOD_VAL,
        min_fan_in_deg=1,
        max_fan_in_deg=MAX_FAN_IN,
        fix_graph=False,
        num_leaf_nodes=6,
        shuffle_node_name=True,
        max_length=int(max_length),
        num_samples=int(num_samples),
        verbose=False,
    )


def jsonl_writer(path: str | Path):
    """Small append-only JSONL writer for one record per graph."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("x", encoding="utf-8")

    def append(record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    def close() -> None:
        handle.close()

    return append, close


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"
