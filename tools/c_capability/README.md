# C0 capability and free-running exploration

Design: `docs/records/2026-11-14/c0-capability-exploration-plan.md`

Status: **GPT-reviewed; CPU tests and a 2-graph model smoke passed; the formal
six-stratum run for s42_e143 and s1234_e143 is complete.**

The smoke evidence is in `docs/records/2026-11-14/c0-capability-smoke.md`.
The formal data and results are under `tmp/analysis/c0-capability-20261114/`;
the report is `docs/records/2026-11-14/c0-capability-report.md`.

This directory implements the approved first C0 capability exploration:

- six fixed strata: N32/64/128 × actual max depth ≤6 / 7–12;
- two fixed weights: old C0 `s42_e143`, `s1234_e143`;
- 64 free-running rounds per graph through the continuous `nointerm` path;
- per-round per-node first-four readouts, VALUE/first-four metrics, `tV`/`t4`,
  and post-completion drift;
- `s2025` is intentionally excluded; `s2026` can be added later with `--ckpt`.

## Files

| File | Role |
|---|---|
| `c_common.py` | torch-free constants, depth/selection helpers, node metadata, completion/drift definitions |
| `build_data.py` | build the six fixed strata from existing pools; optionally generate missing shallow pools with the official generator |
| `run_capability.py` | load one checkpoint, free-run 64 rounds, write one `.npz` per graph plus `index.jsonl` and `run_meta.json` |
| `summarize.py` | torch-free aggregation into `curves.csv`, `capability_by_H.csv`, `by_node_type.csv`, `by_depth.csv`, `completion.csv`, `drift*.csv`, `examples.json`, `report.md` |
| `analyze_existing.py` | CPU-only re-analysis of existing NPZ files: micro/graph-equal field accuracy, error depth/counts, completion lag, per-graph failure/recovery events with positions, same-graph event order |
| `tests/test_c_common.py` | CPU-only tests for selection, node metadata and drift definitions; no model |
| `tests/test_c_summary.py` | CPU-only tests for graph-equal metrics and empty-bucket support |
| `tests/test_c_analysis.py` | CPU-only tests for the existing-NPZ event/table definitions (14 tests) |

## Data plan

Existing pools and the reason for each choice:

| stratum | source | eligible graphs (actual depth) | action |
|---|---|---|---|
| `N32_shallow` | `tmp/analysis/matched-depth/ADD/testdata_N32.pt` | 1024 (depth 3–6) | use existing |
| `N32_deep` | `tmp/analysis/heldout-seed42/ADD/testdata_N32.pt` | 319 (depth 7–11) | use existing |
| `N64_shallow` | `experiments/evaluation/val_datasets/Tr32Test128/ADD/testdata_N64.pt` | 54 (depth ≤6) | generate 64 with `--allow-generate` |
| `N64_deep` | same N64 file | 954 (depth 7–12) | use existing |
| `N128_shallow` | none | 0 | generate 64 with `--allow-generate` |
| `N128_deep` | `tmp/analysis/heldout-seed42/ADD/testdata_N128.pt` | 755 (depth 7–12) | use existing |

The builder never selects graphs by model correctness.  It shuffles eligible
indices with a fixed seed and records `source_indices`, source SHA256, actual
depth stats and the generator config.  Generated strata use the full 128-name
variable pool (vocab129), matching C0.

## Commands used for the formal run

```bash
# 1) Build data.  --allow-generate is needed for N64_shallow and N128_shallow.
python tools/c_capability/build_data.py --allow-generate

# Dry-run source eligibility only:
python tools/c_capability/build_data.py --dry-run

# 2) Run one weight over all strata.  Use a free GPU; the runner is serial.
python tools/c_capability/run_capability.py \
  --weight s42_e143 --strata all --device cuda:0 \
  --rounds 64 --batch-size 4 --num_threads 4

python tools/c_capability/run_capability.py \
  --weight s1234_e143 --strata all --device cuda:0 \
  --rounds 64 --batch-size 4 --num_threads 4

# Smoke only (writes at most 2 graphs per stratum, still 64 rounds):
python tools/c_capability/run_capability.py \
  --weight s42_e143 --strata N32_shallow --limit-graphs 2 \
  --device cuda:0 --rounds 8

# 3) Summarize both weights:
python tools/c_capability/summarize.py \
  --run-dir tmp/analysis/c0-capability-20261114/runs

# 4) CPU-only re-analysis of the existing NPZ traces (no model):
python -I -B tools/c_capability/analyze_existing.py \
  --run-dir tmp/analysis/c0-capability-20261114/runs \
  --out-dir tmp/analysis/c0-capability-20261114/analysis --overwrite-ok
```

`s2026` can be added after training finishes without changing the data or
metrics:

```bash
python tools/c_capability/run_capability.py \
  --ckpt tmp/runs/oldC0-newseed-150ep/checkpoints/<run>/<ckpt> \
  --weight-name s2026_eXXX --strata all --device cuda:0
```

## What is recorded

Per graph `.npz`:

- `pred_syntax/variable/operation/value`: `[rounds, n_nodes]` int16 readouts;
- `true_*`: `[n_nodes]` fixed truth;
- `positions`, `depth`, `is_leaf`, `is_internal`, `is_sink`;
- `value_correct`, `first4_correct`: `[rounds, n_nodes]` bool;
- `graph_value_all`, `graph_first4_all`: `[rounds]` bool;
- `tV`, `t4`, `actual_max_depth`, `completion_ref_round`, `n_rounds`.

`index.jsonl` adds weight/stratum/graph identity and source index.  The index is
append-only and flushed per graph, so a killed run leaves completed graphs.

## Definitions

- **Numeric node set**: non-PAD positions with `dataset_info[:, :, 0] >= 0`
  and final VALUE label `< mod_val`; excludes PAD, operators and
  problem-statement constants.  This is the node definition position (leaf
  variable, internal variable, result node).
- **VALUE node accuracy**: `value_correct / numeric nodes`, count-weighted.
- **first-4 node accuracy**: all four factors (`SYNTAX`, `VARIABLE`,
  `OPERATION`, `VALUE`) correct at the same numeric node.
- **tV**: first 1-based round where every numeric node's VALUE is correct.
- **t4**: first 1-based round where every numeric node's first four fields are
  correct.
- **Post-completion drift**: from the round after `tV` (or `t4`), the first
  relative round with a graph-level failure, whether it later recovers, and
  final correctness.  `tV` and `t4` are reported separately; they are not
  collapsed into one “completion”.
- **Other-only drift**: after `tV`, rounds with all VALUE correct but first-four
  not all correct.

## Review points

1. Generated `N64_shallow` and `N128_shallow` use fresh official-generator
   samples, not a mix with the 54 existing N64 shallow graphs.
2. `s1234_e143` is fixed for comparability; it is not the best s1234 checkpoint
   on the C0′ MixedN test.
3. The runner sets 64 loops explicitly; it does not use the old config's
   `max_val_loop=36`.
4. `tV` and `t4` drift are separate; summary consumers must not merge them.
5. The formal run used GPU6 only; seed2026 training on GPU0 was untouched.  The
   first pass is an exploration budget, not a final evaluation protocol.  It
   found that first-4 collapse in this ADD task is dominated by the VARIABLE
   readout, while VALUE remains high much longer; see the report.
6. `analyze_existing.py` only reads the recorded NPZ files.  It does not load a
   model, rewrite the run directory, or establish hidden-state causality;
   `stable_recovered` means the last two post-completion observations are
   correct, not merely the final round.  `value_and_variable_preserved` is an
   independent boolean, not the mutually exclusive `primary_event`; the
   `value_error_*` tables use `tV <= H` (`completed_by_H`).
