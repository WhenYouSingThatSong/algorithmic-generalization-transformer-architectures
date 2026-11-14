# B mechanism experiment code (v2, 2026-11-14)

Design: `docs/records/2026-11-14/b-mechanism-design.md`
Deferred control: `docs/records/2026-11-14/b-local-self-consistent-deferred.md`

Status: **draft; smoke-tested on 4 heldout graphs** (selection, early own/teacher,
late all first-batch conditions, sustained/switch controls) and one training
batch for the corruption audit.  Not a full experiment; no mechanism result is
claimed here.

## Models

- A′: config `A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa`,
  group `Nodes32-ADD - L4H16D384_DeBERTa`,
  run `seed-42 - 2026-11-09-15:42:45`,
  base `tmp/runs/A-L4D384fast/checkpoints`,
  ckpt `epoch=86-val_loss=0.0000.ckpt`.
- B: config `B-M6-Nodes32-ADD-L4H16D384-DeBERTa`,
  group `Nodes32-ADD - L4H16D384_DeBERTa_recorrection`,
  run `seed-42 - 2026-11-07-17:47:37`,
  base `tmp/runs/B-M6/checkpoints`,
  ckpt `last.ckpt`.

Data: `tmp/analysis/heldout-seed42/ADD/testdata_N32.pt` primary;
`tmp/analysis/matched-depth/ADD/testdata_N32.pt` secondary.

Single GPU, serial, threads 4–8:

```bash
export CUDA_VISIBLE_DEVICES=1          # training uses GPU0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

All entrypoints insert their own directory into `sys.path`; `python -I` works.

## 1. Build common-complete manifest

```bash
python -I tools/b_mechanism/b_mechanism_select.py \
  --ds_file tmp/analysis/heldout-seed42/ADD/testdata_N32.pt \
  --scan_graphs 1024 --n_graphs 1024 --batch_size 8 --k 4 --extra_rounds 4 \
  --a_config A-L4D384fast-Nodes32-ADD-L4H16D384-DeBERTa \
  --a_group 'Nodes32-ADD - L4H16D384_DeBERTa' \
  --a_run 'seed-42 - 2026-11-09-15:42:45' \
  --a_base tmp/runs/A-L4D384fast/checkpoints \
  --a_ckpt 'epoch=86-val_loss=0.0000.ckpt' \
  --b_config B-M6-Nodes32-ADD-L4H16D384-DeBERTa \
  --b_group 'Nodes32-ADD - L4H16D384_DeBERTa_recorrection' \
  --b_run 'seed-42 - 2026-11-07-17:47:37' \
  --b_base tmp/runs/B-M6/checkpoints --b_ckpt last.ckpt \
  --out tmp/analysis/b-mechanism/manifest_N32.json
```

Manifest records:

- `T` = **first** common complete round, `T_kind=first_common_complete`;
- per-chain `early_own_before/after` and `early_teacher_before/after`;
- primary chains only: single-parent u, q1 not directly r, q2 present and not
  directly r;
- late sinks, leaf `var_pos`/`const_pos`, unrelated nodes that cannot reach
  u/q1/q2;
- data/checkpoint/code SHA256.

## 2. Early 2×2 trajectories

```bash
python -I tools/b_mechanism/b_mechanism_early.py \
  --manifest tmp/analysis/b-mechanism/manifest_N32.json \
  --ds_file tmp/analysis/heldout-seed42/ADD/testdata_N32.pt \
  --config B-M6-Nodes32-ADD-L4H16D384-DeBERTa \
  --group 'Nodes32-ADD - L4H16D384_DeBERTa_recorrection' \
  --run 'seed-42 - 2026-11-07-17:47:37' \
  --base tmp/runs/B-M6/checkpoints --ckpt last.ckpt \
  --start own --timing after_u --horizon 6 \
  --out tmp/analysis/b-mechanism/B_early_own_after.json
```

Repeat for `--start teacher`, `--timing before_u`, and for A′.  The runner
skips chains whose manifest `early_<start>_<timing>` is false.

## 3. Late timing

```bash
python -I tools/b_mechanism/b_mechanism_late.py \
  --manifest tmp/analysis/b-mechanism/manifest_N32.json \
  --ds_file tmp/analysis/heldout-seed42/ADD/testdata_N32.pt \
  --config B-M6-Nodes32-ADD-L4H16D384-DeBERTa \
  --group 'Nodes32-ADD - L4H16D384_DeBERTa_recorrection' \
  --run 'seed-42 - 2026-11-07-17:47:37' \
  --base tmp/runs/B-M6/checkpoints --ckpt last.ckpt \
  --conditions late_r,late_u,sink,leaf_value,leaf_constant,leaf_constant_scratch,leaf_constant_pinned,unrelated \
  --delta 3 --u_delta 1 --horizon 6 \
  --out tmp/analysis/b-mechanism/B_late.json
```

- `late_r`: time-matched R-only; `late_u`: time-matched U-only/internal.
- Each target gets a same-start `clean` continuation (except D2 scratch).
- D1 `leaf_constant` stores `new_truth`.
- D2 `leaf_constant_scratch` uses `max(horizon, manifest n_loops)`.
- D3 `leaf_constant_pinned` forces the new constant every round.
- Records include VALUE and first-four node states per round.

## 4. Controls

Local self-consistent-error is **deferred and not implemented**.  Use:

```bash
# hold r wrong for 4 rounds, then release
python -I tools/b_mechanism/b_mechanism_controls.py ... --mode sustained_r \
  --hold_rounds 4 --horizon 10 --out .../B_sustained.json

# hold r wrong, then switch r to a second wrong value at round 5
python -I tools/b_mechanism/b_mechanism_controls.py ... --mode switch_held_r \
  --hold_rounds 4 --switch_at 5 --horizon 10 --out .../B_switch.json
```

`sustained_r` is the no-switch control for `switch_held_r`.

## 5. Training corruption audit

```bash
python -I tools/b_mechanism/b_corruption_audit.py \
  --ds_file experiments/data/Tr32Test128-ADD/train_data.pt \
  --config B-M6-Nodes32-ADD-L4H16D384-DeBERTa \
  --group 'Nodes32-ADD - L4H16D384_DeBERTa_recorrection' \
  --run 'seed-42 - 2026-11-07-17:47:37' \
  --base tmp/runs/B-M6/checkpoints --ckpt last.ckpt \
  --batch_size 8 --n_batches 4 --trials 100 --seed 20261114 \
  --out tmp/analysis/b-mechanism/corruption_audit.json
```

The audit exports all `last` candidates and their **pre-corruption** VALUE, then
separately reports normal-probability sampled replacements.  It does not train.

## 6. Summary (N0/N1/N2)

```bash
python -I tools/b_mechanism/b_mechanism_summary.py \
  --inputs tmp/analysis/b-mechanism/B_early_own_after.json tmp/analysis/b-mechanism/B_late.json \
  --out_csv tmp/analysis/b-mechanism/summary.csv \
  --out_json tmp/analysis/b-mechanism/summary.json
```

N0 = nodes correct in the same-start clean continuation at post_round 1; N1 =
N0 nodes wrong under injection at post_round 1; N2 = N1 nodes correct in a later
round.  Reports first-four and VALUE separately, EMPTY at first round, and max
descendant edge distance.  D1/D2/D3 truth-changing conditions are excluded from
this summary and analyzed separately.

## 7. Smoke status

Small smoke was run on GPU1 on 4 heldout graphs:

- selection: 4 common-complete, 7 chains, 8 sinks, 8 leaves;
- early own after: 28 records; early teacher before: 28 records;
- late: 53 emitted records; D1/D2/D3 leaf modes: 16 records;
- controls: sustained 7 records; switch 7 records;
- audit: 128 candidates, all before=EMPTY; 200 normal trials, 692 actual
  replacements, all before=EMPTY, after mostly numeric.

These are pipeline checks, not results.

## 8. Pre-run checks

1. manifest `early_*` counts > 0;
2. leaf `var_pos`/`const_pos` agree with token layout;
3. injection changes only intended VALUE position(s);
4. clean control has no corrupted target;
5. D2/D3 actual inputs keep the new constant;
6. summary has non-zero N0 before interpreting N1/N2.