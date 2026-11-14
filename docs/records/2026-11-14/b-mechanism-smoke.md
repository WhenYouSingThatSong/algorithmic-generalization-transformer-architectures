# B mechanism smoke summary (2026-11-14)

Status: pipeline smoke only, **not a mechanism result**.  4 heldout N32 graphs,
B-M6 s42, GPU1; training seed2026 was on GPU0 and untouched.

## Selection

- scanned 4, common complete 4;
- 7 primary chains, 8 sinks, 8 leaves;
- early own/teacher before/after counts each 7;
- A′/B clean token states identical: 4/4.

## Runners

- early own after: 28 records; early teacher before: 28 records;
- late first batch: 53 records; D1/D2/D3 leaf modes: 16 records;
- sustained_r: 7 records; switch_held_r: 7 records;
- summary: produced CSV/JSON with N0/N1/N2 and first-four fields.

## Corruption audit

One batch, B-M6 s42, training split:

- `candidates`: 128 positions, **before = EMPTY 128/128**;
- normal sampling: 200 trials, 692 actual replacements,
  **before = EMPTY 692/692**;
- after: numeric 643, N/A 25, EMPTY 24;
- after_is_true: 31.

This is consistent with the review's hypothesis that the `last` path selects
newly available slots that are EMPTY in the depth-wise input.  It is only one
batch and is not yet a training-history conclusion.

## Caveat

The smoke verifies code paths, shapes, identity hashes, output fields, and the
`S[:-1]` fix.  It does not establish any scientific result; full runs still
need manifest eligibility checks and the planned denominators.

## Post-review fix smoke (2026-11-14, second pass)

Pipeline-only checks after R1/R3/R2/R7/R8 fixes:

- selection, 4 heldout graphs, batch 8: 4 common-complete, 9 chains, 16 sinks,
  16 leaves; all `early_*` common flags equal `_a and _b`; all graphs satisfy
  `0 <= T < n_loops`.
- same selection with batch 4 gives the same `(T, n_loops)` for all 4 graphs.
- early own after: 36 records (9 chains x 4 conditions), `event_id` present.
- late: 66 records; D1/D2/D3 contain `new_truth` and
  `input_nodes_new`/`output_nodes_new`; 25 unique clean records after
  target-level dedup.
- controls: sustained 9 records + 9 clean; switch 9 records + 9 clean;
  `condition` and `start/timing` populated.
- summary over early/late/controls: 7 rows, A/B sources separate, input-order
  invariant, `duplicate_cleans=0`.

These are still pipeline checks, not mechanism results.

## GPT follow-up CPU regressions (2026-11-14 14:53 CST)

The previous switch smoke did not assert that the second value was actually
written. At `66440b8`, its accepted switch rounds were all masked by the hold
branch. The existence/count of switch records is not evidence of a switch.

After the follow-up fix:

- `python -I tools/b_mechanism/tests/test_review_regressions.py`: 8 tests pass.
- Common eligibility includes one-model-ineligible cases, missing keys and
  inconsistent common/per-model flags; out-of-budget T is rejected.
- Actual summary CLI tests keep A/B, own/teacher and before/after separate,
  accept controls, reject conflicting clean records, and give identical output
  under reversed input order.
- Source-extracted controls hook writes first-wrong N times, second-wrong once
  at N+1, then releases; all tested accepted schedules execute the switch.
- D1/D2/D3 new-truth labels agree at changed constant, leaf and descendant;
  unchanged constants and original labels are preserved.
- Syntax compilation of all 8 Python files and `git diff --check` pass.

These are standard-library CPU tests. Functions are loaded from source AST with
tensor/model doubles; no checkpoint, real tensor rollout or GPU smoke was run
locally. Server follow-up should inspect actual input trajectories and new-truth
fields on a small model sample before interpreting affected experiment results.
