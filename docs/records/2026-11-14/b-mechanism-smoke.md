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
