# B mechanism code v2 — changelog

Date: 2026-11-14.

## Review fixes

- Entrypoints insert their own directory before importing `b_mechanism_common`;
  `python -I ... --help` passes for all six entrypoints.
- Corruption audit uses `S[:-1]` to match training's `batch_ext[:-1]`, not
  `S[1:-1]`; audit records seed/trial ID and data/checkpoint/code SHA256.
- Selection records A′/B early eligibility for own/teacher states, so early
  comparisons use a common intersection rather than independently skipped
  models.
- Selection primary group is single-parent only, requires q2, and no longer
  falls back to multi-parent chains.  Multi-parent remains a separate future
  group.
- Selection separates `scan_graphs` from selected `n_graphs`, uses per-graph
  `n_loops`, and marks T as `first_common_complete`.
- Unrelated nodes exclude ancestors/descendants of u, q1, and q2, not only u.
- Late runner writes a same-start clean continuation for each target, stores D1
  `new_truth`, gives D2 a full problem budget, and records first-four node
  states.
- Controls no longer implement local self-consistent error.  `sustained_r` is
  the held-wrong-parent control; `switch_held_r` switches from a held state and
  is paired with `sustained_r` as the no-switch control.
- Added `b_mechanism_summary.py` for N0/N1/N2, VALUE and first-four metrics,
  EMPTY counts, and max descendant edge distance.
- Runners verify manifest data/checkpoint hashes.

## Deferred

Local self-consistent-error is deferred: see
`b-local-self-consistent-deferred.md`.  It is not run and
must not be cited.

## Smoke

See `docs/b-mechanism-smoke.md`.  Smoke is pipeline-only; no scientific claim.
