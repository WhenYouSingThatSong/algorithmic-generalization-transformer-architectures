# B mechanism code — changelog

Date: 2026-11-14.

## GPT follow-up: switch execution and constant truth

User authorized necessary fixes after reviewing `reproduction@e27cc6b`.

- Fixed the switch range: the previous `[1, hold_rounds]` validation accepted
  only rounds masked by the hold branch. The implemented schedule is now
  validated as `switch_at == hold_rounds + 1 <= horizon`: hold N forwards,
  inject the second value once, then release. The sustained control releases
  directly after N forwards. This is not a stable-error-state switch experiment.
- Include changed input constants in `recompute_truth()` as well as graph
  nodes. D1/D2/D3 `*_nodes_new[const_pos]` now uses the modified problem label.
  Old-truth fields retain their old-reference meaning.
- Added eight CPU regression tests for common eligibility, graph budgets,
  summary pairing, controls, switching and new truth. All pass; no GPU model
  inference was performed locally. See `b-mechanism-smoke.md` for scope.
- Clarified first-common-complete T and provisional evaluation definitions in
  the design. No changes to training or metric formulas.

## GPT review fixes (post-v2)

Historical first fix pass (`66440b8`); its switch-range claim below is superseded
by the follow-up above, and constant truth needed the additional correction.

- R1: `early_*` flag schema is unified through `early_flag_key`; selection
  writes common and per-model `_a`/`_b` flags plus `event_id`; runners reject
  missing/inconsistent flags instead of silently proceeding.
- R3: selection searches `T` only within each graph's own `n_loops`; runners
  validate `0 <= T < n_loops`; batch-size changes no longer alter accepted
  graph budgets.
- R2: summary keys include source/run identity and full target identity; clean
  and injected records are paired without merging A′/B, start/timing, or
  different input files; duplicate clean records are checked.
- R7: controls records carry `condition`/`start`/`timing`, emit a same-start
  clean, and reject `switch_at` outside `[1, hold_rounds]`.
- R8: D1/D2/D3 records carry `new_truth` plus `input_nodes_new` /
  `output_nodes_new`; old-label fields are marked `truth_ref=old`.
- Summary output now states `post_round_1` / `any_later_round` scope explicitly;
  metric definitions remain unsettled.

## v2 review fixes

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

## Long-horizon pilot summary (2026-11-14)

- Added `b_mechanism_horizon_summary.py`, which consumes early/late/controls
  trajectory JSONs and implements the four user-approved evaluation choices:
  full-window per-round curves, stable recovery = last two in-window rounds
  correct (`L=2`), graph-equal primary with node-weighted secondary, and
  VALUE / first-four correctness reported in parallel.
- H = 2/4/8/16/32/64 is truncated from the same trajectory; no re-runs.
- Direct injected positions are derived from the actual intervention: early
  2x2 conditions for `b_mechanism_early.py`, `injection.pos` for late/control
  records.  Damage is assessed on model outputs; the injected input state
  itself is not counted as damage, so a slot rewritten in the first forward is
  reported as no direct damage, not as a trivial one-round recovery.
- For `sustained_r`, H starts after release (`hold_rounds` offset).  The forced
  hold window is reported separately in `sustained_hold.csv` and excluded from
  the H table.
- Zero-denominator graphs are reported explicitly and never assigned 0%/100%.
- Added `tests/test_horizon_summary.py` (7 CPU checks: H truncation, L=2
  stability, only-last, graph/node weighting, clean-referenced damage,
  injected-position roles, post-release offset).  Together with the existing
  8 review regressions, 15/15 pass.  This is pipeline validation only.

## Smoke

See `b-mechanism-smoke.md`.  Smoke is pipeline-only; no scientific claim.
