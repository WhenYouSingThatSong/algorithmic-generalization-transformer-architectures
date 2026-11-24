# C smoke checks — 2026-11-19

- Bundle: `tmp/analysis/abc-workspace-20261119/C-smoke`
- Shared data: `tmp/analysis/abc-workspace-20261119/data`
- Commit: `d92915a9c2c3233d20ce79e4a5037457fd491775`
- GPU: 0 (the only GPU that became idle; all others were occupied). Queue: 7/7 tasks exited 0.
- Models: `C_s2026`, `C_s42`, `C_s1234`; each is `nointerm=True`, strict-loaded at the registered SHA-256.
- This is an execution/interface smoke, not a capability estimate or the fixed main batch.

## Scope and manifests

- `tasks.json` SHA-256 `b13f87df400c4d2596dd43284b6f906b3ae5eda6a2ae44441404cb05358690d8`:
  80 fresh development cases, 5 shapes × 16; unique input fingerprint per case.
- `models.json` SHA-256 `73e99dcbc603f419962e93bce46021b84ed1d9737936ec325dd9ac372d1bfd19`:
  `C_s2026` (40babb20…), `C_s42` (4a00d6a2…), `C_s1234` (0bca3a2b…). Only C models were loaded here.
- Smoke used the first ordinal of each shape, not an outcome-selected subset:
  `w2_f2:000`, `w4_f2:000`, `w6_f2:000`, `w4_f1:000`, `w4_f3:000`.
- Per-run meta records fixed `physical_nodes=32`, `padded_tokens=256`, six-factor rows and
  `mod_val=23`, `n_vars=128`; no truncation. Assignment smoke has four updates; program smoke
  has two windows. Fixed diagnostics are at protocol stages only.

## Records and queue receipts

| run | records | status | forwards | seconds |
|---|---:|---|---:|---:|
| assignment-C_s2026 | 165 | all complete | 2,940 | 18.9 |
| assignment-C_s42 | 165 | all complete | 2,940 | 18.4 |
| assignment-C_s1234 | 165 | all complete | 2,940 | 19.4 |
| program-C_s2026 | 55 | all complete | 924 | 10.8 |
| program-C_s42 | 55 | all complete | 924 | 10.9 |
| program-C_s1234 | 55 | all complete | 924 | 11.4 |
| summary | — | complete | — | 2.1 |

Total 660 records; no `invalid_commit` in this smoke. Assignment per model is 5 cases ×
33 records (3 old-task controls + 4 stages × [3 branches + unchanged + same-value write] +
2 fixed diagnostic stages × 5 probes). Program per model is 5 cases × 11 records
(2 stages × 3 branches + 1 fixed diagnostic stage × 5 probes).

## Independent verifier

- Script: `tmp/analysis/abc-workspace-20261119/checks/verify_c_smoke.py`,
  SHA-256 `f63978fd6df8fede005177b5563ac800fd58928a76024dbd13590679b6c3b4b2`.
- Result: `{"status":"ok","errors":0,"tokens_mod":23,"tokenizer_n_vars":128,"common_initial_first4":1}`.
- It rechecked, for all 660 records: task-token layout/order, 32 result nodes, 256-token
  limit, raw result VALUE=`EMPTY` (24), six-factor aux fields zero, independent symbolic
  truth at all nodes, conditional truth for actual submitted inputs, all `measures` keys,
  assignment boundary inputs/changed task rows/previous inputs, program actual-commit
  inheritance and final decoded commit, forward accounting, coverage/common denominators,
  program first/final/all-stage status and 128/256/512 threshold rows, and probe repaired/
  damaged pair intersections.
- C-specific limits of the verifier: it does **not** rebuild a hidden start state from
  decoded rows and does not require A/B-style identical trajectories for C same-value
  writes. Those are checked separately below on GPU.

## C hidden-interface checks (manual, one model per process)

Script `tmp/analysis/abc-workspace-20261119/checks/check_hidden_rows.py`,
SHA-256 `5b0aa9e3b375d20e899f4dc8cd94ce36106b15b0f229ff0234a6ef936d16cec3`.
At `w2_f2:000`, stage 0, after the local depth + 8 budget (22 rounds):

| model | edit positions | only specified rows changed | outside rows equal | edited rows = token embedding | same-value write only specified rows | reset positions | reset equals write_rows | reset outside equal |
|---|---:|---|---|---|---:|---|---|---|
| C_s2026 | [0] | yes | yes | yes | yes | [172,178] | yes | yes |
| C_s42 | [0] | yes | yes | yes | yes | [172,178] | yes | yes |
| C_s1234 | [0] | yes | yes | yes | yes | [172,178] | yes | yes |
| A_s42 (contrast) | [0] | yes | yes | yes | n/a (0 positions; raw row unchanged) | [172,178] | yes | yes |

- For C, `reset_results` is exactly `write_rows` on the result rows: the whole result
  hidden row is reinitialized from the token embedding; all non-result rows are unchanged.
- For A_s42, `reset_results` changed **only field 3 (VALUE)** at rows 172 and 178; the other
  five fields and all other rows were unchanged. This is the intended C-vs-A/B semantic
  difference, not a shared reset.
- C's local input write changes only the specified constant row; every other hidden element
  is byte-identical. Same-value write has the same row restriction. No per-round decode is
  fed back: `runtime.run` uses `decode` only for logging and passes the continuous hidden
  to `lit.model` inside the loop; `lit.nointerm` is required by `Runtime`.

## Program boundary and invalid-commit path

- For every program record, the next window's first `w` boundary inputs equal that branch's
  previous `actual_commit` exactly; the remaining inputs equal the fixed exogenous
  constants. Each commit equals the model's decoded final output at the corresponding
  terminal slot. No reference/gold value is used to fill a boundary.
- No natural non-numeric commit appeared. A separate protocol call confirmed
  `emitted_inputs` rejects `None`, and `evaluate` refuses non-numeric inputs; the targeted
  summary test `test_invalid_commit_not_dropped_and_probe_is_paired` passed, keeping the
  failed branch in the denominator. This is a protocol path check, not a model result.

## Fixed outputs

| file | SHA-256 |
|---|---|
| `summary/summary.csv` | `66e5640b5e3fe40051064be4905df0356f3b2fdcad67c0699a3e35d5cc186f37` |
| `summary/coverage.csv` | `cdda43991139ff95fae4a682bc43e9083373e6d180b18ad3653dec2b3f50895b` |
| `summary/programs.csv` | `c594a46f6aa9537657f695b5c2653ba4bcd752d2196365d88b3bc53e34f2321d` |
| `summary/probe_pairs.csv` | `a2ae95d0a3a9eccfbc1bef955e974fac84f60a1c77fb9c478bb8e79401594429` |
| `summary/meta.json` | `4c21d1cbd57985529831dbbed00e7077099a735183558c10797c9b5512f9f3aa` |
| `complete.json` | `255bed946f74f8591031168498724888bde1978d93634809a2bbb81daa982cee` |

## Caveats

- One case per shape, two windows; not a capability estimate. Common initial four-attribute
  qualified case is only `w4_f2:000` in this smoke (coverage is reported per model too).
- All correctness fields are decoded token VALUE/four-attribute behavior. C's hidden vector
  is continuously fed, but program cross-window inputs have the explicit discrete numeric
  commit boundary for all families.
- C `reset_results` reinitializes whole hidden rows; A/B only clears VALUE. This is not an
  equal perturbation and must not be pooled as one reset type.
- The main run will use the same manifests and the fixed 16/8 cases per shape; smoke results
  do not authorize changing tasks, budgets, or weights.
