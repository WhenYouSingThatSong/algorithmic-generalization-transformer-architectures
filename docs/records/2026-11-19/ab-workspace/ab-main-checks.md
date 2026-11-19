# AB main checks — 2026-11-19

- Bundle: `tmp/analysis/abc-workspace-20261119/AB-main`
- Shared data: `tmp/analysis/abc-workspace-20261119/data`
- Prepare branch: `87f7a2a4bcc275ffa196e831eba6a154819877b4`
- Queue-plan HEAD: `3c360ece3e6fe41dcf9f4ba07c977f489efda1a7`
- GPU: 7 (AB serial; GPU 0–5 not used)
- Queue: 5/5 tasks exited 0; `complete.json` written.
- Independent verifier: `tmp/analysis/abc-workspace-20261119/checks/verify_ab_smoke.py`
  invoked with `--bundle tmp/analysis/abc-workspace-20261119/AB-main`; result
  `{"status":"ok","errors":0}` after rebuilding all records and four summary tables.

## Provenance note

The shared worktree HEAD moved from `87f7a2a` to `3c360ec` before the main queue
recorded its plan. `3c360ec` changes only
`docs/records/2026-11-19/c-state-reuse/*` and `tools/state_reuse/export_c_report_tables.py`;
`git diff --name-only 87f7a2a..3c360ec -- tools/abc_workspace experiments` is empty.
The AB main plan contains the same `tools/abc_workspace`/`experiments` code hashes as the
smoke run, and all of those hashes still match the working tree. No AB source changed
during execution. The branch SHA is recorded honestly rather than replaced with the
prepare SHA.

## Scope

- `tasks.json` `b13f87df400c4d2596dd43284b6f906b3ae5eda6a2ae44441404cb05358690d8`:
  80 assignment cases (5 shapes ×16) and 40 program cases (5 shapes ×8) from the same
  80-case manifest. No replacement or outcome-based supplementation.
- `models.json` `73e99dcbc603f419962e93bce46021b84ed1d9737936ec325dd9ac372d1bfd19`.
  A_s42 and B_s42 strict-loaded; all five registry hashes were rechecked.
- Assignment: 4 updates, 80 cases/model, 1,940 records/model.
  Probes only at stages 1 and 4 for the first two ordinals of each shape
  (2×5=10 probe cases), as fixed by `probe_per_stratum=2`.
- Program: 20 windows, 40 cases/model, 2,600 records/model. Probes at stages
  1, 4, 10, and 20 for the same first two ordinals; 10 probe cases/model.
- `physical_nodes=32`, `padded_tokens=256`; 526 cumulative logical nodes at window 20
  (520 executed arithmetic nodes), with the external numeric commit boundary.

## Queue totals

| run | records | forwards | seconds |
|---|---:|---:|---:|
| assignment-A_s42 | 1,940 | 33,936 | 141.0 |
| assignment-B_s42 | 1,940 | 33,936 | 141.1 |
| program-A_s42 | 2,600 | 43,680 | 196.5 |
| program-B_s42 | 2,600 | 43,680 | 191.0 |
| summary | — | — | 1.0 |

Total 9,080 records. All statuses are `complete`; no natural invalid commit occurred.
Assignment forward totals include the unrecorded baseline run per case; the verifier
reconstructed the totals exactly.

## Independent verification

1. **Code/weights/data identity.** Every code hash in `plan.json` matches the current
   working tree; both model paths and hashes match `models.json`; tokenizer is mod 23,
   128 variable names, six factors. Every raw result row is `EMPTY`; auxiliary fields are
   FALSE; no sequence exceeds 256 tokens.
2. **Actual serialized task and independent truth.** For all 9,080 records the verifier
   rebuilt `task_tokens` from the symbolic frame and matched every one of the 32 node
   VALUEs per record against an independent ADD evaluation of the serialized parent
   equations. Metrics (`value`, `first4`, final, last-two, first, stable, reerror, task
   rows) were recomputed from the recorded trace and gold for every record.
3. **Assignment inheritance and reset.** For all four stages, carry/reset/scratch starts
   equal the previous branch output plus exactly the current constant rows; reset_results
   changes only VALUE fields of the affected result rows to `EMPTY`; scratch starts with
   all result VALUEs `EMPTY` and constants preserved; unchanged and same-value-write
   controls are identical to old-clean. Stage 4 returns to the original task inputs.
4. **Program cross-window boundary.** Every one of the 2,400 branch-stage records has the
   next window's first `w` values equal to that branch's previous `actual_commit`, and the
   rest equal to fixed exogenous constants. Every commit value equals the model's decoded
   final output at the matching terminal slot. No reference value is injected; changed
   task rows and cumulative local budgets match the recorded trace.
5. **Fixed diagnostic forks.** All 600 probe records (2 models × [10 assignment probe
   cases ×2 stages + 10 program probe cases ×4 stages] ×5 probes) use the fixed stage
   list. `continue`, `refresh_task`, `reset_results`, `refresh_and_reset`, and
   same-input `scratch` starts were rebuilt from the post-run carry state and match
   exactly. Every snapshot hash matches.
6. **Summaries.** `summary.csv` (1,460 groups), `coverage.csv` (4 rows),
   `programs.csv` (240 trajectories), and `probe_pairs.csv` (480 paired rows) were
   independently rebuilt from records with the same cohort and denominator rules;
   all counts, macros, program failure positions, thresholds, and repaired/damaged
   intersections matched.
7. **Invalid-commit path.** No natural invalid commit appeared. The protocol-level
   `emitted_inputs` rejection of `None` and the targeted summary test
   (`test_invalid_commit_not_dropped_and_probe_is_paired`) passed; failed branches would
   remain in the denominator. No result is inferred from that artificial test.

## Summary hashes

| file | SHA-256 |
|---|---|
| `summary/summary.csv` | `9791c0d4005f86377400048eb6830749aa3c1bd80becae704f303dbbb18713ed` |
| `summary/coverage.csv` | `d659a899ded38c72bef339530e6480b9abaf66a01f1fe2fc9715d42be62465e6` |
| `summary/programs.csv` | `d5972c149b2b42eeb718ccbdb53250e0e8d01bba4f28db10f734961eda2db5f3` |
| `summary/probe_pairs.csv` | `9c485284fa4bac17347b7b71f3af4829cca44a65fb20b95279619e93d4a8e86a` |
| `summary/meta.json` | `9967eb0cfa4f5e3d7bbcfb87f6b03d54844cb7888254076daeeae943ecabf3ad` |

## Caveats

- Fixed development tasks, one seed per model, decoded VALUE/four-attribute behavior;
  no final-test or multi-seed claim.
- The program has an explicit numeric commit at each window boundary for both families;
  this is not a pure continuous latent workspace and not native 526-node graph handling.
- C lane was not run here. C-specific interface/reset checks remain Agent C's work.
- The main plan HEAD is `3c360ec`; the AB code identity is unchanged from `87f7a2a`.