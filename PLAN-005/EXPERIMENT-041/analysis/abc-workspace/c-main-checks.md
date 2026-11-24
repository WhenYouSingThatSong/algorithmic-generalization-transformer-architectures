# C main checks — 2026-11-19

- C main bundle: `tmp/analysis/abc-workspace-20261119/C-main`
- Shared data: `tmp/analysis/abc-workspace-20261119/data`
- Commit: `d92915a9c2c3233d20ce79e4a5037457fd491775`
- GPU: 0; queue 7/7 tasks exit 0.
- Models: C_s2026 (40babb20…), C_s42 (4a00d6a2…), C_s1234 (0bca3a2b…), all `nointerm=True`.
- Manifests: tasks `b13f87df…`, models `73e99dcb…`; exact same files as AB/C smoke.
- Smoke checks: [C-smoke/checks.md](c-smoke-checks.md) (copy; original in tmp).
- This is the fixed main batch, not an independent final test or a capability-generalization claim.

## Queue and records

| run | records | forwards | seconds |
|---|---:|---:|---:|
| assignment-C_s2026 | 1,940 | 35,280 | 156.8 |
| assignment-C_s42 | 1,940 | 35,280 | 147.5 |
| assignment-C_s1234 | 1,940 | 35,280 | 147.9 |
| program-C_s2026 | 2,575 | 40,210 | 179.4 |
| program-C_s42 | 2,585 | 41,916 | 185.5 |
| program-C_s1234 | 2,540 | 37,894 | 169.2 |
| summary | — | — | 37.8 |

- Assignment: 80 fixed tasks (5 shapes ×16); program: 40 tasks (5×8), 20 windows.
- Total 13,520 records. No failed queue task; all model runs have `meta.status=complete` and records hash matching.
- Program record counts below the nominal 2,600 because non-numeric carry commits stop probe generation after the blocking stage; all three branch records remain at every stage.

## Independent verifier

- Script: `tmp/analysis/abc-workspace-20261119/checks/verify_c_smoke.py`,
  SHA-256 `3358f5fd3c18847e7ee1cb026a30a98e1877dd36ad016fad41997b181bbbf88d`
  (same logic as smoke verifier, adapted to C main's invalid-commit/sparse-probe path).
- Result: `{"status":"ok","errors":0,"common_initial_first4":24,"records":{...}}`;
  output `tmp/analysis/abc-workspace-20261119/checks/c-main-verify.out`,
  SHA-256 `1e0eb8dcfdebcaf63aef0a2710248643f5e50ef907ce75be7116de16e46551eb`.
- It rechecked all 13,520 records: token layout/order, 32 results, 256 tokens, raw result
  VALUE=EMPTY (24), six-factor aux zero, independent truth, actual-input conditional truth,
  assignment boundaries/changed rows/previous inputs, program actual-commit inheritance,
  forward accounting, coverage/common denominators, program success/threshold rows, and
  probe repaired/damaged intersections.
- C non-numeric commit semantics checked at transition: a complete stage with a `None`
  commit must be followed by `invalid_commit` at the next stage; at the final stage it is
  counted by terminal failure. It is not silently converted to a numeric value.

## Non-numeric carry commits

Only the `carry` branch produces non-numeric commits; `reset_results` and `scratch` do not
in this batch.

| model | invalid records | trajectories with first invalid | complete carry with any None | of which final stage |
|---|---:|---:|---:|---:|
| C_s2026 | 198 | 18 | 19 | 1 |
| C_s42 | 90 | 12 | 13 | 1 |
| C_s1234 | 278 | 24 | 24 | 0 |

The complete-with-None record is the last stage before blocking (or the final stage); all
later stages are `invalid_commit` and remain in the denominator. No truth/zero value is
backfilled.

## Smoke-interface checks carried forward

The C smoke hidden checks (three C seeds + A_s42 contrast) found:

- local input write changed only the specified task row; every other hidden element equal;
- same-value write had the same row restriction;
- C `reset_results` equaled whole-row `write_rows` on the result rows, with all other rows
  unchanged; A_s42 reset changed only field 3 (VALUE).

These are in `tmp/analysis/abc-workspace-20261119/checks/check_hidden_rows.py` and
`c-smoke-checks.md`; main uses the same runtime code/hashes.

## Caveats

- 32 physical positions / 256 padded tokens; program cumulative logical nodes 526 and
  required ancestor operations 520; this is not a claim that the model natively processes a
  526-node graph in one forward or has 526 simultaneously active positions.
- C program uses explicit numeric commits at window boundaries; it is a hybrid continuous
  computation/discrete-commit protocol, not latent-only continuity.
- C reset is whole result-hidden-row reinitialization; A/B reset only VALUE. Do not pool
  the two reset types.
- All correctness is decoded-field behavior. Main results and probe diagnostics are
  descriptive; they do not identify a single internal error location or repair mechanism.
