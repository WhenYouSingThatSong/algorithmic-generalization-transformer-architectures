# AB smoke checks — 2026-11-19

- Bundle: `tmp/analysis/state-reuse-exec-20261119/AB-smoke`
- Fixed commit: `6ef63e36f51113fa5c60f95d6d107c8521df4482`
- GPU: `0` (Agent C used GPU2; no shared source/edit while queue ran)
- Plan SHA-256: `6c8bcf560639d15919e16db29ef0f8318f81b6829ffa4c101e6cafbc1f4bbc7d`
- Source: reviewed `reassign-main-A` / `reassign-main-B`; first saved delta, graphs 32–33; H4.
- `python -m unittest discover -s tools/state_reuse/tests -v`: 9 tests passed.

## Queue outputs

| run | records | records SHA-256 | summary records/common |
|---|---:|---|---:|
| repeat-A | 18 | `158b0733733c6734aecd58a3e23292cc68c1894c70a80de15159cffa868c84aa` | 36 / 18 |
| repeat-B | 18 | `403e8b261cf52ca5decb0cf035dbe9b4f8a64b5286f9022cd65c35b45802ce01` | |
| reuse-A | 4 | `65e58f84cb67cb0c73c5df8c44ee11e46d30572283494c652dfdb9a99fa18b5a` | 8 / 4 |
| reuse-B | 4 | `4e37ff9d20725fd7913f88a69f5c17a18bc5862b15923f28f7de37c487b258e9` | |

`repeat` has two graphs × two schedules. `same_input` has four episodes (three stage transitions) and `alternating_inputs` has five episodes (four transitions); fourteen transitions per model, twenty-eight total. `reuse` has two graphs × two episodes; four records per model.

Machine-readable check result: `ab-smoke-evidence.json`, SHA-256
`07dc216bff3661d4ed363447642c0a711c5c8a0fbc46fe794b99165266c4e725`.
The validator used the actual tokenizer and `tools/state_reuse/core.py`; it did not load or run the model.

## Repeat checks (A and B)

- **Return to original:** all four graph/schedule groups per model ended with `new_truth == original_truth` for every node; 4/4 A, 4/4 B.
- **Stage chain:** 14/14 transitions per model have `stage[i].old_truth == stage[i-1].new_truth`.
- **Stage inheritance:** starting from the previous branch output, `clean` and `sham` had zero changed cells; `warm` and `warm_pinned` changed only the intended constant VALUE; `clear` changed only the constant and the affected VALUE cells. No unintended field/position replacement was found.
- **Scratch:** every node VALUE in every scratch initial state was `EMPTY`; no solved answer was written into a scratch start. `EMPTY` was checked with the tokenizer id, not a string guess.
- **Actual equation and edit scope:** for internal ADD equations (all parents are nodes), `core.expression_positions` accepted the actual token rows and operand identities. For each stage, `raw_task` differed from the previous branch task only at the changed constant cell.
- **Protocol:** `clean == sham` in all records; `warm_pinned` inputs carried the current new constant; auxiliary fields were FALSE in all six-factor initial tensors; initial full tensors have six branches and the original sequence length.
- **Independent truth:** the runner's plan truth was independently reconstructed by `core.evaluate`; the final return and every stage chain above matched it.

## Reuse checks (A and B)

- **Dead slots:** in both graphs, `a < b`, neither old target appears in any original parent list, and the a/b slots are sinks. The second stage's `new_parents` contains the new a; `core.expression_positions` confirms the actual operand row references the new a variable.
- **Token edits:** `raw_task` differs from the previous stage only at `symbolic_edits`; those are exactly the operand rows returned by `expression_positions`. The target/result row identity is unchanged, token length is unchanged, and no old equation operand remains.
- **Start state:** the six-factor branch starts have the original sequence length; all raw auxiliary fields are FALSE. Scratch nodes are `EMPTY`; `clean == sham` in all records.
- **Pinning and inheritance:** first-four pinned cells in `warm_pinned` inputs match the recorded pin values; stage-2 branch starts inherit the previous branch output except the explicit operand edits and the `clear` target VALUE. Archive lengths are 1 and 2 for the two episodes, and the archived model outputs are not read back into any start.
- **Independent truth:** reuse stage truth was reconstructed with `core.evaluate`; changed/unchanged regions and the second-slot consumption were checked from the actual parent graph.

## Summary / coverage

- `repeat-summary`: complete, 36 records, 18 common A/B cases.
- `reuse-summary`: complete, 8 records, 4 common A/B cases.
- Both coverage files contain graphs 32 and 33 for both models; no zero-eligible or missing-branch case.
- H4 smoke delivery has no H16 main-table rows by design; the smoke summary and records are the evidence, not a missing H16 result.

## Caveats

- This is a two-graph H4 execution check, not a result about model capability.
- The observations are decoded token values; they do not prove an internal mechanism.
- No C run was performed in this bundle. The C lane is Agent C's responsibility.
