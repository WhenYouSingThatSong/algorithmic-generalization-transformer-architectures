# Deferred: local self-consistent error control

Date: 2026-11-14. Status: **DEFERRED / NOT RUN**.

This experiment (artificially setting a locally self-consistent wrong subgraph,
e.g. r=5, u=7, q=16) is postponed.  Reasons:

- the current implementation can leave bypass parents at old values, so the
  constructed start is not actually self-consistent;
- a correct construction must either recompute all affected descendants under
  the preserved parent values, or replace the full affected subgraph and be
  called "subgraph self-consistent", not "local self-consistent";
- the sustained-wrong-parent plus switch controls can answer part of the parent
  dependence question without this artificial state.

Do not report any result from this control.  The first batch, late timing,
sustained wrong parent, switch-from-held state, cyhB replication, and the
training corruption audit are independent of it.

Future work: implement full affected-subgraph consistency and add a no-switch
control.  Until then, the claim "B reopens a coherent wrong state" is open, not
supported or refuted.