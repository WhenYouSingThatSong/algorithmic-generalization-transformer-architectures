# Tools

This directory holds project tools that are meant to be tracked and reviewed.

Current migration:

- `b_mechanism/` — B mechanism experiment code, design references, and smoke
  instructions.  Moved out of the ignored `tmp/` workspace on 2026-11-14.

Policy:

- `tools/` = tracked, reusable code and entrypoints.
- `docs/records/` = tracked design/experiment records.
- `tmp/` remains the local workspace for data, checkpoints, analysis outputs,
  archives, trash, and one-off historical scripts.
- Other `tmp/scripts` tools are intentionally left in place for now; moving
  them requires updating their `tmp/analysis`/`tmp/runs` default paths and
  launcher references.
