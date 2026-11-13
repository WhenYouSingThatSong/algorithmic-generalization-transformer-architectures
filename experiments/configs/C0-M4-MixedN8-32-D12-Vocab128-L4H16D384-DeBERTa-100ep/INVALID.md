# INVALID CONFIG — DO NOT USE

This config pairs a vocab129 tokenizer (`val_dag_config.num_nodes=128`) with the
old `C0-MixedN8-32-D12-ADD` data, which was generated with `n_vars=32` and
encodes N/A as **32** at non-VARIABLE positions. In vocab129, 32 means x_32.

Use instead:
`experiments/configs/C0-M4-MixedN8-32-D12-Vocab128-2Val-L4H16D384-DeBERTa-100ep`
with data `experiments/data/C0-MixedN8-32-D12-Vocab128-ADD`.

`preflight_check.py` now catches this as `data.*_na_index FAIL`.
