#!/usr/bin/env python3
"""New N32/depth<=6 train/validation graphs, official tokenizer and generator."""
import argparse
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P
sys.path.insert(0, str(HERE.parent / 'b_mechanism'))
from trajectory_io import configure_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=P.STUDY / 'data')
    p.add_argument('--train-count', type=int, default=1024)
    p.add_argument('--val-count', type=int, default=128)
    p.add_argument('--num_threads', type=int, default=4)
    p.add_argument('--exclude', type=Path, nargs='*', default=None,
                   help='override exact-graph overlap references; default original train/val and six diagnostic files')
    a = p.parse_args()
    if min(a.train_count, a.val_count) < 1:
        p.error('positive counts required')
    configure_runtime()
    sys.path.insert(0, str(HERE.parent / 'c_capability'))
    import run_capability as R
    from generate_data import generate_dataset
    from tokenizers import build_factored_tokenizer
    import numpy as np
    import torch
    import random
    import hashlib

    _, _, dc = R.load_configs(P.CONFIG)
    tok = build_factored_tokenizer(dc)
    if list(tok.vocab_sizes) != [7, 129, 2, 25, 2, 2]:
        raise ValueError('expected existing vocab129 ADD model')
    pad = tok.syntax_tok2idx[tok.PAD]
    def hashes(tokens):
        return [hashlib.sha256(row[row[:, 0] != pad].contiguous().numpy().tobytes()).hexdigest()
                for row in tokens]
    defaults = [P.ROOT / 'experiments' / dc.data_dir / dc.dag_config.data_file_name,
                P.ROOT / 'experiments' / dc.data_dir / dc.val_dag_config.data_file_name]
    defaults += [R.c_common.DEFAULT_DATA_DIR / f'{name}.pt' for name in R.c_common.STRATA]
    excluded, refs = set(), []
    for path in (defaults if a.exclude is None else a.exclude):
        blob = torch.load(path, map_location='cpu', weights_only=False)
        hs = hashes(blob['tokenized_dataset'])
        excluded.update(hs)
        refs.append(dict(path=str(path), sha256=P.sha(path), graphs=len(hs)))
        del blob
    a.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='building', protocol='C_long_pilot_v1', config=P.CONFIG,
                    vocab_sizes=list(tok.vocab_sizes), exclusion_references=refs,
                    overlap_definition='exact final token sequence ignoring trailing PAD; not graph-isomorphism',
                    config_sha256=P.sha(R.EXPERIMENTS / 'configs' / P.CONFIG / 'data_config.yaml'), splits={})
    P.write_json(a.out_dir / 'manifest.json', manifest)
    for split, count, seed in [('train', a.train_count, P.SEED), ('val', a.val_count, P.SEED + 1)]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        cfg = R.EasyDict(dict(dc.dag_config))
        cfg.num_samples = count
        tokens, info = generate_dataset(cfg, tok)
        depths = info[..., 0].max(dim=1).values.tolist()
        numeric = (info[..., 0] >= 0) & (tokens[..., 3] < tok.mod_val)
        if not (numeric.sum(1) == 32).all() or max(depths) > 6:
            raise ValueError('generated graph count/depth differs from old C0 distribution')
        hs = hashes(tokens)
        if len(set(hs)) != count or excluded.intersection(hs):
            raise ValueError('duplicate/overlapping graph; report before changing fixed seeds')
        excluded.update(hs)
        path = a.out_dir / f'{split}.pt'
        torch.save(dict(tokenized_dataset=tokens, dataset_info=info,
                        spec=R.c_common.spec_from_tensors(tokens, info, pad),
                        meta=dict(split=split, seed=seed, generator_config=dict(cfg), vocab_sizes=list(tok.vocab_sizes))), path)
        manifest['splits'][split] = dict(path=str(path), sha256=P.sha(path), count=count, seed=seed,
                                          depth_counts=dict(Counter(depths)), graph_hashes=hs)
    manifest['status'] = 'complete'
    P.write_json(a.out_dir / 'manifest.json', manifest)
    print(a.out_dir / 'manifest.json')


if __name__ == '__main__':
    main()
