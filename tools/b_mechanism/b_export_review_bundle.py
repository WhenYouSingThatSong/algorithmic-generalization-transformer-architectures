#!/usr/bin/env python3
"""Export complete existing evidence for specified late intervention cases.

Streams JSONL, keeps only requested graphs, pairs using the public clean key.
Copies every recorded input/output node and round, not only damaged nodes.
CPU only: torch is imported solely to read the original tensor dataset.
No model, checkpoint loading, rollout, or inferred causal labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b_main_summary as M


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def graph_structure(info_rows):
    """Decode dataset_info directly, independently of manifest child lists.

    Schema from generate_data.py/model.py: depth, parent, op, parent, op,
    parent; INVALID_NUM=-1. Leaf dependencies may point to constant tokens.
    """
    return {str(pos): dict(depth=int(row[0]),
                           parents=[int(v) for v in row[1::2] if v != -1],
                           operations=[int(v) for v in row[2::2] if v != -1])
            for pos, row in enumerate(info_rows) if row[0] >= 0}


def select_pairs(path, meta, cases):
    """Return exact late cases and complete matched clean records or fail."""
    wanted = {(int(c['graph_id']), c['condition'], int(c['target_index']))
              for c in cases}
    if not wanted or len(wanted) != len(cases):
        raise ValueError('cases must be nonempty and unique')
    gids = {k[0] for k in wanted}
    args = meta.get('args') or {}
    source = M.S.source_identity(str(path), meta)
    cleans, injected = {}, {}
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for raw in f:
            digest.update(raw)
            if not raw.strip():
                continue
            rec = json.loads(raw)
            if rec.get('graph_id') not in gids:
                continue
            if M.S.record_kind(rec) != 'late':
                raise ValueError('this exporter expects late runner records')
            if rec.get('condition') == 'clean':
                key = M.clean_key(rec, source, args)
                if key in cleans and cleans[key] != rec:
                    raise ValueError(f'conflicting clean records: {key}')
                cleans[key] = rec
                continue
            case_key = (rec['graph_id'], rec.get('condition'), rec.get('target_index'))
            if case_key in wanted:
                if case_key in injected:
                    raise ValueError(f'duplicate intervention case: {case_key}')
                injected[case_key] = rec
    missing = wanted - injected.keys()
    if missing:
        raise ValueError(f'missing intervention cases: {sorted(missing)}')
    pairs = []
    for key in sorted(wanted):
        rec = injected[key]
        clean = cleans.get(M.clean_key(rec, source, args))
        if clean is None:
            raise ValueError(f'missing matching clean: {key}')
        if not rec.get('rounds') or len(rec['rounds']) != len(clean.get('rounds', [])):
            raise ValueError(f'empty or unequal observation windows: {key}')
        pairs.append(dict(case=dict(graph_id=key[0], condition=key[1], target_index=key[2]),
                          clean=clean, injected=rec))
    return pairs, digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='existing late JSONL')
    p.add_argument('--cases', required=True, type=Path, help='JSON list of exact cases')
    p.add_argument('--out_dir', required=True, type=Path, help='new directory only')
    a = p.parse_args()
    if a.input.suffix != '.jsonl':
        raise ValueError('--input must be a JSONL sidecar')
    if a.out_dir.exists():
        raise FileExistsError(a.out_dir)
    meta_path = a.input.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    args = meta['args']
    # Resolve the actual paths recorded by the run; run from repository root.
    dataset_path, manifest_path = Path(args['ds_file']), Path(args['manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    dataset_sha = sha256_file(dataset_path)
    if dataset_sha != manifest.get('ds_sha256'):
        raise ValueError('dataset hash differs from manifest or manifest lacks hash')
    cases = json.loads(a.cases.read_text(encoding='utf-8'))
    pairs, trajectory_sha = select_pairs(a.input, meta, cases)
    gids = sorted({pair['case']['graph_id'] for pair in pairs})
    graphs = [g for g in manifest['graphs'] if g['graph_id'] in gids]
    if sorted(g['graph_id'] for g in graphs) != gids:
        raise ValueError('selected graph missing/duplicated in manifest')
    import torch  # server CPU dependency; does not import any model code
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=True)
    raw_graphs = []
    for gid in gids:
        if gid < 0 or gid >= len(dataset['tokenized_dataset']):
            raise ValueError(f'graph_id outside original dataset: {gid}')
        raw_graphs.append(dict(
            graph_id=gid,
            tokenized_dataset=dataset['tokenized_dataset'][gid].tolist(),
            dataset_info=dataset['dataset_info'][gid].tolist()))
    bundle = dict(
        schema_version=1, cases=cases, pairs=pairs, raw_graphs=raw_graphs,
        run_meta=meta,
        manifest={**{k: v for k, v in manifest.items() if k != 'graphs'}, 'graphs': graphs},
        provenance=dict(trajectory=str(a.input), trajectory_sha256=trajectory_sha,
                        meta=str(meta_path), meta_sha256=sha256_file(meta_path),
                        dataset=str(dataset_path), dataset_sha256=dataset_sha,
                        manifest=str(manifest_path), manifest_sha256=sha256_file(manifest_path)),
        notes=['All original recorded rounds and node fields retained.',
               'tokenized_dataset is the original label tensor; not reconstructed model input.',
               'Raw dataset rows retain original positions and padding.',
               'No model execution; no claim that exclusive roles or causal mechanisms are verified.'])
    a.out_dir.mkdir(parents=True, exist_ok=False)
    output = a.out_dir / 'review_bundle.json'
    output.write_text(json.dumps(bundle, ensure_ascii=False), encoding='utf-8')
    # Small review artifact: trajectories and derived graph relations only.
    # Keep the raw token tensors in the server-local bundle, out of git.
    evidence = {k: v for k, v in bundle.items() if k not in ('raw_graphs', 'notes')}
    evidence['dataset_graphs'] = [dict(graph_id=g['graph_id'],
                                     nodes=graph_structure(g['dataset_info']))
                                  for g in raw_graphs]
    evidence['notes'] = ['Complete recorded clean/injected trajectories; no raw dataset tensors.',
                         'dataset_graphs derived directly from original dataset_info; '
                         'manifest node_truth retained separately for cross-checking.',
                         'operations are encoded IDs; no causal classification performed.']
    (a.out_dir / 'review_evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), pairs=len(pairs), graphs=len(gids),
                          bytes=output.stat().st_size)))


if __name__ == '__main__':
    main()
