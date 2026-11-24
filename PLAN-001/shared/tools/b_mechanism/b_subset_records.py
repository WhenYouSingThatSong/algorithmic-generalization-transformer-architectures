#!/usr/bin/env python3
"""Write record subsets matched to a teacher-key set, without running models.

Reads JSONL sidecars written by the B runners.  The teacher inputs define the
allowed (graph_id, event_id) pairs; each source JSONL is filtered to those
pairs and written as a small JSON record file with the original run args.
This is CPU-only and does not touch the raw large JSON files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rec_event_id(rec):
    eid = rec.get('event_id')
    if eid is None:
        eid = (rec.get('chain') or {}).get('event_id')
    return eid


def key_set(paths):
    keys = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                eid = rec_event_id(rec)
                if eid is None:
                    raise SystemExit(f'{path}: record without event_id')
                keys.add((int(rec['graph_id']), str(eid)))
    return keys


def meta_for(sidecar):
    base = str(sidecar)
    if not base.endswith('.jsonl'):
        raise SystemExit(f'expected JSONL sidecar: {sidecar}')
    return Path(base[:-6] + '.meta.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher_inputs', nargs='+', required=True,
                   help='teacher JSONL sidecars defining graph_id/event_id keys')
    p.add_argument('--inputs', nargs='+', required=True,
                   help='source JSONL sidecars to filter')
    p.add_argument('--out_dir', required=True)
    a = p.parse_args()
    keys = key_set(a.teacher_inputs)
    os.makedirs(a.out_dir, exist_ok=True)
    for src in a.inputs:
        meta_path = meta_for(src)
        meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
        args = dict(meta.get('args') or {})
        kept = []
        with open(src) as f:
            for line in f:
                rec = json.loads(line)
                eid = rec_event_id(rec)
                if eid is not None and (int(rec['graph_id']), str(eid)) in keys:
                    kept.append(rec)
        name = Path(src).name[:-6] + '.teacher_subset.json'
        out = Path(a.out_dir) / name
        args['out'] = str(out)
        payload = dict(args=args, records=kept,
                       subset=dict(source=str(src), teacher_inputs=list(a.teacher_inputs),
                                   n_keys=len(keys), n_records=len(kept)))
        out.write_text(json.dumps(payload), encoding='utf-8')
        print(json.dumps(dict(out=str(out), records=len(kept), keys=len(keys)),
                         ensure_ascii=True))


if __name__ == '__main__':
    main()
