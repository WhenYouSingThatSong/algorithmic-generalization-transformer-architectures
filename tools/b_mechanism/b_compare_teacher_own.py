#!/usr/bin/env python3
"""Compare teacher-start and own-start trajectories by identity.

Reads existing JSONL sidecars; no model execution.  Keys use the public
target identity (graph, chain/target, timing, start), excluding source, so a
teacher64 record can be matched to the own record for the same target.
Outputs per model/start/timing/condition match/missing/duplicate/conflict
counts and whether the serialized rounds are identical.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_main_summary as M  # noqa: E402


def meta_for(sidecar):
    base = str(sidecar)
    if not base.endswith('.jsonl'):
        raise SystemExit(f'expected JSONL sidecar: {sidecar}')
    return Path(base[:-6] + '.meta.json')


def load_file(path):
    meta_path = meta_for(path)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    args = dict(meta.get('args') or {})
    model = M.H.model_label(args)
    return args, model


def record_identity(rec):
    """Target identity, deliberately excluding start (teacher vs own)."""
    kind = M.S.record_kind(rec)
    gid = rec.get('graph_id')
    timing = str(rec.get('timing'))
    if kind == 'early':
        eid = rec.get('event_id') or (rec.get('chain') or {}).get('event_id')
        return (kind, gid, rec.get('chain_index'), eid, timing)
    target_json = json.dumps(rec.get('target'), sort_keys=True)
    return (kind, gid, rec.get('target_index'), target_json, timing)


def add_records(paths):
    records = {}
    stats = defaultdict(lambda: dict(n=0, duplicate=0, conflict=0))
    for path in paths:
        args, model = load_file(path)
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('condition') == 'clean':
                    continue
                start, timing = M.S.record_start_timing(rec, args)
                cond = str(rec.get('condition'))
                key = (model, timing, cond, record_identity(rec))
                s = stats[(model, timing, cond, start)]
                s['n'] += 1
                if key in records:
                    s['duplicate'] += 1
                    if json.dumps(records[key].get('rounds'), sort_keys=True) != \
                            json.dumps(rec.get('rounds'), sort_keys=True):
                        s['conflict'] += 1
                    continue
                records[key] = rec
    return records, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher_inputs', nargs='+', required=True)
    p.add_argument('--own_inputs', nargs='+', required=True)
    p.add_argument('--out_csv', required=True)
    a = p.parse_args()
    teacher, tstats = add_records(a.teacher_inputs)
    own, ostats = add_records(a.own_inputs)
    groups = sorted({k[:3] for k in set(tstats) | set(ostats)})
    rows = []
    for g in groups:
        tkeys = {k for k in teacher if k[:3] == g}
        okeys = {k for k in own if k[:3] == g}
        inter = tkeys & okeys
        identical = sum(
            json.dumps(teacher[k].get('rounds'), sort_keys=True) ==
            json.dumps(own[k].get('rounds'), sort_keys=True)
            for k in inter)
        tstates = [k for k in tstats if k[:3] == g]
        ostates = [k for k in ostats if k[:3] == g]
        rows.append(dict(
            model=g[0], timing=g[1], condition=g[2],
            teacher_start=','.join(sorted({k[3] for k in tstates})),
            own_start=','.join(sorted({k[3] for k in ostates})),
            teacher_n=len(tkeys), own_n=len(okeys),
            intersection=len(inter),
            missing_in_own=len(tkeys - okeys),
            missing_in_teacher=len(okeys - tkeys),
            duplicate_teacher=sum(tstats[k]['duplicate'] for k in tstates),
            duplicate_own=sum(ostats[k]['duplicate'] for k in ostates),
            conflict_teacher=sum(tstats[k]['conflict'] for k in tstates),
            conflict_own=sum(ostats[k]['conflict'] for k in ostates),
            rounds_identical=identical,
            rounds_different=len(inter) - identical,
        ))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(rows=len(rows), teacher=len(teacher), own=len(own),
                          out=a.out_csv), ensure_ascii=True))


if __name__ == '__main__':
    main()