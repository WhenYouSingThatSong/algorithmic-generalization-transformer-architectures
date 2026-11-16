#!/usr/bin/env python3
"""Export complete discrete chains for B wrong_plus_11 residual cases.

CPU-only.  The input is a completed `run_b.py --multi_parent` records.jsonl.
The selection rule is fixed: fixed-candidate records whose target is correct,
whose unchanged branch is correct in the last two outputs, and whose
wrong_plus_11 branch leaves at least one descendant wrong in both last two
outputs.  For each selected target, export target, ancestors, descendants and
their direct parents with paired unchanged/wrong_plus_11 VALUE reads for all
16 rounds.  No model execution and no mechanism interpretation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

VALUE_INDEX = 3
CLEAN_BRANCH = 'unchanged'
WRONG_BRANCH = 'wrong_plus_11'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_records(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def parent_map(rec):
    raw = rec.get('all_parents') or {}
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


def child_map(parents):
    children = defaultdict(list)
    for child, ps in parents.items():
        for p in ps:
            children[p].append(child)
    return {k: sorted(v) for k, v in children.items()}


def values_ok(rows, index, truth):
    return [row[index][VALUE_INDEX] == truth[index][VALUE_INDEX] for row in rows]


def node_status(clean_outputs, wrong_outputs, index, truth):
    clean_ok = values_ok(clean_outputs, index, truth)
    wrong_ok = values_ok(wrong_outputs, index, truth)
    n = len(clean_ok)
    damage = [t + 1 for t in range(n) if clean_ok[t] and not wrong_ok[t]]
    persistent_start = None
    for t in range(n):
        # Only count a persistent run when the unchanged peer is correct
        # throughout it; a clean branch that is still EMPTY/wrong is not
        # evidence of intervention-specific persistence.
        if all(clean_ok[k] and not wrong_ok[k] for k in range(t, n)):
            persistent_start = t + 1
            break
    final_two_clean = n >= 2 and all(clean_ok[-2:])
    final_two_wrong = final_two_clean and all(not x for x in wrong_ok[-2:])
    return dict(clean_ok=clean_ok, wrong_ok=wrong_ok,
                damaged_any=bool(damage),
                first_damage_round=damage[0] if damage else None,
                persistent_start=persistent_start,
                final_two_clean_correct=final_two_clean,
                final_two_wrong=final_two_wrong)


def target_cases(records):
    found = {}
    for rec in records:
        if rec.get('cohort') != 'fixed_candidates':
            continue
        branches = rec.get('branches') or {}
        if CLEAN_BRANCH not in branches or WRONG_BRANCH not in branches:
            continue
        pos = rec['node_positions']
        truth = rec['truth4']
        clean = branches[CLEAN_BRANCH]['outputs']
        wrong = branches[WRONG_BRANCH]['outputs']
        hit = []
        for p in rec.get('descendant_positions', []):
            if p not in pos:
                continue
            st = node_status(clean, wrong, pos.index(p), truth)
            if st['final_two_wrong']:
                hit.append(p)
        if hit:
            found[(rec['graph_id'], rec['target_pos'])] = dict(
                target_pos=rec['target_pos'], hit=hit, record=rec)
    return found


def scope_nodes(rec, parents):
    target = rec['target_pos']
    nodes = {target} | set(rec.get('parent_positions', []))
    nodes |= set(rec.get('descendant_positions', []))
    pending = list(nodes)
    while pending:
        p = pending.pop()
        for q in parents.get(p, []):
            if q not in nodes:
                nodes.add(q)
                pending.append(q)
    return sorted(nodes)


def node_role(rec, p, status):
    if p == rec['target_pos']:
        return 'target'
    if p in set(rec.get('parent_positions', [])):
        return 'target_parent'
    if p in set(rec.get('descendant_positions', [])):
        if status['persistent_start'] is not None:
            return 'persistent_descendant'
        if status['final_two_wrong']:
            return 'final_wrong_descendant'
        if status['damaged_any']:
            return 'transient_descendant'
        return 'undamaged_descendant'
    return 'input_ancestor'


def build_rows(records, selected):
    rows = []
    summaries = []
    for rec in records:
        key = (rec['graph_id'], rec['target_pos'])
        if rec.get('cohort') != 'fixed_candidates' or key not in selected:
            continue
        branches = rec.get('branches') or {}
        if CLEAN_BRANCH not in branches or WRONG_BRANCH not in branches:
            continue
        pos = rec['node_positions']
        truth = rec['truth4']
        parents = parent_map(rec)
        children = child_map(parents)
        clean_branch = branches[CLEAN_BRANCH]
        wrong_branch = branches[WRONG_BRANCH]
        clean_outputs = clean_branch['outputs']
        wrong_outputs = wrong_branch['outputs']
        if len(clean_outputs) != len(wrong_outputs) or len(clean_outputs) < 2:
            raise ValueError(f'unpaired branch windows: {rec.get("case_id")}')
        nodes = scope_nodes(rec, parents)
        statuses = {p: node_status(clean_outputs, wrong_outputs, pos.index(p), truth)
                    for p in nodes}
        target = rec['target_pos']
        ti = pos.index(target)
        damaged = [p for p in rec.get('descendant_positions', []) if p in pos
                   and statuses[p]['damaged_any']]
        persistent = [p for p in damaged if statuses[p]['persistent_start'] is not None]
        final_wrong = [p for p in damaged if statuses[p]['final_two_wrong']]
        first_damage = min((statuses[p]['first_damage_round'] for p in damaged
                            if statuses[p]['first_damage_round'] is not None), default=None)
        first_persistent = min((statuses[p]['persistent_start'] for p in persistent), default=None)
        first_damage_nodes = sorted(p for p in damaged
                                    if statuses[p]['first_damage_round'] == first_damage)
        first_persistent_nodes = sorted(p for p in persistent
                                       if statuses[p]['persistent_start'] == first_persistent)
        depths = rec.get('depths') or {}
        def depth(p):
            return int(depths[str(p)]) if str(p) in depths else None
        summaries.append(dict(
            case_id=rec.get('case_id'), timing=rec.get('timing'), graph_id=rec['graph_id'],
            target_pos=target, true_value=truth[ti][VALUE_INDEX],
            wrong_value=wrong_branch.get('assigned_value'), fan_in=rec.get('fan_in'),
            state_round=rec.get('state_round'),
            n_scope_nodes=len(nodes), n_descendants=len(rec.get('descendant_positions', [])),
            n_damaged_any=len(damaged), n_persistent=len(persistent),
            n_transient=len(damaged) - len(final_wrong), n_final_two_wrong=len(final_wrong),
            first_damage_round=first_damage,
            first_damage_nodes=';'.join(map(str, first_damage_nodes)),
            first_damage_depths=';'.join(str(depth(p)) for p in first_damage_nodes),
            first_persistent_round=first_persistent,
            first_persistent_nodes=';'.join(map(str, first_persistent_nodes)),
            first_persistent_depths=';'.join(str(depth(p)) for p in first_persistent_nodes),
            target_initial_clean_value=clean_branch['initial'][ti][VALUE_INDEX],
            target_initial_wrong_value=wrong_branch['initial'][ti][VALUE_INDEX],
            target_first_output_clean_value=clean_outputs[0][ti][VALUE_INDEX],
            target_first_output_wrong_value=wrong_outputs[0][ti][VALUE_INDEX],
            target_first_output_correct_in_wrong=(
                wrong_outputs[0][ti][VALUE_INDEX] == truth[ti][VALUE_INDEX]),
            target_final_two_clean=all(x[ti][VALUE_INDEX] == truth[ti][VALUE_INDEX]
                                       for x in clean_outputs[-2:]),
            target_final_two_correct_in_wrong=all(
                x[ti][VALUE_INDEX] == truth[ti][VALUE_INDEX] for x in wrong_outputs[-2:]),
            clean_all_final_two_correct=all(
                statuses[p]['final_two_clean_correct'] for p in nodes)))
        for p in nodes:
            i = pos.index(p)
            st = statuses[p]
            for t in range(1, len(clean_outputs) + 1):
                ci = clean_branch['initial'][i] if t == 1 else clean_outputs[t - 2][i]
                wi = wrong_branch['initial'][i] if t == 1 else wrong_outputs[t - 2][i]
                co = clean_outputs[t - 1][i]
                wo = wrong_outputs[t - 1][i]
                rows.append(dict(
                    case_id=rec.get('case_id'), timing=rec.get('timing'),
                    graph_id=rec['graph_id'], target_pos=target,
                    true_value=truth[ti][VALUE_INDEX],
                    wrong_value=wrong_branch.get('assigned_value'),
                    fan_in=rec.get('fan_in'), state_round=rec.get('state_round'),
                    node_pos=p, node_depth=depth(p), node_role=node_role(rec, p, st),
                    parents=';'.join(map(str, parents.get(p, []))),
                    children=';'.join(map(str, children.get(p, []))),
                    initial_clean=clean_branch['initial'][i][VALUE_INDEX],
                    initial_wrong=wrong_branch['initial'][i][VALUE_INDEX],
                    round=t,
                    clean_input=ci[VALUE_INDEX], wrong_input=wi[VALUE_INDEX],
                    clean_output=co[VALUE_INDEX], wrong_output=wo[VALUE_INDEX],
                    clean_input_correct=ci[VALUE_INDEX] == truth[i][VALUE_INDEX],
                    wrong_input_correct=wi[VALUE_INDEX] == truth[i][VALUE_INDEX],
                    clean_output_correct=co[VALUE_INDEX] == truth[i][VALUE_INDEX],
                    wrong_output_correct=wo[VALUE_INDEX] == truth[i][VALUE_INDEX],
                    is_damage_round=(co[VALUE_INDEX] == truth[i][VALUE_INDEX]
                                     and wo[VALUE_INDEX] != truth[i][VALUE_INDEX]),
                    damaged_any=st['damaged_any'],
                    first_damage_round=st['first_damage_round'],
                    persistent_start=st['persistent_start'],
                    final_two_clean_correct=st['final_two_clean_correct'],
                    final_two_wrong=st['final_two_wrong']))
    return rows, summaries


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'refusing to write empty table: {path}')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    a = p.parse_args()
    records = load_records(a.records)
    selected = target_cases(records)
    if len(selected) != 5:
        raise SystemExit(f'expected 5 residual targets, found {len(selected)}: '
                         f'{sorted(selected)}')
    rows, summaries = build_rows(records, selected)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(a.out_dir / 'b-multiparent-residual-chains.csv', rows)
    write_csv(a.out_dir / 'b-multiparent-residual-summary.csv', summaries)
    meta = dict(
        source=str(a.records), source_sha256=sha256(a.records),
        script_sha256=sha256(__file__),
        selection=('fixed_candidates; target correct; unchanged final-two correct; '
                   'wrong_plus_11 leaves >=1 descendant wrong in final two'),
        value_index=VALUE_INDEX,
        target_cases=sorted(f'{g}:{t}' for g, t in selected),
        chain_rows=len(rows), summary_rows=len(summaries),
        notes=['Paired unchanged (clean) and wrong_plus_11 discrete VALUE reads.',
               'round=1 input is the branch initial state; later input is previous output.',
               'first_persistent_round is the first round after which the node remains '
               'wrong (while its unchanged branch remains correct).'])
    with open(a.out_dir / 'b-multiparent-residual-meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(targets=len(selected), rows=len(rows),
                          summaries=len(summaries), out=str(a.out_dir)),
                     ensure_ascii=True))


if __name__ == '__main__':
    main()
