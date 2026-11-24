#!/usr/bin/env python3
"""Audit exported five-case evidence, using dataset-derived edges (CPU only).

ADD-only arithmetic check: compare each damaged output to its actual INPUT
parent values. This is a behavioral consistency check, not an attribution of
the model's internal algorithm. Missing/non-numeric parents remain unclassified.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def reachable(children, start):
    seen, queue = {start}, [start]
    for pos in queue:
        for child in children[pos]:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def value_at(round_record, side, pos):
    return round_record.get(side, {}).get(str(pos), {}).get('pred4', {}).get('VALUE')


def audit(evidence, modulus):
    graphs = {g['graph_id']: {int(p): n for p, n in g['nodes'].items()}
              for g in evidence['dataset_graphs']}
    rows, summaries = [], []
    for pair in evidence['pairs']:
        rec, clean = pair['injected'], pair['clean']
        gid = rec['graph_id']
        graph = graphs[gid]
        if any(op != 0 for n in graph.values() for op in n['operations']):
            raise ValueError('only ADD operation ID 0 is supported')
        children = defaultdict(list)
        for pos, node in graph.items():
            for parent in node['parents']:
                children[parent].append(pos)
        truth = rec['node_truth']
        differences = []
        if {int(p) for p in truth} != set(graph):
            differences.append('node_set')
        for pos, node in graph.items():
            old = truth.get(str(pos), {})
            if (set(p for p in node['parents'] if p in graph) != set(old.get('parents', []))
                    or set(children[pos]) != set(old.get('children', []))):
                differences.append(pos)
        injection = rec['injection']['pos']
        descendants = reachable(children, injection)
        exclusive = (reachable(children, rec['target']['for_u']) - descendants
                     if rec['condition'] == 'unrelated' else set())
        if len(rec['rounds']) != len(clean['rounds']):
            raise ValueError('unequal windows')
        damaged, exclusive_damaged = set(), set()
        for index, (rr, cr) in enumerate(zip(rec['rounds'], clean['rounds']), 1):
            for pos in sorted(graph):
                out = rr['output_nodes'][str(pos)]
                cout = cr['output_nodes'][str(pos)]
                expected_truth = out['true4']['VALUE']
                if cout['pred4']['VALUE'] != expected_truth or out['pred4']['VALUE'] == expected_truth:
                    continue
                damaged.add(pos)
                if pos in exclusive:
                    exclusive_damaged.add(pos)
                parents = graph[pos]['parents']
                parent_values = [value_at(rr, 'input_nodes', p) for p in parents]
                numeric = bool(parents) and all(isinstance(v, str) and v.isdigit() for v in parent_values)
                expected = str(sum(map(int, parent_values)) % modulus) if numeric else None
                rows.append(dict(graph_id=gid, condition=rec['condition'],
                                 target_index=rec['target_index'], injection_pos=injection,
                                 node_pos=pos, round=index, reachable_from_injection=pos in descendants,
                                 protected_exclusive=pos in exclusive,
                                 parents=json.dumps(parents), parent_input_values=json.dumps(parent_values),
                                 true_value=expected_truth,
                                 input_value=value_at(rr, 'input_nodes', pos),
                                 output_value=value_at(rr, 'output_nodes', pos),
                                 clean_output=value_at(cr, 'output_nodes', pos),
                                 expected_from_input_parents=expected,
                                 matches_parent_expression=(out['pred4']['VALUE'] == expected) if numeric else None))
        summaries.append(dict(case=pair['case'], checked_nodes=len(graph),
                              structure_differences=differences, damaged_nodes=sorted(damaged),
                              protected_exclusive_damaged_nodes=sorted(exclusive_damaged)))
    return rows, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--modulus', type=int, required=True,
                   help='must match the source model/data configuration')
    a = p.parse_args()
    if a.modulus < 2:
        raise ValueError('invalid modulus')
    raw = a.input.read_bytes()
    rows, cases = audit(json.loads(raw), a.modulus)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    with (a.out_dir / 'damaged-rounds.csv').open('w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (a.out_dir / 'audit.json').write_text(json.dumps(dict(
        input=str(a.input), input_sha256=hashlib.sha256(raw).hexdigest(),
        modulus=a.modulus, operation='ADD (ID 0)', cases=cases,
        damaged_node_rounds=len(rows)), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(cases, ensure_ascii=False))


if __name__ == '__main__':
    main()
