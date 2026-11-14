#!/usr/bin/env python3
"""Create a fixed graph_id-range subset of an existing selection manifest.

No model is loaded and no graph is re-selected.  This is only a manifest view
for the teacher-64 and long-horizon-32 runs; it never tops up outside the
requested range.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--start_graph', type=int, required=True)
    p.add_argument('--end_graph', type=int, required=True,
                   help='exclusive upper graph_id')
    p.add_argument('--out', required=True)
    p.add_argument('--allow_empty', action='store_true')
    a = p.parse_args()
    if a.start_graph < 0 or a.end_graph <= a.start_graph:
        p.error('require 0 <= start_graph < end_graph')

    with open(a.manifest) as f:
        parent = json.load(f)
    graphs = [g for g in parent.get('graphs', [])
              if a.start_graph <= int(g['graph_id']) < a.end_graph]
    if not graphs and not a.allow_empty:
        raise SystemExit(
            f'no selected graphs in [{a.start_graph}, {a.end_graph}); '
            'refusing to write an empty subset (use --allow_empty to override)')

    out = dict(parent)
    out['subset_of'] = os.path.abspath(a.manifest)
    out['subset_start_graph'] = a.start_graph
    out['subset_end_graph'] = a.end_graph
    out['n_graphs'] = len(graphs)
    out['graphs'] = graphs
    if 'candidates' in parent:
        out['candidates'] = [
            c for c in parent['candidates']
            if a.start_graph <= int(c['graph_id']) < a.end_graph]
    out['parent_counts'] = parent.get('counts', {})
    cs = out.get('candidates', graphs)
    counts = dict(scanned=len(cs), candidates_total=len(cs),
                  common_complete=sum(bool(g.get('common_complete', g.get('T') is not None)) for g in cs),
                  not_common_complete=sum(not bool(g.get('common_complete', g.get('T') is not None)) for g in cs),
                  candidates_with_chains=sum(bool(g.get('chains')) for g in graphs))
    for key, getter in (
            ('chains', lambda g: g.get('chains', [])),
            ('sinks', lambda g: (g.get('late') or {}).get('sinks', [])),
            ('leaves', lambda g: (g.get('late') or {}).get('leaves', [])),
            ('unrelated', lambda g: g.get('unrelated', []))):
        counts[key] = sum(len(getter(g)) for g in graphs)
        counts['subset_' + key] = counts[key]
    counts['subset_graphs'] = len(graphs)
    for key in ('early_own_before', 'early_own_after', 'early_teacher_before', 'early_teacher_after'):
        counts[key] = sum(bool(ch.get(key)) for g in graphs for ch in g.get('chains', []))
        counts[key + '_graphs'] = sum(any(ch.get(key) for ch in g.get('chains', [])) for g in graphs)
    out['counts'] = counts

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(dict(subset_graphs=len(graphs),
                          subset_chains=counts['subset_chains'],
                          subset_sinks=counts['subset_sinks'],
                          subset_leaves=counts['subset_leaves'],
                          subset_unrelated=counts['subset_unrelated'],
                          out=a.out), indent=2))


if __name__ == '__main__':
    main()
